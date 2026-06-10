import asyncio
from dataclasses import dataclass

from futures_lab.audit import AuditLog
from futures_lab.binance_streams import BinanceStreamRecorder
from futures_lab.candidate_outcomes import CandidateOutcomeTracker
from futures_lab.config import Settings
from futures_lab.context_polling import BinanceContextPoller
from futures_lab.market_state import MarketStateBook
from futures_lab.models import Decision, MarketState, PaperState, RiskVerdict
from futures_lab.paper import PaperBroker
from futures_lab.recon_log import ReconLogger
from futures_lab.regime_outcomes import RegimeOutcomeTracker
from futures_lab.risk import RiskEngine
from futures_lab.shadow import ShadowTradeTracker
from futures_lab.strategy import HitAndRunStrategy


@dataclass
class TradingRuntime:
    settings: Settings
    audit: AuditLog
    state_book: MarketStateBook
    recorder: BinanceStreamRecorder
    context: BinanceContextPoller
    strategy: HitAndRunStrategy
    risk: RiskEngine
    paper: PaperBroker
    shadow: ShadowTradeTracker
    regime_outcomes: RegimeOutcomeTracker
    candidate_outcomes: CandidateOutcomeTracker
    recon_log: ReconLogger

    def __post_init__(self) -> None:
        self.latest_market: MarketState | None = None
        self.latest_decision: Decision | None = None
        self.latest_risk: RiskVerdict | None = None
        self._task: asyncio.Task | None = None
        self._stop: asyncio.Event | None = None

    @classmethod
    def create(cls, settings: Settings) -> "TradingRuntime":
        audit = AuditLog(settings)
        state_book = MarketStateBook(settings)
        return cls(
            settings=settings,
            audit=audit,
            state_book=state_book,
            recorder=BinanceStreamRecorder(settings=settings, state=state_book, audit=audit),
            context=BinanceContextPoller(settings=settings, state=state_book, audit=audit),
            strategy=HitAndRunStrategy(settings),
            risk=RiskEngine(settings),
            paper=PaperBroker(settings),
            shadow=ShadowTradeTracker(settings),
            regime_outcomes=RegimeOutcomeTracker(settings),
            candidate_outcomes=CandidateOutcomeTracker(settings),
            recon_log=ReconLogger(settings),
        )

    def is_running(self) -> bool:
        return self.recorder.is_running()

    def start(self) -> None:
        self.recorder.start()
        self.context.start()
        if self._task is None or self._task.done():
            self._stop = asyncio.Event()
            self._task = asyncio.create_task(self._decision_loop(), name="decision-loop")
        self.audit.write("runtime_start", {"symbol": self.settings.symbol.upper()})

    async def stop(self) -> None:
        if self._stop is not None:
            self._stop.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self.recorder.stop()
        await self.context.stop()
        for event in self.shadow.close_all(self.latest_market, reason="session_end"):
            self.audit.write("shadow_trade", event)
            self.recon_log.write_shadow_trade(event)
        for event in self.regime_outcomes.close_all(self.latest_market, reason="session_end"):
            self.audit.write("regime_outcome", event)
            self.recon_log.write_regime_outcome(event)
        for event in self.candidate_outcomes.close_all(self.latest_market, reason="session_end"):
            self.audit.write("candidate_outcome", event)
            self.recon_log.write_candidate_outcome(event)
        self.audit.write("runtime_stop", {"symbol": self.settings.symbol.upper()})

    def market(self) -> MarketState:
        self.latest_market = self.state_book.snapshot()
        return self.latest_market

    def decide_once(self) -> tuple[MarketState, Decision, RiskVerdict]:
        market = self.market()
        decision = self.strategy.decide(market)
        risk = self.risk.evaluate(
            decision,
            market,
            self.paper.state(),
            candidate_quality=self.candidate_outcomes.live_edge_quality_snapshot(),
        )
        self.latest_decision = decision
        self.latest_risk = risk
        return market, decision, risk

    def paper_state(self) -> PaperState:
        return self.paper.state()

    async def _decision_loop(self) -> None:
        assert self._stop is not None
        interval = max(0.1, self.settings.decision_interval_ms / 1000)
        while not self._stop.is_set():
            loop_started = asyncio.get_running_loop().time()
            market, decision, risk = self.decide_once()
            decision_latency_ms = (asyncio.get_running_loop().time() - loop_started) * 1000
            self.recon_log.write_feature(market)
            self.recon_log.write_decision(market, decision, risk, decision_latency_ms=decision_latency_ms)
            closed = self.paper.mark(market)
            if closed is not None:
                self.audit.write("paper_close", closed.model_dump())
                self.recon_log.write_paper_trade(closed)
            for event in self.shadow.mark(market):
                self.audit.write("shadow_trade", event)
                self.recon_log.write_shadow_trade(event)
            for event in self.regime_outcomes.mark(market):
                self.audit.write("regime_outcome", event)
                self.recon_log.write_regime_outcome(event)
            for event in self.candidate_outcomes.mark(market):
                self.audit.write("candidate_outcome", event)
                self.recon_log.write_candidate_outcome(event)
            if risk.allowed:
                opened = self.paper.open_from_decision(decision)
                if opened is not None:
                    self.audit.write(
                        "paper_open",
                        {
                            "symbol": opened.symbol,
                            "side": opened.side.value,
                            "mode": opened.mode.value,
                            "trade_profile": opened.trade_profile,
                            "exit_policy": opened.exit_policy,
                            "entry_price": opened.entry_price,
                            "take_profit_price": opened.take_profit_price,
                            "stop_loss_price": opened.stop_loss_price,
                            "confidence": opened.confidence,
                            "leverage": opened.leverage,
                            "notional_usd": opened.notional_usd,
                        },
                    )
            shadow_opened = self.shadow.open_from_decision(decision, opened_at=market.last_received_at)
            if shadow_opened is not None:
                self.audit.write("shadow_trade", shadow_opened)
                self.recon_log.write_shadow_trade(shadow_opened)
            regime_opened = self.regime_outcomes.open_from_decision(decision, market, opened_at=market.last_received_at)
            if regime_opened is not None:
                self.audit.write("regime_outcome", regime_opened)
                self.recon_log.write_regime_outcome(regime_opened)
            for event in self.candidate_outcomes.open_from_decision(decision, market, opened_at=market.last_received_at):
                self.audit.write("candidate_outcome", event)
                self.recon_log.write_candidate_outcome(event)
            await asyncio.sleep(interval)
