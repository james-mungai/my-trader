import asyncio
from dataclasses import dataclass

from futures_lab.audit import AuditLog
from futures_lab.binance_streams import BinanceStreamRecorder
from futures_lab.config import Settings
from futures_lab.market_state import MarketStateBook
from futures_lab.models import Decision, MarketState, PaperState, RiskVerdict
from futures_lab.paper import PaperBroker
from futures_lab.recon_log import ReconLogger
from futures_lab.risk import RiskEngine
from futures_lab.strategy import HitAndRunStrategy


@dataclass
class TradingRuntime:
    settings: Settings
    audit: AuditLog
    state_book: MarketStateBook
    recorder: BinanceStreamRecorder
    strategy: HitAndRunStrategy
    risk: RiskEngine
    paper: PaperBroker
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
            strategy=HitAndRunStrategy(settings),
            risk=RiskEngine(settings),
            paper=PaperBroker(settings),
            recon_log=ReconLogger(settings),
        )

    def is_running(self) -> bool:
        return self.recorder.is_running()

    def start(self) -> None:
        self.recorder.start()
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
        self.audit.write("runtime_stop", {"symbol": self.settings.symbol.upper()})

    def market(self) -> MarketState:
        self.latest_market = self.state_book.snapshot()
        return self.latest_market

    def decide_once(self) -> tuple[MarketState, Decision, RiskVerdict]:
        market = self.market()
        decision = self.strategy.decide(market)
        risk = self.risk.evaluate(decision, market, self.paper.state())
        self.latest_decision = decision
        self.latest_risk = risk
        return market, decision, risk

    def paper_state(self) -> PaperState:
        return self.paper.state()

    async def _decision_loop(self) -> None:
        assert self._stop is not None
        interval = max(0.1, self.settings.decision_interval_ms / 1000)
        while not self._stop.is_set():
            market, decision, risk = self.decide_once()
            self.recon_log.write_feature(market)
            self.recon_log.write_decision(market, decision, risk)
            closed = self.paper.mark(market)
            if closed is not None:
                self.audit.write("paper_close", closed.model_dump())
                self.recon_log.write_paper_trade(closed)
            if risk.allowed:
                opened = self.paper.open_from_decision(decision)
                if opened is not None:
                    self.audit.write(
                        "paper_open",
                        {
                            "symbol": opened.symbol,
                            "side": opened.side.value,
                            "mode": opened.mode.value,
                            "entry_price": opened.entry_price,
                            "take_profit_price": opened.take_profit_price,
                            "stop_loss_price": opened.stop_loss_price,
                            "confidence": opened.confidence,
                            "leverage": opened.leverage,
                            "notional_usd": opened.notional_usd,
                        },
                    )
            await asyncio.sleep(interval)
