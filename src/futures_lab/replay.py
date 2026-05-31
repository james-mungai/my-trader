import gzip
import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator, TextIO

from futures_lab.audit import AuditLog
from futures_lab.candidate_outcomes import CandidateOutcomeTracker
from futures_lab.config import Settings
from futures_lab.cross_market import build_cross_market_context
from futures_lab.hostile_replay import HostileReplayBroker, HostileReplayStats
from futures_lab.market_state import MarketStateBook
from futures_lab.models import DecisionAction, MarketState, PaperPosition, PaperTrade, Side
from futures_lab.paper import PaperBroker
from futures_lab.regime_outcomes import RegimeOutcomeTracker
from futures_lab.risk import RiskEngine
from futures_lab.shadow import ShadowTradeTracker
from futures_lab.strategy import HitAndRunStrategy


@dataclass(frozen=True)
class ReplayMessage:
    received_at: datetime
    payload: dict


@dataclass
class ReplayPositionStats:
    symbol: str
    side: str
    mode: str
    entry_price: float
    opened_at: datetime
    status: str
    last_price: float | None = None
    last_seen_at: datetime | None = None
    closed_at: datetime | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    unrealized_pnl_usd: float = 0.0
    net_unrealized_pnl_usd: float = 0.0
    max_favorable_move_pct: float = 0.0
    max_adverse_move_pct: float = 0.0
    max_favorable_pnl_usd: float = 0.0
    max_adverse_pnl_usd: float = 0.0
    taker_fee_bps: float = 4.0

    @classmethod
    def from_position(cls, position: PaperPosition, settings: Settings) -> "ReplayPositionStats":
        return cls(
            symbol=position.symbol,
            side=position.side.value,
            mode=position.mode.value,
            entry_price=position.entry_price,
            opened_at=position.opened_at,
            status="open",
            taker_fee_bps=settings.taker_fee_bps,
        )

    @property
    def time_in_trade_seconds(self) -> float | None:
        end = self.closed_at or self.last_seen_at
        if end is None:
            return None
        return (end - self.opened_at).total_seconds()

    def update(self, position: PaperPosition, market: MarketState) -> None:
        if market.mid_price is None:
            return
        self.last_price = market.mid_price
        self.last_seen_at = market.last_received_at
        gross = _gross_pnl(position, market.mid_price)
        net = gross - _round_trip_fees(position, self.taker_fee_bps)
        move = _underlying_move_pct(position, market.mid_price)
        self.unrealized_pnl_usd = gross
        self.net_unrealized_pnl_usd = net
        self.max_favorable_pnl_usd = max(self.max_favorable_pnl_usd, gross)
        self.max_adverse_pnl_usd = min(self.max_adverse_pnl_usd, gross)
        self.max_favorable_move_pct = max(self.max_favorable_move_pct, move)
        self.max_adverse_move_pct = min(self.max_adverse_move_pct, move)

    def close(self, trade: PaperTrade, status: str = "closed") -> None:
        self.status = status
        self.closed_at = trade.closed_at
        self.exit_price = trade.exit_price
        self.exit_reason = trade.exit_reason
        self.unrealized_pnl_usd = trade.gross_pnl_usd
        self.net_unrealized_pnl_usd = trade.net_pnl_usd

    def model_dump(self) -> dict:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "mode": self.mode,
            "entry_price": self.entry_price,
            "opened_at": self.opened_at,
            "status": self.status,
            "last_price": self.last_price,
            "last_seen_at": self.last_seen_at,
            "closed_at": self.closed_at,
            "exit_price": self.exit_price,
            "exit_reason": self.exit_reason,
            "time_in_trade_seconds": self.time_in_trade_seconds,
            "unrealized_pnl_usd": self.unrealized_pnl_usd,
            "net_unrealized_pnl_usd": self.net_unrealized_pnl_usd,
            "max_favorable_move_pct": self.max_favorable_move_pct,
            "max_adverse_move_pct": self.max_adverse_move_pct,
            "max_favorable_pnl_usd": self.max_favorable_pnl_usd,
            "max_adverse_pnl_usd": self.max_adverse_pnl_usd,
        }


@dataclass
class ReplaySummary:
    files: list[str]
    replay_mode: str = "normal"
    messages: int = 0
    decisions: int = 0
    proposals: int = 0
    risk_allowed: int = 0
    paper_opens: int = 0
    paper_closes: int = 0
    net_pnl_usd: float = 0.0
    trades: list[PaperTrade] = field(default_factory=list)
    position_stats: list[ReplayPositionStats] = field(default_factory=list)
    open_position: ReplayPositionStats | None = None
    unrealized_pnl_usd: float = 0.0
    net_unrealized_pnl_usd: float = 0.0
    markov: dict = field(default_factory=dict)
    shadow_opens: int = 0
    shadow_closes: int = 0
    shadow_net_pnl_usd: float = 0.0
    shadow_events: list[dict] = field(default_factory=list)
    regime_outcome_opens: int = 0
    regime_outcome_closes: int = 0
    regime_outcome_events: list[dict] = field(default_factory=list)
    candidate_outcome_opens: int = 0
    candidate_outcome_closes: int = 0
    candidate_outcome_events: list[dict] = field(default_factory=list)
    hostile_replay: HostileReplayStats | None = None

    @property
    def wins(self) -> int:
        return sum(1 for trade in self.trades if trade.net_pnl_usd > 0)

    @property
    def losses(self) -> int:
        return sum(1 for trade in self.trades if trade.net_pnl_usd <= 0)

    @property
    def win_rate(self) -> float | None:
        if not self.trades:
            return None
        return self.wins / len(self.trades)

    def model_dump(self) -> dict:
        return {
            "files": self.files,
            "replay_mode": self.replay_mode,
            "messages": self.messages,
            "decisions": self.decisions,
            "proposals": self.proposals,
            "risk_allowed": self.risk_allowed,
            "paper_opens": self.paper_opens,
            "paper_closes": self.paper_closes,
            "net_pnl_usd": self.net_pnl_usd,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.win_rate,
            "trades": [trade.model_dump() for trade in self.trades],
            "position_stats": [stats.model_dump() for stats in self.position_stats],
            "open_position": self.open_position.model_dump() if self.open_position else None,
            "unrealized_pnl_usd": self.unrealized_pnl_usd,
            "net_unrealized_pnl_usd": self.net_unrealized_pnl_usd,
            "markov": self.markov,
            "shadow_opens": self.shadow_opens,
            "shadow_closes": self.shadow_closes,
            "shadow_net_pnl_usd": self.shadow_net_pnl_usd,
            "shadow_events": self.shadow_events,
            "regime_outcome_opens": self.regime_outcome_opens,
            "regime_outcome_closes": self.regime_outcome_closes,
            "regime_outcome_events": self.regime_outcome_events,
            "candidate_outcome_opens": self.candidate_outcome_opens,
            "candidate_outcome_closes": self.candidate_outcome_closes,
            "candidate_outcome_events": self.candidate_outcome_events,
            "hostile_replay": self.hostile_replay.model_dump() if self.hostile_replay else None,
        }


def _gross_pnl(position: PaperPosition, exit_price: float) -> float:
    direction = 1 if position.side == Side.long else -1
    return (exit_price - position.entry_price) * position.quantity * direction


def _round_trip_fees(position: PaperPosition, taker_fee_bps: float) -> float:
    return position.notional_usd * 2 * (taker_fee_bps / 10_000)


def _underlying_move_pct(position: PaperPosition, price: float) -> float:
    direction = 1 if position.side == Side.long else -1
    return ((price - position.entry_price) / position.entry_price) * direction


def discover_raw_files(settings: Settings, pattern: str | None = None) -> list[Path]:
    raw_dir = Path(settings.data_dir) / "raw_ws"
    if pattern is not None:
        return sorted(raw_dir.glob(pattern))
    symbols = [settings.symbol.upper()]
    anchor = settings.cross_market_anchor_symbol.upper()
    if settings.cross_market_enabled and anchor not in symbols:
        symbols.append(anchor)
    files = []
    for symbol in symbols:
        files.extend(raw_dir.glob(f"{symbol}_*_*.jsonl"))
        files.extend(raw_dir.glob(f"{symbol}_*_*.jsonl.gz"))
    return sorted(files)


@contextmanager
def _open_text(path: Path) -> Iterator[TextIO]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            yield handle
        return
    with path.open("r", encoding="utf-8") as handle:
        yield handle


def load_messages(
    paths: Iterable[Path],
    include_depth: bool = False,
    book_ticker_min_interval_ms: int = 100,
) -> list[ReplayMessage]:
    messages: list[ReplayMessage] = []
    last_book_ticker_at: datetime | None = None
    for path in paths:
        with _open_text(path) as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                received_at = datetime.fromisoformat(row["received_at"])
                envelope = row["message"]
                payload = envelope.get("data", envelope)
                event_type = payload.get("e")
                if event_type == "depthUpdate" and not include_depth:
                    continue
                if event_type == "bookTicker" and last_book_ticker_at is not None:
                    elapsed_ms = (received_at - last_book_ticker_at).total_seconds() * 1000
                    if elapsed_ms < book_ticker_min_interval_ms:
                        continue
                if event_type == "bookTicker":
                    last_book_ticker_at = received_at
                messages.append(ReplayMessage(received_at=received_at, payload=payload))
    return sorted(messages, key=lambda msg: msg.received_at)


def replay_files(
    settings: Settings,
    paths: Iterable[Path],
    decision_interval_ms: int | None = None,
    include_depth: bool = False,
    book_ticker_min_interval_ms: int = 100,
    flatten_at_end: bool = False,
    audit: AuditLog | None = None,
    hostile: bool | None = None,
) -> ReplaySummary:
    path_list = [Path(path) for path in paths]
    hostile_enabled = settings.hostile_replay_enabled if hostile is None else hostile
    book_settings = settings
    if hostile_enabled:
        book_settings = settings.model_copy(update={"max_exchange_event_lag_ms": 86_400_000})
    state_book = MarketStateBook(book_settings)
    state_book.set_connected(True)
    anchor_symbol = settings.cross_market_anchor_symbol.upper()
    cross_market_book = None
    if settings.cross_market_enabled and anchor_symbol != settings.symbol.upper():
        cross_market_book = MarketStateBook(book_settings.model_copy(update={"symbol": anchor_symbol}))
        cross_market_book.set_connected(True)
    strategy = HitAndRunStrategy(settings)
    risk = RiskEngine(settings)
    paper = HostileReplayBroker(settings) if hostile_enabled else PaperBroker(settings)
    shadow = ShadowTradeTracker(settings)
    regime_outcomes = RegimeOutcomeTracker(settings)
    candidate_outcomes = CandidateOutcomeTracker(settings)
    summary = ReplaySummary(
        files=[str(path) for path in path_list],
        replay_mode="hostile" if hostile_enabled else "normal",
        hostile_replay=paper.stats if isinstance(paper, HostileReplayBroker) else None,
    )
    interval_ms = decision_interval_ms if decision_interval_ms is not None else settings.decision_interval_ms
    last_sample_at: datetime | None = None
    active_stats: ReplayPositionStats | None = None
    last_market: MarketState | None = None
    last_message_at: datetime | None = None

    for message in load_messages(
        path_list,
        include_depth=include_depth,
        book_ticker_min_interval_ms=book_ticker_min_interval_ms,
    ):
        summary.messages += 1
        last_message_at = message.received_at
        symbol = _payload_symbol(message.payload)
        if cross_market_book is not None and symbol == anchor_symbol:
            cross_market_book.ingest(message.payload, received_at=message.received_at)
            _refresh_cross_market_context(settings, state_book, cross_market_book, message.received_at)
            continue
        state_book.ingest(message.payload, received_at=message.received_at)
        if cross_market_book is not None:
            _refresh_cross_market_context(settings, state_book, cross_market_book, message.received_at)

        should_sample = last_sample_at is None
        if last_sample_at is not None:
            elapsed_ms = (message.received_at - last_sample_at).total_seconds() * 1000
            should_sample = elapsed_ms >= interval_ms
        if not should_sample:
            continue

        last_sample_at = message.received_at
        market = state_book.snapshot(current=message.received_at)
        last_market = market
        if paper.open_position is not None and active_stats is not None:
            active_stats.update(paper.open_position, market)

        closed = paper.mark(market, timestamp=message.received_at)
        if closed is not None:
            summary.paper_closes += 1
            summary.trades.append(closed)
            if active_stats is not None:
                active_stats.close(closed)
                summary.position_stats.append(active_stats)
                active_stats = None
        for event in shadow.mark(market, timestamp=message.received_at):
            summary.shadow_closes += 1
            summary.shadow_net_pnl_usd += float(event.get("net_pnl_usd") or 0.0)
            summary.shadow_events.append(event)
        for event in regime_outcomes.mark(market, timestamp=message.received_at):
            summary.regime_outcome_closes += 1
            summary.regime_outcome_events.append(event)
        for event in candidate_outcomes.mark(market, timestamp=message.received_at):
            summary.candidate_outcome_closes += 1
            summary.candidate_outcome_events.append(event)

        decision = strategy.decide(market)
        verdict = risk.evaluate(decision, market, paper.state())
        summary.decisions += 1
        if decision.action in {DecisionAction.propose_long, DecisionAction.propose_short}:
            summary.proposals += 1
        if verdict.allowed:
            summary.risk_allowed += 1
            if isinstance(paper, HostileReplayBroker):
                opened = paper.open_from_decision(decision, market, opened_at=message.received_at)
            else:
                opened = paper.open_from_decision(decision, opened_at=message.received_at)
            if opened is not None:
                summary.paper_opens += 1
                active_stats = ReplayPositionStats.from_position(opened, settings)
                active_stats.update(opened, market)
        shadow_opened = shadow.open_from_decision(decision, opened_at=message.received_at)
        if shadow_opened is not None:
            summary.shadow_opens += 1
            summary.shadow_events.append(shadow_opened)
        regime_opened = regime_outcomes.open_from_decision(decision, market, opened_at=message.received_at)
        if regime_opened is not None:
            summary.regime_outcome_opens += 1
            summary.regime_outcome_events.append(regime_opened)
        for event in candidate_outcomes.open_from_decision(decision, market, opened_at=message.received_at):
            summary.candidate_outcome_opens += 1
            summary.candidate_outcome_events.append(event)

    if flatten_at_end and paper.open_position is not None and last_market is not None and last_market.mid_price is not None:
        if active_stats is not None:
            active_stats.update(paper.open_position, last_market)
        closed = paper.close(last_market.mid_price, "session_end", closed_at=last_message_at)
        summary.paper_closes += 1
        summary.trades.append(closed)
        if active_stats is not None:
            active_stats.close(closed, status="flattened")
            summary.position_stats.append(active_stats)
            active_stats = None

    if flatten_at_end and last_market is not None and last_market.mid_price is not None:
        for event in shadow.close_all(last_market, reason="session_end", timestamp=last_message_at):
            summary.shadow_closes += 1
            summary.shadow_net_pnl_usd += float(event.get("net_pnl_usd") or 0.0)
            summary.shadow_events.append(event)
        for event in regime_outcomes.close_all(last_market, reason="session_end", timestamp=last_message_at):
            summary.regime_outcome_closes += 1
            summary.regime_outcome_events.append(event)
        for event in candidate_outcomes.close_all(last_market, reason="session_end", timestamp=last_message_at):
            summary.candidate_outcome_closes += 1
            summary.candidate_outcome_events.append(event)

    if active_stats is not None and paper.open_position is not None:
        if last_market is not None:
            active_stats.update(paper.open_position, last_market)
        summary.open_position = active_stats
        summary.unrealized_pnl_usd = active_stats.unrealized_pnl_usd
        summary.net_unrealized_pnl_usd = active_stats.net_unrealized_pnl_usd

    summary.net_pnl_usd = paper.state().realized_pnl_usd
    if hasattr(strategy, "sequence_summary"):
        summary.markov = strategy.sequence_summary()
    if audit is not None:
        audit.write("replay_complete", summary.model_dump())
    return summary


def _payload_symbol(payload: dict) -> str | None:
    symbol = payload.get("s")
    return str(symbol).upper() if symbol else None


def _refresh_cross_market_context(
    settings: Settings,
    state_book: MarketStateBook,
    cross_market_book: MarketStateBook,
    current: datetime,
) -> None:
    primary = state_book.snapshot(current=current)
    anchor = cross_market_book.snapshot(current=current)
    context = build_cross_market_context(settings, primary, anchor, updated_at=current)
    state_book.set_cross_market_context(context, updated_at=current)
