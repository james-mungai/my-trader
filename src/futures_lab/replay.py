import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

from futures_lab.audit import AuditLog
from futures_lab.config import Settings
from futures_lab.market_state import MarketStateBook
from futures_lab.models import DecisionAction, PaperTrade
from futures_lab.paper import PaperBroker
from futures_lab.risk import RiskEngine
from futures_lab.strategy import HitAndRunStrategy


@dataclass(frozen=True)
class ReplayMessage:
    received_at: datetime
    payload: dict


@dataclass
class ReplaySummary:
    files: list[str]
    messages: int = 0
    decisions: int = 0
    proposals: int = 0
    risk_allowed: int = 0
    paper_opens: int = 0
    paper_closes: int = 0
    net_pnl_usd: float = 0.0
    trades: list[PaperTrade] = field(default_factory=list)

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
        }


def discover_raw_files(settings: Settings, pattern: str | None = None) -> list[Path]:
    raw_dir = Path(settings.data_dir) / "raw_ws"
    glob_pattern = pattern or f"{settings.symbol.upper()}_*_*.jsonl"
    return sorted(raw_dir.glob(glob_pattern))


def load_messages(
    paths: Iterable[Path],
    include_depth: bool = False,
    book_ticker_min_interval_ms: int = 100,
) -> list[ReplayMessage]:
    messages: list[ReplayMessage] = []
    last_book_ticker_at: datetime | None = None
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
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
    audit: AuditLog | None = None,
) -> ReplaySummary:
    path_list = [Path(path) for path in paths]
    state_book = MarketStateBook(settings)
    state_book.set_connected(True)
    strategy = HitAndRunStrategy(settings)
    risk = RiskEngine(settings)
    paper = PaperBroker(settings)
    summary = ReplaySummary(files=[str(path) for path in path_list])
    interval_ms = decision_interval_ms if decision_interval_ms is not None else settings.decision_interval_ms
    last_sample_at: datetime | None = None

    for message in load_messages(
        path_list,
        include_depth=include_depth,
        book_ticker_min_interval_ms=book_ticker_min_interval_ms,
    ):
        summary.messages += 1
        state_book.ingest(message.payload, received_at=message.received_at)

        should_sample = last_sample_at is None
        if last_sample_at is not None:
            elapsed_ms = (message.received_at - last_sample_at).total_seconds() * 1000
            should_sample = elapsed_ms >= interval_ms
        if not should_sample:
            continue

        last_sample_at = message.received_at
        market = state_book.snapshot(current=message.received_at)
        closed = paper.mark(market, timestamp=message.received_at)
        if closed is not None:
            summary.paper_closes += 1
            summary.trades.append(closed)

        decision = strategy.decide(market)
        verdict = risk.evaluate(decision, market, paper.state())
        summary.decisions += 1
        if decision.action in {DecisionAction.propose_long, DecisionAction.propose_short}:
            summary.proposals += 1
        if verdict.allowed:
            summary.risk_allowed += 1
            opened = paper.open_from_decision(decision, opened_at=message.received_at)
            if opened is not None:
                summary.paper_opens += 1

    summary.net_pnl_usd = paper.state().realized_pnl_usd
    if audit is not None:
        audit.write("replay_complete", summary.model_dump())
    return summary
