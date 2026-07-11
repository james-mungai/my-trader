import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from futures_lab.audit import AuditLog
from futures_lab.binance_streams import BinanceStreamRecorder
from futures_lab.config import Settings
from futures_lab.first_touch import _micro_signal, _squeeze_breakout_signal
from futures_lab.market_state import MarketStateBook
from futures_lab.models import MarketState


SignalFunction = Callable[[dict[str, Any]], float]


@dataclass(frozen=True)
class ShadowArenaConfig:
    target_bps: float = 60.0
    stop_bps: float = 60.0
    cost_bps: float = 10.0
    horizon_seconds: int = 21_600
    cooldown_seconds: int = 60
    warmup_seconds: int = 180
    account_equity_usd: float = 100.0
    account_exposure: float = 4.95
    max_trades_per_arm: int = 300
    checkpoints: tuple[int, ...] = (100, 200, 300)
    health_log_seconds: int = 60

    def __post_init__(self) -> None:
        if self.target_bps <= 0 or self.stop_bps <= 0:
            raise ValueError("Target and stop barriers must be positive.")
        if self.cost_bps < 0:
            raise ValueError("Modeled cost cannot be negative.")
        if self.horizon_seconds <= 0 or self.cooldown_seconds < 0 or self.warmup_seconds < 0:
            raise ValueError("Arena timing values are invalid.")
        if self.account_equity_usd <= 0 or self.account_exposure <= 0:
            raise ValueError("Account equity and exposure must be positive.")

    @property
    def modeled_notional_usd(self) -> float:
        return self.account_equity_usd * self.account_exposure


@dataclass
class ArenaPosition:
    side: int
    entry_price: float
    opened_at: datetime
    signal_score: float
    trigger: str
    max_favorable_bps: float = 0.0
    max_adverse_bps: float = 0.0


@dataclass
class ArenaArm:
    name: str
    signal: SignalFunction | None = None
    position: ArenaPosition | None = None
    opened: int = 0
    closed: int = 0
    targets: int = 0
    stops: int = 0
    timeouts: int = 0
    session_ends: int = 0
    cumulative_net_bps: float = 0.0
    cumulative_net_pnl_usd: float = 0.0
    last_closed_at: datetime | None = None
    emitted_checkpoints: set[int] = field(default_factory=set)


class ArenaEventLog:
    def __init__(self, settings: Settings) -> None:
        self.directory = Path(settings.data_dir) / "shadow_arena"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / f"{settings.symbol.upper()}_shadow_arena.jsonl"

    def write(self, event: str, payload: dict[str, Any], *, timestamp: datetime | None = None) -> None:
        row = {
            "ts": (timestamp or datetime.now(timezone.utc)).isoformat(),
            "event": event,
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, separators=(",", ":"), default=str))
            handle.write("\n")


class ShadowArena:
    """Forward-only, independent first-touch ledgers sharing one public market feed."""

    def __init__(
        self,
        settings: Settings,
        config: ShadowArenaConfig,
        *,
        state: MarketStateBook | None = None,
        audit: AuditLog | None = None,
        recorder: BinanceStreamRecorder | None = None,
    ) -> None:
        self.settings = settings
        self.config = config
        self.audit = audit or AuditLog(settings)
        self.state = state or MarketStateBook(settings)
        self.recorder = recorder or BinanceStreamRecorder(settings, self.state, self.audit)
        self.events = ArenaEventLog(settings)
        self.arms = {
            "squeeze_breakout": ArenaArm("squeeze_breakout", _squeeze_breakout_signal),
            "raw_micro_momentum": ArenaArm("raw_micro_momentum", _micro_signal),
            "squeeze_matched_coin": ArenaArm("squeeze_matched_coin"),
            "clock_coin_control": ArenaArm("clock_coin_control"),
        }
        self.started_at: datetime | None = None
        self.last_health_at: datetime | None = None
        self.last_price: float | None = None
        self.last_market_at: datetime | None = None

    def start(self) -> None:
        self.started_at = datetime.now(timezone.utc)
        self.recorder.start()
        self.events.write(
            "arena_start",
            {
                "symbol": self.settings.symbol.upper(),
                "arms": list(self.arms),
                "target_bps": self.config.target_bps,
                "stop_bps": self.config.stop_bps,
                "cost_bps": self.config.cost_bps,
                "break_even_win_rate": self.break_even_win_rate,
                "horizon_seconds": self.config.horizon_seconds,
                "cooldown_seconds": self.config.cooldown_seconds,
                "warmup_seconds": self.config.warmup_seconds,
                "modeled_notional_usd": self.config.modeled_notional_usd,
                "paper_only": True,
            },
            timestamp=self.started_at,
        )

    async def stop(self) -> None:
        market = self.state.snapshot()
        timestamp = self._market_timestamp(market)
        price = self._price(market) or self.last_price
        if price is not None:
            for arm in self.arms.values():
                if arm.position is not None:
                    self._close(arm, price, timestamp, "session_end")
        self.events.write("arena_stop", {"summary": self.summary()}, timestamp=timestamp)
        await self.recorder.stop()

    @property
    def break_even_win_rate(self) -> float:
        denominator = self.config.target_bps + self.config.stop_bps
        return (self.config.stop_bps + self.config.cost_bps) / denominator

    def tick(self, *, current: datetime | None = None) -> MarketState:
        market = self.state.snapshot(current=current)
        timestamp = self._market_timestamp(market, fallback=current)
        price = self._price(market)
        if price is not None:
            self.last_price = price
            self.last_market_at = timestamp
            for arm in self.arms.values():
                self._mark(arm, price, timestamp)

        if price is None or not self._market_ready(market):
            self._write_health(market, timestamp)
            return market

        row = market.model_dump(mode="python")
        squeeze_score = _squeeze_breakout_signal(row)
        squeeze_opened = self._try_signal_arm(
            self.arms["squeeze_breakout"], squeeze_score, price, timestamp, trigger="squeeze"
        )
        self._try_signal_arm(
            self.arms["raw_micro_momentum"], _micro_signal(row), price, timestamp, trigger="micro"
        )
        if squeeze_opened:
            self._try_signal_arm(
                self.arms["squeeze_matched_coin"],
                float(self._coin_side(timestamp, "squeeze")),
                price,
                timestamp,
                trigger="squeeze_matched_coin",
            )
        self._try_signal_arm(
            self.arms["clock_coin_control"],
            float(self._coin_side(timestamp, "clock")),
            price,
            timestamp,
            trigger="clock_coin",
        )
        self._write_health(market, timestamp)
        return market

    def all_arms_complete(self) -> bool:
        limit = self.config.max_trades_per_arm
        return limit > 0 and all(arm.closed >= limit for arm in self.arms.values())

    def summary(self) -> dict[str, Any]:
        return {
            "target_bps": self.config.target_bps,
            "stop_bps": self.config.stop_bps,
            "cost_bps": self.config.cost_bps,
            "break_even_win_rate": self.break_even_win_rate,
            "arms": {name: self._arm_summary(arm) for name, arm in self.arms.items()},
        }

    def _try_signal_arm(
        self,
        arm: ArenaArm,
        score: float,
        price: float,
        timestamp: datetime,
        *,
        trigger: str,
    ) -> bool:
        if arm.position is not None or score == 0.0:
            return False
        if self.config.max_trades_per_arm > 0 and arm.closed >= self.config.max_trades_per_arm:
            return False
        if arm.last_closed_at is not None:
            elapsed = (timestamp - arm.last_closed_at).total_seconds()
            if elapsed < self.config.cooldown_seconds:
                return False
        side = 1 if score > 0 else -1
        arm.position = ArenaPosition(
            side=side,
            entry_price=price,
            opened_at=timestamp,
            signal_score=score,
            trigger=trigger,
        )
        arm.opened += 1
        self.events.write(
            "arena_open",
            {
                "arm": arm.name,
                "trade_number": arm.opened,
                "side": "long" if side > 0 else "short",
                "entry_price": price,
                "target_price": price * (1 + side * self.config.target_bps / 10_000),
                "stop_price": price * (1 - side * self.config.stop_bps / 10_000),
                "signal_score": score,
                "trigger": trigger,
            },
            timestamp=timestamp,
        )
        return True

    def _mark(self, arm: ArenaArm, price: float, timestamp: datetime) -> None:
        position = arm.position
        if position is None:
            return
        signed_bps = position.side * (price / position.entry_price - 1) * 10_000
        position.max_favorable_bps = max(position.max_favorable_bps, signed_bps)
        position.max_adverse_bps = min(position.max_adverse_bps, signed_bps)
        if signed_bps >= self.config.target_bps:
            self._close(arm, price, timestamp, "take_profit")
        elif signed_bps <= -self.config.stop_bps:
            self._close(arm, price, timestamp, "stop_loss")
        elif (timestamp - position.opened_at).total_seconds() >= self.config.horizon_seconds:
            self._close(arm, price, timestamp, "timeout")

    def _close(self, arm: ArenaArm, price: float, timestamp: datetime, reason: str) -> None:
        position = arm.position
        if position is None:
            return
        gross_bps = position.side * (price / position.entry_price - 1) * 10_000
        net_bps = gross_bps - self.config.cost_bps
        pnl_usd = self.config.modeled_notional_usd * net_bps / 10_000
        arm.closed += 1
        arm.cumulative_net_bps += net_bps
        arm.cumulative_net_pnl_usd += pnl_usd
        arm.last_closed_at = timestamp
        if reason == "take_profit":
            arm.targets += 1
        elif reason == "stop_loss":
            arm.stops += 1
        elif reason == "timeout":
            arm.timeouts += 1
        elif reason == "session_end":
            arm.session_ends += 1
        self.events.write(
            "arena_close",
            {
                "arm": arm.name,
                "trade_number": arm.closed,
                "side": "long" if position.side > 0 else "short",
                "entry_price": position.entry_price,
                "exit_price": price,
                "exit_reason": reason,
                "gross_bps": gross_bps,
                "cost_bps": self.config.cost_bps,
                "net_bps": net_bps,
                "net_pnl_usd": pnl_usd,
                "cumulative_net_bps": arm.cumulative_net_bps,
                "cumulative_net_pnl_usd": arm.cumulative_net_pnl_usd,
                "max_favorable_bps": position.max_favorable_bps,
                "max_adverse_bps": position.max_adverse_bps,
                "holding_seconds": max(0.0, (timestamp - position.opened_at).total_seconds()),
                "signal_score": position.signal_score,
                "trigger": position.trigger,
            },
            timestamp=timestamp,
        )
        arm.position = None
        self._emit_checkpoint(arm, timestamp)

    def _emit_checkpoint(self, arm: ArenaArm, timestamp: datetime) -> None:
        for checkpoint in self.config.checkpoints:
            if arm.closed >= checkpoint and checkpoint not in arm.emitted_checkpoints:
                arm.emitted_checkpoints.add(checkpoint)
                self.events.write(
                    "arena_checkpoint",
                    {"arm": arm.name, "checkpoint": checkpoint, "summary": self._arm_summary(arm)},
                    timestamp=timestamp,
                )

    def _market_ready(self, market: MarketState) -> bool:
        return bool(
            market.connected
            and market.observed_seconds >= self.config.warmup_seconds
            and market.data_age_seconds is not None
            and market.data_age_seconds <= self.settings.stale_after_seconds
            and market.mark_price is not None
        )

    def _write_health(self, market: MarketState, timestamp: datetime) -> None:
        if self.last_health_at is not None:
            if (timestamp - self.last_health_at).total_seconds() < self.config.health_log_seconds:
                return
        self.last_health_at = timestamp
        self.events.write(
            "arena_health",
            {
                "connected": market.connected,
                "observed_seconds": market.observed_seconds,
                "data_age_seconds": market.data_age_seconds,
                "mark_price": market.mark_price,
                "book_event_lag_ms": market.book_event_lag_ms,
                "trade_event_lag_ms": market.trade_event_lag_ms,
                "avg_book_event_lag_30s_ms": market.avg_book_event_lag_30s_ms,
                "avg_trade_event_lag_30s_ms": market.avg_trade_event_lag_30s_ms,
                "open_positions": [name for name, arm in self.arms.items() if arm.position is not None],
                "closed_trades": {name: arm.closed for name, arm in self.arms.items()},
            },
            timestamp=timestamp,
        )

    def _arm_summary(self, arm: ArenaArm) -> dict[str, Any]:
        resolved = arm.targets + arm.stops
        win_rate = arm.targets / resolved if resolved else None
        interval = _wilson_interval(arm.targets, resolved) if resolved else None
        return {
            "opened": arm.opened,
            "closed": arm.closed,
            "open": arm.position is not None,
            "targets": arm.targets,
            "stops": arm.stops,
            "timeouts": arm.timeouts,
            "session_ends": arm.session_ends,
            "resolved_win_rate": win_rate,
            "win_rate_wilson_95": interval,
            "beats_break_even": win_rate > self.break_even_win_rate if win_rate is not None else None,
            "cumulative_net_bps": arm.cumulative_net_bps,
            "average_net_bps": arm.cumulative_net_bps / arm.closed if arm.closed else None,
            "cumulative_net_pnl_usd": arm.cumulative_net_pnl_usd,
        }

    def _price(self, market: MarketState) -> float | None:
        return market.mark_price if market.mark_price is not None and market.mark_price > 0 else None

    def _market_timestamp(self, market: MarketState, *, fallback: datetime | None = None) -> datetime:
        return market.last_received_at or fallback or datetime.now(timezone.utc)

    def _coin_side(self, timestamp: datetime, namespace: str) -> int:
        bucket = int(timestamp.timestamp()) // max(1, self.config.cooldown_seconds)
        key = f"{self.settings.symbol.upper()}:{namespace}:{bucket}".encode("ascii")
        return 1 if hashlib.sha256(key).digest()[0] & 1 else -1


def summarize_shadow_arena(settings: Settings) -> dict[str, Any]:
    directory = Path(settings.data_dir) / "shadow_arena"
    paths = sorted(directory.glob("*.jsonl")) if directory.exists() else []
    closes: dict[str, list[dict[str, Any]]] = {}
    opens: dict[str, dict[str, Any]] = {}
    start: dict[str, Any] | None = None
    stop: dict[str, Any] | None = None
    health: dict[str, Any] | None = None
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                event = row.get("event")
                arm = row.get("arm")
                if event == "arena_start":
                    start = row
                elif event == "arena_stop":
                    stop = row
                elif event == "arena_health":
                    health = row
                elif event == "arena_open" and arm:
                    opens[arm] = row
                elif event == "arena_close" and arm:
                    closes.setdefault(arm, []).append(row)
                    opens.pop(arm, None)

    target_bps = float((start or {}).get("target_bps", 60.0))
    stop_bps = float((start or {}).get("stop_bps", 60.0))
    cost_bps = float((start or {}).get("cost_bps", 10.0))
    break_even = (stop_bps + cost_bps) / (target_bps + stop_bps)
    arm_names = sorted(set(closes) | set(opens) | set((start or {}).get("arms", [])))
    summaries = {
        arm: _summarize_close_rows(closes.get(arm, []), arm in opens, break_even) for arm in arm_names
    }
    return {
        "files": [str(path) for path in paths],
        "started_at": (start or {}).get("ts"),
        "stopped_at": (stop or {}).get("ts"),
        "last_health": health,
        "target_bps": target_bps,
        "stop_bps": stop_bps,
        "cost_bps": cost_bps,
        "break_even_win_rate": break_even,
        "arms": summaries,
    }


def _summarize_close_rows(rows: list[dict[str, Any]], is_open: bool, break_even: float) -> dict[str, Any]:
    targets = sum(row.get("exit_reason") == "take_profit" for row in rows)
    stops = sum(row.get("exit_reason") == "stop_loss" for row in rows)
    timeouts = sum(row.get("exit_reason") == "timeout" for row in rows)
    session_ends = sum(row.get("exit_reason") == "session_end" for row in rows)
    resolved = targets + stops
    win_rate = targets / resolved if resolved else None
    net_bps = sum(float(row.get("net_bps", 0.0)) for row in rows)
    net_pnl = sum(float(row.get("net_pnl_usd", 0.0)) for row in rows)
    sides: dict[str, dict[str, Any]] = {}
    for side in ("long", "short"):
        selected = [row for row in rows if row.get("side") == side]
        side_targets = sum(row.get("exit_reason") == "take_profit" for row in selected)
        side_stops = sum(row.get("exit_reason") == "stop_loss" for row in selected)
        side_resolved = side_targets + side_stops
        sides[side] = {
            "trades": len(selected),
            "targets": side_targets,
            "stops": side_stops,
            "resolved_win_rate": side_targets / side_resolved if side_resolved else None,
            "net_bps": sum(float(row.get("net_bps", 0.0)) for row in selected),
        }
    blocks = []
    for start in range(0, len(rows), 25):
        block = rows[start : start + 25]
        block_targets = sum(row.get("exit_reason") == "take_profit" for row in block)
        block_stops = sum(row.get("exit_reason") == "stop_loss" for row in block)
        block_resolved = block_targets + block_stops
        blocks.append(
            {
                "trades": f"{start + 1}-{start + len(block)}",
                "targets": block_targets,
                "stops": block_stops,
                "resolved_win_rate": block_targets / block_resolved if block_resolved else None,
                "net_bps": sum(float(row.get("net_bps", 0.0)) for row in block),
            }
        )
    return {
        "closed": len(rows),
        "open": is_open,
        "targets": targets,
        "stops": stops,
        "timeouts": timeouts,
        "session_ends": session_ends,
        "resolved_win_rate": win_rate,
        "win_rate_wilson_95": _wilson_interval(targets, resolved) if resolved else None,
        "beats_break_even": win_rate > break_even if win_rate is not None else None,
        "total_net_bps": net_bps,
        "average_net_bps": net_bps / len(rows) if rows else None,
        "modeled_net_pnl_usd": net_pnl,
        "by_side": sides,
        "chronological_blocks_25": blocks,
    }


def _wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float] | None:
    if trials <= 0:
        return None
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]
