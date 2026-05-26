from dataclasses import dataclass, field
from datetime import datetime

from futures_lab.config import Settings
from futures_lab.exit_shadow import ExitShadowEvaluator
from futures_lab.models import Decision, MarketState, Side, utc_now


@dataclass
class ShadowPosition:
    signal_id: str
    symbol: str
    side: Side
    entry_price: float
    take_profit_price: float
    stop_loss_price: float
    stake_usd: float
    notional_usd: float
    leverage: int
    target_move_pct: float
    stop_move_pct: float
    confidence: float
    opened_at: datetime
    blocked_by: list[str] = field(default_factory=list)
    source: str = "blocked_stateful_continuation"
    max_favorable_move_pct: float = 0.0
    max_adverse_move_pct: float = 0.0
    exit_shadow: ExitShadowEvaluator | None = None


@dataclass
class ShadowTradeTracker:
    settings: Settings
    positions: list[ShadowPosition] = field(default_factory=list)
    opened_count: int = 0
    closed_count: int = 0

    def open_from_decision(self, decision: Decision, opened_at: datetime | None = None) -> dict | None:
        if not self.settings.write_shadow_trade_log:
            return None
        signal = (decision.evidence.get("stateful_momentum_filter") or {}).get("shadow_trade")
        if not signal:
            return None

        self.opened_count += 1
        signal_id = f"shadow-{self.opened_count:06d}"
        actual_opened_at = opened_at or decision.timestamp
        position = ShadowPosition(
            signal_id=signal_id,
            symbol=decision.symbol,
            side=Side.long if signal["side"] == "long" else Side.short,
            entry_price=float(signal["entry_price"]),
            take_profit_price=float(signal["take_profit_price"]),
            stop_loss_price=float(signal["stop_loss_price"]),
            stake_usd=float(signal["stake_usd"]),
            notional_usd=float(signal["notional_usd"]),
            leverage=int(signal["leverage"]),
            target_move_pct=float(signal["target_move_pct"]),
            stop_move_pct=float(signal["stop_move_pct"]),
            confidence=float(signal["confidence"]),
            opened_at=actual_opened_at,
            blocked_by=list(signal.get("blocked_by") or []),
            source=str(signal.get("source") or "blocked_stateful_continuation"),
        )
        position.exit_shadow = ExitShadowEvaluator(
            settings=self.settings,
            symbol=position.symbol,
            side=position.side,
            entry_price=position.entry_price,
            notional_usd=position.notional_usd,
            opened_at=actual_opened_at,
        )
        self.positions.append(position)
        return self._event("open", position)

    def mark(self, market: MarketState, timestamp: datetime | None = None) -> list[dict]:
        if not self.positions or market.mid_price is None:
            return []
        if not market.connected:
            return []
        if market.data_age_seconds is not None and market.data_age_seconds > self.settings.stale_after_seconds:
            return []

        closed = []
        remaining = []
        current = timestamp or market.last_received_at or utc_now()
        for position in self.positions:
            self._update_excursion(position, market.mid_price)
            if position.exit_shadow is not None:
                position.exit_shadow.mark(market, current)
            reason = self._exit_reason(position, market.mid_price)
            if reason is None:
                remaining.append(position)
                continue
            closed.append(self._close_event(position, market.mid_price, reason, current))
        self.positions = remaining
        return closed

    def close_all(self, market: MarketState | None, reason: str = "session_end", timestamp: datetime | None = None) -> list[dict]:
        if not self.positions:
            return []
        if market is None or market.mid_price is None:
            return []
        current = timestamp or market.last_received_at or utc_now()
        for position in self.positions:
            self._update_excursion(position, market.mid_price)
            if position.exit_shadow is not None:
                position.exit_shadow.mark(market, current)
        closed = [self._close_event(position, market.mid_price, reason, current) for position in self.positions]
        self.positions = []
        return closed

    def _exit_reason(self, position: ShadowPosition, price: float) -> str | None:
        if position.side == Side.long:
            if price >= position.take_profit_price:
                return "take_profit"
            if price <= position.stop_loss_price:
                return "stop_loss"
            return None
        if price <= position.take_profit_price:
            return "take_profit"
        if price >= position.stop_loss_price:
            return "stop_loss"
        return None

    def _update_excursion(self, position: ShadowPosition, price: float) -> None:
        move = self._underlying_move_pct(position, price)
        position.max_favorable_move_pct = max(position.max_favorable_move_pct, move)
        position.max_adverse_move_pct = min(position.max_adverse_move_pct, move)

    def _close_event(self, position: ShadowPosition, exit_price: float, reason: str, closed_at: datetime) -> dict:
        self.closed_count += 1
        gross = self._gross_pnl(position, exit_price)
        fees = position.notional_usd * 2 * (self.settings.taker_fee_bps / 10_000)
        exit_shadow = {}
        if position.exit_shadow is not None:
            exit_shadow = position.exit_shadow.close_at_actual(exit_price, reason, closed_at)
        return self._event(
            "close",
            position,
            {
                "exit_price": exit_price,
                "exit_reason": reason,
                "closed_at": closed_at.isoformat(),
                "gross_pnl_usd": gross,
                "fees_usd": fees,
                "net_pnl_usd": gross - fees,
                "exit_shadow": exit_shadow,
                "max_favorable_move_pct": position.max_favorable_move_pct,
                "max_adverse_move_pct": position.max_adverse_move_pct,
                "time_in_trade_seconds": (closed_at - position.opened_at).total_seconds(),
            },
        )

    def _event(self, event: str, position: ShadowPosition, extra: dict | None = None) -> dict:
        row = {
            "event": event,
            "signal_id": position.signal_id,
            "symbol": position.symbol,
            "side": position.side.value,
            "source": position.source,
            "entry_price": position.entry_price,
            "take_profit_price": position.take_profit_price,
            "stop_loss_price": position.stop_loss_price,
            "stake_usd": position.stake_usd,
            "notional_usd": position.notional_usd,
            "leverage": position.leverage,
            "target_move_pct": position.target_move_pct,
            "stop_move_pct": position.stop_move_pct,
            "confidence": position.confidence,
            "opened_at": position.opened_at.isoformat(),
            "blocked_by": position.blocked_by,
        }
        if extra:
            row.update(extra)
        return row

    def _gross_pnl(self, position: ShadowPosition, exit_price: float) -> float:
        direction = 1 if position.side == Side.long else -1
        quantity = position.notional_usd / position.entry_price
        return (exit_price - position.entry_price) * quantity * direction

    def _underlying_move_pct(self, position: ShadowPosition, price: float) -> float:
        direction = 1 if position.side == Side.long else -1
        return ((price - position.entry_price) / position.entry_price) * direction
