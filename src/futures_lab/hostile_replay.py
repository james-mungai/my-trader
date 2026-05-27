from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from futures_lab.config import Settings
from futures_lab.models import Decision, MarketState, PaperPosition, PaperTrade, Side
from futures_lab.paper import PaperBroker


@dataclass
class HostileReplayStats:
    enabled: bool = True
    open_rejections: int = 0
    stale_book_rejections: int = 0
    maker_queue_rejections: int = 0
    entry_fills: int = 0
    exit_fills: int = 0
    entry_slippage_usd: float = 0.0
    exit_slippage_usd: float = 0.0
    stop_penalties_usd: float = 0.0
    rejection_reasons: dict[str, int] = field(default_factory=dict)

    def reject(self, reason: str) -> None:
        self.open_rejections += 1
        self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) + 1
        if reason == "stale_book":
            self.stale_book_rejections += 1
        if reason == "maker_queue":
            self.maker_queue_rejections += 1

    def model_dump(self) -> dict:
        return {
            "enabled": self.enabled,
            "open_rejections": self.open_rejections,
            "stale_book_rejections": self.stale_book_rejections,
            "maker_queue_rejections": self.maker_queue_rejections,
            "entry_fills": self.entry_fills,
            "exit_fills": self.exit_fills,
            "entry_slippage_usd": round(self.entry_slippage_usd, 8),
            "exit_slippage_usd": round(self.exit_slippage_usd, 8),
            "stop_penalties_usd": round(self.stop_penalties_usd, 8),
            "rejection_reasons": dict(sorted(self.rejection_reasons.items())),
        }


class HostileReplayBroker(PaperBroker):
    """Replay-only paper broker that pessimizes fills and rejects stale books."""

    def __init__(self, settings: Settings):
        super().__init__(settings=settings)
        self.hostile_stats = HostileReplayStats()

    @property
    def stats(self) -> HostileReplayStats:
        return self.hostile_stats

    def open_from_decision(
        self,
        decision: Decision,
        market: MarketState,
        opened_at: datetime | None = None,
    ) -> PaperPosition | None:
        rejection = self._open_rejection_reason(decision, market)
        if rejection is not None:
            self.hostile_stats.reject(rejection)
            return None

        side = self._decision_side(decision)
        if side is None:
            return None

        reference = decision.entry_price or market.mid_price
        if reference is None:
            self.hostile_stats.reject("missing_reference_price")
            return None
        fill_price = self._entry_fill_price(side, market, reference)
        adjusted = self._decision_with_hostile_fill(decision, side, fill_price)
        opened = super().open_from_decision(adjusted, opened_at=opened_at)
        if opened is not None:
            self.hostile_stats.entry_fills += 1
            self.hostile_stats.entry_slippage_usd += abs(fill_price - reference)
        return opened

    def mark(self, market: MarketState, timestamp: datetime | None = None) -> PaperTrade | None:
        self._reset_day_if_needed()
        if self.open_position is None or market.mid_price is None:
            return None
        if not market.connected:
            return None
        if market.data_age_seconds is not None and market.data_age_seconds > self.settings.stale_after_seconds:
            return None

        pos = self.open_position
        current = timestamp or datetime.now(tz=pos.opened_at.tzinfo)
        self._update_excursion(pos, market.mid_price)
        if self.exit_shadow is not None:
            self.exit_shadow.mark(market, current)

        exit_reason = self._exit_reason(pos, market, current)
        if exit_reason is None:
            return None
        exit_price, stop_penalty_usd = self._exit_fill_price(pos, market, exit_reason)
        trade = super().close(exit_price, exit_reason, closed_at=current)
        self.hostile_stats.exit_fills += 1
        self.hostile_stats.exit_slippage_usd += abs(exit_price - market.mid_price)
        self.hostile_stats.stop_penalties_usd += stop_penalty_usd
        trade.exit_shadow.setdefault("hostile_execution", {})
        trade.exit_shadow["hostile_execution"].update(
            {
                "market_mid_price": market.mid_price,
                "hostile_exit_price": exit_price,
                "exit_slippage_usd": abs(exit_price - market.mid_price),
                "stop_penalty_usd": stop_penalty_usd,
                "latency_penalty_bps": self._latency_penalty_bps(),
            }
        )
        return trade

    def _open_rejection_reason(self, decision: Decision, market: MarketState) -> str | None:
        if market.best_bid is None or market.best_ask is None:
            return "missing_top_of_book"
        max_book_age_seconds = max(0, self.settings.hostile_replay_max_book_age_ms) / 1000
        if market.data_age_seconds is not None and market.data_age_seconds > max_book_age_seconds:
            return "stale_book"
        max_lag = max(0.0, self.settings.hostile_replay_max_event_lag_ms)
        if market.exchange_event_lag_ms is not None and market.exchange_event_lag_ms > max_lag:
            return "stale_book"
        if self.settings.default_entry_order_type.strip().lower() == "maker" and not self._maker_queue_fill_allowed(decision):
            return "maker_queue"
        return None

    def _maker_queue_fill_allowed(self, decision: Decision) -> bool:
        probability = max(0.0, min(1.0, self.settings.hostile_replay_maker_queue_fill_probability))
        edge = (decision.evidence.get("edge_router") or {}).get("selected_candidate") or {}
        score = float(edge.get("score") or decision.confidence or 0.0)
        return score >= (1.0 - probability)

    def _decision_side(self, decision: Decision) -> Side | None:
        if decision.action.value == "propose_long":
            return Side.long
        if decision.action.value == "propose_short":
            return Side.short
        return None

    def _decision_with_hostile_fill(self, decision: Decision, side: Side, fill_price: float) -> Decision:
        target = decision.target_move_pct
        stop = decision.stop_move_pct
        take_profit = decision.take_profit_price
        stop_loss = decision.stop_loss_price
        if target is not None:
            take_profit = fill_price * (1 + target) if side == Side.long else fill_price * (1 - target)
        if stop is not None:
            stop_loss = fill_price * (1 - stop) if side == Side.long else fill_price * (1 + stop)
        evidence = dict(decision.evidence)
        evidence["hostile_replay_entry"] = {
            "original_entry_price": decision.entry_price,
            "hostile_entry_price": fill_price,
            "entry_slippage_bps": self.settings.hostile_replay_entry_slippage_bps,
            "latency_penalty_bps": self._latency_penalty_bps(),
        }
        return decision.model_copy(
            update={
                "entry_price": fill_price,
                "take_profit_price": take_profit,
                "stop_loss_price": stop_loss,
                "evidence": evidence,
            }
        )

    def _entry_fill_price(self, side: Side, market: MarketState, reference: float) -> float:
        base = market.best_ask if side == Side.long else market.best_bid
        base = base or reference
        penalty = (self.settings.hostile_replay_entry_slippage_bps + self._latency_penalty_bps()) / 10_000
        if side == Side.long:
            return base * (1 + penalty)
        return base * (1 - penalty)

    def _exit_fill_price(self, pos: PaperPosition, market: MarketState, reason: str) -> tuple[float, float]:
        reference = market.mid_price or pos.entry_price
        base = market.best_bid if pos.side == Side.long else market.best_ask
        base = base or reference
        stop_penalty_bps = self.settings.hostile_replay_stop_penalty_bps if _is_adverse_exit(reason) else 0.0
        penalty = (self.settings.hostile_replay_exit_slippage_bps + self._latency_penalty_bps() + stop_penalty_bps) / 10_000
        if pos.side == Side.long:
            fill = base * (1 - penalty)
        else:
            fill = base * (1 + penalty)
        stop_penalty_usd = abs(base - fill) if stop_penalty_bps else 0.0
        return fill, stop_penalty_usd

    def _latency_penalty_bps(self) -> float:
        latency_ms = (
            max(0, self.settings.hostile_replay_feed_latency_ms)
            + max(0, self.settings.hostile_replay_decision_latency_ms)
            + max(0, self.settings.hostile_replay_order_latency_ms)
        )
        dynamic = (latency_ms / 1000) * max(0.0, self.settings.hostile_replay_latency_bps_per_second)
        return max(0.0, self.settings.hostile_replay_latency_penalty_bps) + dynamic

    def _exit_reason(self, pos: PaperPosition, market: MarketState, current: datetime) -> str | None:
        if pos.side == Side.long:
            if market.mid_price is not None and market.mid_price >= pos.take_profit_price:
                return "take_profit"
            if self.settings.enable_price_stop and market.mid_price is not None and market.mid_price <= pos.stop_loss_price:
                return "stop_loss"
        else:
            if market.mid_price is not None and market.mid_price <= pos.take_profit_price:
                return "take_profit"
            if self.settings.enable_price_stop and market.mid_price is not None and market.mid_price >= pos.stop_loss_price:
                return "stop_loss"
        if self._should_close_mfe_trailing(pos, market):
            return "mfe_trailing_stop"
        if pos.max_adverse_move_pct <= -abs(self.settings.emergency_max_adverse_move_pct):
            return "emergency_adverse_move"
        if self._should_close_fast_failure(pos, current):
            return "fast_failure"
        if self._should_close_max_hold(pos, current):
            return "max_hold"
        return None


def _is_adverse_exit(reason: str) -> bool:
    return reason in {"stop_loss", "emergency_adverse_move", "fast_failure", "max_hold", "mfe_trailing_stop"}
