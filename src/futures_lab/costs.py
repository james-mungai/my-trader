from __future__ import annotations

from dataclasses import dataclass

from futures_lab.config import Settings
from futures_lab.models import MarketState


@dataclass(frozen=True)
class EffectiveCost:
    entry_order_type: str
    exit_order_type: str
    maker_fee_bps: float
    taker_fee_bps: float
    entry_fee_bps: float
    exit_fee_bps: float
    fee_bps: float
    spread_cross_bps: float
    slippage_bps: float
    latency_penalty_bps: float
    total_cost_bps: float

    @property
    def total_cost_pct(self) -> float:
        return self.total_cost_bps / 10_000

    def required_target_pct(self, multiple: float) -> float:
        return self.total_cost_pct * max(0.0, multiple)

    def model_dump(self) -> dict:
        return {
            "entry_order_type": self.entry_order_type,
            "exit_order_type": self.exit_order_type,
            "maker_fee_bps": self.maker_fee_bps,
            "taker_fee_bps": self.taker_fee_bps,
            "entry_fee_bps": self.entry_fee_bps,
            "exit_fee_bps": self.exit_fee_bps,
            "fee_bps": self.fee_bps,
            "spread_cross_bps": self.spread_cross_bps,
            "slippage_bps": self.slippage_bps,
            "latency_penalty_bps": self.latency_penalty_bps,
            "total_cost_bps": self.total_cost_bps,
            "expected_round_trip_cost_bps": self.total_cost_bps,
            "total_cost_pct": self.total_cost_pct,
        }


def estimate_effective_cost(
    settings: Settings,
    market: MarketState,
    entry_order_type: str | None = None,
    exit_order_type: str | None = None,
) -> EffectiveCost:
    entry = _order_type(entry_order_type or settings.default_entry_order_type)
    exit_ = _order_type(exit_order_type or settings.default_exit_order_type)
    spread_bps = max(0.0, market.spread_bps or 0.0)
    entry_fee = _fee_bps(settings, entry)
    exit_fee = _fee_bps(settings, exit_)
    taker_sides = int(entry == "taker") + int(exit_ == "taker")
    spread_cross = (spread_bps / 2.0) * taker_sides
    slippage = max(0.0, settings.expected_slippage_bps) * taker_sides
    latency = max(0.0, settings.latency_adverse_selection_bps)
    fee = entry_fee + exit_fee
    return EffectiveCost(
        entry_order_type=entry,
        exit_order_type=exit_,
        maker_fee_bps=max(0.0, settings.maker_fee_bps),
        taker_fee_bps=max(0.0, settings.taker_fee_bps),
        entry_fee_bps=entry_fee,
        exit_fee_bps=exit_fee,
        fee_bps=fee,
        spread_cross_bps=spread_cross,
        slippage_bps=slippage,
        latency_penalty_bps=latency,
        total_cost_bps=fee + spread_cross + slippage + latency,
    )


def _order_type(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in {"maker", "taker"}:
        return "taker"
    return normalized


def _fee_bps(settings: Settings, order_type: str) -> float:
    if order_type == "maker":
        return max(0.0, settings.maker_fee_bps)
    return max(0.0, settings.taker_fee_bps)
