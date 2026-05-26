from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from futures_lab.config import Settings
from futures_lab.models import MarketState, Side


@dataclass
class ExitPolicyState:
    name: str
    closed: bool = False
    exit_price: float | None = None
    exit_reason: str | None = None
    closed_at: datetime | None = None
    activated: bool = False
    max_favorable_move_pct: float = 0.0
    max_adverse_move_pct: float = 0.0
    partial_fraction: float = 0.0
    partial_taken: bool = False
    partial_price: float | None = None
    partial_move_pct: float | None = None
    partial_closed_at: datetime | None = None


@dataclass
class ExitShadowEvaluator:
    settings: Settings
    symbol: str
    side: Side
    entry_price: float
    notional_usd: float
    opened_at: datetime
    policies: dict[str, ExitPolicyState] = field(init=False)

    def __post_init__(self) -> None:
        partial = self._partial_fraction()
        self.policies = {
            "fixed_tp_stop": ExitPolicyState("fixed_tp_stop"),
            "fee_breakeven_trail": ExitPolicyState("fee_breakeven_trail"),
            "mfe_trailing_stop": ExitPolicyState("mfe_trailing_stop"),
            "partial_tp": ExitPolicyState("partial_tp", partial_fraction=partial),
            "partial_tp_alt": ExitPolicyState("partial_tp_alt", partial_fraction=partial),
            "time_decay": ExitPolicyState("time_decay"),
        }

    def mark(self, market: MarketState, timestamp: datetime) -> None:
        if not self.settings.exit_shadow_enabled or market.mid_price is None:
            return
        price = market.mid_price
        move = self._move_pct(price)
        elapsed = max(0.0, (timestamp - self.opened_at).total_seconds())
        round_trip_fee_pct = self._round_trip_fee_pct()
        fee_trail_activation = round_trip_fee_pct * self.settings.exit_shadow_fee_trail_activation_fee_multiple
        fee_trail_floor = round_trip_fee_pct * self.settings.exit_shadow_fee_trail_floor_fee_multiple
        time_decay_mfe = round_trip_fee_pct * self.settings.exit_shadow_time_decay_min_mfe_fee_multiple

        for policy in self.policies.values():
            self._update_excursion(policy, move)

        fee_policy = self.policies["fee_breakeven_trail"]
        if not fee_policy.closed:
            if move >= fee_trail_activation:
                fee_policy.activated = True
            if fee_policy.activated and move <= fee_trail_floor:
                self._close_policy(fee_policy, price, "fee_breakeven_trail", timestamp)

        mfe_policy = self.policies["mfe_trailing_stop"]
        if not mfe_policy.closed:
            if move >= self.settings.exit_shadow_mfe_trail_activation_pct:
                mfe_policy.activated = True
            trail_distance = max(
                self.settings.exit_shadow_mfe_trail_distance_pct,
                (market.realized_vol_60s_pct or 0.0) * 0.75,
            )
            if mfe_policy.activated and (mfe_policy.max_favorable_move_pct - move) >= trail_distance:
                self._close_policy(mfe_policy, price, "mfe_trailing_stop", timestamp)

        self._mark_partial_policy(
            self.policies["partial_tp"],
            price,
            move,
            timestamp,
            self.settings.exit_shadow_partial_take_profit_pct,
        )
        self._mark_partial_policy(
            self.policies["partial_tp_alt"],
            price,
            move,
            timestamp,
            self.settings.exit_shadow_partial_take_profit_alt_pct,
        )

        decay_policy = self.policies["time_decay"]
        if not decay_policy.closed and elapsed >= self.settings.exit_shadow_time_decay_seconds:
            flow_faded = self._flow_faded(market)
            if decay_policy.max_favorable_move_pct < time_decay_mfe and (move <= 0.0 or flow_faded):
                self._close_policy(decay_policy, price, "time_decay", timestamp)

    def close_at_actual(self, exit_price: float, exit_reason: str, closed_at: datetime) -> dict:
        if not self.settings.exit_shadow_enabled:
            return {}
        for policy in self.policies.values():
            if not policy.closed:
                self._close_policy(policy, exit_price, f"actual_{exit_reason}", closed_at)
        return self.summary()

    def summary(self) -> dict:
        rows = {name: self._policy_summary(policy) for name, policy in self.policies.items()}
        fixed = rows.get("fixed_tp_stop") or {}
        fixed_net = fixed.get("net_pnl_usd")
        for row in rows.values():
            if fixed_net is not None and row.get("net_pnl_usd") is not None:
                row["net_vs_fixed_usd"] = row["net_pnl_usd"] - fixed_net
        best_name = None
        best_net = None
        for name, row in rows.items():
            net = row.get("net_pnl_usd")
            if net is not None and (best_net is None or net > best_net):
                best_name = name
                best_net = net
        return {
            "enabled": True,
            "round_trip_fee_pct": self._round_trip_fee_pct(),
            "policies": rows,
            "best_policy": best_name,
            "best_net_pnl_usd": best_net,
        }

    def _mark_partial_policy(
        self,
        policy: ExitPolicyState,
        price: float,
        move: float,
        timestamp: datetime,
        threshold: float,
    ) -> None:
        if policy.closed:
            return
        if not policy.partial_taken and move >= threshold:
            policy.activated = True
            policy.partial_taken = True
            policy.partial_price = price
            policy.partial_move_pct = move
            policy.partial_closed_at = timestamp

    def _update_excursion(self, policy: ExitPolicyState, move: float) -> None:
        policy.max_favorable_move_pct = max(policy.max_favorable_move_pct, move)
        policy.max_adverse_move_pct = min(policy.max_adverse_move_pct, move)

    def _close_policy(self, policy: ExitPolicyState, price: float, reason: str, closed_at: datetime) -> None:
        policy.closed = True
        policy.exit_price = price
        policy.exit_reason = reason
        policy.closed_at = closed_at

    def _policy_summary(self, policy: ExitPolicyState) -> dict:
        if policy.exit_price is None or policy.exit_reason is None or policy.closed_at is None:
            return {
                "closed": False,
                "activated": policy.activated,
                "max_favorable_move_pct": policy.max_favorable_move_pct,
                "max_adverse_move_pct": policy.max_adverse_move_pct,
            }
        final_move = self._move_pct(policy.exit_price)
        if policy.partial_taken and policy.partial_move_pct is not None:
            remaining = 1.0 - policy.partial_fraction
            gross = (
                policy.partial_fraction * policy.partial_move_pct * self.notional_usd
                + remaining * final_move * self.notional_usd
            )
        else:
            gross = final_move * self.notional_usd
        fees = self.notional_usd * 2 * (self.settings.taker_fee_bps / 10_000)
        return {
            "closed": True,
            "activated": policy.activated,
            "exit_price": policy.exit_price,
            "exit_reason": policy.exit_reason,
            "closed_at": policy.closed_at.isoformat(),
            "gross_pnl_usd": gross,
            "fees_usd": fees,
            "net_pnl_usd": gross - fees,
            "max_favorable_move_pct": policy.max_favorable_move_pct,
            "max_adverse_move_pct": policy.max_adverse_move_pct,
            "time_in_trade_seconds": (policy.closed_at - self.opened_at).total_seconds(),
            "partial_taken": policy.partial_taken,
            "partial_fraction": policy.partial_fraction,
            "partial_price": policy.partial_price,
            "partial_move_pct": policy.partial_move_pct,
            "partial_closed_at": policy.partial_closed_at.isoformat() if policy.partial_closed_at else None,
        }

    def _move_pct(self, price: float) -> float:
        direction = 1 if self.side == Side.long else -1
        return ((price - self.entry_price) / self.entry_price) * direction

    def _flow_faded(self, market: MarketState) -> bool:
        r60 = market.return_60s_pct or 0.0
        buy_10s = market.taker_buy_ratio_10s
        if self.side == Side.long:
            return r60 <= 0.0 or (buy_10s is not None and buy_10s < 0.50)
        return r60 >= 0.0 or (buy_10s is not None and buy_10s > 0.50)

    def _round_trip_fee_pct(self) -> float:
        return 2 * (self.settings.taker_fee_bps / 10_000)

    def _partial_fraction(self) -> float:
        return min(1.0, max(0.0, self.settings.exit_shadow_partial_fraction))
