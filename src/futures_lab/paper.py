from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from futures_lab.config import Settings
from futures_lab.exit_shadow import ExitShadowEvaluator

from futures_lab.models import (
    Decision,
    DecisionAction,
    MarketState,
    PaperPosition,
    PaperState,
    PaperTrade,
    Side,
    utc_now,
)


@dataclass
class RangeExitCounterfactual:
    symbol: str
    side: Side
    trade_profile: str
    entry_price: float
    actual_exit_price: float
    quantity: float
    stake_usd: float
    notional_usd: float
    leverage: int
    take_profit_price: float
    structural_invalidation_price: float | None
    structural_adverse_move_pct: float | None
    opened_at: datetime
    actual_closed_at: datetime
    actual_exit_reason: str
    actual_net_pnl_usd: float
    taker_fee_bps: float
    emergency_adverse_move_pct: float
    inner_loss_move_pct: float
    rough_isolated_liquidation_price: float | None
    rough_cross_liquidation_price: float | None
    max_horizon_seconds: int
    max_favorable_move_pct: float = 0.0
    max_adverse_move_pct: float = 0.0
    max_favorable_price: float | None = None
    max_adverse_price: float | None = None
    max_favorable_at: datetime | None = None
    max_adverse_at: datetime | None = None
    hit_events: list[dict[str, Any]] = field(default_factory=list)
    _hit_names: set[str] = field(default_factory=set)

    @property
    def id(self) -> str:
        return f"{self.symbol}:{self.opened_at.isoformat()}:{self.actual_closed_at.isoformat()}:{self.side.value}"

    def open_event(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side.value,
            "trade_profile": self.trade_profile,
            "entry_price": self.entry_price,
            "actual_exit_price": self.actual_exit_price,
            "take_profit_price": self.take_profit_price,
            "structural_invalidation_price": self.structural_invalidation_price,
            "rough_isolated_liquidation_price": self.rough_isolated_liquidation_price,
            "rough_cross_liquidation_price": self.rough_cross_liquidation_price,
            "opened_at": self.opened_at.isoformat(),
            "actual_closed_at": self.actual_closed_at.isoformat(),
            "actual_exit_reason": self.actual_exit_reason,
            "actual_net_pnl_usd": self.actual_net_pnl_usd,
            "notional_usd": self.notional_usd,
            "leverage": self.leverage,
            "max_horizon_seconds": self.max_horizon_seconds,
        }

    def mark(self, price: float, current: datetime) -> dict[str, Any] | None:
        self._update_excursion(price, current)
        for name, threshold in [
            ("inner_loss", self._inner_loss_price()),
            ("rough_isolated_liquidation", self.rough_isolated_liquidation_price),
        ]:
            if threshold is not None and self._crossed_adverse(price, threshold):
                self._record_hit(name, price, threshold, current)

        if self._crossed_favorable(price, self.take_profit_price):
            return self._close("target_after_time_decay", price, current, threshold=self.take_profit_price)
        if (
            self.structural_invalidation_price is not None
            and self._crossed_adverse(price, self.structural_invalidation_price)
        ):
            return self._close(
                "structural_invalidation_after_time_decay",
                price,
                current,
                threshold=self.structural_invalidation_price,
            )
        emergency_price = self._emergency_price()
        if self._crossed_adverse(price, emergency_price):
            return self._close("emergency_after_time_decay", price, current, threshold=emergency_price)
        if (
            self.rough_cross_liquidation_price is not None
            and self._crossed_adverse(price, self.rough_cross_liquidation_price)
        ):
            return self._close(
                "rough_cross_liquidation_after_time_decay",
                price,
                current,
                threshold=self.rough_cross_liquidation_price,
            )
        elapsed = (current - self.opened_at).total_seconds()
        if self.max_horizon_seconds > 0 and elapsed >= self.max_horizon_seconds:
            return self._close("counterfactual_max_horizon", price, current)
        return None

    def close_at_session_end(self, price: float, current: datetime, reason: str = "session_end") -> dict[str, Any]:
        self._update_excursion(price, current)
        return self._close(f"counterfactual_{reason}", price, current)

    def _close(
        self,
        reason: str,
        price: float,
        current: datetime,
        threshold: float | None = None,
    ) -> dict[str, Any]:
        if threshold is not None:
            self._record_hit(reason, price, threshold, current)
        counterfactual_net = self._net_pnl(price)
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side.value,
            "trade_profile": self.trade_profile,
            "entry_price": self.entry_price,
            "actual_exit_price": self.actual_exit_price,
            "counterfactual_exit_price": price,
            "actual_exit_reason": self.actual_exit_reason,
            "counterfactual_exit_reason": reason,
            "opened_at": self.opened_at.isoformat(),
            "actual_closed_at": self.actual_closed_at.isoformat(),
            "counterfactual_closed_at": current.isoformat(),
            "seconds_after_actual_exit": (current - self.actual_closed_at).total_seconds(),
            "seconds_from_entry": (current - self.opened_at).total_seconds(),
            "actual_net_pnl_usd": self.actual_net_pnl_usd,
            "counterfactual_net_pnl_usd": counterfactual_net,
            "net_delta_vs_actual_usd": counterfactual_net - self.actual_net_pnl_usd,
            "counterfactual_false_positive_exit": reason == "target_after_time_decay",
            "max_favorable_move_pct": self.max_favorable_move_pct,
            "max_adverse_move_pct": self.max_adverse_move_pct,
            "max_favorable_price": self.max_favorable_price,
            "max_adverse_price": self.max_adverse_price,
            "max_favorable_at": self.max_favorable_at.isoformat() if self.max_favorable_at else None,
            "max_adverse_at": self.max_adverse_at.isoformat() if self.max_adverse_at else None,
            "max_favorable_net_pnl_usd": self._net_pnl(self.max_favorable_price)
            if self.max_favorable_price is not None
            else None,
            "max_adverse_net_pnl_usd": self._net_pnl(self.max_adverse_price)
            if self.max_adverse_price is not None
            else None,
            "hit_events": self.hit_events,
            "rough_isolated_liquidation_price": self.rough_isolated_liquidation_price,
            "rough_cross_liquidation_price": self.rough_cross_liquidation_price,
            "structural_invalidation_price": self.structural_invalidation_price,
            "take_profit_price": self.take_profit_price,
        }

    def _record_hit(self, name: str, price: float, threshold: float, current: datetime) -> None:
        if name in self._hit_names:
            return
        self._hit_names.add(name)
        self.hit_events.append(
            {
                "name": name,
                "price": price,
                "threshold": threshold,
                "timestamp": current.isoformat(),
                "seconds_after_actual_exit": (current - self.actual_closed_at).total_seconds(),
                "seconds_from_entry": (current - self.opened_at).total_seconds(),
            }
        )

    def _update_excursion(self, price: float, current: datetime) -> None:
        move = self._move_pct(price)
        if self.max_favorable_price is None or move > self.max_favorable_move_pct:
            self.max_favorable_move_pct = move
            self.max_favorable_price = price
            self.max_favorable_at = current
        if self.max_adverse_price is None or move < self.max_adverse_move_pct:
            self.max_adverse_move_pct = move
            self.max_adverse_price = price
            self.max_adverse_at = current

    def _move_pct(self, price: float) -> float:
        direction = 1 if self.side == Side.long else -1
        return ((price - self.entry_price) / self.entry_price) * direction

    def _net_pnl(self, price: float | None) -> float:
        if price is None:
            return 0.0
        direction = 1 if self.side == Side.long else -1
        gross = (price - self.entry_price) * self.quantity * direction
        fees = (self.notional_usd * 2) * (self.taker_fee_bps / 10_000)
        return gross - fees

    def _crossed_favorable(self, price: float, threshold: float) -> bool:
        if self.side == Side.long:
            return price >= threshold
        return price <= threshold

    def _crossed_adverse(self, price: float, threshold: float) -> bool:
        if self.side == Side.long:
            return price <= threshold
        return price >= threshold

    def _inner_loss_price(self) -> float:
        if self.side == Side.long:
            return self.entry_price * (1 - abs(self.inner_loss_move_pct))
        return self.entry_price * (1 + abs(self.inner_loss_move_pct))

    def _emergency_price(self) -> float:
        if self.side == Side.long:
            return self.entry_price * (1 - abs(self.emergency_adverse_move_pct))
        return self.entry_price * (1 + abs(self.emergency_adverse_move_pct))


@dataclass
class PaperBroker:
    settings: Settings
    realized_pnl_usd: float = 0.0
    trades_today: int = 0
    open_position: PaperPosition | None = None
    last_trade: PaperTrade | None = None
    exit_shadow: ExitShadowEvaluator | None = None
    day: str = field(default_factory=lambda: date.today().isoformat())
    range_exit_counterfactuals: list[RangeExitCounterfactual] = field(default_factory=list)
    audit_events: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def state(self) -> PaperState:
        self._reset_day_if_needed()
        return PaperState(
            day=self.day,
            account_equity_usd=self.settings.account_equity_usd,
            realized_pnl_usd=self.realized_pnl_usd,
            trades_today=self.trades_today,
            daily_target_hit=self.realized_pnl_usd >= self.settings.daily_target_usd,
            daily_max_loss_hit=self.realized_pnl_usd <= -abs(self.settings.daily_max_loss_usd),
            open_position=self.open_position,
            last_trade=self.last_trade,
        )

    def reset(self) -> PaperState:
        self.realized_pnl_usd = 0.0
        self.trades_today = 0
        self.open_position = None
        self.last_trade = None
        self.exit_shadow = None
        self.range_exit_counterfactuals.clear()
        self.audit_events.clear()
        self.day = date.today().isoformat()
        return self.state()

    def open_from_decision(
        self,
        decision: Decision,
        opened_at: datetime | None = None,
        notional_usd: float | None = None,
    ) -> PaperPosition | None:
        self._reset_day_if_needed()
        if self.open_position is not None:
            return None
        if decision.action not in {DecisionAction.propose_long, DecisionAction.propose_short}:
            return None
        if (
            decision.entry_price is None
            or decision.take_profit_price is None
            or decision.stop_loss_price is None
            or decision.leverage is None
            or decision.mode is None
        ):
            return None
        side = Side.long if decision.action == DecisionAction.propose_long else Side.short
        if notional_usd is not None and notional_usd > 0:
            notional = float(notional_usd)
            stake = notional / decision.leverage
        else:
            stake = self.settings.stake_usd
            notional = stake * decision.leverage
        quantity = notional / decision.entry_price
        actual_opened_at = opened_at or utc_now()
        trade_profile = str(decision.evidence.get("trade_profile") or decision.mode.value)
        exit_policy = self._exit_policy_for_profile(trade_profile)
        structural_risk = self._structural_risk_for_decision(decision, side)
        self.open_position = PaperPosition(
            symbol=decision.symbol,
            side=side,
            mode=decision.mode,
            trade_profile=trade_profile,
            exit_policy=exit_policy,
            entry_price=decision.entry_price,
            quantity=quantity,
            stake_usd=stake,
            notional_usd=notional,
            leverage=decision.leverage,
            take_profit_price=decision.take_profit_price,
            stop_loss_price=decision.stop_loss_price,
            structural_invalidation_price=structural_risk.get("invalidation_price"),
            structural_adverse_move_pct=structural_risk.get("structural_adverse_move_pct"),
            opened_at=actual_opened_at,
            confidence=decision.confidence,
        )
        self.exit_shadow = ExitShadowEvaluator(
            settings=self.settings,
            symbol=decision.symbol,
            side=side,
            entry_price=decision.entry_price,
            notional_usd=notional,
            opened_at=actual_opened_at,
        )
        return self.open_position

    def mark(self, market: MarketState, timestamp: datetime | None = None) -> PaperTrade | None:
        self._reset_day_if_needed()
        if market.mid_price is None:
            return None
        if not market.connected:
            return None
        if market.data_age_seconds is not None and market.data_age_seconds > self.settings.stale_after_seconds:
            return None
        current = timestamp or utc_now()
        self._mark_range_exit_counterfactuals(market.mid_price, current)
        if self.open_position is None:
            return None
        pos = self.open_position
        self._update_excursion(pos, market.mid_price)
        if self.exit_shadow is not None:
            self.exit_shadow.mark(market, current)
        if pos.side == Side.long:
            if market.mid_price >= pos.take_profit_price:
                return self.close(market.mid_price, "take_profit", closed_at=current)
            if self.settings.enable_price_stop and market.mid_price <= pos.stop_loss_price:
                return self.close(market.mid_price, "stop_loss", closed_at=current)
            if self._should_close_structural_invalidation(pos, market.mid_price):
                return self.close(market.mid_price, "structural_invalidation", closed_at=current)
        else:
            if market.mid_price <= pos.take_profit_price:
                return self.close(market.mid_price, "take_profit", closed_at=current)
            if self.settings.enable_price_stop and market.mid_price >= pos.stop_loss_price:
                return self.close(market.mid_price, "stop_loss", closed_at=current)
            if self._should_close_structural_invalidation(pos, market.mid_price):
                return self.close(market.mid_price, "structural_invalidation", closed_at=current)
        if self._should_close_range_time_decay(pos, market, current):
            return self.close(market.mid_price, "range_time_decay", closed_at=current)
        if self._should_close_mfe_trailing(pos, market):
            return self.close(market.mid_price, "mfe_trailing_stop", closed_at=current)
        if pos.max_adverse_move_pct <= -abs(self._emergency_adverse_limit(pos)):
            return self.close(market.mid_price, "emergency_adverse_move", closed_at=current)
        if self._should_close_fast_failure(pos, current):
            return self.close(market.mid_price, "fast_failure", closed_at=current)
        if self._should_close_max_hold(pos, current):
            return self.close(market.mid_price, "max_hold", closed_at=current)
        return None

    def close_open_position(
        self,
        market: MarketState | None,
        reason: str = "session_end",
        closed_at: datetime | None = None,
    ) -> PaperTrade | None:
        if self.open_position is None or market is None or market.mid_price is None:
            return None
        current = closed_at or market.last_received_at or utc_now()
        self._update_excursion(self.open_position, market.mid_price)
        if self.exit_shadow is not None:
            self.exit_shadow.mark(market, current)
        return self.close(market.mid_price, reason, closed_at=current)

    def _update_excursion(self, pos: PaperPosition, price: float) -> None:
        direction = 1 if pos.side == Side.long else -1
        move = ((price - pos.entry_price) / pos.entry_price) * direction
        pos.max_favorable_move_pct = max(pos.max_favorable_move_pct, move)
        pos.max_adverse_move_pct = min(pos.max_adverse_move_pct, move)

    def _structural_risk_for_decision(self, decision: Decision, side: Side) -> dict:
        risk_by_side = decision.evidence.get("range_bound_structural_risk")
        if not isinstance(risk_by_side, dict):
            return {}
        risk = risk_by_side.get(side.value)
        return risk if isinstance(risk, dict) else {}

    def _should_close_structural_invalidation(self, pos: PaperPosition, price: float) -> bool:
        if not self.settings.range_bound_structural_exit_enabled:
            return False
        if pos.trade_profile != "range_bound_support_resistance" or pos.structural_invalidation_price is None:
            return False
        if pos.side == Side.long:
            return price <= pos.structural_invalidation_price
        return price >= pos.structural_invalidation_price

    def _emergency_adverse_limit(self, pos: PaperPosition) -> float:
        limit = abs(self.settings.emergency_max_adverse_move_pct)
        if (
            pos.trade_profile == "range_bound_support_resistance"
            and pos.structural_adverse_move_pct is not None
            and self.settings.range_bound_structural_risk_enabled
        ):
            return max(limit, abs(pos.structural_adverse_move_pct))
        return limit

    def _should_close_fast_failure(self, pos: PaperPosition, current: datetime) -> bool:
        if not self.settings.enable_fast_failure_exit:
            return False
        if pos.mode.value != "fast":
            return False
        elapsed_seconds = (current - pos.opened_at).total_seconds()
        if elapsed_seconds < self.settings.fast_failure_seconds:
            return False
        return pos.max_favorable_move_pct < self.settings.fast_failure_min_favorable_move_pct

    def _should_close_max_hold(self, pos: PaperPosition, current: datetime) -> bool:
        if self.settings.max_position_seconds <= 0:
            return False
        return (current - pos.opened_at).total_seconds() >= self.settings.max_position_seconds

    def _should_close_mfe_trailing(self, pos: PaperPosition, market: MarketState) -> bool:
        if pos.exit_policy != "mfe_trailing_stop" or market.mid_price is None:
            return False
        if pos.max_favorable_move_pct < self.settings.paper_mfe_trail_activation_pct:
            return False
        trail_distance = max(
            self.settings.paper_mfe_trail_distance_pct,
            (market.realized_vol_60s_pct or 0.0) * self.settings.paper_mfe_trail_vol_multiplier,
        )
        return (pos.max_favorable_move_pct - self._move_pct(pos, market.mid_price)) >= trail_distance

    def _should_close_range_time_decay(
        self,
        pos: PaperPosition,
        market: MarketState,
        current: datetime,
    ) -> bool:
        if not self.settings.range_bound_time_decay_exit_enabled:
            return False
        if pos.trade_profile != "range_bound_support_resistance" or market.mid_price is None:
            return False
        elapsed_seconds = (current - pos.opened_at).total_seconds()
        if elapsed_seconds < self.settings.range_bound_time_decay_seconds:
            return False
        round_trip_fee_pct = 2 * (self.settings.taker_fee_bps / 10_000)
        required_mfe = round_trip_fee_pct * self.settings.range_bound_time_decay_min_mfe_fee_multiple
        if pos.max_favorable_move_pct >= required_mfe:
            return False
        move = self._move_pct(pos, market.mid_price)
        if move <= -abs(self.settings.range_bound_time_decay_adverse_move_pct):
            return True
        if not self._range_flow_faded(pos, market):
            return False
        required_flow_adverse = abs(self.settings.range_bound_time_decay_flow_adverse_move_pct)
        if pos.structural_adverse_move_pct is not None:
            structural_threshold = abs(pos.structural_adverse_move_pct) * abs(
                self.settings.range_bound_time_decay_structural_fraction
            )
            required_flow_adverse = max(
                required_flow_adverse,
                min(abs(self.settings.range_bound_time_decay_adverse_move_pct), structural_threshold),
            )
        return move <= -required_flow_adverse

    def _range_flow_faded(self, pos: PaperPosition, market: MarketState) -> bool:
        r60 = market.return_60s_pct or 0.0
        buy_10s = market.taker_buy_ratio_10s
        if pos.side == Side.long:
            return r60 <= 0.0 or (buy_10s is not None and buy_10s < 0.50)
        return r60 >= 0.0 or (buy_10s is not None and buy_10s > 0.50)

    def _move_pct(self, pos: PaperPosition, price: float) -> float:
        direction = 1 if pos.side == Side.long else -1
        return ((price - pos.entry_price) / pos.entry_price) * direction

    def _exit_policy_for_profile(self, trade_profile: str) -> str:
        policy = self.settings.paper_exit_policy.strip().lower()
        if policy != "mfe_trailing_stop":
            return "fixed_tp_stop"
        allowed_profiles = {
            item.strip()
            for item in self.settings.paper_mfe_trailing_profiles.split(",")
            if item.strip()
        }
        if allowed_profiles and trade_profile not in allowed_profiles:
            return "fixed_tp_stop"
        return "mfe_trailing_stop"

    def close(self, exit_price: float, reason: str, closed_at: datetime | None = None) -> PaperTrade:
        if self.open_position is None:
            raise ValueError("No paper position is open.")
        pos = self.open_position
        direction = 1 if pos.side == Side.long else -1
        gross = (exit_price - pos.entry_price) * pos.quantity * direction
        fees = (pos.notional_usd * 2) * (self.settings.taker_fee_bps / 10_000)
        net = gross - fees
        actual_closed_at = closed_at or utc_now()
        exit_shadow = {}
        if self.exit_shadow is not None:
            exit_shadow = self.exit_shadow.close_at_actual(exit_price, reason, actual_closed_at)
        trade = PaperTrade(
            symbol=pos.symbol,
            side=pos.side,
            mode=pos.mode,
            trade_profile=pos.trade_profile,
            exit_policy=pos.exit_policy,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=pos.quantity,
            stake_usd=pos.stake_usd,
            notional_usd=pos.notional_usd,
            leverage=pos.leverage,
            structural_invalidation_price=pos.structural_invalidation_price,
            structural_adverse_move_pct=pos.structural_adverse_move_pct,
            gross_pnl_usd=gross,
            fees_usd=fees,
            net_pnl_usd=net,
            exit_reason=reason,
            opened_at=pos.opened_at,
            closed_at=actual_closed_at,
            exit_shadow=exit_shadow,
        )
        self._start_range_exit_counterfactual(pos, trade)
        self.open_position = None
        self.exit_shadow = None
        self.last_trade = trade
        self.realized_pnl_usd += net
        self.trades_today += 1
        return trade

    def drain_audit_events(self) -> list[tuple[str, dict[str, Any]]]:
        events = list(self.audit_events)
        self.audit_events.clear()
        return events

    def close_range_exit_counterfactuals(
        self,
        market: MarketState | None,
        reason: str = "session_end",
        timestamp: datetime | None = None,
    ) -> list[tuple[str, dict[str, Any]]]:
        if market is None or market.mid_price is None:
            return []
        current = timestamp or market.last_received_at or utc_now()
        closed = [
            ("range_exit_counterfactual_close", counterfactual.close_at_session_end(market.mid_price, current, reason))
            for counterfactual in self.range_exit_counterfactuals
        ]
        self.range_exit_counterfactuals.clear()
        return closed

    def _start_range_exit_counterfactual(self, pos: PaperPosition, trade: PaperTrade) -> None:
        if not self.settings.range_bound_exit_counterfactual_enabled:
            return
        if pos.trade_profile != "range_bound_support_resistance" or trade.exit_reason != "range_time_decay":
            return
        counterfactual = RangeExitCounterfactual(
            symbol=pos.symbol,
            side=pos.side,
            trade_profile=pos.trade_profile,
            entry_price=pos.entry_price,
            actual_exit_price=trade.exit_price,
            quantity=pos.quantity,
            stake_usd=pos.stake_usd,
            notional_usd=pos.notional_usd,
            leverage=pos.leverage,
            take_profit_price=pos.take_profit_price,
            structural_invalidation_price=pos.structural_invalidation_price,
            structural_adverse_move_pct=pos.structural_adverse_move_pct,
            opened_at=pos.opened_at,
            actual_closed_at=trade.closed_at,
            actual_exit_reason=trade.exit_reason,
            actual_net_pnl_usd=trade.net_pnl_usd,
            taker_fee_bps=self.settings.taker_fee_bps,
            emergency_adverse_move_pct=self._emergency_adverse_limit(pos),
            inner_loss_move_pct=self.settings.range_bound_stop_move_pct,
            rough_isolated_liquidation_price=self._rough_liquidation_price(
                pos,
                collateral_usd=pos.stake_usd,
            ),
            rough_cross_liquidation_price=self._rough_liquidation_price(
                pos,
                collateral_usd=self.settings.account_equity_usd
                * self.settings.range_bound_exit_counterfactual_cross_wallet_fraction,
            ),
            max_horizon_seconds=self.settings.range_bound_exit_counterfactual_horizon_seconds,
        )
        self.range_exit_counterfactuals.append(counterfactual)
        self.audit_events.append(("range_exit_counterfactual_open", counterfactual.open_event()))

    def _mark_range_exit_counterfactuals(self, price: float, current: datetime) -> None:
        if not self.range_exit_counterfactuals:
            return
        active: list[RangeExitCounterfactual] = []
        for counterfactual in self.range_exit_counterfactuals:
            event = counterfactual.mark(price, current)
            if event is None:
                active.append(counterfactual)
            else:
                self.audit_events.append(("range_exit_counterfactual_close", event))
        self.range_exit_counterfactuals = active

    def _rough_liquidation_price(self, pos: PaperPosition, collateral_usd: float) -> float | None:
        if collateral_usd <= 0 or pos.notional_usd <= 0:
            return None
        close_fee_pct = self.settings.taker_fee_bps / 10_000
        adverse_buffer = (
            collateral_usd / pos.notional_usd
            - self.settings.range_bound_exit_counterfactual_maintenance_margin_pct
            - close_fee_pct
        )
        if adverse_buffer <= 0:
            return pos.entry_price
        if pos.side == Side.long:
            return pos.entry_price * (1 - adverse_buffer)
        return pos.entry_price * (1 + adverse_buffer)

    def _reset_day_if_needed(self) -> None:
        today = date.today().isoformat()
        if self.day == today:
            return
        self.day = today
        self.realized_pnl_usd = 0.0
        self.trades_today = 0
        self.last_trade = None
