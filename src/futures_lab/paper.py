from dataclasses import dataclass, field
from datetime import date, datetime

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
class PaperBroker:
    settings: Settings
    realized_pnl_usd: float = 0.0
    trades_today: int = 0
    open_position: PaperPosition | None = None
    last_trade: PaperTrade | None = None
    exit_shadow: ExitShadowEvaluator | None = None
    day: str = field(default_factory=lambda: date.today().isoformat())

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
        self.day = date.today().isoformat()
        return self.state()

    def open_from_decision(self, decision: Decision, opened_at: datetime | None = None) -> PaperPosition | None:
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
        stake = self.settings.stake_usd
        notional = stake * decision.leverage
        quantity = notional / decision.entry_price
        actual_opened_at = opened_at or utc_now()
        self.open_position = PaperPosition(
            symbol=decision.symbol,
            side=side,
            mode=decision.mode,
            entry_price=decision.entry_price,
            quantity=quantity,
            stake_usd=stake,
            notional_usd=notional,
            leverage=decision.leverage,
            take_profit_price=decision.take_profit_price,
            stop_loss_price=decision.stop_loss_price,
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
        if self.open_position is None or market.mid_price is None:
            return None
        if not market.connected:
            return None
        if market.data_age_seconds is not None and market.data_age_seconds > self.settings.stale_after_seconds:
            return None
        pos = self.open_position
        current = timestamp or utc_now()
        self._update_excursion(pos, market.mid_price)
        if self.exit_shadow is not None:
            self.exit_shadow.mark(market, current)
        if pos.side == Side.long:
            if market.mid_price >= pos.take_profit_price:
                return self.close(market.mid_price, "take_profit", closed_at=current)
            if self.settings.enable_price_stop and market.mid_price <= pos.stop_loss_price:
                return self.close(market.mid_price, "stop_loss", closed_at=current)
        else:
            if market.mid_price <= pos.take_profit_price:
                return self.close(market.mid_price, "take_profit", closed_at=current)
            if self.settings.enable_price_stop and market.mid_price >= pos.stop_loss_price:
                return self.close(market.mid_price, "stop_loss", closed_at=current)
        if pos.max_adverse_move_pct <= -abs(self.settings.emergency_max_adverse_move_pct):
            return self.close(market.mid_price, "emergency_adverse_move", closed_at=current)
        if self._should_close_fast_failure(pos, current):
            return self.close(market.mid_price, "fast_failure", closed_at=current)
        if self._should_close_max_hold(pos, current):
            return self.close(market.mid_price, "max_hold", closed_at=current)
        return None

    def _update_excursion(self, pos: PaperPosition, price: float) -> None:
        direction = 1 if pos.side == Side.long else -1
        move = ((price - pos.entry_price) / pos.entry_price) * direction
        pos.max_favorable_move_pct = max(pos.max_favorable_move_pct, move)
        pos.max_adverse_move_pct = min(pos.max_adverse_move_pct, move)

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
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=pos.quantity,
            stake_usd=pos.stake_usd,
            notional_usd=pos.notional_usd,
            leverage=pos.leverage,
            gross_pnl_usd=gross,
            fees_usd=fees,
            net_pnl_usd=net,
            exit_reason=reason,
            opened_at=pos.opened_at,
            closed_at=actual_closed_at,
            exit_shadow=exit_shadow,
        )
        self.open_position = None
        self.exit_shadow = None
        self.last_trade = trade
        self.realized_pnl_usd += net
        self.trades_today += 1
        return trade

    def _reset_day_if_needed(self) -> None:
        today = date.today().isoformat()
        if self.day == today:
            return
        self.day = today
        self.realized_pnl_usd = 0.0
        self.trades_today = 0
        self.open_position = None
        self.last_trade = None
        self.exit_shadow = None
