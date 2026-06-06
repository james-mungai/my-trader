from dataclasses import dataclass

from futures_lab.config import Settings
from futures_lab.costs import estimate_effective_cost
from futures_lab.models import Decision, DecisionAction, MarketState, PaperState, RiskVerdict


@dataclass
class RiskEngine:
    settings: Settings

    def evaluate(self, decision: Decision, market: MarketState, paper: PaperState) -> RiskVerdict:
        blockers: list[str] = []
        if decision.action not in {DecisionAction.propose_long, DecisionAction.propose_short}:
            return RiskVerdict(allowed=False, reason="No trade proposal.", blockers=["decision is wait/close"])
        if decision.confidence < self.settings.min_confidence:
            blockers.append(f"confidence {decision.confidence:.2f} < {self.settings.min_confidence:.2f}")
        if paper.open_position is not None:
            blockers.append("paper position already open")
        if paper.last_trade is not None and market.last_received_at is not None:
            elapsed_since_close = (market.last_received_at - paper.last_trade.closed_at).total_seconds()
            if elapsed_since_close < self.settings.trade_cooldown_seconds:
                blockers.append(
                    "trade cooldown active: "
                    f"{elapsed_since_close:.0f}s < {self.settings.trade_cooldown_seconds}s"
                )
        if paper.daily_target_hit:
            blockers.append("daily target already hit")
        if paper.daily_max_loss_hit:
            blockers.append("daily max loss hit")
        if self.settings.max_trades_per_day > 0 and paper.trades_today >= self.settings.max_trades_per_day:
            blockers.append("max trades per day hit")
        if market.data_age_seconds is None or market.data_age_seconds > self.settings.stale_after_seconds:
            blockers.append("market data stale")
        lag_ms = (
            market.avg_hot_event_lag_30s_ms
            if market.avg_hot_event_lag_30s_ms is not None
            else market.hot_event_lag_ms
        )
        if lag_ms is not None and lag_ms > self.settings.max_exchange_event_lag_ms:
            blockers.append(f"hot exchange event lag too high: {lag_ms:.0f}ms")
        if market.spread_bps is None or market.spread_bps > self.settings.max_spread_bps:
            blockers.append("spread too wide")
        if decision.target_move_pct is not None:
            effective_cost = estimate_effective_cost(self.settings, market)
            required_target_pct = effective_cost.required_target_pct(self.settings.min_gross_target_fee_multiple)
            if decision.target_move_pct < required_target_pct:
                blockers.append(
                    "target does not clear effective costs: "
                    f"{decision.target_move_pct:.4%} < {required_target_pct:.4%} "
                    f"(cost={effective_cost.total_cost_bps:.2f}bps)"
                )
        if decision.stop_move_pct is not None and decision.leverage is not None:
            leveraged_stop_loss = decision.stop_move_pct * decision.leverage
            if leveraged_stop_loss > 0.60:
                blockers.append(f"leveraged stop risks {leveraged_stop_loss:.1%} of stake")

        if blockers:
            return RiskVerdict(allowed=False, reason="Risk blocked proposal.", blockers=blockers)
        return RiskVerdict(allowed=True, reason="Proposal passed paper risk checks.")

