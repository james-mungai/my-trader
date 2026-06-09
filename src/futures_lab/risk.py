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
        lag_ms = market.book_freshness_lag_ms if market.book_freshness_lag_ms is not None else market.hot_freshness_lag_ms
        if lag_ms is not None and lag_ms > self.settings.max_exchange_event_lag_ms:
            blockers.append(f"book exchange event lag too high: {lag_ms:.0f}ms")
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
            blockers.extend(self._paper_live_edge_blockers(decision, effective_cost.total_cost_bps))
        if decision.stop_move_pct is not None and decision.leverage is not None:
            leveraged_stop_loss = decision.stop_move_pct * decision.leverage
            if leveraged_stop_loss > 0.60:
                blockers.append(f"leveraged stop risks {leveraged_stop_loss:.1%} of stake")

        if blockers:
            return RiskVerdict(allowed=False, reason="Risk blocked proposal.", blockers=blockers)
        return RiskVerdict(allowed=True, reason="Proposal passed paper risk checks.")

    def _paper_live_edge_blockers(self, decision: Decision, cost_bps: float) -> list[str]:
        if not self.settings.paper_live_edge_gate_enabled:
            return []
        blockers: list[str] = []
        target_bps = float(decision.target_move_pct or 0.0) * 10_000
        required_target_bps = cost_bps * self.settings.paper_live_min_target_cost_multiple
        if target_bps < required_target_bps:
            blockers.append(
                "paper live target does not clear edge costs: "
                f"{target_bps:.2f}bps < {required_target_bps:.2f}bps "
                f"(cost={cost_bps:.2f}bps, multiple={self.settings.paper_live_min_target_cost_multiple:.2f})"
            )

        selected = self._selected_edge_candidate(decision)
        if selected is None:
            blockers.append("paper live edge gate missing selected candidate")
            return blockers

        expected_ev_bps = self._float_value(selected.get("expected_ev_bps"))
        if expected_ev_bps is None or expected_ev_bps < self.settings.paper_live_min_expected_ev_bps:
            blockers.append(
                "paper live expected EV too low: "
                f"{expected_ev_bps if expected_ev_bps is not None else 'missing'} "
                f"< {self.settings.paper_live_min_expected_ev_bps:.2f}bps"
            )
        score = self._float_value(selected.get("score"))
        if score is None or score < self.settings.paper_live_min_score:
            blockers.append(
                "paper live score too low: "
                f"{score if score is not None else 'missing'} < {self.settings.paper_live_min_score:.2f}"
            )
        probability = self._float_value(selected.get("p_hit_tp_before_sl"))
        if probability is None or probability < self.settings.paper_live_min_tp_probability:
            blockers.append(
                "paper live TP probability too low: "
                f"{probability if probability is not None else 'missing'} < {self.settings.paper_live_min_tp_probability:.2f}"
            )
        return blockers

    def _selected_edge_candidate(self, decision: Decision) -> dict | None:
        edge_router = decision.evidence.get("edge_router") or {}
        if not isinstance(edge_router, dict):
            return None
        selected = edge_router.get("selected") or edge_router.get("selected_candidate")
        return selected if isinstance(selected, dict) else None

    @staticmethod
    def _float_value(value: object) -> float | None:
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

