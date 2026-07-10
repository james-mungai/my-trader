from dataclasses import dataclass

from futures_lab.config import Settings
from futures_lab.costs import estimate_effective_cost
from futures_lab.models import Decision, DecisionAction, MarketState, PaperState, RiskVerdict


@dataclass
class RiskEngine:
    settings: Settings

    def evaluate(
        self,
        decision: Decision,
        market: MarketState,
        paper: PaperState,
        candidate_quality: dict | None = None,
    ) -> RiskVerdict:
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
            blockers.extend(self._paper_live_rolling_quality_blockers(decision, candidate_quality))
        selected = self._selected_edge_candidate(decision)
        if self._is_range_bound_candidate(selected):
            blockers.extend(self._range_bound_structural_risk_blockers(decision, selected))
            blockers.extend(self._range_bound_entry_quality_blockers(market, selected))
        elif decision.stop_move_pct is not None and decision.notional_usd is not None:
            effective_cost = estimate_effective_cost(self.settings, market)
            account_equity = max(0.000001, self.settings.account_equity_usd)
            account_risk_fraction = (
                (decision.stop_move_pct + effective_cost.total_cost_pct)
                * decision.notional_usd
                / account_equity
            )
            if account_risk_fraction > self.settings.max_account_risk_per_trade_fraction:
                blockers.append(
                    "account risk per trade too high: "
                    f"{account_risk_fraction:.1%} > {self.settings.max_account_risk_per_trade_fraction:.1%}"
                )

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
        for blocker in selected.get("blockers") or []:
            if self._is_soft_range_confirmation_blocker(selected, str(blocker)):
                continue
            blockers.append(f"paper live selected candidate blocked: {blocker}")

        expected_ev_bps = self._float_value(selected.get("expected_ev_bps"))
        min_expected_ev_bps = self._paper_live_min_expected_ev_bps(selected)
        if expected_ev_bps is None or expected_ev_bps < min_expected_ev_bps:
            blockers.append(
                "paper live expected EV too low: "
                f"{expected_ev_bps if expected_ev_bps is not None else 'missing'} "
                f"< {min_expected_ev_bps:.2f}bps"
            )
        score = self._float_value(selected.get("score"))
        min_score = self._paper_live_min_score(selected)
        if score is None or score < min_score:
            blockers.append(
                "paper live score too low: "
                f"{score if score is not None else 'missing'} < {min_score:.2f}"
            )
        probability = self._float_value(selected.get("p_hit_tp_before_sl"))
        min_probability = self._paper_live_min_tp_probability(selected)
        if probability is None or probability < min_probability:
            blockers.append(
                "paper live TP probability too low: "
                f"{probability if probability is not None else 'missing'} < {min_probability:.2f}"
            )
        return blockers

    def _paper_live_rolling_quality_blockers(self, decision: Decision, candidate_quality: dict | None) -> list[str]:
        if not self.settings.paper_live_rolling_edge_monitor_enabled:
            return []
        selected = self._selected_edge_candidate(decision)
        if self._is_range_bound_candidate(selected) and not self.settings.range_bound_rolling_edge_monitor_enabled:
            return []
        if not candidate_quality or not candidate_quality.get("enabled"):
            if self.settings.paper_live_rolling_block_until_ready:
                return ["paper live rolling edge monitor warming up: missing candidate quality snapshot"]
            return []
        if not candidate_quality.get("ready"):
            if self.settings.paper_live_rolling_block_until_ready:
                accepted_count = int(candidate_quality.get("accepted_count") or 0)
                rejected_count = int(candidate_quality.get("rejected_count") or 0)
                min_accepted = int(candidate_quality.get("min_accepted") or self.settings.paper_live_rolling_min_accepted)
                min_rejected = int(candidate_quality.get("min_rejected") or self.settings.paper_live_rolling_min_rejected)
                return [
                    "paper live rolling edge monitor warming up: "
                    f"accepted_count={accepted_count}/{min_accepted}, "
                    f"rejected_count={rejected_count}/{min_rejected}"
                ]
            return []
        if not candidate_quality.get("block"):
            return []
        return [
            "paper live rolling edge monitor blocked opens: "
            f"{candidate_quality.get('reason') or 'accepted candidates are not outperforming rejected candidates'}"
        ]

    def _selected_edge_candidate(self, decision: Decision) -> dict | None:
        edge_router = decision.evidence.get("edge_router") or {}
        if not isinstance(edge_router, dict):
            return None
        selected = edge_router.get("selected_candidate") or edge_router.get("selected")
        return selected if isinstance(selected, dict) else None

    def _range_bound_structural_risk_blockers(self, decision: Decision, selected: dict | None) -> list[str]:
        if not self.settings.range_bound_structural_risk_enabled:
            return []
        side = (selected or {}).get("side")
        risk_by_side = decision.evidence.get("range_bound_structural_risk")
        risk = risk_by_side.get(side) if isinstance(risk_by_side, dict) else None
        if not isinstance(risk, dict):
            return ["range structural risk missing"]
        blockers = [f"range structural risk blocked: {item}" for item in risk.get("blockers") or []]
        structural_adverse = self._float_value(risk.get("structural_adverse_move_pct"))
        account_drawdown = self._float_value(risk.get("account_drawdown_fraction"))
        if structural_adverse is None:
            blockers.append("range structural adverse move missing")
        elif structural_adverse > self.settings.range_bound_max_structural_adverse_move_pct:
            blockers.append(
                "range structural adverse move too wide: "
                f"{structural_adverse:.2%} > {self.settings.range_bound_max_structural_adverse_move_pct:.2%}"
            )
        if account_drawdown is None:
            blockers.append("range account drawdown missing")
        elif account_drawdown > self.settings.range_bound_max_account_drawdown_fraction:
            blockers.append(
                "range account drawdown too high: "
                f"{account_drawdown:.1%} > {self.settings.range_bound_max_account_drawdown_fraction:.1%}"
            )
        return blockers

    def _range_bound_entry_quality_blockers(self, market: MarketState, selected: dict | None) -> list[str]:
        if not self.settings.range_bound_entry_quality_gate_enabled:
            return []
        if not self._is_range_bound_candidate(selected):
            return []
        side = str((selected or {}).get("side") or "").lower()
        if side not in {"long", "short"}:
            return ["range entry quality side missing"]

        signed = 1.0 if side == "long" else -1.0
        taker_buy_ratio = self._float_value(market.taker_buy_ratio_10s)
        flow_alignment = (
            None
            if taker_buy_ratio is None
            else taker_buy_ratio if side == "long" else 1.0 - taker_buy_ratio
        )
        book_imbalance = self._float_value(market.book_imbalance_top) or 0.0
        depth_imbalance = (
            self._float_value(market.depth_imbalance_top5)
            if market.depth_imbalance_top5 is not None
            else book_imbalance
        )
        pressure = (book_imbalance + (depth_imbalance or 0.0)) / 2.0
        pressure_alignment = (
            self._clamp((pressure + 1.0) / 2.0)
            if side == "long"
            else self._clamp((-pressure + 1.0) / 2.0)
        )
        ofi_1s = signed * (self._float_value(market.order_flow_imbalance_1s) or 0.0)
        aggression_1s = signed * (self._float_value(market.taker_aggression_imbalance_1s) or 0.0)
        microprice_bps = signed * (
            self._float_value(market.microprice_mid_bps)
            if market.microprice_mid_bps is not None
            else self._float_value(market.vamp_mid_bps) or 0.0
        )
        return_15s = signed * (self._float_value(market.return_15s_pct) or 0.0)
        return_60s = signed * (self._float_value(market.return_60s_pct) or 0.0)

        passes: list[str] = []
        misses: list[str] = []
        if flow_alignment is not None and flow_alignment >= self.settings.range_bound_entry_min_flow_alignment:
            passes.append(f"flow={flow_alignment:.3f}")
        else:
            misses.append(f"flow={flow_alignment if flow_alignment is not None else 'missing'}")
        if pressure_alignment >= self.settings.range_bound_entry_min_pressure_alignment:
            passes.append(f"pressure={pressure_alignment:.3f}")
        else:
            misses.append(f"pressure={pressure_alignment:.3f}")
        if ofi_1s >= self.settings.range_bound_entry_min_ofi_1s:
            passes.append(f"ofi_1s={ofi_1s:.3f}")
        else:
            misses.append(f"ofi_1s={ofi_1s:.3f}")
        if aggression_1s >= self.settings.range_bound_entry_min_aggression_1s:
            passes.append(f"aggression_1s={aggression_1s:.3f}")
        else:
            misses.append(f"aggression_1s={aggression_1s:.3f}")
        if microprice_bps >= self.settings.range_bound_entry_min_microprice_bps:
            passes.append(f"microprice_bps={microprice_bps:.3f}")
        else:
            misses.append(f"microprice_bps={microprice_bps:.3f}")

        adverse_15s = return_15s < -abs(self.settings.range_bound_entry_max_adverse_return_15s_pct)
        adverse_60s = return_60s < -abs(self.settings.range_bound_entry_max_adverse_return_60s_pct)
        if not adverse_15s:
            passes.append(f"return_15s={return_15s:.4%}")
        else:
            misses.append(f"return_15s={return_15s:.4%}")
        if not adverse_60s:
            passes.append(f"return_60s={return_60s:.4%}")
        else:
            misses.append(f"return_60s={return_60s:.4%}")

        required = max(1, self.settings.range_bound_entry_min_confirmations)
        blockers: list[str] = []
        if adverse_15s:
            blockers.append("range entry 15s return is adverse")
        if adverse_60s:
            blockers.append("range entry 60s return is adverse")
        if len(passes) < required:
            blockers.append(
                "range entry quality not confirmed: "
                f"confirmations={len(passes)}/{required}; "
                f"passes={passes}; misses={misses}"
            )
        return blockers

    def _paper_live_min_expected_ev_bps(self, selected: dict) -> float:
        if self._is_range_bound_candidate(selected):
            return self.settings.range_bound_paper_live_min_expected_ev_bps
        return self.settings.paper_live_min_expected_ev_bps

    def _paper_live_min_score(self, selected: dict) -> float:
        if self._is_range_bound_candidate(selected):
            return self.settings.range_bound_paper_live_min_score
        return self.settings.paper_live_min_score

    def _paper_live_min_tp_probability(self, selected: dict) -> float:
        if self._is_range_bound_candidate(selected):
            return self.settings.range_bound_paper_live_min_tp_probability
        return self.settings.paper_live_min_tp_probability

    @staticmethod
    def _is_range_bound_candidate(selected: dict | None) -> bool:
        if not isinstance(selected, dict):
            return False
        return selected.get("family") == "range_bound" or selected.get("strategy") == "range_bound_support_resistance"

    def _is_soft_range_confirmation_blocker(self, selected: dict | None, blocker: str) -> bool:
        if not self.settings.range_bound_soft_confirmation_blockers_enabled:
            return False
        if not self._is_range_bound_candidate(selected):
            return False
        return blocker in {
            "higher_timeframe_hostile",
            "range_htf_edge_not_confirmed",
            "range_local_edge_not_confirmed",
            "range_flow_not_confirmed",
            "range_pressure_not_confirmed",
        }

    @staticmethod
    def _float_value(value: object) -> float | None:
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, value))

