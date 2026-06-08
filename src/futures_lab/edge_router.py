from __future__ import annotations

from dataclasses import dataclass, field

from futures_lab.config import Settings
from futures_lab.costs import EffectiveCost, estimate_effective_cost
from futures_lab.cross_market import cross_market_gate
from futures_lab.models import MarketState, Side


@dataclass(frozen=True)
class EdgeCandidate:
    strategy: str
    family: str
    side: Side
    entry_type: str
    exit_type: str
    target_bps: float
    stop_bps: float
    max_hold_ms: int
    expected_cost_bps: float
    score: float
    p_hit_tp_before_sl: float
    expected_ev_bps: float
    blockers: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    exit_plan: dict = field(default_factory=dict)

    @property
    def viable(self) -> bool:
        return not self.blockers

    def model_dump(self) -> dict:
        return {
            "strategy": self.strategy,
            "family": self.family,
            "side": self.side.value,
            "entry_type": self.entry_type,
            "exit_type": self.exit_type,
            "target_bps": self.target_bps,
            "stop_bps": self.stop_bps,
            "max_hold_ms": self.max_hold_ms,
            "expected_cost_bps": self.expected_cost_bps,
            "score": self.score,
            "p_hit_tp_before_sl": self.p_hit_tp_before_sl,
            "expected_ev_bps": self.expected_ev_bps,
            "viable": self.viable,
            "blockers": self.blockers,
            "reasons": self.reasons,
            "exit_plan": self.exit_plan,
        }


@dataclass(frozen=True)
class BaselineCandidateInput:
    side: Side
    score: float
    target_bps: float
    stop_bps: float
    max_hold_ms: int = 60_000
    reasons: list[str] = field(default_factory=list)


@dataclass
class EdgeRouter:
    settings: Settings

    def evaluate(
        self,
        market: MarketState,
        baseline: BaselineCandidateInput | None = None,
    ) -> dict:
        if not self.settings.edge_router_shadow_enabled:
            return {"enabled": False, "candidates": [], "selected": None}
        mode = "paper_candidate" if self.settings.edge_router_paper_enabled else "shadow"
        candidates = self._microstructure_candidates(market)
        if baseline is not None:
            candidates.append(self._baseline_candidate(market, baseline))
        candidates_sorted = sorted(
            candidates,
            key=lambda candidate: (candidate.viable, candidate.expected_ev_bps, candidate.score),
            reverse=True,
        )
        selected = next((candidate for candidate in candidates_sorted if candidate.viable), None)
        return {
            "enabled": True,
            "mode": mode,
            "selected": selected.model_dump() if selected else None,
            "candidate_count": len(candidates_sorted),
            "viable_count": sum(1 for candidate in candidates_sorted if candidate.viable),
            "candidates": [candidate.model_dump() for candidate in candidates_sorted],
        }

    def _microstructure_candidates(self, market: MarketState) -> list[EdgeCandidate]:
        candidates = [
            self._taker_impulse_candidate(market, Side.long),
            self._taker_impulse_candidate(market, Side.short),
            self._liquidation_continuation_candidate(market, Side.long),
            self._liquidation_continuation_candidate(market, Side.short),
            self._liquidation_exhaustion_bounce_candidate(market, Side.long),
            self._liquidation_exhaustion_bounce_candidate(market, Side.short),
        ]
        if self.settings.maker_reversion_shadow_enabled:
            candidates.extend(
                [
                    self._maker_reversion_candidate(market, Side.long),
                    self._maker_reversion_candidate(market, Side.short),
                ]
            )
        return candidates

    def _baseline_candidate(self, market: MarketState, baseline: BaselineCandidateInput) -> EdgeCandidate:
        return self._build_candidate(
            market=market,
            strategy="stateful_momentum_baseline",
            family="baseline",
            side=baseline.side,
            target_bps=baseline.target_bps,
            stop_bps=baseline.stop_bps,
            max_hold_ms=baseline.max_hold_ms,
            score=baseline.score,
            reasons=baseline.reasons or ["current deterministic strategy score"],
        )

    def _taker_impulse_candidate(self, market: MarketState, side: Side) -> EdgeCandidate:
        signed = self._signed_features(market, side)
        micro_score = self._bounded(
            0.22 * self._positive_unit(signed["ofi_1s"])
            + 0.14 * self._positive_unit(signed["ofi_5s"])
            + 0.18 * self._positive_unit(signed["aggression_1s"])
            + 0.12 * self._positive_unit(signed["aggression_5s"])
            + 0.14 * self._positive_unit(signed["microprice_pressure"])
            + 0.12 * self._positive_unit(signed["vamp_pressure"])
            + 0.08 * self._positive_unit(signed["depth_pressure"])
        )
        cost = estimate_effective_cost(self.settings, market)
        candidate = self._build_candidate(
            market=market,
            strategy=f"taker_impulse_{side.value}",
            family="taker_impulse",
            side=side,
            target_bps=self.settings.fast_target_move_pct * 10_000,
            stop_bps=self.settings.fast_stop_move_pct * 10_000,
            max_hold_ms=self.settings.edge_router_impulse_max_hold_ms,
            score=micro_score,
            reasons=[
                "OFI/aggression/microprice/VAMP/depth impulse score with 1s/5s agreement",
                f"ofi_1s={market.order_flow_imbalance_1s}",
                f"ofi_5s={market.order_flow_imbalance_5s}",
                f"aggression_1s={market.taker_aggression_imbalance_1s}",
                f"aggression_5s={market.taker_aggression_imbalance_5s}",
            ],
            cost=cost,
            exit_plan=self._taker_impulse_exit_plan(cost),
        )
        blockers = self._taker_impulse_blockers(market, signed)
        if blockers:
            return self._with_blockers(candidate, blockers)
        return candidate

    def _liquidation_continuation_candidate(self, market: MarketState, side: Side) -> EdgeCandidate:
        relevant_liq = self._continuation_liquidation_notional(market, side)
        liq_score = min(1.0, relevant_liq / 250_000)
        phase_score = self._liquidation_phase_score(market, side, {"liquidation_impulse", "cascade_continuation"})
        signed = self._signed_features(market, side)
        score = self._bounded(
            0.25 * liq_score
            + 0.25 * phase_score
            + 0.25 * self._positive_unit(signed["ofi_1s"])
            + 0.15 * self._positive_unit(signed["aggression_1s"])
            + 0.10 * self._positive_unit(signed["depth_pressure"])
        )
        candidate = self._build_candidate(
            market=market,
            strategy=f"liquidation_continuation_{side.value}",
            family="liquidation_continuation",
            side=side,
            target_bps=max(self.settings.fast_target_move_pct * 10_000, 22.0),
            stop_bps=self.settings.fast_stop_move_pct * 10_000,
            max_hold_ms=self.settings.edge_router_liquidation_max_hold_ms,
            score=score,
            reasons=[
                f"liquidation_notional={relevant_liq}",
                f"liquidation_phase={market.liquidation_phase}",
                f"liquidation_phase_side={market.liquidation_phase_side}",
                "forceOrder treated as sampled largest-pulse event flag",
            ],
        )
        if relevant_liq <= 0:
            return self._with_blocker(candidate, "no_liquidation_pulse")
        if phase_score <= 0:
            return self._with_blocker(candidate, "liquidation_phase_not_continuation")
        return candidate

    def _liquidation_exhaustion_bounce_candidate(self, market: MarketState, side: Side) -> EdgeCandidate:
        relevant_liq = self._exhaustion_liquidation_notional(market, side)
        liq_score = min(1.0, relevant_liq / 250_000)
        phase_score = self._liquidation_phase_score(market, side, {"exhaustion_candidate", "reclaim_or_failed_reclaim"})
        signed = self._signed_features(market, side)
        score = self._bounded(
            0.20 * liq_score
            + 0.30 * phase_score
            + 0.18 * self._positive_unit(signed["microprice_pressure"])
            + 0.14 * self._positive_unit(signed["vamp_pressure"])
            + 0.12 * self._positive_unit(signed["refill_pressure"])
            + 0.06 * self._positive_unit(signed["ofi_1s"])
        )
        candidate = self._build_candidate(
            market=market,
            strategy=f"liquidation_exhaustion_bounce_{side.value}",
            family="liquidation_exhaustion_bounce",
            side=side,
            target_bps=self.settings.fast_target_move_pct * 10_000,
            stop_bps=max(8.0, self.settings.fast_stop_move_pct * 10_000 * 0.8),
            max_hold_ms=self.settings.edge_router_liquidation_max_hold_ms,
            score=score,
            reasons=[
                f"exhaustion_liquidation_notional={relevant_liq}",
                f"liquidation_phase={market.liquidation_phase}",
                f"liquidation_phase_side={market.liquidation_phase_side}",
                "forceOrder treated as sampled largest-pulse event flag",
                "delayed bounce requires exhaustion/reclaim phase",
            ],
        )
        if relevant_liq <= 0:
            return self._with_blocker(candidate, "no_exhaustion_pulse")
        if phase_score <= 0:
            return self._with_blocker(candidate, "cascade_not_exhausted")
        return candidate

    def _maker_reversion_candidate(self, market: MarketState, side: Side) -> EdgeCandidate:
        toxicity = self._maker_toxicity_score(market)
        queue_score = self._maker_queue_score(market, side)
        fair_value_offset = self._maker_fair_value_offset(market, side)
        score = self._bounded(0.45 * (1.0 - toxicity) + 0.35 * queue_score + 0.20 * fair_value_offset)
        cost = estimate_effective_cost(self.settings, market, entry_order_type="maker", exit_order_type="maker")
        candidate = self._build_candidate(
            market=market,
            strategy=f"maker_reversion_{side.value}",
            family="maker_reversion",
            side=side,
            target_bps=self.settings.maker_reversion_target_bps,
            stop_bps=self.settings.maker_reversion_stop_bps,
            max_hold_ms=self.settings.maker_reversion_max_hold_ms,
            score=score,
            reasons=[
                "passive maker reversion shadow candidate for calm low-toxicity regimes",
                f"toxicity_score={round(toxicity, 4)}",
                f"queue_score={round(queue_score, 4)}",
                f"fair_value_offset={round(fair_value_offset, 4)}",
            ],
            cost=cost,
            exit_plan={
                "baseline": "maker_shadow_only",
                "timeout_ms": self.settings.maker_reversion_max_hold_ms,
                "queue_score": round(queue_score, 4),
                "toxicity_score": round(toxicity, 4),
                "adverse_selection_guard": "hostile_replay_maker_adverse_selection_bps",
            },
        )
        blockers = self._maker_reversion_blockers(market, toxicity, queue_score)
        blockers.append("maker_shadow_only")
        return self._with_blockers(candidate, blockers)

    def _build_candidate(
        self,
        market: MarketState,
        strategy: str,
        family: str,
        side: Side,
        target_bps: float,
        stop_bps: float,
        max_hold_ms: int,
        score: float,
        reasons: list[str],
        cost: EffectiveCost | None = None,
        exit_plan: dict | None = None,
    ) -> EdgeCandidate:
        cost = cost or estimate_effective_cost(self.settings, market)
        p_hit = self._bounded(score)
        ev = p_hit * target_bps - (1.0 - p_hit) * stop_bps - cost.total_cost_bps
        blockers = self._base_blockers(market, side, target_bps, cost.total_cost_bps, score, ev)
        return EdgeCandidate(
            strategy=strategy,
            family=family,
            side=side,
            entry_type=cost.entry_order_type,
            exit_type=cost.exit_order_type,
            target_bps=target_bps,
            stop_bps=stop_bps,
            max_hold_ms=max_hold_ms,
            expected_cost_bps=cost.total_cost_bps,
            score=score,
            p_hit_tp_before_sl=p_hit,
            expected_ev_bps=ev,
            blockers=blockers,
            reasons=reasons,
            exit_plan=exit_plan or self._default_exit_plan(max_hold_ms),
        )

    def _taker_impulse_blockers(self, market: MarketState, signed: dict[str, float]) -> list[str]:
        blockers = []
        if signed["ofi_1s"] < self.settings.taker_impulse_min_ofi_1s:
            blockers.append("impulse_ofi_1s_not_aligned")
        if signed["ofi_5s"] < self.settings.taker_impulse_min_ofi_5s:
            blockers.append("impulse_ofi_5s_not_aligned")
        if signed["aggression_1s"] < self.settings.taker_impulse_min_aggression_1s:
            blockers.append("impulse_aggression_1s_not_aligned")
        if signed["aggression_5s"] < self.settings.taker_impulse_min_aggression_5s:
            blockers.append("impulse_aggression_5s_not_aligned")
        if signed["microprice_mid_bps"] <= self.settings.taker_impulse_min_pressure_bps:
            blockers.append("impulse_microprice_not_on_side")
        if signed["vamp_mid_bps"] <= self.settings.taker_impulse_min_pressure_bps:
            blockers.append("impulse_vamp_not_on_side")
        depth_pressure_ok = signed["depth_pressure"] >= self.settings.taker_impulse_min_depth_pressure
        refill_pressure_ok = signed["refill_pressure"] >= self.settings.taker_impulse_min_refill_pressure
        if not (depth_pressure_ok or refill_pressure_ok):
            blockers.append("impulse_depth_not_thinning_or_refilling")
        if market.spread_bps is None or market.spread_bps > self.settings.taker_impulse_max_spread_bps:
            blockers.append("impulse_spread_too_wide")
        if (
            market.spread_bps_std_5s is not None
            and market.spread_bps_std_5s > self.settings.taker_impulse_max_spread_std_bps
        ):
            blockers.append("impulse_spread_unstable")
        book_lag_ms = market.book_freshness_lag_ms if market.book_freshness_lag_ms is not None else market.hot_freshness_lag_ms
        if book_lag_ms is not None and book_lag_ms > self.settings.taker_impulse_max_event_lag_ms:
            blockers.append("impulse_book_lagged")
        trade_lag_ms = market.trade_freshness_lag_ms
        if trade_lag_ms is not None and trade_lag_ms > self.settings.taker_impulse_max_event_lag_ms:
            blockers.append("impulse_trade_lagged")
        return blockers

    def _base_blockers(
        self,
        market: MarketState,
        side: Side,
        target_bps: float,
        cost_bps: float,
        score: float,
        ev_bps: float,
    ) -> list[str]:
        blockers = []
        if market.data_age_seconds is None or market.data_age_seconds > self.settings.stale_after_seconds:
            blockers.append("stale_data")
        lag_ms = market.book_freshness_lag_ms if market.book_freshness_lag_ms is not None else market.hot_freshness_lag_ms
        if lag_ms is not None and lag_ms > self.settings.max_exchange_event_lag_ms:
            blockers.append("exchange_event_lagged")
        if market.spread_bps is None or market.spread_bps > self.settings.max_spread_bps:
            blockers.append("spread_too_wide")
        if market.mid_price is None:
            blockers.append("missing_mid_price")
        if score < self.settings.edge_router_min_score:
            blockers.append("score_below_min")
        if target_bps < cost_bps * self.settings.edge_router_target_cost_multiple:
            blockers.append("target_below_cost_multiple")
        if ev_bps < self.settings.edge_router_min_ev_bps:
            blockers.append("ev_below_min")
        bias_side = market.higher_timeframe_bias_side
        bias_strength = market.higher_timeframe_bias_strength
        if bias_side in {"long", "short"} and bias_side != side.value and bias_strength >= self.settings.higher_timeframe_gate_min_strength:
            blockers.append("higher_timeframe_hostile")
        cross_gate = cross_market_gate(self.settings, market, side, score)
        if not cross_gate["allowed"]:
            blockers.append(cross_gate["blocker"])
        return blockers

    def _signed_features(self, market: MarketState, side: Side) -> dict[str, float]:
        sign = 1.0 if side == Side.long else -1.0
        bid_refill = market.bid_depth_refill_rate_5s or 0.0
        ask_refill = market.ask_depth_refill_rate_5s or 0.0
        bid_evap = market.bid_depth_evaporation_rate_5s or 0.0
        ask_evap = market.ask_depth_evaporation_rate_5s or 0.0
        refill_pressure = (bid_refill + ask_evap) if side == Side.long else (ask_refill + bid_evap)
        return {
            "ofi_1s": sign * (market.order_flow_imbalance_1s or 0.0),
            "ofi_5s": sign * (market.order_flow_imbalance_5s or 0.0),
            "aggression_1s": sign * (market.taker_aggression_imbalance_1s or 0.0),
            "aggression_5s": sign * (market.taker_aggression_imbalance_5s or 0.0),
            "microprice_mid_bps": sign * (market.microprice_mid_bps or 0.0),
            "vamp_mid_bps": sign * (market.vamp_mid_bps or 0.0),
            "microprice_pressure": sign * self._scaled_bps(market.microprice_mid_bps, market.spread_bps),
            "vamp_pressure": sign * self._scaled_bps(market.vamp_mid_bps, market.spread_bps),
            "depth_pressure": sign * (market.depth_imbalance_top5 or market.book_imbalance_top or 0.0),
            "refill_pressure": refill_pressure,
        }

    def _maker_reversion_blockers(self, market: MarketState, toxicity: float, queue_score: float) -> list[str]:
        blockers = []
        if market.spread_bps is None or market.spread_bps > self.settings.maker_reversion_max_spread_bps:
            blockers.append("maker_spread_too_wide")
        if (
            market.spread_bps_std_5s is not None
            and market.spread_bps_std_5s > self.settings.maker_reversion_max_spread_std_bps
        ):
            blockers.append("maker_spread_unstable")
        if abs(market.order_flow_imbalance_1s or 0.0) > self.settings.maker_reversion_max_ofi:
            blockers.append("maker_ofi_toxic")
        if abs(market.order_flow_imbalance_5s or 0.0) > self.settings.maker_reversion_max_ofi:
            blockers.append("maker_ofi_5s_toxic")
        if abs(market.taker_aggression_imbalance_1s or 0.0) > self.settings.maker_reversion_max_aggression:
            blockers.append("maker_aggression_toxic")
        if abs(market.taker_aggression_imbalance_5s or 0.0) > self.settings.maker_reversion_max_aggression:
            blockers.append("maker_aggression_5s_toxic")
        if (market.realized_vol_60s_pct or 0.0) > self.settings.maker_reversion_max_vol_60s_pct:
            blockers.append("maker_vol_too_high")
        if (market.liquidation_notional_30s or 0.0) > 0:
            blockers.append("maker_liquidation_pulse")
        if market.liquidation_phase != "normal":
            blockers.append("maker_liquidation_phase_active")
        if queue_score < self.settings.maker_reversion_min_queue_score:
            blockers.append("maker_queue_score_low")
        if toxicity > 0.45:
            blockers.append("maker_toxicity_high")
        return blockers

    def _maker_toxicity_score(self, market: MarketState) -> float:
        spread_instability = self._bounded((market.spread_bps_std_5s or 0.0) / max(0.01, self.settings.maker_reversion_max_spread_std_bps))
        ofi = max(abs(market.order_flow_imbalance_1s or 0.0), abs(market.order_flow_imbalance_5s or 0.0))
        aggression = max(abs(market.taker_aggression_imbalance_1s or 0.0), abs(market.taker_aggression_imbalance_5s or 0.0))
        vol = self._bounded((market.realized_vol_60s_pct or 0.0) / max(0.000001, self.settings.maker_reversion_max_vol_60s_pct))
        liquidation = 1.0 if (market.liquidation_notional_30s or 0.0) > 0 or market.liquidation_phase != "normal" else 0.0
        return self._bounded(0.25 * spread_instability + 0.25 * ofi + 0.25 * aggression + 0.15 * vol + 0.10 * liquidation)

    def _maker_queue_score(self, market: MarketState, side: Side) -> float:
        if side == Side.long:
            own_qty = market.best_bid_qty
            opposite_qty = market.best_ask_qty
            refill = market.bid_depth_refill_rate_5s or 0.0
            evap = market.bid_depth_evaporation_rate_5s or 0.0
        else:
            own_qty = market.best_ask_qty
            opposite_qty = market.best_bid_qty
            refill = market.ask_depth_refill_rate_5s or 0.0
            evap = market.ask_depth_evaporation_rate_5s or 0.0
        if own_qty is None or opposite_qty is None:
            return 0.0
        total = own_qty + opposite_qty
        balance = own_qty / total if total > 0 else 0.0
        stability = self._bounded(0.5 + refill - evap)
        return self._bounded(0.55 * balance + 0.45 * stability)

    def _maker_fair_value_offset(self, market: MarketState, side: Side) -> float:
        sign = 1.0 if side == Side.long else -1.0
        micro = sign * (market.microprice_mid_bps or 0.0)
        vamp = sign * (market.vamp_mid_bps or 0.0)
        # Maker reversion wants fair value slightly away from our passive quote, not a runaway impulse.
        return self._bounded(1.0 - abs((micro + vamp) / max(0.1, 2 * (market.spread_bps or 1.0))))

    def _continuation_liquidation_notional(self, market: MarketState, side: Side) -> float:
        if side == Side.short:
            return market.long_liquidation_notional_30s or 0.0
        return market.short_liquidation_notional_30s or 0.0

    def _exhaustion_liquidation_notional(self, market: MarketState, side: Side) -> float:
        if side == Side.long:
            return market.long_liquidation_notional_30s or 0.0
        return market.short_liquidation_notional_30s or 0.0

    def _liquidation_phase_score(self, market: MarketState, side: Side, allowed_phases: set[str]) -> float:
        if market.liquidation_phase not in allowed_phases:
            return 0.0
        if market.liquidation_phase_side != side.value:
            return 0.0
        return self._bounded(market.liquidation_phase_confidence)

    def _with_blocker(self, candidate: EdgeCandidate, blocker: str) -> EdgeCandidate:
        return self._with_blockers(candidate, [blocker])

    def _with_blockers(self, candidate: EdgeCandidate, blockers: list[str]) -> EdgeCandidate:
        return EdgeCandidate(
            strategy=candidate.strategy,
            family=candidate.family,
            side=candidate.side,
            entry_type=candidate.entry_type,
            exit_type=candidate.exit_type,
            target_bps=candidate.target_bps,
            stop_bps=candidate.stop_bps,
            max_hold_ms=candidate.max_hold_ms,
            expected_cost_bps=candidate.expected_cost_bps,
            score=candidate.score,
            p_hit_tp_before_sl=candidate.p_hit_tp_before_sl,
            expected_ev_bps=candidate.expected_ev_bps,
            blockers=sorted(set(candidate.blockers + blockers)),
            reasons=candidate.reasons,
            exit_plan=candidate.exit_plan,
        )

    def _default_exit_plan(self, max_hold_ms: int) -> dict:
        return {
            "baseline": "fixed_tp_stop",
            "timeout_ms": max_hold_ms,
        }

    def _taker_impulse_exit_plan(self, cost: EffectiveCost) -> dict:
        trail_activation_bps = cost.total_cost_bps * self.settings.taker_impulse_trail_activation_cost_multiple
        return {
            "baseline": "fixed_tp_stop",
            "preferred": "mfe_trailing_stop_after_cost_paid",
            "trail_activation_bps": round(trail_activation_bps, 4),
            "soft_exit_signals": [
                "ofi_flip",
                "microprice_reclaims_or_loses_mid",
                "spread_expansion",
                "exit_side_depth_disappears",
            ],
            "timeout_ms": self.settings.edge_router_impulse_max_hold_ms,
        }

    def _positive_unit(self, value: float) -> float:
        return self._bounded((value + 1.0) / 2.0)

    def _scaled_bps(self, value: float | None, spread_bps: float | None) -> float:
        if value is None:
            return 0.0
        scale = max(0.1, spread_bps or 1.0)
        return max(-1.0, min(1.0, value / scale))

    def _bounded(self, value: float) -> float:
        return round(max(0.0, min(1.0, value)), 4)
