from __future__ import annotations

from dataclasses import dataclass, field

from futures_lab.config import Settings
from futures_lab.costs import estimate_effective_cost
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
            "mode": "shadow",
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
            0.30 * self._positive_unit(signed["ofi_1s"])
            + 0.22 * self._positive_unit(signed["aggression_1s"])
            + 0.18 * self._positive_unit(signed["microprice_pressure"])
            + 0.15 * self._positive_unit(signed["vamp_pressure"])
            + 0.10 * self._positive_unit(signed["depth_pressure"])
            + 0.05 * self._positive_unit(signed["refill_pressure"])
        )
        return self._build_candidate(
            market=market,
            strategy=f"taker_impulse_{side.value}",
            family="taker_impulse",
            side=side,
            target_bps=self.settings.fast_target_move_pct * 10_000,
            stop_bps=self.settings.fast_stop_move_pct * 10_000,
            max_hold_ms=self.settings.edge_router_impulse_max_hold_ms,
            score=micro_score,
            reasons=[
                "OFI/aggression/microprice/VAMP/depth impulse score",
                f"ofi_1s={market.order_flow_imbalance_1s}",
                f"aggression_1s={market.taker_aggression_imbalance_1s}",
            ],
        )

    def _liquidation_continuation_candidate(self, market: MarketState, side: Side) -> EdgeCandidate:
        relevant_liq = self._continuation_liquidation_notional(market, side)
        liq_score = min(1.0, relevant_liq / 250_000)
        signed = self._signed_features(market, side)
        score = self._bounded(
            0.35 * liq_score
            + 0.25 * self._positive_unit(signed["ofi_1s"])
            + 0.20 * self._positive_unit(signed["aggression_1s"])
            + 0.20 * self._positive_unit(signed["depth_pressure"])
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
            reasons=[f"liquidation_notional={relevant_liq}", "forced-flow continuation proxy"],
        )
        if relevant_liq <= 0:
            return self._with_blocker(candidate, "no_liquidation_pulse")
        return candidate

    def _liquidation_exhaustion_bounce_candidate(self, market: MarketState, side: Side) -> EdgeCandidate:
        relevant_liq = self._exhaustion_liquidation_notional(market, side)
        liq_score = min(1.0, relevant_liq / 250_000)
        signed = self._signed_features(market, side)
        score = self._bounded(
            0.30 * liq_score
            + 0.25 * self._positive_unit(signed["microprice_pressure"])
            + 0.20 * self._positive_unit(signed["vamp_pressure"])
            + 0.15 * self._positive_unit(signed["refill_pressure"])
            + 0.10 * self._positive_unit(signed["ofi_1s"])
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
            reasons=[f"exhaustion_liquidation_notional={relevant_liq}", "delayed bounce proxy"],
        )
        if relevant_liq <= 0:
            return self._with_blocker(candidate, "no_exhaustion_pulse")
        return candidate

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
    ) -> EdgeCandidate:
        cost = estimate_effective_cost(self.settings, market)
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
        )

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
            "aggression_1s": sign * (market.taker_aggression_imbalance_1s or 0.0),
            "microprice_pressure": sign * self._scaled_bps(market.microprice_mid_bps, market.spread_bps),
            "vamp_pressure": sign * self._scaled_bps(market.vamp_mid_bps, market.spread_bps),
            "depth_pressure": sign * (market.depth_imbalance_top5 or market.book_imbalance_top or 0.0),
            "refill_pressure": refill_pressure,
        }

    def _continuation_liquidation_notional(self, market: MarketState, side: Side) -> float:
        if side == Side.short:
            return market.long_liquidation_notional_30s or 0.0
        return market.short_liquidation_notional_30s or 0.0

    def _exhaustion_liquidation_notional(self, market: MarketState, side: Side) -> float:
        if side == Side.long:
            return market.long_liquidation_notional_30s or 0.0
        return market.short_liquidation_notional_30s or 0.0

    def _with_blocker(self, candidate: EdgeCandidate, blocker: str) -> EdgeCandidate:
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
            blockers=sorted(set(candidate.blockers + [blocker])),
            reasons=candidate.reasons,
        )

    def _positive_unit(self, value: float) -> float:
        return self._bounded((value + 1.0) / 2.0)

    def _scaled_bps(self, value: float | None, spread_bps: float | None) -> float:
        if value is None:
            return 0.0
        scale = max(0.1, spread_bps or 1.0)
        return max(-1.0, min(1.0, value / scale))

    def _bounded(self, value: float) -> float:
        return round(max(0.0, min(1.0, value)), 4)
