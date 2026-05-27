from __future__ import annotations

from dataclasses import dataclass, field

from futures_lab.config import Settings
from futures_lab.models import MarketState, Side


@dataclass(frozen=True)
class LiquidationPhaseSnapshot:
    phase: str
    side: str
    confidence: float
    pulse_side: str
    pulse_notional: float
    blockers: list[str] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)

    def model_dump(self) -> dict:
        return {
            "phase": self.phase,
            "side": self.side,
            "confidence": self.confidence,
            "pulse_side": self.pulse_side,
            "pulse_notional": self.pulse_notional,
            "blockers": self.blockers,
            "evidence": self.evidence,
        }


def classify_liquidation_phase(market: MarketState, settings: Settings) -> LiquidationPhaseSnapshot:
    sell_pulse = market.long_liquidation_notional_30s or 0.0
    buy_pulse = market.short_liquidation_notional_30s or 0.0
    pulse_side = "sell" if sell_pulse >= buy_pulse else "buy"
    pulse_notional = max(sell_pulse, buy_pulse)
    continuation_side = Side.short if pulse_side == "sell" else Side.long
    bounce_side = Side.long if pulse_side == "sell" else Side.short
    continuation = _signed_features(market, continuation_side)
    bounce = _signed_features(market, bounce_side)

    pulse_score = _bounded(pulse_notional / settings.liquidation_phase_impulse_notional)
    continuation_pressure = _pressure_score(continuation)
    bounce_pressure = _pressure_score(bounce)
    refill_pressure = _bounded(bounce["refill_pressure"])
    spread_stable = _spread_stable(market, settings)
    basis_pressure = _basis_pressure(market, continuation_side, settings)
    oi_pressure = _oi_pressure(market, settings)

    evidence = {
        "force_order_stream_is_sampled_largest_pulse": True,
        "sell_pulse_notional_30s": sell_pulse,
        "buy_pulse_notional_30s": buy_pulse,
        "continuation_side": continuation_side.value,
        "bounce_side": bounce_side.value,
        "continuation_pressure": continuation_pressure,
        "bounce_pressure": bounce_pressure,
        "refill_pressure": refill_pressure,
        "spread_stable": spread_stable,
        "basis_pressure": basis_pressure,
        "oi_pressure": oi_pressure,
    }

    if pulse_notional >= settings.liquidation_phase_impulse_notional:
        if continuation_pressure >= settings.liquidation_phase_min_continuation_pressure:
            confidence = _bounded(0.45 * pulse_score + 0.35 * continuation_pressure + 0.10 * basis_pressure + 0.10 * oi_pressure)
            return LiquidationPhaseSnapshot(
                phase="cascade_continuation",
                side=continuation_side.value,
                confidence=confidence,
                pulse_side=pulse_side,
                pulse_notional=pulse_notional,
                evidence=evidence,
            )
        if (
            bounce_pressure >= settings.liquidation_phase_min_exhaustion_pressure
            and refill_pressure >= settings.liquidation_phase_min_refill_pressure
            and spread_stable
        ):
            reclaimed = _reclaimed_after_pulse(market, bounce_side)
            phase = "reclaim_or_failed_reclaim" if reclaimed else "exhaustion_candidate"
            confidence = _bounded(0.40 * pulse_score + 0.35 * bounce_pressure + 0.15 * refill_pressure + 0.10 * int(reclaimed))
            return LiquidationPhaseSnapshot(
                phase=phase,
                side=bounce_side.value,
                confidence=confidence,
                pulse_side=pulse_side,
                pulse_notional=pulse_notional,
                evidence=evidence,
            )
        confidence = _bounded(0.65 * pulse_score + 0.35 * max(continuation_pressure, bounce_pressure))
        return LiquidationPhaseSnapshot(
            phase="liquidation_impulse",
            side=continuation_side.value,
            confidence=confidence,
            pulse_side=pulse_side,
            pulse_notional=pulse_notional,
            blockers=["phase_unresolved"],
            evidence=evidence,
        )

    if (
        pulse_notional >= settings.liquidation_phase_pressure_notional
        or continuation_pressure >= settings.liquidation_phase_min_pressure_building
    ):
        confidence = _bounded(0.35 * pulse_score + 0.45 * continuation_pressure + 0.10 * basis_pressure + 0.10 * oi_pressure)
        return LiquidationPhaseSnapshot(
            phase="pressure_building",
            side=continuation_side.value,
            confidence=confidence,
            pulse_side=pulse_side if pulse_notional > 0 else "none",
            pulse_notional=pulse_notional,
            evidence=evidence,
        )

    return LiquidationPhaseSnapshot(
        phase="normal",
        side="none",
        confidence=0.0,
        pulse_side="none",
        pulse_notional=0.0,
        evidence=evidence,
    )


def _signed_features(market: MarketState, side: Side) -> dict[str, float]:
    sign = 1.0 if side == Side.long else -1.0
    bid_refill = market.bid_depth_refill_rate_5s or 0.0
    ask_refill = market.ask_depth_refill_rate_5s or 0.0
    bid_evap = market.bid_depth_evaporation_rate_5s or 0.0
    ask_evap = market.ask_depth_evaporation_rate_5s or 0.0
    refill_pressure = (bid_refill + ask_evap) if side == Side.long else (ask_refill + bid_evap)
    return {
        "ofi_1s": sign * (market.order_flow_imbalance_1s or 0.0),
        "aggression_1s": sign * (market.taker_aggression_imbalance_1s or 0.0),
        "depth_pressure": sign * (market.depth_imbalance_top5 or market.book_imbalance_top or 0.0),
        "microprice_mid_bps": sign * (market.microprice_mid_bps or 0.0),
        "vamp_mid_bps": sign * (market.vamp_mid_bps or 0.0),
        "refill_pressure": refill_pressure,
    }


def _pressure_score(features: dict[str, float]) -> float:
    return _bounded(
        0.25 * _positive_unit(features["ofi_1s"])
        + 0.25 * _positive_unit(features["aggression_1s"])
        + 0.20 * _positive_unit(features["depth_pressure"])
        + 0.15 * _positive_unit(features["microprice_mid_bps"])
        + 0.15 * _positive_unit(features["vamp_mid_bps"])
    )


def _basis_pressure(market: MarketState, side: Side, settings: Settings) -> float:
    basis = market.mark_last_basis_bps
    if basis is None:
        return 0.0
    signed = basis if side == Side.long else -basis
    return _bounded(signed / max(settings.liquidation_phase_mark_basis_bps, 0.1))


def _oi_pressure(market: MarketState, settings: Settings) -> float:
    oi = market.open_interest_change_5m_pct
    if oi is None:
        return 0.0
    return _bounded(abs(oi) / max(settings.liquidation_phase_oi_change_pct, 0.0001))


def _spread_stable(market: MarketState, settings: Settings) -> bool:
    if market.spread_bps is None:
        return False
    if market.spread_bps > settings.taker_impulse_max_spread_bps:
        return False
    if market.spread_bps_std_5s is not None and market.spread_bps_std_5s > settings.taker_impulse_max_spread_std_bps:
        return False
    return True


def _reclaimed_after_pulse(market: MarketState, side: Side) -> bool:
    r15 = market.return_15s_pct or 0.0
    micro = market.microprice_mid_bps or 0.0
    if side == Side.long:
        return r15 > 0 and micro > 0
    return r15 < 0 and micro < 0


def _positive_unit(value: float) -> float:
    return _bounded((value + 1.0) / 2.0)


def _bounded(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 4)
