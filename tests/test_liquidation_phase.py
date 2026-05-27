from futures_lab.config import Settings
from futures_lab.liquidation_phase import classify_liquidation_phase
from futures_lab.models import MarketState, Regime


def _market(**overrides) -> MarketState:
    base = {
        "symbol": "ETHUSDT",
        "connected": True,
        "data_age_seconds": 0.1,
        "observed_seconds": 300,
        "mid_price": 100.0,
        "spread_bps": 0.5,
        "spread_bps_std_5s": 0.05,
        "long_liquidation_notional_30s": 0.0,
        "short_liquidation_notional_30s": 0.0,
        "order_flow_imbalance_1s": 0.0,
        "taker_aggression_imbalance_1s": 0.0,
        "depth_imbalance_top5": 0.0,
        "microprice_mid_bps": 0.0,
        "vamp_mid_bps": 0.0,
        "bid_depth_refill_rate_5s": 0.0,
        "ask_depth_refill_rate_5s": 0.0,
        "bid_depth_evaporation_rate_5s": 0.0,
        "ask_depth_evaporation_rate_5s": 0.0,
        "mark_last_basis_bps": 0.0,
        "open_interest_change_5m_pct": 0.0,
        "return_15s_pct": 0.0,
        "regime": Regime.directional,
    }
    base.update(overrides)
    return MarketState(**base)


def test_sell_cascade_classifies_as_short_continuation():
    phase = classify_liquidation_phase(
        _market(
            long_liquidation_notional_30s=600_000,
            order_flow_imbalance_1s=-0.90,
            taker_aggression_imbalance_1s=-0.88,
            depth_imbalance_top5=-0.75,
            microprice_mid_bps=-0.45,
            vamp_mid_bps=-0.40,
            bid_depth_evaporation_rate_5s=0.30,
            mark_last_basis_bps=-3.0,
            open_interest_change_5m_pct=0.002,
        ),
        Settings(),
    )

    assert phase.phase == "cascade_continuation"
    assert phase.side == "short"
    assert phase.pulse_side == "sell"
    assert phase.evidence["force_order_stream_is_sampled_largest_pulse"] is True


def test_sell_cascade_exhaustion_classifies_as_long_reclaim():
    phase = classify_liquidation_phase(
        _market(
            long_liquidation_notional_30s=600_000,
            order_flow_imbalance_1s=0.72,
            taker_aggression_imbalance_1s=0.70,
            depth_imbalance_top5=0.55,
            microprice_mid_bps=0.40,
            vamp_mid_bps=0.35,
            bid_depth_refill_rate_5s=0.24,
            ask_depth_evaporation_rate_5s=0.10,
            return_15s_pct=0.0004,
        ),
        Settings(),
    )

    assert phase.phase == "reclaim_or_failed_reclaim"
    assert phase.side == "long"
    assert phase.pulse_side == "sell"


def test_no_pulse_and_weak_pressure_is_normal():
    phase = classify_liquidation_phase(_market(), Settings())

    assert phase.phase == "normal"
    assert phase.side == "none"
