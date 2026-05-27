from futures_lab.config import Settings
from futures_lab.edge_router import BaselineCandidateInput, EdgeRouter
from futures_lab.models import MarketState, Regime, Side


def _market(**overrides) -> MarketState:
    base = {
        "symbol": "ETHUSDT",
        "connected": True,
        "data_age_seconds": 0.1,
        "observed_seconds": 300,
        "best_bid": 100.0,
        "best_ask": 100.01,
        "best_bid_qty": 20.0,
        "best_ask_qty": 20.0,
        "mid_price": 100.005,
        "spread_bps": 0.5,
        "spread_bps_std_5s": 0.05,
        "last_trade_price": 100.0,
        "mark_price": 100.0,
        "funding_rate": 0.0,
        "return_15s_pct": -0.0004,
        "return_60s_pct": -0.001,
        "return_180s_pct": -0.002,
        "realized_vol_60s_pct": 0.00005,
        "realized_vol_180s_pct": 0.00008,
        "range_180s_pct": 0.006,
        "range_high_180s": 101.0,
        "range_low_180s": 100.0,
        "range_position_180s": 0.55,
        "taker_buy_ratio_10s": 0.22,
        "taker_buy_ratio_30s": 0.34,
        "book_imbalance_top": -0.45,
        "depth_imbalance_top5": -0.70,
        "order_flow_imbalance_1s": -0.95,
        "order_flow_imbalance_5s": -0.80,
        "taker_aggression_imbalance_1s": -0.92,
        "taker_aggression_imbalance_5s": -0.78,
        "microprice_mid_bps": -0.40,
        "vamp_mid_bps": -0.35,
        "bid_depth_evaporation_rate_5s": 0.40,
        "ask_depth_refill_rate_5s": 0.20,
        "bid_depth_refill_rate_5s": 0.02,
        "ask_depth_evaporation_rate_5s": 0.02,
        "higher_timeframe_bias_side": "short",
        "higher_timeframe_bias_strength": 0.72,
        "avg_event_lag_30s_ms": 20.0,
        "regime": Regime.directional,
    }
    base.update(overrides)
    return MarketState(**base)


def test_router_selects_viable_taker_impulse_short():
    router = EdgeRouter(Settings(EDGE_ROUTER_MIN_EV_BPS=0.0))

    result = router.evaluate(_market())

    assert result["enabled"] is True
    assert result["mode"] == "shadow"
    assert result["selected"] is not None
    assert result["selected"]["strategy"] == "taker_impulse_short"
    assert result["selected"]["side"] == "short"
    assert result["selected"]["viable"] is True
    assert result["selected"]["exit_plan"]["baseline"] == "fixed_tp_stop"
    assert result["selected"]["exit_plan"]["preferred"] == "mfe_trailing_stop_after_cost_paid"
    assert "ofi_flip" in result["selected"]["exit_plan"]["soft_exit_signals"]
    assert any(candidate["strategy"] == "taker_impulse_long" for candidate in result["candidates"])


def test_router_blocks_target_when_cost_multiple_fails():
    router = EdgeRouter(
        Settings(
            TAKER_FEE_BPS=8.0,
            EXPECTED_SLIPPAGE_BPS=2.0,
            LATENCY_ADVERSE_SELECTION_BPS=2.0,
            EDGE_ROUTER_TARGET_COST_MULTIPLE=2.0,
        )
    )

    result = router.evaluate(_market())
    impulse_short = next(candidate for candidate in result["candidates"] if candidate["strategy"] == "taker_impulse_short")

    assert impulse_short["viable"] is False
    assert "target_below_cost_multiple" in impulse_short["blockers"]


def test_taker_impulse_requires_one_and_five_second_confirmation():
    router = EdgeRouter(Settings(EDGE_ROUTER_MIN_EV_BPS=0.0))

    result = router.evaluate(
        _market(
            order_flow_imbalance_5s=0.10,
            taker_aggression_imbalance_5s=0.05,
        )
    )
    impulse_short = next(candidate for candidate in result["candidates"] if candidate["strategy"] == "taker_impulse_short")

    assert impulse_short["viable"] is False
    assert "impulse_ofi_5s_not_aligned" in impulse_short["blockers"]
    assert "impulse_aggression_5s_not_aligned" in impulse_short["blockers"]


def test_taker_impulse_blocks_unstable_or_lagged_book():
    router = EdgeRouter(Settings(EDGE_ROUTER_MIN_EV_BPS=0.0))

    result = router.evaluate(_market(spread_bps_std_5s=1.2, avg_event_lag_30s_ms=900.0))
    impulse_short = next(candidate for candidate in result["candidates"] if candidate["strategy"] == "taker_impulse_short")

    assert "impulse_spread_unstable" in impulse_short["blockers"]
    assert "impulse_book_lagged" in impulse_short["blockers"]


def test_router_can_report_paper_candidate_mode_without_executing():
    router = EdgeRouter(Settings(EDGE_ROUTER_PAPER_ENABLED=True, EDGE_ROUTER_MIN_EV_BPS=0.0))

    result = router.evaluate(_market())

    assert result["mode"] == "paper_candidate"


def test_liquidation_continuation_requires_relevant_pulse():
    router = EdgeRouter(Settings(EDGE_ROUTER_MIN_EV_BPS=0.0))

    without_pulse = router.evaluate(_market())
    blocked = next(
        candidate for candidate in without_pulse["candidates"] if candidate["strategy"] == "liquidation_continuation_short"
    )

    with_pulse = router.evaluate(_market(long_liquidation_notional_30s=500_000))
    allowed = next(
        candidate for candidate in with_pulse["candidates"] if candidate["strategy"] == "liquidation_continuation_short"
    )

    assert "no_liquidation_pulse" in blocked["blockers"]
    assert "no_liquidation_pulse" not in allowed["blockers"]


def test_liquidation_router_prefers_sell_cascade_short_continuation():
    router = EdgeRouter(Settings(EDGE_ROUTER_MIN_EV_BPS=0.0))

    result = router.evaluate(
        _market(
            long_liquidation_notional_30s=600_000,
            liquidation_phase="cascade_continuation",
            liquidation_phase_side="short",
            liquidation_phase_confidence=0.92,
        )
    )
    continuation = next(
        candidate for candidate in result["candidates"] if candidate["strategy"] == "liquidation_continuation_short"
    )
    early_bounce = next(
        candidate for candidate in result["candidates"] if candidate["strategy"] == "liquidation_exhaustion_bounce_long"
    )

    assert continuation["viable"] is True
    assert "liquidation_phase_not_continuation" not in continuation["blockers"]
    assert "cascade_not_exhausted" in early_bounce["blockers"]


def test_liquidation_router_allows_delayed_long_bounce_after_reclaim():
    router = EdgeRouter(Settings(EDGE_ROUTER_MIN_EV_BPS=0.0))

    result = router.evaluate(
        _market(
            higher_timeframe_bias_side="neutral",
            higher_timeframe_bias_strength=0.0,
            long_liquidation_notional_30s=600_000,
            liquidation_phase="reclaim_or_failed_reclaim",
            liquidation_phase_side="long",
            liquidation_phase_confidence=0.90,
            order_flow_imbalance_1s=0.74,
            taker_aggression_imbalance_1s=0.70,
            microprice_mid_bps=0.40,
            vamp_mid_bps=0.35,
            depth_imbalance_top5=0.55,
            bid_depth_refill_rate_5s=0.25,
            ask_depth_evaporation_rate_5s=0.12,
        )
    )
    bounce = next(
        candidate for candidate in result["candidates"] if candidate["strategy"] == "liquidation_exhaustion_bounce_long"
    )
    continuation = next(
        candidate for candidate in result["candidates"] if candidate["strategy"] == "liquidation_continuation_short"
    )

    assert bounce["viable"] is True
    assert "cascade_not_exhausted" not in bounce["blockers"]
    assert "liquidation_phase_not_continuation" in continuation["blockers"]


def test_baseline_candidate_is_logged_with_router_candidates():
    router = EdgeRouter(Settings(EDGE_ROUTER_MIN_EV_BPS=0.0))
    baseline = BaselineCandidateInput(
        side=Side.short,
        score=0.83,
        target_bps=20.0,
        stop_bps=15.0,
        reasons=["test baseline"],
    )

    result = router.evaluate(_market(), baseline=baseline)

    assert any(candidate["strategy"] == "stateful_momentum_baseline" for candidate in result["candidates"])
    assert result["candidate_count"] == 7
