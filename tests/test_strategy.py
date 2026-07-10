from datetime import datetime, timedelta, timezone

import pytest

from futures_lab.config import Settings
from futures_lab.models import DecisionAction, MarketState, Regime
from futures_lab.strategy import HitAndRunStrategy


def _market(**overrides) -> MarketState:
    base = {
        "symbol": "BTCUSDT",
        "connected": True,
        "data_age_seconds": 0.1,
        "observed_seconds": 300,
        "best_bid": 100.0,
        "best_ask": 100.01,
        "best_bid_qty": 12.0,
        "best_ask_qty": 8.0,
        "mid_price": 100.005,
        "spread_bps": 0.5,
        "last_trade_price": 100.0,
        "mark_price": 100.0,
        "funding_rate": 0.0,
        "return_15s_pct": 0.0001,
        "return_60s_pct": -0.0005,
        "return_180s_pct": 0.0002,
        "realized_vol_60s_pct": 0.00004,
        "realized_vol_180s_pct": 0.00006,
        "range_180s_pct": 0.007,
        "range_high_180s": 101.0,
        "range_low_180s": 100.0,
        "range_position_180s": 0.05,
        "taker_buy_ratio_10s": 0.78,
        "taker_buy_ratio_30s": 0.66,
        "book_imbalance_top": 0.2,
        "regime": Regime.sideways,
    }
    base.update(overrides)
    return MarketState(**base)


def test_strategy_waits_on_stale_regime():
    strategy = HitAndRunStrategy(Settings())

    decision = strategy.decide(_market(regime=Regime.stale, data_age_seconds=5.0))

    assert decision.action == DecisionAction.wait
    assert "regime=stale" in decision.reason


def test_strategy_proposes_fast_long_near_range_low():
    strategy = HitAndRunStrategy(Settings(MIN_CONFIDENCE=0.70))

    decision = strategy.decide(_market())

    assert decision.action == DecisionAction.propose_long
    assert decision.mode is not None
    assert decision.leverage == 200
    assert decision.take_profit_price is not None
    assert decision.take_profit_price > decision.entry_price


def test_strategy_logs_shadow_edge_router_candidates():
    strategy = HitAndRunStrategy(Settings(MIN_CONFIDENCE=0.70, EDGE_ROUTER_SHADOW_ENABLED=True))

    decision = strategy.decide(
        _market(
            order_flow_imbalance_1s=0.88,
            taker_aggression_imbalance_1s=0.84,
            microprice_mid_bps=0.30,
            vamp_mid_bps=0.28,
            depth_imbalance_top5=0.50,
            bid_depth_refill_rate_5s=0.22,
            ask_depth_evaporation_rate_5s=0.18,
        )
    )

    edge_router = decision.evidence["edge_router"]
    assert edge_router["enabled"] is True
    assert edge_router["mode"] == "shadow"
    assert edge_router["candidate_count"] == 7
    assert any(candidate["strategy"] == "stateful_momentum_baseline" for candidate in edge_router["candidates"])
    assert any(candidate["strategy"] == "taker_impulse_long" for candidate in edge_router["candidates"])


def test_strategy_proposes_short_near_range_high():
    strategy = HitAndRunStrategy(Settings(MIN_CONFIDENCE=0.70))

    decision = strategy.decide(
        _market(
            range_position_180s=0.96,
            taker_buy_ratio_10s=0.18,
            taker_buy_ratio_30s=0.30,
            book_imbalance_top=-0.25,
            return_15s_pct=-0.0001,
        )
    )

    assert decision.action == DecisionAction.propose_short
    assert decision.stop_loss_price is not None
    assert decision.stop_loss_price > decision.entry_price


def test_strategy_blocks_short_when_30s_taker_flow_is_buy_biased():
    strategy = HitAndRunStrategy(Settings(MIN_CONFIDENCE=0.70, MAX_SHORT_TAKER_BUY_RATIO_30S=0.60))

    decision = strategy.decide(
        _market(
            range_position_180s=0.96,
            taker_buy_ratio_10s=0.18,
            taker_buy_ratio_30s=0.74,
            book_imbalance_top=-0.25,
            return_15s_pct=-0.0001,
        )
    )

    assert decision.action == DecisionAction.wait
    assert "30s taker buy ratio" in decision.reason


@pytest.mark.parametrize(
    ("sign", "expected_action"),
    [
        (1.0, DecisionAction.propose_long),
        (-1.0, DecisionAction.propose_short),
    ],
)
def test_first_touch_micro_momentum_is_side_neutral(sign, expected_action):
    strategy = HitAndRunStrategy(
        Settings(
            MIN_CONFIDENCE=0.50,
            STRATEGY_VARIANT="first_touch_micro_momentum",
            EDGE_ROUTER_SHADOW_ENABLED=False,
        )
    )

    decision = strategy.decide(
        _market(
            regime=Regime.volatile,
            order_flow_imbalance_1s=0.80 * sign,
            order_flow_imbalance_5s=0.50 * sign,
            taker_aggression_imbalance_1s=0.75 * sign,
            taker_aggression_imbalance_5s=0.45 * sign,
            microprice_mid_bps=0.30 * sign,
            vamp_mid_bps=0.25 * sign,
            depth_imbalance_top5=0.55 * sign,
            book_imbalance_top=0.40 * sign,
            taker_buy_ratio_30s=0.80 if sign < 0 else 0.20,
        )
    )

    assert decision.action == expected_action
    assert decision.target_move_pct == pytest.approx(0.006)
    assert decision.stop_move_pct == pytest.approx(0.007)
    assert decision.leverage == 150
    assert decision.evidence["first_touch_micro"]["score_is_probability"] is False
    assert "edge_router" not in decision.evidence
    assert "higher_timeframe_context" not in decision.evidence


def test_first_touch_uses_only_fresh_microstructure_inputs():
    settings = Settings(
        MIN_CONFIDENCE=0.50,
        STRATEGY_VARIANT="first_touch_micro_momentum",
        EDGE_ROUTER_SHADOW_ENABLED=True,
        ENABLE_MARKOV_STATE_MACHINE=True,
        HIGHER_TIMEFRAME_ENABLED=True,
        CROSS_MARKET_ENABLED=True,
    )
    market = _market(
        regime=Regime.volatile,
        range_position_180s=None,
        taker_buy_ratio_10s=None,
        order_flow_imbalance_1s=0.80,
        order_flow_imbalance_5s=0.50,
        taker_aggression_imbalance_1s=0.75,
        taker_aggression_imbalance_5s=0.45,
        microprice_mid_bps=0.30,
        vamp_mid_bps=0.25,
        depth_imbalance_top5=0.55,
        avg_book_event_lag_30s_ms=100,
        avg_trade_event_lag_30s_ms=120,
        higher_timeframe_bias_side="short",
        higher_timeframe_bias_strength=1.0,
        btc_order_flow_imbalance_1s=-1.0,
    )

    decision = HitAndRunStrategy(settings).decide(market)

    assert decision.action == DecisionAction.propose_long
    assert set(decision.evidence) == {
        "strategy_variant",
        "first_touch_micro",
        "freshness",
        "effective_cost",
        "trade_profile",
    }

    lagged = HitAndRunStrategy(settings).decide(
        market.model_copy(update={"avg_trade_event_lag_30s_ms": 2_500})
    )
    assert lagged.action == DecisionAction.wait
    assert "trade exchange event lag too high" in lagged.reason


def test_liquidity_sweep_reversal_proposes_long_after_low_reclaim():
    strategy = HitAndRunStrategy(Settings(MIN_CONFIDENCE=0.70, STRATEGY_VARIANT="liquidity_sweep_reversal"))

    decision = strategy.decide(
        _market(
            range_position_180s=0.08,
            return_15s_pct=0.0007,
            return_60s_pct=-0.001,
            taker_buy_ratio_10s=0.78,
            book_imbalance_top=0.35,
            depth_imbalance_top5=0.55,
            short_liquidation_notional_30s=25_000,
        )
    )

    assert decision.action == DecisionAction.propose_long
    assert decision.evidence["strategy_variant"] == "liquidity_sweep_reversal"


def test_liquidity_sweep_reversal_proposes_short_after_high_rejection():
    strategy = HitAndRunStrategy(Settings(MIN_CONFIDENCE=0.70, STRATEGY_VARIANT="liquidity_sweep_reversal"))

    decision = strategy.decide(
        _market(
            range_position_180s=0.94,
            return_15s_pct=-0.0007,
            return_60s_pct=0.001,
            taker_buy_ratio_10s=0.20,
            taker_buy_ratio_30s=0.30,
            book_imbalance_top=-0.35,
            depth_imbalance_top5=-0.55,
            long_liquidation_notional_30s=25_000,
        )
    )

    assert decision.action == DecisionAction.propose_short
    assert decision.evidence["strategy_variant"] == "liquidity_sweep_reversal"


def _range_context(position: float, support_distance: float, resistance_distance: float) -> dict:
    return {
        "bias": {"side": "neutral", "strength": 0.05, "reason": "range test"},
        "timeframes": {
            "15m": {
                "interval": "15m",
                "range_position": position,
                "range_pct": 0.014,
                "trend_score": 0.08,
                "structure": "range_support_test" if position < 0.5 else "range_resistance_test",
                "support_distance_pct": support_distance,
                "resistance_distance_pct": resistance_distance,
            },
            "30m": {
                "interval": "30m",
                "range_position": position,
                "range_pct": 0.018,
                "trend_score": -0.04,
                "structure": "balanced",
                "support_distance_pct": support_distance,
                "resistance_distance_pct": resistance_distance,
            },
            "1h": {
                "interval": "1h",
                "range_position": position,
                "range_pct": 0.022,
                "trend_score": 0.02,
                "structure": "balanced",
                "support_distance_pct": support_distance,
                "resistance_distance_pct": resistance_distance,
            },
        },
    }


def test_range_bound_strategy_proposes_long_near_multi_timeframe_support():
    strategy = HitAndRunStrategy(
        Settings(
            MIN_CONFIDENCE=0.70,
            STRATEGY_VARIANT="range_bound_support_resistance",
            RANGE_BOUND_TARGET_MOVE_PCT=0.005,
            RANGE_BOUND_STOP_MOVE_PCT=0.0025,
            RANGE_BOUND_LEVERAGE=200,
        )
    )

    decision = strategy.decide(
        _market(
            symbol="ETHUSDT",
            range_position_180s=0.08,
            taker_buy_ratio_10s=0.64,
            taker_buy_ratio_30s=0.58,
            book_imbalance_top=0.22,
            depth_imbalance_top5=0.30,
            higher_timeframe_context=_range_context(0.10, support_distance=0.001, resistance_distance=0.012),
            higher_timeframe_context_age_seconds=60,
            higher_timeframe_bias_side="neutral",
            higher_timeframe_bias_strength=0.05,
        )
    )

    assert decision.action == DecisionAction.propose_long
    assert decision.evidence["strategy_variant"] == "range_bound_support_resistance"
    assert decision.evidence["trade_profile"] == "range_bound_support_resistance"
    assert decision.target_move_pct == 0.005
    assert decision.stop_move_pct == 0.0025
    assert decision.leverage == 200
    assert decision.evidence["edge_router"]["selected"]["target_bps"] == 50.0
    assert decision.evidence["edge_router"]["selected_candidate"]["strategy"] == "range_bound_support_resistance"
    assert decision.evidence["edge_router"]["selected_candidate"]["family"] == "range_bound"
    assert decision.evidence["edge_router"]["selected_candidate"]["max_hold_ms"] == 21_600_000


def test_range_bound_strategy_proposes_short_near_multi_timeframe_resistance():
    strategy = HitAndRunStrategy(
        Settings(
            MIN_CONFIDENCE=0.70,
            STRATEGY_VARIANT="range_bound_support_resistance",
            RANGE_BOUND_TARGET_MOVE_PCT=0.005,
            RANGE_BOUND_STOP_MOVE_PCT=0.0025,
            RANGE_BOUND_LEVERAGE=200,
            MAX_SHORT_TAKER_BUY_RATIO_30S=0.60,
        )
    )

    decision = strategy.decide(
        _market(
            symbol="ETHUSDT",
            range_position_180s=0.94,
            taker_buy_ratio_10s=0.22,
            taker_buy_ratio_30s=0.34,
            book_imbalance_top=-0.24,
            depth_imbalance_top5=-0.35,
            return_15s_pct=-0.0002,
            higher_timeframe_context=_range_context(0.92, support_distance=0.013, resistance_distance=0.001),
            higher_timeframe_context_age_seconds=60,
            higher_timeframe_bias_side="neutral",
            higher_timeframe_bias_strength=0.05,
        )
    )

    assert decision.action == DecisionAction.propose_short
    assert decision.evidence["trade_profile"] == "range_bound_support_resistance"
    assert decision.evidence["edge_router"]["selected_candidate"]["strategy"] == "range_bound_support_resistance"
    assert decision.take_profit_price < decision.entry_price


def test_range_bound_strategy_scales_leverage_to_structural_box():
    strategy = HitAndRunStrategy(
        Settings(
            MIN_CONFIDENCE=0.70,
            STRATEGY_VARIANT="range_bound_support_resistance",
            RANGE_BOUND_LEVERAGE=200,
            RANGE_BOUND_MIN_LEVERAGE=40,
            RANGE_BOUND_MAX_ACCOUNT_DRAWDOWN_FRACTION=0.65,
        )
    )

    decision = strategy.decide(
        _market(
            symbol="ETHUSDT",
            range_position_180s=0.08,
            taker_buy_ratio_10s=0.64,
            taker_buy_ratio_30s=0.58,
            book_imbalance_top=0.22,
            depth_imbalance_top5=0.30,
            higher_timeframe_context=_range_context(0.10, support_distance=0.030, resistance_distance=0.012),
            higher_timeframe_context_age_seconds=60,
            higher_timeframe_bias_side="neutral",
            higher_timeframe_bias_strength=0.05,
        )
    )

    risk = decision.evidence["range_bound_structural_risk"]["long"]
    assert decision.action == DecisionAction.propose_long
    assert decision.stop_move_pct == pytest.approx(0.0305)
    assert decision.leverage < 200
    assert risk["dynamic_leverage_applied"] is True
    assert risk["account_drawdown_fraction"] <= 0.65


def test_range_bound_selected_candidate_blocks_thin_target_room():
    strategy = HitAndRunStrategy(
        Settings(
            MIN_CONFIDENCE=0.70,
            STRATEGY_VARIANT="range_bound_support_resistance",
            RANGE_BOUND_TARGET_MOVE_PCT=0.005,
            RANGE_BOUND_CANDIDATE_MIN_TARGET_ROOM_MULTIPLE=1.20,
        )
    )

    decision = strategy.decide(
        _market(
            symbol="ETHUSDT",
            range_position_180s=0.08,
            taker_buy_ratio_10s=0.64,
            taker_buy_ratio_30s=0.58,
            book_imbalance_top=0.22,
            depth_imbalance_top5=0.30,
            higher_timeframe_context=_range_context(0.10, support_distance=0.001, resistance_distance=0.0055),
            higher_timeframe_context_age_seconds=60,
            higher_timeframe_bias_side="neutral",
            higher_timeframe_bias_strength=0.05,
        )
    )

    selected = decision.evidence["edge_router"]["selected_candidate"]
    assert decision.action == DecisionAction.propose_long
    assert selected["strategy"] == "range_bound_support_resistance"
    assert "range_target_room_buffer_too_thin" in selected["blockers"]


def test_range_bound_strategy_waits_without_range_context():
    strategy = HitAndRunStrategy(Settings(MIN_CONFIDENCE=0.70, STRATEGY_VARIANT="range_bound_support_resistance"))

    decision = strategy.decide(_market(range_position_180s=0.08, taker_buy_ratio_10s=0.64))

    assert decision.action == DecisionAction.wait


def test_momentum_pullback_proposes_long_in_bullish_structure():
    strategy = HitAndRunStrategy(Settings(MIN_CONFIDENCE=0.70, STRATEGY_VARIANT="momentum_pullback"))

    decision = strategy.decide(
        _market(
            range_position_180s=0.55,
            return_15s_pct=-0.0001,
            return_60s_pct=0.0008,
            return_180s_pct=0.0025,
            taker_buy_ratio_10s=0.76,
            book_imbalance_top=0.35,
            depth_imbalance_top5=0.50,
            open_interest_change_5m_pct=0.0015,
        )
    )

    assert decision.action == DecisionAction.propose_long
    assert decision.evidence["strategy_variant"] == "momentum_pullback"


def test_momentum_pullback_proposes_short_in_bearish_structure():
    strategy = HitAndRunStrategy(Settings(MIN_CONFIDENCE=0.70, STRATEGY_VARIANT="momentum_pullback"))

    decision = strategy.decide(
        _market(
            range_position_180s=0.45,
            return_15s_pct=0.0001,
            return_60s_pct=-0.0008,
            return_180s_pct=-0.0025,
            taker_buy_ratio_10s=0.24,
            taker_buy_ratio_30s=0.35,
            book_imbalance_top=-0.35,
            depth_imbalance_top5=-0.50,
            open_interest_change_5m_pct=0.0015,
        )
    )

    assert decision.action == DecisionAction.propose_short
    assert decision.evidence["strategy_variant"] == "momentum_pullback"


def test_stateful_momentum_waits_until_sequence_confirms_long():
    strategy = HitAndRunStrategy(
        Settings(MIN_CONFIDENCE=0.70, STRATEGY_VARIANT="stateful_momentum", FEE_EDGE_QUALITY_GATE_ENABLED=False)
    )
    impulse = _market(
        range_position_180s=0.55,
        return_15s_pct=0.0002,
        return_60s_pct=0.001,
        return_180s_pct=0.0025,
        taker_buy_ratio_10s=0.76,
        book_imbalance_top=0.35,
        depth_imbalance_top5=0.45,
        open_interest_change_5m_pct=0.001,
    )

    early = strategy.decide(impulse)
    assert early.action == DecisionAction.wait
    assert "confirmed sequence" in early.reason

    strategy.decide(impulse.model_copy(update={"return_15s_pct": -0.0003, "taker_buy_ratio_10s": 0.56}))
    strategy.decide(impulse.model_copy(update={"return_15s_pct": 0.00035, "taker_buy_ratio_10s": 0.72}))
    confirmed = strategy.decide(impulse.model_copy(update={"return_15s_pct": 0.0004, "taker_buy_ratio_10s": 0.74}))

    assert confirmed.action == DecisionAction.propose_long
    assert confirmed.evidence["market_sequence"]["state"] == "long_continuation_confirmed"


def test_stateful_momentum_waits_until_sequence_confirms_short():
    strategy = HitAndRunStrategy(
        Settings(MIN_CONFIDENCE=0.70, STRATEGY_VARIANT="stateful_momentum", FEE_EDGE_QUALITY_GATE_ENABLED=False)
    )
    impulse = _market(
        range_position_180s=0.45,
        return_15s_pct=-0.0002,
        return_60s_pct=-0.001,
        return_180s_pct=-0.0025,
        taker_buy_ratio_10s=0.24,
        taker_buy_ratio_30s=0.35,
        book_imbalance_top=-0.35,
        depth_imbalance_top5=-0.45,
        open_interest_change_5m_pct=0.001,
    )

    early = strategy.decide(impulse)
    assert early.action == DecisionAction.wait
    assert "confirmed sequence" in early.reason

    strategy.decide(impulse.model_copy(update={"return_15s_pct": 0.0003, "taker_buy_ratio_10s": 0.44}))
    strategy.decide(impulse.model_copy(update={"return_15s_pct": -0.00035, "taker_buy_ratio_10s": 0.28}))
    confirmed = strategy.decide(impulse.model_copy(update={"return_15s_pct": -0.0004, "taker_buy_ratio_10s": 0.26}))

    assert confirmed.action == DecisionAction.propose_short
    assert confirmed.evidence["market_sequence"]["state"] == "short_continuation_confirmed"


def test_stateful_momentum_accepts_confirmed_long_breakout_location():
    strategy = HitAndRunStrategy(
        Settings(MIN_CONFIDENCE=0.70, STRATEGY_VARIANT="stateful_momentum", FEE_EDGE_QUALITY_GATE_ENABLED=False)
    )
    impulse = _market(
        range_position_180s=0.97,
        return_15s_pct=0.00025,
        return_60s_pct=0.0009,
        return_180s_pct=0.0026,
        taker_buy_ratio_10s=0.74,
        book_imbalance_top=0.25,
        depth_imbalance_top5=0.55,
        open_interest_change_5m_pct=0.001,
    )

    strategy.decide(impulse)
    strategy.decide(impulse.model_copy(update={"return_15s_pct": -0.00025, "taker_buy_ratio_10s": 0.56}))
    strategy.decide(impulse.model_copy(update={"return_15s_pct": 0.00032, "taker_buy_ratio_10s": 0.70}))
    confirmed = strategy.decide(impulse.model_copy(update={"return_15s_pct": 0.00036, "taker_buy_ratio_10s": 0.72}))

    assert confirmed.action == DecisionAction.propose_long
    assert confirmed.confidence >= 0.70
    assert confirmed.evidence["market_sequence"]["state"] == "long_continuation_confirmed"
    assert confirmed.evidence["stateful_momentum_filter"]["blocked"] is False
    assert confirmed.evidence["stateful_momentum_filter"]["target_feasible"] is True


def test_stateful_momentum_accepts_confirmed_short_breakdown_location():
    strategy = HitAndRunStrategy(
        Settings(MIN_CONFIDENCE=0.70, STRATEGY_VARIANT="stateful_momentum", FEE_EDGE_QUALITY_GATE_ENABLED=False)
    )
    impulse = _market(
        range_position_180s=0.08,
        return_15s_pct=-0.00025,
        return_60s_pct=-0.0009,
        return_180s_pct=-0.0026,
        taker_buy_ratio_10s=0.26,
        taker_buy_ratio_30s=0.35,
        book_imbalance_top=-0.25,
        depth_imbalance_top5=-0.55,
        open_interest_change_5m_pct=0.001,
    )

    strategy.decide(impulse)
    strategy.decide(impulse.model_copy(update={"return_15s_pct": 0.00025, "taker_buy_ratio_10s": 0.44}))
    strategy.decide(impulse.model_copy(update={"return_15s_pct": -0.00032, "taker_buy_ratio_10s": 0.30}))
    confirmed = strategy.decide(impulse.model_copy(update={"return_15s_pct": -0.00036, "taker_buy_ratio_10s": 0.28}))

    assert confirmed.action == DecisionAction.propose_short
    assert confirmed.confidence >= 0.70
    assert confirmed.evidence["market_sequence"]["state"] == "short_continuation_confirmed"
    assert confirmed.evidence["stateful_momentum_filter"]["blocked"] is False
    assert confirmed.evidence["stateful_momentum_filter"]["target_feasible"] is True


def test_stateful_momentum_uses_higher_timeframe_aligned_fast_profile():
    strategy = HitAndRunStrategy(
        Settings(MIN_CONFIDENCE=0.70, STRATEGY_VARIANT="stateful_momentum", FEE_EDGE_QUALITY_GATE_ENABLED=False)
    )
    impulse = _market(
        range_position_180s=0.08,
        return_15s_pct=-0.00025,
        return_60s_pct=-0.0009,
        return_180s_pct=-0.0026,
        taker_buy_ratio_10s=0.26,
        taker_buy_ratio_30s=0.35,
        book_imbalance_top=-0.25,
        depth_imbalance_top5=-0.55,
        open_interest_change_5m_pct=0.001,
        higher_timeframe_context={"bias": {"side": "short", "strength": 0.55, "reason": "1h/4h downtrend"}},
        higher_timeframe_context_age_seconds=30,
        higher_timeframe_bias_side="short",
        higher_timeframe_bias_strength=0.55,
        higher_timeframe_bias_reason="1h/4h downtrend",
    )

    strategy.decide(impulse)
    strategy.decide(impulse.model_copy(update={"return_15s_pct": 0.00025, "taker_buy_ratio_10s": 0.44}))
    strategy.decide(impulse.model_copy(update={"return_15s_pct": -0.00032, "taker_buy_ratio_10s": 0.30}))
    confirmed = strategy.decide(impulse.model_copy(update={"return_15s_pct": -0.00036, "taker_buy_ratio_10s": 0.28}))

    assert confirmed.action == DecisionAction.propose_short
    assert confirmed.target_move_pct == 0.002
    assert confirmed.stop_move_pct == 0.0015
    assert confirmed.evidence["trade_profile"] == "htf_aligned_fast"
    assert confirmed.evidence["stateful_momentum_filter"]["higher_timeframe_gate"]["profile"] == "htf_aligned_fast"


def test_stateful_momentum_blocks_htf_short_when_5m_is_still_bouncing():
    strategy = HitAndRunStrategy(Settings(MIN_CONFIDENCE=0.70, STRATEGY_VARIANT="stateful_momentum"))
    impulse = _market(
        range_position_180s=0.08,
        return_15s_pct=-0.00025,
        return_60s_pct=-0.0009,
        return_180s_pct=-0.0026,
        taker_buy_ratio_10s=0.26,
        taker_buy_ratio_30s=0.35,
        book_imbalance_top=-0.25,
        depth_imbalance_top5=-0.55,
        open_interest_change_5m_pct=0.001,
        higher_timeframe_context={
            "bias": {"side": "short", "strength": 0.55, "reason": "1h/4h downtrend"},
            "timeframes": {
                "5m": {
                    "structure": "uptrend_breakout",
                    "trend_score": 0.72,
                    "range_position": 0.82,
                    "taker_buy_ratio": 0.52,
                }
            },
        },
        higher_timeframe_context_age_seconds=30,
        higher_timeframe_bias_side="short",
        higher_timeframe_bias_strength=0.55,
        higher_timeframe_bias_reason="1h/4h downtrend",
    )

    strategy.decide(impulse)
    strategy.decide(impulse.model_copy(update={"return_15s_pct": 0.00025, "taker_buy_ratio_10s": 0.44}))
    strategy.decide(impulse.model_copy(update={"return_15s_pct": -0.00032, "taker_buy_ratio_10s": 0.30}))
    confirmed = strategy.decide(
        impulse.model_copy(
            update={
                "return_15s_pct": -0.00034,
                "return_60s_pct": -0.0002,
                "taker_buy_ratio_10s": 0.28,
            }
        )
    )

    assert confirmed.action == DecisionAction.wait
    stateful_filter = confirmed.evidence["stateful_momentum_filter"]
    assert "local_execution_countertrend" in stateful_filter["blockers"]
    assert stateful_filter["local_execution_gate"]["structure_5m"] == "uptrend_breakout"
    assert stateful_filter["local_execution_gate"]["allowed"] is False
    assert "shadow_trade" in stateful_filter


def test_stateful_momentum_blocks_counter_higher_timeframe_trade_unless_exceptional():
    strategy = HitAndRunStrategy(
        Settings(
            MIN_CONFIDENCE=0.70,
            STRATEGY_VARIANT="stateful_momentum",
            HIGHER_TIMEFRAME_COUNTERTREND_MIN_QUALITY=0.99,
        )
    )
    impulse = _market(
        range_position_180s=0.97,
        return_15s_pct=0.00025,
        return_60s_pct=0.0009,
        return_180s_pct=0.0026,
        taker_buy_ratio_10s=0.74,
        taker_buy_ratio_30s=0.72,
        book_imbalance_top=0.25,
        depth_imbalance_top5=0.55,
        open_interest_change_5m_pct=0.001,
        higher_timeframe_context={"bias": {"side": "short", "strength": 0.65, "reason": "1h/4h downtrend"}},
        higher_timeframe_context_age_seconds=30,
        higher_timeframe_bias_side="short",
        higher_timeframe_bias_strength=0.65,
        higher_timeframe_bias_reason="1h/4h downtrend",
    )

    strategy.decide(impulse)
    strategy.decide(impulse.model_copy(update={"return_15s_pct": -0.00025, "taker_buy_ratio_10s": 0.56}))
    strategy.decide(impulse.model_copy(update={"return_15s_pct": 0.00032, "taker_buy_ratio_10s": 0.70}))
    confirmed = strategy.decide(impulse.model_copy(update={"return_15s_pct": 0.00034, "taker_buy_ratio_10s": 0.72}))

    assert confirmed.action == DecisionAction.wait
    stateful_filter = confirmed.evidence["stateful_momentum_filter"]
    assert "higher_timeframe_countertrend" in stateful_filter["blockers"]
    assert stateful_filter["higher_timeframe_gate"]["profile"] == "countertrend_shadow_only"
    assert "shadow_trade" in stateful_filter


def test_stateful_momentum_allows_eth_counter_htf_bounce_when_local_context_confirms():
    strategy = HitAndRunStrategy(
        Settings(
            MIN_CONFIDENCE=0.70,
            STRATEGY_VARIANT="stateful_momentum",
            COUNTER_HTF_BOUNCE_MIN_QUALITY=0.65,
            COUNTER_HTF_BOUNCE_MIN_SCORE=0.70,
            COUNTER_HTF_BOUNCE_MIN_SEQUENCE_CONFIDENCE=0.85,
            COUNTER_HTF_BOUNCE_LIVE_ENABLED=True,
            FEE_EDGE_FAST_MIN_QUALITY=0.65,
            FEE_EDGE_FAST_MIN_SEQUENCE_CONFIDENCE=0.85,
        )
    )
    impulse = _market(
        symbol="ETHUSDT",
        range_180s_pct=0.003,
        range_position_180s=0.92,
        return_15s_pct=0.00035,
        return_60s_pct=0.0012,
        return_180s_pct=0.0028,
        taker_buy_ratio_10s=0.78,
        taker_buy_ratio_30s=0.72,
        book_imbalance_top=0.35,
        depth_imbalance_top5=0.55,
        open_interest_change_5m_pct=0.001,
        higher_timeframe_context={
            "bias": {"side": "short", "strength": 0.62, "reason": "4h/1d downtrend"},
            "timeframes": {
                "5m": {
                    "structure": "uptrend_breakout",
                    "trend_score": 0.68,
                    "range_position": 0.76,
                    "taker_buy_ratio": 0.54,
                },
                "1h": {
                    "structure": "balanced",
                    "trend_score": 0.12,
                    "return_pct": 0.003,
                    "range_position": 0.58,
                },
            },
        },
        higher_timeframe_context_age_seconds=30,
        higher_timeframe_bias_side="short",
        higher_timeframe_bias_strength=0.62,
        higher_timeframe_bias_reason="4h/1d downtrend",
    )

    strategy.decide(impulse)
    strategy.decide(impulse.model_copy(update={"return_15s_pct": -0.00025, "taker_buy_ratio_10s": 0.56}))
    strategy.decide(impulse.model_copy(update={"return_15s_pct": 0.00034, "taker_buy_ratio_10s": 0.74}))
    confirmed = strategy.decide(impulse.model_copy(update={"return_15s_pct": 0.00040, "taker_buy_ratio_10s": 0.76}))

    assert confirmed.action == DecisionAction.propose_long
    assert confirmed.evidence["trade_profile"] == "eth_counter_htf_bounce"
    stateful_filter = confirmed.evidence["stateful_momentum_filter"]
    assert stateful_filter["higher_timeframe_gate"]["profile"] == "eth_counter_htf_bounce"
    assert stateful_filter["higher_timeframe_gate"]["counter_bounce_gate"]["allowed"] is True
    assert confirmed.target_move_pct == 0.002


def test_stateful_momentum_keeps_eth_counter_htf_bounce_shadow_only_by_default():
    strategy = HitAndRunStrategy(Settings(MIN_CONFIDENCE=0.70, STRATEGY_VARIANT="stateful_momentum"))
    impulse = _market(
        symbol="ETHUSDT",
        range_180s_pct=0.0022,
        range_position_180s=0.90,
        return_15s_pct=0.00025,
        return_60s_pct=0.0014,
        return_180s_pct=0.0030,
        taker_buy_ratio_10s=0.78,
        taker_buy_ratio_30s=0.72,
        book_imbalance_top=0.35,
        depth_imbalance_top5=0.55,
        open_interest_change_5m_pct=0.001,
        higher_timeframe_context={
            "bias": {"side": "short", "strength": 0.62, "reason": "4h/1d downtrend"},
            "timeframes": {
                "5m": {
                    "structure": "uptrend_pullback",
                    "trend_score": 0.58,
                    "range_position": 0.70,
                    "taker_buy_ratio": 0.54,
                },
                "1h": {
                    "structure": "downtrend_bounce",
                    "trend_score": -0.08,
                    "return_pct": 0.001,
                    "range_position": 0.54,
                },
            },
        },
        higher_timeframe_context_age_seconds=30,
        higher_timeframe_bias_side="short",
        higher_timeframe_bias_strength=0.62,
        higher_timeframe_bias_reason="4h/1d downtrend",
    )

    strategy.decide(impulse)
    strategy.decide(impulse.model_copy(update={"return_15s_pct": -0.00020, "taker_buy_ratio_10s": 0.56}))
    strategy.decide(impulse.model_copy(update={"return_15s_pct": 0.00028, "taker_buy_ratio_10s": 0.74}))
    confirmed = strategy.decide(impulse.model_copy(update={"return_15s_pct": 0.00030, "taker_buy_ratio_10s": 0.78}))

    assert confirmed.action == DecisionAction.wait
    stateful_filter = confirmed.evidence["stateful_momentum_filter"]
    assert "counter_htf_bounce_shadow_only" in stateful_filter["blockers"]
    assert "shadow_trade" in stateful_filter
    assert "counter-HTF bounce profile is shadow-only" in confirmed.reason
    assert stateful_filter["higher_timeframe_gate"]["profile"] == "eth_counter_htf_bounce"
    gate = stateful_filter["entry_follow_through_gate"]
    assert gate["checks"]["return_15s"] is False
    assert gate["strict_mandatory_confirmed"] is False
    assert gate["counter_htf_bounce_override_confirmed"] is True
    assert gate["allowed"] is True
    assert stateful_filter["fee_edge_gate"]["trade_profile"] == "eth_counter_htf_bounce"
    assert stateful_filter["fee_edge_gate"]["allowed"] is True


def test_stateful_momentum_allows_eth_counter_htf_bounce_when_live_enabled():
    strategy = HitAndRunStrategy(
        Settings(
            MIN_CONFIDENCE=0.70,
            STRATEGY_VARIANT="stateful_momentum",
            COUNTER_HTF_BOUNCE_LIVE_ENABLED=True,
        )
    )
    impulse = _market(
        symbol="ETHUSDT",
        range_180s_pct=0.0022,
        range_position_180s=0.90,
        return_15s_pct=0.00025,
        return_60s_pct=0.0014,
        return_180s_pct=0.0030,
        taker_buy_ratio_10s=0.78,
        taker_buy_ratio_30s=0.72,
        book_imbalance_top=0.35,
        depth_imbalance_top5=0.55,
        open_interest_change_5m_pct=0.001,
        higher_timeframe_context={
            "bias": {"side": "short", "strength": 0.62, "reason": "4h/1d downtrend"},
            "timeframes": {
                "5m": {
                    "structure": "uptrend_pullback",
                    "trend_score": 0.58,
                    "range_position": 0.70,
                    "taker_buy_ratio": 0.54,
                },
                "1h": {
                    "structure": "downtrend_bounce",
                    "trend_score": -0.08,
                    "return_pct": 0.001,
                    "range_position": 0.54,
                },
            },
        },
        higher_timeframe_context_age_seconds=30,
        higher_timeframe_bias_side="short",
        higher_timeframe_bias_strength=0.62,
        higher_timeframe_bias_reason="4h/1d downtrend",
    )

    strategy.decide(impulse)
    strategy.decide(impulse.model_copy(update={"return_15s_pct": -0.00020, "taker_buy_ratio_10s": 0.56}))
    strategy.decide(impulse.model_copy(update={"return_15s_pct": 0.00028, "taker_buy_ratio_10s": 0.74}))
    confirmed = strategy.decide(impulse.model_copy(update={"return_15s_pct": 0.00030, "taker_buy_ratio_10s": 0.78}))

    assert confirmed.action == DecisionAction.propose_long
    stateful_filter = confirmed.evidence["stateful_momentum_filter"]
    assert "counter_htf_bounce_shadow_only" not in stateful_filter["blockers"]
    assert stateful_filter["higher_timeframe_gate"]["profile"] == "eth_counter_htf_bounce"


def test_stateful_momentum_allows_eth_counter_htf_bounce_when_one_hour_leads_relief():
    strategy = HitAndRunStrategy(
        Settings(
            MIN_CONFIDENCE=0.70,
            STRATEGY_VARIANT="stateful_momentum",
            COUNTER_HTF_BOUNCE_LIVE_ENABLED=True,
        )
    )
    impulse = _market(
        symbol="ETHUSDT",
        range_180s_pct=0.0034,
        range_position_180s=0.88,
        return_15s_pct=0.00030,
        return_60s_pct=0.0014,
        return_180s_pct=0.0030,
        taker_buy_ratio_10s=0.78,
        taker_buy_ratio_30s=0.72,
        book_imbalance_top=0.35,
        depth_imbalance_top5=0.55,
        open_interest_change_5m_pct=0.001,
        higher_timeframe_context={
            "bias": {"side": "short", "strength": 0.62, "reason": "4h/1d downtrend"},
            "timeframes": {
                "5m": {
                    "structure": "downtrend_breakdown",
                    "trend_score": -0.92,
                    "range_position": 0.18,
                    "taker_buy_ratio": 0.46,
                },
                "1h": {
                    "structure": "balanced",
                    "trend_score": 0.20,
                    "return_pct": 0.004,
                    "range_position": 0.35,
                },
            },
        },
        higher_timeframe_context_age_seconds=30,
        higher_timeframe_bias_side="short",
        higher_timeframe_bias_strength=0.62,
        higher_timeframe_bias_reason="4h/1d downtrend",
    )

    strategy.decide(impulse)
    strategy.decide(impulse.model_copy(update={"return_15s_pct": -0.00020, "taker_buy_ratio_10s": 0.56}))
    strategy.decide(impulse.model_copy(update={"return_15s_pct": 0.00028, "taker_buy_ratio_10s": 0.74}))
    confirmed = strategy.decide(impulse.model_copy(update={"return_15s_pct": 0.00030, "taker_buy_ratio_10s": 0.78}))

    assert confirmed.action == DecisionAction.propose_long
    stateful_filter = confirmed.evidence["stateful_momentum_filter"]
    counter_gate = stateful_filter["higher_timeframe_gate"]["counter_bounce_gate"]
    assert stateful_filter["higher_timeframe_gate"]["profile"] == "eth_counter_htf_bounce"
    assert counter_gate["allowed"] is True
    assert counter_gate["local_context"]["five_minute_reclaim"] is False
    assert counter_gate["local_context"]["one_hour_led_relief"] is True


def test_stateful_momentum_blocks_weak_neutral_short_without_stronger_quality():
    strategy = HitAndRunStrategy(
        Settings(
            MIN_CONFIDENCE=0.70,
            STRATEGY_VARIANT="stateful_momentum",
            FEE_EDGE_QUALITY_GATE_ENABLED=False,
        )
    )
    impulse = _market(
        range_position_180s=0.08,
        return_15s_pct=-0.00025,
        return_60s_pct=-0.0009,
        return_180s_pct=-0.0026,
        taker_buy_ratio_10s=0.26,
        taker_buy_ratio_30s=0.35,
        book_imbalance_top=-0.25,
        depth_imbalance_top5=-0.55,
        open_interest_change_5m_pct=0.001,
        higher_timeframe_context={"bias": {"side": "neutral", "strength": 0.15, "reason": "mixed"}},
        higher_timeframe_context_age_seconds=30,
        higher_timeframe_bias_side="neutral",
        higher_timeframe_bias_strength=0.15,
        higher_timeframe_bias_reason="mixed",
    )

    strategy.decide(impulse)
    strategy.decide(impulse.model_copy(update={"return_15s_pct": 0.00025, "taker_buy_ratio_10s": 0.44}))
    strategy.decide(impulse.model_copy(update={"return_15s_pct": -0.00032, "taker_buy_ratio_10s": 0.30}))
    confirmed = strategy.decide(impulse.model_copy(update={"return_15s_pct": -0.00036, "taker_buy_ratio_10s": 0.28}))

    assert confirmed.action == DecisionAction.wait
    stateful_filter = confirmed.evidence["stateful_momentum_filter"]
    assert "weak_neutral_short_quality" in stateful_filter["blockers"]
    assert stateful_filter["higher_timeframe_gate"]["profile"] == "weak_or_neutral_htf"
    assert stateful_filter["weak_neutral_short_gate"]["allowed"] is False


def test_stateful_momentum_blocks_weak_neutral_long_without_one_hour_support():
    strategy = HitAndRunStrategy(
        Settings(
            MIN_CONFIDENCE=0.70,
            STRATEGY_VARIANT="stateful_momentum",
            FEE_EDGE_QUALITY_GATE_ENABLED=False,
            WEAK_NEUTRAL_LONG_MIN_QUALITY=0.50,
            WEAK_NEUTRAL_LONG_MIN_SEQUENCE_CONFIDENCE=0.50,
            WEAK_NEUTRAL_LONG_MIN_FOLLOW_SCORE=0.50,
        )
    )
    impulse = _market(
        symbol="ETHUSDT",
        range_position_180s=0.92,
        return_15s_pct=0.00042,
        return_60s_pct=0.0011,
        return_180s_pct=0.0028,
        taker_buy_ratio_10s=0.82,
        taker_buy_ratio_30s=0.68,
        book_imbalance_top=0.35,
        depth_imbalance_top5=0.55,
        open_interest_change_5m_pct=0.001,
        higher_timeframe_context={
            "timeframes": {
                "5m": {
                    "structure": "uptrend_breakout",
                    "trend_score": 0.72,
                    "return_pct": 0.002,
                    "range_position": 0.80,
                    "taker_buy_ratio": 0.55,
                },
                "1h": {
                    "structure": "balanced",
                    "trend_score": 0.10,
                    "return_pct": -0.001,
                    "range_position": 0.78,
                },
            },
        },
        higher_timeframe_context_age_seconds=30,
        higher_timeframe_bias_side="short",
        higher_timeframe_bias_strength=0.20,
        higher_timeframe_bias_reason="weak mixed bias",
    )

    strategy.decide(impulse)
    strategy.decide(impulse.model_copy(update={"return_15s_pct": -0.00025, "taker_buy_ratio_10s": 0.56}))
    strategy.decide(impulse.model_copy(update={"return_15s_pct": 0.00040, "taker_buy_ratio_10s": 0.78}))
    confirmed = strategy.decide(impulse.model_copy(update={"return_15s_pct": 0.00044, "taker_buy_ratio_10s": 0.82}))

    assert confirmed.action == DecisionAction.wait
    stateful_filter = confirmed.evidence["stateful_momentum_filter"]
    assert stateful_filter["higher_timeframe_gate"]["profile"] == "weak_or_neutral_htf"
    assert "weak_neutral_long_quality" in stateful_filter["blockers"]
    assert stateful_filter["weak_neutral_long_gate"]["allowed"] is False
    assert "one_hour_support" in stateful_filter["weak_neutral_long_gate"]["blockers"]


def test_stateful_momentum_allows_weak_neutral_long_with_one_hour_support():
    strategy = HitAndRunStrategy(
        Settings(
            MIN_CONFIDENCE=0.70,
            STRATEGY_VARIANT="stateful_momentum",
            FEE_EDGE_QUALITY_GATE_ENABLED=False,
            WEAK_NEUTRAL_LONG_MIN_QUALITY=0.50,
            WEAK_NEUTRAL_LONG_MIN_SEQUENCE_CONFIDENCE=0.50,
            WEAK_NEUTRAL_LONG_MIN_FOLLOW_SCORE=0.50,
        )
    )
    impulse = _market(
        symbol="ETHUSDT",
        range_position_180s=0.92,
        return_15s_pct=0.00042,
        return_60s_pct=0.0011,
        return_180s_pct=0.0028,
        taker_buy_ratio_10s=0.82,
        taker_buy_ratio_30s=0.68,
        book_imbalance_top=0.35,
        depth_imbalance_top5=0.55,
        open_interest_change_5m_pct=0.001,
        higher_timeframe_context={
            "timeframes": {
                "5m": {
                    "structure": "uptrend_breakout",
                    "trend_score": 0.72,
                    "return_pct": 0.002,
                    "range_position": 0.80,
                    "taker_buy_ratio": 0.55,
                },
                "1h": {
                    "structure": "uptrend_pullback",
                    "trend_score": 0.32,
                    "return_pct": 0.003,
                    "range_position": 0.78,
                },
            },
        },
        higher_timeframe_context_age_seconds=30,
        higher_timeframe_bias_side="short",
        higher_timeframe_bias_strength=0.20,
        higher_timeframe_bias_reason="weak mixed bias",
    )

    strategy.decide(impulse)
    strategy.decide(impulse.model_copy(update={"return_15s_pct": -0.00025, "taker_buy_ratio_10s": 0.56}))
    strategy.decide(impulse.model_copy(update={"return_15s_pct": 0.00040, "taker_buy_ratio_10s": 0.78}))
    confirmed = strategy.decide(impulse.model_copy(update={"return_15s_pct": 0.00044, "taker_buy_ratio_10s": 0.82}))

    assert confirmed.action == DecisionAction.propose_long
    stateful_filter = confirmed.evidence["stateful_momentum_filter"]
    assert stateful_filter["higher_timeframe_gate"]["profile"] == "weak_or_neutral_htf"
    assert stateful_filter["weak_neutral_long_gate"]["allowed"] is True
    assert stateful_filter["weak_neutral_long_gate"]["one_hour_supportive"] is True


def test_stateful_momentum_blocks_fee_thin_trade_when_quality_is_not_enough():
    strategy = HitAndRunStrategy(
        Settings(
            MIN_CONFIDENCE=0.70,
            STRATEGY_VARIANT="stateful_momentum",
            FEE_EDGE_FAST_MIN_QUALITY=0.95,
            FEE_EDGE_FAST_MIN_SEQUENCE_CONFIDENCE=0.85,
        )
    )
    impulse = _market(
        range_180s_pct=0.003,
        range_position_180s=0.95,
        return_15s_pct=0.00035,
        return_60s_pct=0.0012,
        return_180s_pct=0.0028,
        taker_buy_ratio_10s=0.78,
        taker_buy_ratio_30s=0.72,
        book_imbalance_top=0.35,
        depth_imbalance_top5=0.55,
        open_interest_change_5m_pct=0.001,
    )

    strategy.decide(impulse)
    strategy.decide(impulse.model_copy(update={"return_15s_pct": -0.00025, "taker_buy_ratio_10s": 0.56}))
    strategy.decide(impulse.model_copy(update={"return_15s_pct": 0.00034, "taker_buy_ratio_10s": 0.74}))
    confirmed = strategy.decide(impulse.model_copy(update={"return_15s_pct": 0.00040, "taker_buy_ratio_10s": 0.76}))

    assert confirmed.action == DecisionAction.wait
    stateful_filter = confirmed.evidence["stateful_momentum_filter"]
    assert "fee_edge_quality" in stateful_filter["blockers"]
    assert stateful_filter["fee_edge_gate"]["fee_thin_target"] is True
    assert stateful_filter["fee_edge_gate"]["allowed"] is False


def test_stateful_momentum_blocks_entry_without_immediate_follow_through():
    strategy = HitAndRunStrategy(
        Settings(
            MIN_CONFIDENCE=0.70,
            STRATEGY_VARIANT="stateful_momentum",
            FEE_EDGE_QUALITY_GATE_ENABLED=False,
        )
    )
    impulse = _market(
        range_180s_pct=0.003,
        range_position_180s=0.95,
        return_15s_pct=0.00025,
        return_60s_pct=0.0012,
        return_180s_pct=0.0028,
        taker_buy_ratio_10s=0.78,
        taker_buy_ratio_30s=0.72,
        book_imbalance_top=0.35,
        depth_imbalance_top5=0.55,
        open_interest_change_5m_pct=0.001,
    )

    strategy.decide(impulse)
    strategy.decide(impulse.model_copy(update={"return_15s_pct": -0.00025, "taker_buy_ratio_10s": 0.56}))
    strategy.decide(impulse.model_copy(update={"return_15s_pct": 0.00034, "taker_buy_ratio_10s": 0.74}))
    confirmed = strategy.decide(impulse.model_copy(update={"return_15s_pct": 0.00032, "taker_buy_ratio_10s": 0.76}))

    assert confirmed.action == DecisionAction.wait
    stateful_filter = confirmed.evidence["stateful_momentum_filter"]
    assert "entry_follow_through" in stateful_filter["blockers"]
    assert stateful_filter["entry_follow_through_gate"]["checks"]["return_15s"] is False
    assert stateful_filter["entry_follow_through_gate"]["allowed"] is False


def test_stateful_momentum_allows_entry_with_immediate_follow_through():
    strategy = HitAndRunStrategy(
        Settings(
            MIN_CONFIDENCE=0.70,
            STRATEGY_VARIANT="stateful_momentum",
            FEE_EDGE_QUALITY_GATE_ENABLED=False,
        )
    )
    impulse = _market(
        range_180s_pct=0.003,
        range_position_180s=0.95,
        return_15s_pct=0.00035,
        return_60s_pct=0.0012,
        return_180s_pct=0.0028,
        taker_buy_ratio_10s=0.78,
        taker_buy_ratio_30s=0.72,
        book_imbalance_top=0.35,
        depth_imbalance_top5=0.55,
        open_interest_change_5m_pct=0.001,
    )

    strategy.decide(impulse)
    strategy.decide(impulse.model_copy(update={"return_15s_pct": -0.00025, "taker_buy_ratio_10s": 0.56}))
    strategy.decide(impulse.model_copy(update={"return_15s_pct": 0.00034, "taker_buy_ratio_10s": 0.74}))
    confirmed = strategy.decide(impulse.model_copy(update={"return_15s_pct": 0.00040, "taker_buy_ratio_10s": 0.76}))

    assert confirmed.action == DecisionAction.propose_long
    gate = confirmed.evidence["stateful_momentum_filter"]["entry_follow_through_gate"]
    assert gate["allowed"] is True
    assert gate["confirmations"] >= 4


def test_stateful_momentum_allows_htf_aligned_entry_with_strong_flow_before_15s_threshold():
    strategy = HitAndRunStrategy(
        Settings(
            MIN_CONFIDENCE=0.70,
            STRATEGY_VARIANT="stateful_momentum",
            FEE_EDGE_QUALITY_GATE_ENABLED=False,
        )
    )
    impulse = _market(
        range_position_180s=0.08,
        return_15s_pct=-0.00025,
        return_60s_pct=-0.0012,
        return_180s_pct=-0.0028,
        taker_buy_ratio_10s=0.22,
        taker_buy_ratio_30s=0.34,
        book_imbalance_top=-0.35,
        depth_imbalance_top5=-0.55,
        open_interest_change_5m_pct=0.001,
        higher_timeframe_context={"bias": {"side": "short", "strength": 0.55, "reason": "1h/4h downtrend"}},
        higher_timeframe_context_age_seconds=30,
        higher_timeframe_bias_side="short",
        higher_timeframe_bias_strength=0.55,
        higher_timeframe_bias_reason="1h/4h downtrend",
    )

    strategy.decide(impulse)
    strategy.decide(impulse.model_copy(update={"return_15s_pct": 0.00025, "taker_buy_ratio_10s": 0.44}))
    strategy.decide(impulse.model_copy(update={"return_15s_pct": -0.00030, "taker_buy_ratio_10s": 0.30}))
    confirmed = strategy.decide(impulse.model_copy(update={"return_15s_pct": -0.00026, "taker_buy_ratio_10s": 0.28}))

    assert confirmed.action == DecisionAction.propose_short
    gate = confirmed.evidence["stateful_momentum_filter"]["entry_follow_through_gate"]
    assert gate["trade_profile"] == "htf_aligned_fast"
    assert gate["checks"]["return_15s"] is False
    assert gate["strict_mandatory_confirmed"] is False
    assert gate["htf_aligned_override_confirmed"] is True
    assert gate["allowed"] is True


def test_stateful_momentum_suppresses_duplicate_same_regime_signal():
    strategy = HitAndRunStrategy(
        Settings(
            MIN_CONFIDENCE=0.70,
            STRATEGY_VARIANT="stateful_momentum",
            DUPLICATE_SIGNAL_SUPPRESSION_SECONDS=900,
            FEE_EDGE_FAST_MIN_QUALITY=0.60,
            FEE_EDGE_FAST_MIN_SEQUENCE_CONFIDENCE=0.80,
        )
    )
    start = datetime(2026, 5, 21, 12, 0, tzinfo=timezone.utc)
    impulse = _market(
        last_received_at=start,
        range_180s_pct=0.003,
        range_position_180s=0.95,
        return_15s_pct=0.00035,
        return_60s_pct=0.0012,
        return_180s_pct=0.0028,
        taker_buy_ratio_10s=0.78,
        taker_buy_ratio_30s=0.72,
        book_imbalance_top=0.35,
        depth_imbalance_top5=0.55,
        open_interest_change_5m_pct=0.001,
    )

    strategy.decide(impulse)
    strategy.decide(
        impulse.model_copy(
            update={"last_received_at": start + timedelta(seconds=1), "return_15s_pct": -0.00025, "taker_buy_ratio_10s": 0.56}
        )
    )
    strategy.decide(
        impulse.model_copy(
            update={"last_received_at": start + timedelta(seconds=2), "return_15s_pct": 0.00034, "taker_buy_ratio_10s": 0.74}
        )
    )
    first = strategy.decide(
        impulse.model_copy(
            update={"last_received_at": start + timedelta(seconds=3), "return_15s_pct": 0.00040, "taker_buy_ratio_10s": 0.76}
        )
    )
    repeat = strategy.decide(
        impulse.model_copy(
            update={"last_received_at": start + timedelta(seconds=60), "return_15s_pct": 0.00041, "taker_buy_ratio_10s": 0.76}
        )
    )

    assert first.action == DecisionAction.propose_long
    assert repeat.action == DecisionAction.wait
    stateful_filter = repeat.evidence["stateful_momentum_filter"]
    assert "duplicate_signal" in stateful_filter["blockers"]
    assert "shadow_trade" not in stateful_filter


def test_stateful_momentum_blocks_breakout_when_recent_range_cannot_support_target():
    strategy = HitAndRunStrategy(
        Settings(MIN_CONFIDENCE=0.70, STRATEGY_VARIANT="stateful_momentum", ENTRY_FOLLOW_THROUGH_GATE_ENABLED=False)
    )
    impulse = _market(
        range_180s_pct=0.0008,
        range_position_180s=0.02,
        return_15s_pct=-0.00025,
        return_60s_pct=-0.0009,
        return_180s_pct=-0.0026,
        taker_buy_ratio_10s=0.08,
        taker_buy_ratio_30s=0.20,
        book_imbalance_top=-0.90,
        depth_imbalance_top5=-0.55,
        open_interest_change_5m_pct=0.001,
    )

    strategy.decide(impulse)
    strategy.decide(impulse.model_copy(update={"return_15s_pct": 0.00025, "taker_buy_ratio_10s": 0.44}))
    strategy.decide(impulse.model_copy(update={"return_15s_pct": -0.00032, "taker_buy_ratio_10s": 0.12}))
    confirmed = strategy.decide(impulse.model_copy(update={"return_15s_pct": -0.00034, "taker_buy_ratio_10s": 0.10}))

    assert confirmed.action == DecisionAction.wait
    assert "target feasibility blocked" in confirmed.reason
    assert confirmed.evidence["market_sequence"]["state"] == "short_continuation_confirmed"
    assert confirmed.evidence["stateful_momentum_filter"]["blocked"] is True
    assert confirmed.evidence["stateful_momentum_filter"]["blockers"] == ["target_feasibility"]
    assert confirmed.evidence["stateful_momentum_filter"]["target_feasible"] is False
    assert confirmed.evidence["stateful_momentum_filter"]["adaptive_entry_allowed"] is False
    assert "shadow_trade" in confirmed.evidence["stateful_momentum_filter"]
    assert confirmed.evidence["stateful_momentum_filter"]["range_180s_pct"] == 0.0008
    assert confirmed.evidence["stateful_momentum_filter"]["min_required_range_180s_pct"] == 0.0012


def test_stateful_momentum_adaptive_gate_keeps_low_range_long_shadow_only_by_default():
    strategy = HitAndRunStrategy(
        Settings(
            MIN_CONFIDENCE=0.72,
            STRATEGY_VARIANT="stateful_momentum",
            STATEFUL_ADAPTIVE_MIN_SCORE=0.70,
            STATEFUL_ADAPTIVE_MIN_SEQUENCE_CONFIDENCE=0.85,
            FEE_EDGE_QUALITY_GATE_ENABLED=False,
        )
    )
    impulse = _market(
        range_180s_pct=0.0011,
        range_position_180s=0.92,
        return_15s_pct=0.00032,
        return_60s_pct=0.001,
        return_180s_pct=0.0027,
        taker_buy_ratio_10s=0.78,
        taker_buy_ratio_30s=0.72,
        book_imbalance_top=0.40,
        depth_imbalance_top5=0.60,
        open_interest_change_5m_pct=0.001,
    )

    strategy.decide(impulse)
    strategy.decide(impulse.model_copy(update={"return_15s_pct": -0.00025, "taker_buy_ratio_10s": 0.56}))
    strategy.decide(impulse.model_copy(update={"return_15s_pct": 0.00034, "taker_buy_ratio_10s": 0.74}))
    confirmed = strategy.decide(impulse.model_copy(update={"return_15s_pct": 0.00036, "taker_buy_ratio_10s": 0.76}))

    assert confirmed.action == DecisionAction.wait
    assert "adaptive low-range profile is shadow-only" in confirmed.reason
    stateful_filter = confirmed.evidence["stateful_momentum_filter"]
    assert stateful_filter["target_feasible"] is False
    assert stateful_filter["adaptive_entry_allowed"] is True
    assert "adaptive_low_range_shadow_only" in stateful_filter["blockers"]
    assert stateful_filter["adaptive_low_range_live_gate"]["enabled"] is False
    assert stateful_filter["entry_follow_through_gate"]["trade_profile"] == "adaptive_low_range"
    assert "shadow_trade" in stateful_filter


def test_stateful_momentum_adaptive_gate_can_be_live_enabled_with_strict_follow_through():
    strategy = HitAndRunStrategy(
        Settings(
            MIN_CONFIDENCE=0.72,
            STRATEGY_VARIANT="stateful_momentum",
            STATEFUL_ADAPTIVE_MIN_SCORE=0.70,
            STATEFUL_ADAPTIVE_MIN_SEQUENCE_CONFIDENCE=0.85,
            STATEFUL_ADAPTIVE_LIVE_ENABLED=True,
            STATEFUL_ADAPTIVE_LIVE_MIN_FOLLOW_SCORE=0.86,
            STATEFUL_ADAPTIVE_LIVE_MIN_FOLLOW_CONFIRMATIONS=5,
            FEE_EDGE_QUALITY_GATE_ENABLED=False,
        )
    )
    impulse = _market(
        range_180s_pct=0.0011,
        range_position_180s=0.92,
        return_15s_pct=0.00042,
        return_60s_pct=0.0012,
        return_180s_pct=0.0027,
        taker_buy_ratio_10s=0.80,
        taker_buy_ratio_30s=0.72,
        book_imbalance_top=0.40,
        depth_imbalance_top5=0.60,
        open_interest_change_5m_pct=0.001,
    )

    strategy.decide(impulse)
    strategy.decide(impulse.model_copy(update={"return_15s_pct": -0.00025, "taker_buy_ratio_10s": 0.56}))
    strategy.decide(impulse.model_copy(update={"return_15s_pct": 0.00036, "taker_buy_ratio_10s": 0.76}))
    confirmed = strategy.decide(impulse.model_copy(update={"return_15s_pct": 0.00045, "taker_buy_ratio_10s": 0.80}))

    assert confirmed.action == DecisionAction.propose_long
    assert confirmed.mode is not None
    assert confirmed.mode.value == "slow"
    assert confirmed.leverage == 80
    assert confirmed.target_move_pct == 0.002
    assert confirmed.evidence["trade_profile"] == "adaptive_low_range"
    stateful_filter = confirmed.evidence["stateful_momentum_filter"]
    assert stateful_filter["adaptive_low_range_live_gate"]["allowed"] is True
    assert stateful_filter["entry_follow_through_gate"]["confirmations"] == 5

