import numpy as np

from futures_lab.exit_laboratory import (
    ExitOpportunity,
    ExitPolicy,
    _paired_bootstrap_delta,
    _select_exit_candidates,
    simulate_exit_policy,
)
from futures_lab.first_touch import PriceTimeline


def test_fixed_barrier_uses_side_relative_target_and_cost() -> None:
    timeline, opportunity = _timeline([100.0, 100.4, 100.7])
    policy = ExitPolicy(
        family="fixed_barrier",
        name="fixed",
        params={"target_bps": 60.0, "stop_bps": 60.0},
    )

    trade = simulate_exit_policy(
        timeline,
        opportunity,
        policy,
        features=(),
        feature_times=(),
        cost_bps=10.0,
    )

    assert trade.exit_reason == "take_profit"
    assert trade.gross_bps == 60.0
    assert trade.net_bps == 50.0


def test_mfe_trail_activates_only_after_favorable_excursion() -> None:
    timeline, opportunity = _timeline([100.0, 100.3, 100.5, 100.35])
    policy = ExitPolicy(
        family="mfe_trailing",
        name="trail",
        params={
            "target_bps": 100.0,
            "stop_bps": 150.0,
            "activation_bps": 40.0,
            "retrace_bps": 10.0,
        },
    )

    trade = simulate_exit_policy(
        timeline,
        opportunity,
        policy,
        features=(),
        feature_times=(),
        cost_bps=10.0,
    )

    assert trade.exit_reason == "mfe_trailing_stop"
    assert 34.9 < trade.gross_bps < 35.1
    assert trade.mfe_bps >= 50.0 - 1e-6


def test_partial_profit_blends_realized_legs_before_cost() -> None:
    timeline, opportunity = _timeline([100.0, 100.5, 100.2, 100.0])
    policy = ExitPolicy(
        family="partial_profit",
        name="partial",
        params={
            "target_bps": 100.0,
            "stop_bps": 150.0,
            "partial_target_bps": 40.0,
            "partial_fraction": 0.50,
            "move_to_breakeven": True,
        },
    )

    trade = simulate_exit_policy(
        timeline,
        opportunity,
        policy,
        features=(),
        feature_times=(),
        cost_bps=10.0,
    )

    assert trade.exit_reason == "partial_then_breakeven"
    assert trade.partial_taken is True
    assert trade.gross_bps == 20.0
    assert trade.net_bps == 10.0


def test_partial_profit_handles_scale_and_final_target_on_same_mark() -> None:
    timeline, opportunity = _timeline([100.0, 101.1])
    policy = ExitPolicy(
        family="partial_profit",
        name="partial_gap",
        params={
            "target_bps": 100.0,
            "stop_bps": 150.0,
            "partial_target_bps": 40.0,
            "partial_fraction": 0.50,
            "move_to_breakeven": False,
        },
    )

    trade = simulate_exit_policy(
        timeline,
        opportunity,
        policy,
        features=(),
        feature_times=(),
        cost_bps=10.0,
    )

    assert trade.exit_reason == "partial_then_target"
    assert trade.gross_bps == 70.0
    assert trade.net_bps == 60.0


def test_signal_invalidation_is_limited_to_feature_timestamp() -> None:
    times = np.asarray([0, 60_000, 120_000], dtype=np.int64)
    prices = np.asarray([100.0, 100.0, 100.0], dtype=np.float64)
    timeline = PriceTimeline(times, prices, max_gap_ms=60_000)
    opportunity = ExitOpportunity(0, 1, {}, 0, 2)
    opposing = {
        "order_flow_imbalance_1s": -1.0,
        "order_flow_imbalance_5s": -1.0,
        "taker_aggression_imbalance_1s": -1.0,
        "taker_aggression_imbalance_5s": -1.0,
        "microprice_mid_bps": -1.0,
        "vamp_mid_bps": -1.0,
        "depth_imbalance_top5": -1.0,
        "book_imbalance_top": -1.0,
    }
    policy = ExitPolicy(
        family="signal_invalidation",
        name="invalidate",
        params={
            "target_bps": 100.0,
            "stop_bps": 150.0,
            "minimum_hold_seconds": 60,
            "invalidation_threshold": 0.0,
        },
    )

    trade = simulate_exit_policy(
        timeline,
        opportunity,
        policy,
        features=[(60_000, opposing)],
        feature_times=[60_000],
        cost_bps=10.0,
    )

    assert trade.exit_reason == "signal_invalidation"
    assert trade.exit_ms == 60_000
    assert trade.net_bps == -10.0


def test_exit_selection_uses_pre_holdout_robust_floor() -> None:
    weak = _candidate("family", "weak", robust_floor=1.0, qualifies=True)
    strong = _candidate("family", "strong", robust_floor=3.0, qualifies=True)
    lucky = _candidate("family", "lucky", robust_floor=20.0, qualifies=False)

    selected = _select_exit_candidates([weak, strong, lucky])["family"]

    assert selected["policy"] == "strong"
    assert selected["selection_qualified"] is True
    assert selected["qualified_candidate_count"] == 2


def test_paired_bootstrap_aligns_identical_entry_cohorts() -> None:
    timeline, opportunity = _timeline([100.0, 100.7])
    winning = simulate_exit_policy(
        timeline,
        opportunity,
        ExitPolicy("fixed_barrier", "win", {"target_bps": 60.0, "stop_bps": 60.0}),
        features=(),
        feature_times=(),
        cost_bps=10.0,
    )
    baseline = winning.__class__(**{**winning.__dict__, "net_bps": 20.0})

    result = _paired_bootstrap_delta([winning] * 8, [baseline] * 8, samples=200, seed=7)

    assert result["mean_delta_bps"] == 30.0
    assert result["bootstrap_p025_bps"] == 30.0


def _timeline(prices: list[float]) -> tuple[PriceTimeline, ExitOpportunity]:
    times = np.arange(len(prices), dtype=np.int64) * 1_000
    timeline = PriceTimeline(times, np.asarray(prices, dtype=np.float64), max_gap_ms=2_000)
    opportunity = ExitOpportunity(0, 1, {}, 0, len(prices) - 1)
    return timeline, opportunity


def _candidate(family: str, policy: str, *, robust_floor: float, qualifies: bool) -> dict:
    metrics = {
        "average_net_bps_per_trade": robust_floor,
        "max_additive_account_drawdown_pct": 1.0,
    }
    return {
        "family": family,
        "policy": policy,
        "qualifies": qualifies,
        "robust_floor_bps": robust_floor,
        "discovery": metrics,
        "calibration": metrics,
    }
