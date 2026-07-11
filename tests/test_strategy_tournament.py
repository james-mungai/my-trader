from futures_lab.first_touch import LabeledSample
from futures_lab.strategy_tournament import (
    SignalVariant,
    _matched_coin_predictions,
    _period_boundaries,
    _select_family_candidates,
    predict_variant,
)


def test_period_boundaries_are_chronological() -> None:
    features = [(index * 1_000, {}) for index in range(100)]

    calibration, holdout = _period_boundaries(
        features,
        discovery_fraction=0.50,
        calibration_end_fraction=0.70,
    )

    assert calibration == 50_000
    assert holdout == 70_000


def test_squeeze_variant_is_side_symmetric() -> None:
    variant = SignalVariant(
        family="volatility_squeeze_breakout",
        name="test",
        params={"range_max_bps": 15.0, "edge": 0.80, "micro_min": 0.03},
    )
    long_row = _micro_row(1)
    long_row.update({"range_180s_pct": 0.001, "range_position_180s": 0.9, "return_15s_pct": 0.0001})
    short_row = {key: -value for key, value in _micro_row(1).items()}
    short_row.update({"range_180s_pct": 0.001, "range_position_180s": 0.1, "return_15s_pct": -0.0001})

    assert predict_variant(variant, long_row) == 1
    assert predict_variant(variant, short_row) == -1


def test_selection_prefers_qualified_robust_floor() -> None:
    weak = _candidate("family", "weak", floor=1.0, qualifies=True, calibration_trades=30)
    strong = _candidate("family", "strong", floor=2.0, qualifies=True, calibration_trades=15)
    lucky = _candidate("family", "lucky", floor=20.0, qualifies=False, calibration_trades=50)

    selected = _select_family_candidates([weak, strong, lucky])["family"]

    assert selected["variant"] == "strong"
    assert selected["selection_qualified"] is True
    assert selected["qualified_candidate_count"] == 2


def test_matched_coin_preserves_signal_timestamps() -> None:
    samples = [
        LabeledSample(timestamp_ms=index * 1_000, row={}, symmetric_direction=None, symmetric_exit_ms=None, scenarios={})
        for index in range(12)
    ]
    signals = [1, None, -1, None] * 3

    first = _matched_coin_predictions(samples, signals, seed=7, namespace="ETHUSDT")
    second = _matched_coin_predictions(samples, signals, seed=7, namespace="ETHUSDT")

    assert first == second
    assert [value is None for value in first] == [value is None for value in signals]
    assert set(value for value in first if value is not None) == {-1, 1}


def _micro_row(side: int) -> dict[str, float]:
    return {
        "order_flow_imbalance_1s": 0.8 * side,
        "order_flow_imbalance_5s": 0.7 * side,
        "taker_aggression_imbalance_1s": 0.8 * side,
        "taker_aggression_imbalance_5s": 0.7 * side,
        "taker_aggression_imbalance_15s": 0.6 * side,
        "microprice_mid_bps": 0.5 * side,
        "vamp_mid_bps": 0.4 * side,
        "depth_imbalance_top5": 0.5 * side,
        "book_imbalance_top": 0.4 * side,
    }


def _candidate(family: str, variant: str, *, floor: float, qualifies: bool, calibration_trades: int) -> dict:
    metrics = {
        "sequential_trades": calibration_trades,
        "average_net_bps_per_trade": floor,
        "long": {"trades": calibration_trades // 2},
        "short": {"trades": calibration_trades - calibration_trades // 2},
    }
    return {
        "family": family,
        "variant": variant,
        "profile": "profile",
        "qualifies": qualifies,
        "robust_floor_bps": floor,
        "minimum_side_fraction": 0.5,
        "discovery": metrics,
        "calibration": metrics,
    }
