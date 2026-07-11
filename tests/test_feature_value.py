from futures_lab.exit_laboratory import ExitOpportunity, ExitTrade
from futures_lab.feature_value import (
    FEATURE_DEFINITIONS,
    FeatureRecord,
    _auc,
    _benjamini_hochberg,
    _discovery_orientation,
    _select_group_features,
    _spearman,
    extract_feature_values,
)


def test_directional_features_are_side_neutral_after_mirroring() -> None:
    long = extract_feature_values(_opportunity(1, 1, directional_sign=1.0))
    short = extract_feature_values(_opportunity(2, -1, directional_sign=-1.0))

    for name in (
        "signed_ofi_1s",
        "signed_aggression_1s",
        "signed_microprice_bps",
        "signed_return_15s_bps",
        "breakout_edge_position",
        "htf_trend_alignment",
        "htf_return_alignment_bps",
        "htf_edge_position",
        "htf_taker_alignment",
        "signed_btc_ofi_1s",
        "signed_eth_btc_relative_15s_bps",
    ):
        assert long[name] == short[name]


def test_feed_lag_features_are_diagnostic_only() -> None:
    diagnostics = [definition for definition in FEATURE_DEFINITIONS if definition.group == "data_quality"]

    assert {definition.name for definition in diagnostics} == {
        "avg_book_lag_30s_ms",
        "avg_trade_lag_30s_ms",
    }
    assert all(not definition.alpha_eligible for definition in diagnostics)


def test_discovery_orientation_flips_when_lower_values_perform_better() -> None:
    records = [
        _record(1, value=1.0, net_bps=30.0),
        _record(2, value=2.0, net_bps=10.0),
        _record(3, value=3.0, net_bps=-10.0),
        _record(4, value=4.0, net_bps=-30.0),
    ]

    orientation, raw_effect = _discovery_orientation(records, "test_feature")

    assert orientation == -1
    assert raw_effect < 0


def test_rank_metrics_handle_order_and_ties() -> None:
    assert _spearman([1.0, 2.0, 3.0], [-2.0, 0.0, 4.0]) == 1.0
    assert _auc([0.0, 0.0, 1.0, 1.0], [False, False, True, True]) == 1.0


def test_benjamini_hochberg_preserves_original_order() -> None:
    adjusted = _benjamini_hochberg([0.01, 0.04, 0.03, 0.50])

    assert adjusted == [0.04, 0.05333333333333334, 0.05333333333333334, 0.5]


def test_group_selection_ignores_lucky_but_unstable_feature() -> None:
    stable = _feature_row("stable", stable=True, robust=4.0, calibration=5.0)
    lucky = _feature_row("lucky", stable=False, robust=20.0, calibration=30.0)

    selected = _select_group_features([stable, lucky])["flow"]

    assert selected["feature"] == "stable"
    assert selected["selection_qualified"] is True
    assert selected["stable_group_feature_count"] == 1


def test_group_selection_requires_preholdout_fdr_support() -> None:
    suggestive = _feature_row("suggestive", stable=True, robust=8.0, calibration=12.0, fdr=False)
    supported = _feature_row("supported", stable=True, robust=4.0, calibration=7.0, fdr=True)

    selected = _select_group_features([suggestive, supported])["flow"]

    assert selected["feature"] == "supported"
    assert selected["selection_qualified"] is True
    assert selected["fdr_supported_group_feature_count"] == 1


def _opportunity(timestamp_ms: int, side: int, *, directional_sign: float) -> ExitOpportunity:
    bullish_position = 0.80
    row = {
        "order_flow_imbalance_1s": 0.40 * directional_sign,
        "taker_aggression_imbalance_1s": 0.30 * directional_sign,
        "microprice_mid_bps": 0.20 * directional_sign,
        "return_15s_pct": 0.0005 * directional_sign,
        "range_position_180s": bullish_position if side == 1 else 1.0 - bullish_position,
        "btc_order_flow_imbalance_1s": 0.25 * directional_sign,
        "eth_btc_relative_return_15s_pct": 0.0002 * directional_sign,
        "higher_timeframe_context": {
            "timeframes": {
                interval: {
                    "trend_score": 0.50 * directional_sign,
                    "return_pct": 0.0010 * directional_sign,
                    "range_position": bullish_position if side == 1 else 1.0 - bullish_position,
                    "taker_buy_ratio": 0.65 if side == 1 else 0.35,
                }
                for interval in ("15m", "30m", "1h")
            }
        },
    }
    return ExitOpportunity(timestamp_ms, side, row, 0, 1)


def _record(timestamp_ms: int, *, value: float, net_bps: float) -> FeatureRecord:
    opportunity = ExitOpportunity(timestamp_ms, 1, {}, 0, 1)
    trade = ExitTrade(
        timestamp_ms=timestamp_ms,
        side=1,
        exit_ms=timestamp_ms + 1_000,
        exit_reason="test",
        gross_bps=net_bps + 10.0,
        net_bps=net_bps,
        mfe_bps=max(0.0, net_bps),
        mae_bps=min(0.0, net_bps),
        duration_seconds=1.0,
    )
    return FeatureRecord(opportunity, trade, {"test_feature": value})


def _feature_row(
    name: str,
    *,
    stable: bool,
    robust: float,
    calibration: float,
    fdr: bool | None = None,
) -> dict:
    return {
        "feature": name,
        "group": "flow",
        "alpha_eligible": True,
        "stable_preholdout": stable,
        "preholdout_fdr_supported": stable if fdr is None else fdr,
        "robust_preholdout_score_bps": robust,
        "calibration": {
            "top_minus_bottom_bps": calibration,
            "spearman_rho": calibration / 100.0,
        },
    }
