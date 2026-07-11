from futures_lab.exit_laboratory import ExitOpportunity, ExitTrade
from futures_lab.regime_atlas import (
    REGIME_DEFINITIONS,
    RegimeRecord,
    _bootstrap_group_delta,
    _select_gate_candidates,
    candidate_category_sets,
    classify_opportunity,
    derive_regime_thresholds,
)


def test_regime_thresholds_are_derived_from_discovery_inputs() -> None:
    features = [
        (1_000, {"realized_vol_180s_pct": 0.0001, "spread_bps": 0.05}),
        (2_000, {"realized_vol_180s_pct": 0.0002, "spread_bps": 0.06}),
        (3_000, {"realized_vol_180s_pct": 0.0003, "spread_bps": 0.07}),
    ]
    cohort = [
        _opportunity(index * 1_000, 1, micro=0.1 * index, drift_bps=float(index))
        for index in (1, 2, 3)
    ]

    thresholds = derive_regime_thresholds(features, cohort)

    assert 1.0 < thresholds["volatility_bps"]["low_max"] < 2.0
    assert 2.0 < thresholds["volatility_bps"]["high_min"] < 3.0
    assert 0.05 < thresholds["spread_bps"]["low_max"] < 0.06
    assert thresholds["htf_alignment_threshold"] == 0.20


def test_regime_classification_is_mirrored_for_long_and_short() -> None:
    thresholds = _thresholds()
    long = _opportunity(10 * 3_600_000, 1, micro=0.5, drift_bps=5.0, htf=0.5)
    short = _opportunity(10 * 3_600_000, -1, micro=-0.5, drift_bps=-5.0, htf=-0.5)

    long_labels = classify_opportunity(long, thresholds)
    short_labels = classify_opportunity(short, thresholds)

    assert {key: value for key, value in long_labels.items() if key != "entry_side"} == {
        key: value for key, value in short_labels.items() if key != "entry_side"
    }
    assert long_labels["htf_alignment"] == "aligned"
    assert long_labels["entry_side"] == "long"
    assert short_labels["entry_side"] == "short"


def test_entry_side_is_audit_only() -> None:
    entry_side = next(definition for definition in REGIME_DEFINITIONS if definition.name == "entry_side")

    assert entry_side.gate_eligible is False
    assert candidate_category_sets(entry_side) == [
        frozenset({"long"}),
        frozenset({"short"}),
    ]


def test_utc_session_candidates_include_wraparound_pair() -> None:
    session = next(definition for definition in REGIME_DEFINITIONS if definition.name == "utc_session")

    candidates = candidate_category_sets(session)

    assert frozenset({"asia_00_08", "americas_16_24"}) in candidates


def test_gate_selection_uses_pre_holdout_robust_score() -> None:
    weak = _candidate("volatility", ["low"], score=1.0, qualifies=True)
    strong = _candidate("volatility", ["normal"], score=3.0, qualifies=True)
    lucky = _candidate("volatility", ["high"], score=20.0, qualifies=False)

    selected = _select_gate_candidates([weak, strong, lucky])["volatility"]

    assert selected["accepted_categories"] == ["normal"]
    assert selected["selection_qualified"] is True
    assert selected["qualified_candidate_count"] == 2


def test_group_bootstrap_compares_accepted_and_rejected_returns() -> None:
    records = []
    for index, value in enumerate((20.0, 10.0, 20.0, 10.0, -10.0, -20.0, -10.0, -20.0)):
        category = "allow" if index < 4 else "block"
        records.append(_record(index, value, category))

    result = _bootstrap_group_delta(
        records,
        dimension="test",
        accepted_categories=frozenset({"allow"}),
        samples=300,
        seed=11,
        block_size=1,
    )

    assert result["mean_delta_bps"] == 30.0
    assert result["bootstrap_p025_bps"] > 0
    assert result["bootstrap_probability_positive"] == 1.0


def _opportunity(
    timestamp_ms: int,
    side: int,
    *,
    micro: float,
    drift_bps: float,
    htf: float = 0.0,
) -> ExitOpportunity:
    row = {
        "realized_vol_180s_pct": 0.0002,
        "spread_bps": 0.06,
        "return_180s_pct": drift_bps / 10_000,
        "order_flow_imbalance_1s": micro,
        "order_flow_imbalance_5s": micro,
        "taker_aggression_imbalance_1s": micro,
        "taker_aggression_imbalance_5s": micro,
        "microprice_mid_bps": micro,
        "vamp_mid_bps": micro,
        "depth_imbalance_top5": micro,
        "book_imbalance_top": micro,
        "higher_timeframe_context": {
            "timeframes": {
                "15m": {"trend_score": htf},
                "30m": {"trend_score": htf},
                "1h": {"trend_score": htf},
            }
        },
    }
    return ExitOpportunity(timestamp_ms, side, row, 0, 1)


def _thresholds() -> dict:
    return {
        "volatility_bps": {"low_max": 1.0, "high_min": 3.0},
        "spread_bps": {"low_max": 0.05, "high_min": 0.07},
        "flow_strength": {"low_max": 0.2, "high_min": 0.4},
        "breakout_extension_bps": {"low_max": 2.0, "high_min": 8.0},
        "htf_alignment_threshold": 0.20,
        "utc_session_boundaries": [0, 8, 16, 24],
    }


def _candidate(dimension: str, categories: list[str], *, score: float, qualifies: bool) -> dict:
    period = {"accepted": {"average_net_bps_per_trade": score, "trades": 10}}
    return {
        "dimension": dimension,
        "accepted_categories": categories,
        "qualifies": qualifies,
        "robust_gate_score_bps": score,
        "discovery": period,
        "calibration": period,
    }


def _record(index: int, net_bps: float, category: str) -> RegimeRecord:
    opportunity = _opportunity(index * 1_000, 1 if index % 2 == 0 else -1, micro=0.5, drift_bps=5.0)
    trade = ExitTrade(
        timestamp_ms=index * 1_000,
        side=opportunity.side,
        exit_ms=index * 1_000 + 1_000,
        exit_reason="test",
        gross_bps=net_bps + 10.0,
        net_bps=net_bps,
        mfe_bps=max(0.0, net_bps),
        mae_bps=min(0.0, net_bps),
        duration_seconds=1.0,
    )
    return RegimeRecord(opportunity, trade, {"test": category})
