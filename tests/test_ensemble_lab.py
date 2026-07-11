import numpy as np

from futures_lab.ensemble_lab import (
    FROZEN_ORIENTATIONS,
    _ensemble_verdict,
    _expanding_splits,
    _fit_feature_transform,
    _fit_monotonic_calibrator,
    _probability_metrics,
    _threshold_candidate,
    _transform_records,
)
from futures_lab.exit_laboratory import ExitOpportunity, ExitTrade
from futures_lab.feature_value import FeatureRecord


def test_expanding_splits_never_train_on_validation_or_future_rows() -> None:
    splits = _expanding_splits(40, minimum_train=16, validation_size=8)

    assert [(train[0], train[-1], validate[0], validate[-1]) for train, validate in splits] == [
        (0, 15, 16, 23),
        (0, 23, 24, 31),
        (0, 31, 32, 39),
    ]
    assert all(train[-1] < validate[0] for train, validate in splits)


def test_feature_transform_applies_frozen_orientation_and_median_imputation() -> None:
    feature = "signed_aggression_15s"
    records = [
        _record(1, 1, 10.0, {feature: -3.0}),
        _record(2, -1, -10.0, {feature: -2.0}),
        _record(3, 1, 10.0, {feature: -1.0}),
    ]
    transform = _fit_feature_transform(records, (feature,))
    transformed = _transform_records(
        [*records, _record(4, -1, -10.0, {feature: None})],
        transform,
    )

    assert FROZEN_ORIENTATIONS[feature] == -1
    assert transform[0].median == 2.0
    assert transformed[0, 0] > 0
    assert transformed[2, 0] < 0
    assert transformed[3, 0] == 0


def test_monotonic_calibrator_cannot_reverse_score_ordering() -> None:
    scores = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0])
    labels = np.asarray([0, 0, 0, 1, 1])

    calibrator = _fit_monotonic_calibrator(scores, labels)
    probabilities = calibrator.predict(scores)

    assert calibrator.slope >= 0
    assert np.all(np.diff(probabilities) >= 0)
    assert np.all((probabilities > 0) & (probabilities < 1))


def test_probability_metrics_reward_calibrated_ordering() -> None:
    records = [
        _record(1, 1, -50.0),
        _record(2, -1, -50.0),
        _record(3, 1, 50.0),
        _record(4, -1, 50.0),
    ]
    metrics = _probability_metrics(
        records,
        np.asarray([0.10, 0.20, 0.80, 0.90]),
        reference_win_rate=0.50,
    )

    assert metrics["auc"] == 1.0
    assert metrics["brier_skill_vs_discovery_base"] > 0
    assert metrics["expected_calibration_error"] < 0.20
    assert sum(row["rows"] for row in metrics["calibration_bins"]) == 4


def test_threshold_candidate_requires_both_sides_and_independent_improvement() -> None:
    discovery = _separated_records(12, start=0)
    calibration = _separated_records(8, start=100)
    discovery_probabilities = np.asarray([0.90] * 6 + [0.40] * 6)
    calibration_probabilities = np.asarray([0.90] * 4 + [0.40] * 4)

    candidate = _threshold_candidate(
        discovery,
        discovery_probabilities,
        calibration,
        calibration_probabilities,
        threshold=0.60,
        account_exposure=1.0,
        bootstrap_samples=100,
        seed=7,
    )

    assert candidate["basic_qualified"] is True
    assert candidate["qualification_reasons"] == []
    assert candidate["accepted_preholdout_long_trades"] == 5
    assert candidate["accepted_preholdout_short_trades"] == 5
    assert candidate["robust_preholdout_score_bps"] > 0


def test_unqualified_selection_cannot_pass_historical_audit() -> None:
    verdict = _ensemble_verdict(
        False,
        {},
        {},
        {},
    )

    assert verdict == "selection_failed"


def _separated_records(count: int, *, start: int) -> list[FeatureRecord]:
    accepted = count // 2
    return [
        _record(
            start + index,
            1 if index % 2 == 0 else -1,
            50.0 if index < accepted else -70.0,
        )
        for index in range(count)
    ]


def _record(
    timestamp_ms: int,
    side: int,
    net_bps: float,
    values: dict[str, float | None] | None = None,
) -> FeatureRecord:
    opportunity = ExitOpportunity(timestamp_ms, side, {}, 0, 1)
    trade = ExitTrade(
        timestamp_ms=timestamp_ms,
        side=side,
        exit_ms=timestamp_ms + 1_000,
        exit_reason="test",
        gross_bps=net_bps + 10.0,
        net_bps=net_bps,
        mfe_bps=max(0.0, net_bps),
        mae_bps=min(0.0, net_bps),
        duration_seconds=1.0,
    )
    return FeatureRecord(opportunity, trade, values or {})
