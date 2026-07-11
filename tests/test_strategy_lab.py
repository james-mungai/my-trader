import numpy as np
import pytest

pytest.importorskip("sklearn")

from futures_lab.strategy_lab import (
    _apply_platt_calibrator,
    _fit_platt_calibrator,
    _stable_feature_indices,
)


def test_stable_feature_selection_rejects_sparse_and_constant_columns() -> None:
    fit = np.asarray(
        [
            [1.0, np.nan, 5.0],
            [2.0, np.nan, 5.0],
            [3.0, 1.0, 5.0],
            [4.0, np.nan, 5.0],
        ]
    )
    calibration = np.asarray([[5.0, np.nan, 5.0], [6.0, np.nan, 5.0]])
    holdout = np.asarray([[7.0, np.nan, 5.0], [8.0, 2.0, 5.0]])

    assert _stable_feature_indices(
        fit,
        calibration,
        holdout,
        minimum_coverage=0.75,
    ) == [0]


def test_platt_calibration_preserves_probability_ordering() -> None:
    raw = np.asarray([0.10, 0.25, 0.35, 0.65, 0.75, 0.90])
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    calibrator = _fit_platt_calibrator(raw, labels)
    calibrated = _apply_platt_calibrator(calibrator, raw)

    assert np.all(np.diff(calibrated) > 0)
    assert np.all((calibrated > 0) & (calibrated < 1))
