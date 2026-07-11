from __future__ import annotations

import math
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from futures_lab.first_touch import (
    MODEL_FEATURES,
    LabeledSample,
    _deterministic_predictions,
    _evaluate_predictions,
    _feature_vector,
    _iso,
    break_even_win_rate,
    label_samples,
    load_feature_samples,
    load_mark_price_timeline,
)


MODEL_CONFIDENCE_THRESHOLDS = (0.50, 0.55, 0.58, 0.60, 0.65, 0.70)


def run_symmetric_strategy_lab(
    runs_root: Path,
    *,
    symbol: str = "ETHUSDT",
    barrier_bps: float = 60.0,
    cost_bps: float = 10.0,
    horizon_seconds: int = 21_600,
    sample_seconds: int = 60,
    max_gap_seconds: int = 5,
    train_fraction: float = 0.70,
    calibration_fraction: float = 0.25,
    minimum_feature_coverage: float = 0.80,
    account_exposure: float = 4.95,
) -> dict[str, Any]:
    if barrier_bps <= cost_bps:
        raise ValueError("Barrier must exceed round-trip cost")
    if not 0.5 <= train_fraction < 0.9:
        raise ValueError("train_fraction must be between 0.5 and 0.9")
    if not 0.15 <= calibration_fraction <= 0.40:
        raise ValueError("calibration_fraction must be between 0.15 and 0.40")

    timeline, mark_stats = load_mark_price_timeline(
        runs_root,
        symbol=symbol,
        max_gap_seconds=max_gap_seconds,
    )
    features, feature_stats = load_feature_samples(
        runs_root,
        symbol=symbol,
        sample_seconds=sample_seconds,
    )
    labeled = label_samples(
        timeline,
        features,
        target_bps=barrier_bps,
        stop_bps_values=[barrier_bps],
        cost_bps=cost_bps,
        horizon_seconds=horizon_seconds,
    )
    if len(labeled) < 100:
        raise ValueError(f"Only {len(labeled)} labelable samples were found")

    purge_ms = horizon_seconds * 1000
    holdout_index = max(1, min(len(labeled) - 1, int(len(labeled) * train_fraction)))
    holdout_start_ms = labeled[holdout_index].timestamp_ms
    pre_holdout = [row for row in labeled if row.timestamp_ms < holdout_start_ms - purge_ms]
    holdout = [row for row in labeled if row.timestamp_ms >= holdout_start_ms]

    calibration_index = max(
        1,
        min(len(pre_holdout) - 1, int(len(pre_holdout) * (1.0 - calibration_fraction))),
    )
    calibration_start_ms = pre_holdout[calibration_index].timestamp_ms
    fit_rows = [row for row in pre_holdout if row.timestamp_ms < calibration_start_ms - purge_ms]
    calibration_rows = [row for row in pre_holdout if row.timestamp_ms >= calibration_start_ms]
    if min(len(fit_rows), len(calibration_rows), len(holdout)) < 100:
        raise ValueError(
            "Insufficient purged fit/calibration/holdout split: "
            f"fit={len(fit_rows)} calibration={len(calibration_rows)} holdout={len(holdout)}"
        )

    fit_matrix = _feature_matrix(fit_rows)
    calibration_matrix = _feature_matrix(calibration_rows)
    holdout_matrix = _feature_matrix(holdout)
    selected_indices = _stable_feature_indices(
        fit_matrix,
        calibration_matrix,
        holdout_matrix,
        minimum_coverage=minimum_feature_coverage,
    )
    selected_features = [MODEL_FEATURES[index] for index in selected_indices]
    if len(selected_features) < 3:
        raise ValueError(f"Only {len(selected_features)} stable features passed coverage checks")

    feature_coverage = _feature_coverage_rows(
        fit_matrix,
        calibration_matrix,
        holdout_matrix,
        selected_indices,
    )

    deterministic_periods: dict[str, dict[str, Any]] = {}
    for period_name, period_rows in (
        ("fit", fit_rows),
        ("calibration", calibration_rows),
        ("holdout", holdout),
    ):
        for name, predictions in _deterministic_predictions(period_rows).items():
            deterministic_periods.setdefault(name, {})[period_name] = _evaluate_predictions(
                period_rows,
                predictions,
                barrier_bps,
                target_bps=barrier_bps,
                cost_bps=cost_bps,
                account_exposure=account_exposure,
            )
    deterministic = {
        name: periods["holdout"] for name, periods in deterministic_periods.items()
    }

    model_reports = {}
    factories: dict[str, Callable[[], Any]] = {
        "regularized_logistic": _logistic_pipeline,
        "hist_gradient_boosting": _boosting_pipeline,
    }
    for name, factory in factories.items():
        model_reports[name] = _fit_calibrate_score_model(
            factory(),
            fit_rows,
            calibration_rows,
            holdout,
            fit_matrix[:, selected_indices],
            calibration_matrix[:, selected_indices],
            holdout_matrix[:, selected_indices],
            selected_features,
            barrier_bps=barrier_bps,
            cost_bps=cost_bps,
            account_exposure=account_exposure,
        )

    return {
        "study": "purged_symmetric_first_touch_strategy_lab",
        "symbol": symbol,
        "parameters": {
            "barrier_bps": barrier_bps,
            "round_trip_cost_bps": cost_bps,
            "horizon_seconds": horizon_seconds,
            "sample_seconds": sample_seconds,
            "max_gap_seconds": max_gap_seconds,
            "train_fraction": train_fraction,
            "calibration_fraction_of_pre_holdout": calibration_fraction,
            "purge_seconds": horizon_seconds,
            "minimum_feature_coverage": minimum_feature_coverage,
            "account_exposure_multiple": account_exposure,
        },
        "economics": {
            "net_win_bps": barrier_bps - cost_bps,
            "net_loss_bps": -(barrier_bps + cost_bps),
            "random_direction_win_rate": 0.50,
            "fee_adjusted_break_even_win_rate": break_even_win_rate(
                barrier_bps,
                barrier_bps,
                cost_bps,
            ),
        },
        "data": {
            **mark_stats,
            **feature_stats,
            "labelable_samples": len(labeled),
            "fit_samples_after_purge": len(fit_rows),
            "calibration_samples": len(calibration_rows),
            "holdout_samples": len(holdout),
            "holdout_resolved_samples": sum(row.symmetric_direction is not None for row in holdout),
            "fit_end": _iso(fit_rows[-1].timestamp_ms),
            "calibration_start": _iso(calibration_start_ms),
            "holdout_start": _iso(holdout_start_ms),
            "last_sample_at": _iso(labeled[-1].timestamp_ms),
        },
        "feature_selection": {
            "available_features": len(MODEL_FEATURES),
            "selected_features": selected_features,
            "selected_count": len(selected_features),
            "coverage": feature_coverage,
        },
        "deterministic_strategy_periods": deterministic_periods,
        "holdout_deterministic_strategies": deterministic,
        "holdout_models": model_reports,
        "interpretation_notes": [
            "Every strategy uses the same favorable and adverse barrier distance.",
            "Fit, probability calibration, and final holdout are chronological and separated by the full label horizon.",
            "Sparse features are excluded rather than used as fingerprints for historical run configuration.",
            "Overlapping holdout rows inform calibration diagnostics, but PnL metrics use sequential non-overlapping trades.",
            "The holdout becomes development data once inspected; future paper data remains necessary confirmation.",
        ],
    }


def _feature_matrix(samples: list[LabeledSample]) -> np.ndarray:
    return np.asarray([_feature_vector(sample.row) for sample in samples], dtype=np.float64)


def _stable_feature_indices(
    fit: np.ndarray,
    calibration: np.ndarray,
    holdout: np.ndarray,
    *,
    minimum_coverage: float,
) -> list[int]:
    output = []
    for index in range(fit.shape[1]):
        coverage = min(
            float(np.isfinite(fit[:, index]).mean()),
            float(np.isfinite(calibration[:, index]).mean()),
            float(np.isfinite(holdout[:, index]).mean()),
        )
        finite = fit[np.isfinite(fit[:, index]), index]
        if coverage >= minimum_coverage and len(finite) >= 2 and float(np.std(finite)) > 1e-12:
            output.append(index)
    return output


def _feature_coverage_rows(
    fit: np.ndarray,
    calibration: np.ndarray,
    holdout: np.ndarray,
    selected_indices: list[int],
) -> list[dict[str, Any]]:
    selected = set(selected_indices)
    return [
        {
            "feature": name,
            "fit_coverage": float(np.isfinite(fit[:, index]).mean()),
            "calibration_coverage": float(np.isfinite(calibration[:, index]).mean()),
            "holdout_coverage": float(np.isfinite(holdout[:, index]).mean()),
            "selected": index in selected,
        }
        for index, name in enumerate(MODEL_FEATURES)
    ]


def _logistic_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.25,
                    max_iter=2_000,
                    random_state=17,
                ),
            ),
        ]
    )


def _boosting_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=180,
                    max_leaf_nodes=15,
                    max_depth=3,
                    l2_regularization=1.0,
                    early_stopping=False,
                    random_state=17,
                ),
            ),
        ]
    )


def _fit_calibrate_score_model(
    model: Any,
    fit_rows: list[LabeledSample],
    calibration_rows: list[LabeledSample],
    holdout_rows: list[LabeledSample],
    fit_matrix: np.ndarray,
    calibration_matrix: np.ndarray,
    holdout_matrix: np.ndarray,
    feature_names: list[str],
    *,
    barrier_bps: float,
    cost_bps: float,
    account_exposure: float,
) -> dict[str, Any]:
    fit_mask, fit_labels = _resolved_mask_and_labels(fit_rows)
    calibration_mask, calibration_labels = _resolved_mask_and_labels(calibration_rows)
    holdout_mask, holdout_labels = _resolved_mask_and_labels(holdout_rows)
    model.fit(fit_matrix[fit_mask], fit_labels)

    raw_calibration = model.predict_proba(calibration_matrix[calibration_mask])[:, 1]
    calibrator = _fit_platt_calibrator(raw_calibration, calibration_labels)
    raw_holdout = model.predict_proba(holdout_matrix)[:, 1]
    probabilities = _apply_platt_calibrator(calibrator, raw_holdout)
    resolved_probabilities = probabilities[holdout_mask]

    thresholds = {}
    for threshold in MODEL_CONFIDENCE_THRESHOLDS:
        predictions = [
            1 if probability >= threshold else -1 if probability <= 1.0 - threshold else None
            for probability in probabilities
        ]
        thresholds[f"confidence_{threshold:.2f}"] = _evaluate_predictions(
            holdout_rows,
            predictions,
            barrier_bps,
            target_bps=barrier_bps,
            cost_bps=cost_bps,
            account_exposure=account_exposure,
        )

    direction_predictions = resolved_probabilities >= 0.5
    direction_accuracy = float(np.mean(direction_predictions == holdout_labels))
    raw_resolved = raw_holdout[holdout_mask]
    report = {
        "fit_resolved_rows": int(fit_mask.sum()),
        "calibration_resolved_rows": int(calibration_mask.sum()),
        "holdout_resolved_rows": int(holdout_mask.sum()),
        "direction_accuracy": direction_accuracy,
        "direction_accuracy_wilson_95": list(
            _wilson_interval_local(int(np.sum(direction_predictions == holdout_labels)), len(holdout_labels))
        ),
        "raw_brier_score": float(brier_score_loss(holdout_labels, raw_resolved)),
        "calibrated_brier_score": float(brier_score_loss(holdout_labels, resolved_probabilities)),
        "raw_log_loss": float(log_loss(holdout_labels, raw_resolved, labels=[0, 1])),
        "calibrated_log_loss": float(log_loss(holdout_labels, resolved_probabilities, labels=[0, 1])),
        "platt_slope": float(calibrator.coef_[0][0]),
        "platt_intercept": float(calibrator.intercept_[0]),
        "confidence_calibration": _confidence_calibration_rows(
            holdout_rows,
            probabilities.tolist(),
        ),
        "thresholds": thresholds,
    }
    coefficients = _standardized_logistic_coefficients(model, feature_names)
    if coefficients:
        report["top_standardized_coefficients"] = coefficients
    return report


def _resolved_mask_and_labels(samples: list[LabeledSample]) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray([sample.symmetric_direction is not None for sample in samples], dtype=bool)
    labels = np.asarray(
        [1 if sample.symmetric_direction == 1 else 0 for sample in samples if sample.symmetric_direction is not None],
        dtype=np.int64,
    )
    return mask, labels


def _fit_platt_calibrator(probabilities: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    logits = _probability_logits(probabilities).reshape(-1, 1)
    calibrator = LogisticRegression(C=1.0, max_iter=1_000, random_state=17)
    calibrator.fit(logits, labels)
    return calibrator


def _apply_platt_calibrator(calibrator: LogisticRegression, probabilities: np.ndarray) -> np.ndarray:
    logits = _probability_logits(probabilities).reshape(-1, 1)
    return calibrator.predict_proba(logits)[:, 1]


def _probability_logits(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _confidence_calibration_rows(
    samples: list[LabeledSample],
    probabilities: list[float],
) -> list[dict[str, Any]]:
    rows = []
    for lower in np.arange(0.50, 1.0, 0.05):
        upper = min(1.0, lower + 0.05)
        bucket = []
        for sample, probability in zip(samples, probabilities, strict=True):
            if sample.symmetric_direction is None:
                continue
            confidence = max(probability, 1.0 - probability)
            if lower <= confidence < upper or (upper >= 1.0 and confidence == 1.0):
                predicted_side = 1 if probability >= 0.5 else -1
                bucket.append((confidence, predicted_side == sample.symmetric_direction))
        if bucket:
            rows.append(
                {
                    "lower": round(float(lower), 2),
                    "upper": round(float(upper), 2),
                    "count": len(bucket),
                    "average_confidence": mean(confidence for confidence, _ in bucket),
                    "actual_accuracy": mean(correct for _, correct in bucket),
                }
            )
    return rows


def _standardized_logistic_coefficients(model: Any, feature_names: list[str]) -> list[dict[str, Any]]:
    estimator = model.named_steps.get("model") if hasattr(model, "named_steps") else None
    if not isinstance(estimator, LogisticRegression):
        return []
    pairs = sorted(
        zip(feature_names, estimator.coef_[0], strict=True),
        key=lambda item: abs(float(item[1])),
        reverse=True,
    )[:15]
    return [{"feature": name, "coefficient": float(value)} for name, value in pairs]


def _wilson_interval_local(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    z = 1.959963984540054
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt((rate * (1.0 - rate) + z * z / (4.0 * total)) / total) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)
