from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from futures_lab.exit_laboratory import (
    REFERENCE_POLICY,
    ExitOpportunity,
    evaluate_policy,
    lock_reference_cohort,
    summarize_exit_trades,
)
from futures_lab.feature_value import (
    FeatureRecord,
    _auc,
    _compact_trade_metrics,
    _coin_side,
    _iso,
    extract_feature_values,
)
from futures_lab.first_touch import PriceTimeline, load_feature_samples, load_mark_price_timeline
from futures_lab.strategy_tournament import _period_boundaries


FROZEN_SHORTLIST = (
    "signed_aggression_15s",
    "signed_taker_ratio_10s",
    "htf_return_alignment_bps",
    "favorable_funding_carry_bps",
)

FROZEN_ORIENTATIONS = {
    "signed_aggression_15s": -1,
    "signed_taker_ratio_10s": -1,
    "htf_return_alignment_bps": -1,
    "favorable_funding_carry_bps": 1,
}

RIDGE_C_GRID = (0.01, 0.03, 0.10, 0.30, 1.0)
PROBABILITY_THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70)


@dataclass(frozen=True)
class EnsembleSpec:
    name: str
    family: str
    description: str
    kind: str
    features: tuple[str, ...]
    selection_eligible: bool = True


@dataclass(frozen=True)
class FeatureTransform:
    feature: str
    orientation: int
    median: float
    scale: float
    coverage: float


@dataclass(frozen=True)
class MonotonicCalibrator:
    intercept: float
    slope: float

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return _sigmoid(self.intercept + self.slope * np.asarray(scores, dtype=np.float64))


ENSEMBLE_SPECS = (
    EnsembleSpec(
        name="single_aggression_benchmark",
        family="single_feature_benchmark",
        description="Proposal 5's strongest single input, retained as a non-promotable benchmark.",
        kind="single",
        features=("signed_aggression_15s",),
        selection_eligible=False,
    ),
    EnsembleSpec(
        name="equal_weight_core",
        family="transparent_ensemble",
        description="Equal average of the four discovery-oriented robust z-scores.",
        kind="equal",
        features=FROZEN_SHORTLIST,
    ),
    EnsembleSpec(
        name="consensus_vote_core",
        family="transparent_ensemble",
        description="Fraction of the four oriented inputs above their discovery medians.",
        kind="vote",
        features=FROZEN_SHORTLIST,
    ),
    EnsembleSpec(
        name="ridge_logistic_core",
        family="regularized_ensemble",
        description="L2-logistic blend with regularization chosen by expanding discovery validation.",
        kind="ridge",
        features=FROZEN_SHORTLIST,
    ),
)


def run_ensemble_calibration_lab(
    runs_root: Path,
    *,
    symbol: str = "ETHUSDT",
    cost_bps: float = 10.0,
    sample_seconds: int = 60,
    max_gap_seconds: int = 5,
    discovery_fraction: float = 0.50,
    calibration_end_fraction: float = 0.70,
    horizon_seconds: int = 21_600,
    account_exposure: float = 4.95,
    matched_coin_seeds: int = 32,
    bootstrap_samples: int = 2_000,
) -> dict[str, Any]:
    if cost_bps < 0 or sample_seconds <= 0 or horizon_seconds <= 0:
        raise ValueError("Ensemble cost, sampling, and horizon settings must be valid")
    if matched_coin_seeds < 8 or bootstrap_samples < 100:
        raise ValueError("Ensemble controls require at least 8 coin seeds and 100 bootstrap samples")

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
    if len(features) < 500:
        raise ValueError(f"Only {len(features)} feature samples were found")

    calibration_start_ms, holdout_start_ms = _period_boundaries(
        features,
        discovery_fraction=discovery_fraction,
        calibration_end_fraction=calibration_end_fraction,
    )
    horizon_ms = horizon_seconds * 1_000
    period_features = {
        "discovery": [row for row in features if row[0] < calibration_start_ms - horizon_ms],
        "calibration": [
            row for row in features if calibration_start_ms <= row[0] < holdout_start_ms - horizon_ms
        ],
        "holdout": [row for row in features if row[0] >= holdout_start_ms],
    }

    period_records: dict[str, list[FeatureRecord]] = {}
    cohort_stats: dict[str, dict[str, Any]] = {}
    baseline: dict[str, dict[str, Any]] = {}
    for period, rows in period_features.items():
        cohort, stats = lock_reference_cohort(
            timeline,
            rows,
            cost_bps=cost_bps,
            horizon_seconds=horizon_seconds,
        )
        trades = evaluate_policy(timeline, cohort, REFERENCE_POLICY, cost_bps=cost_bps)
        records = [
            FeatureRecord(opportunity, trade, extract_feature_values(opportunity))
            for opportunity, trade in zip(cohort, trades, strict=True)
        ]
        period_records[period] = records
        cohort_stats[period] = stats
        baseline[period] = _compact_trade_metrics(
            summarize_exit_trades(trades, account_exposure=account_exposure)
        )

    discovery_records = period_records["discovery"]
    if len(discovery_records) < 32 or len(period_records["calibration"]) < 12:
        raise ValueError(
            "The locked ensemble cohort is too small: "
            f"discovery={len(discovery_records)} calibration={len(period_records['calibration'])}"
        )
    frozen_transform = _fit_feature_transform(discovery_records, FROZEN_SHORTLIST)
    reference_win_rate = mean(_record_label(record) for record in discovery_records)

    candidates = [
        _fit_candidate(
            timeline,
            period_records,
            spec,
            cost_bps=cost_bps,
            reference_win_rate=reference_win_rate,
            account_exposure=account_exposure,
            matched_coin_seeds=matched_coin_seeds,
            bootstrap_samples=bootstrap_samples,
            namespace=f"{symbol}:{spec.name}",
            coin_namespace=f"{symbol}:ensemble_quality_gate",
        )
        for spec in ENSEMBLE_SPECS
    ]
    eligible = [candidate for candidate in candidates if candidate["selection_eligible"]]
    qualified = [candidate for candidate in eligible if candidate["selection_qualified"]]
    selected = max(
        qualified or eligible,
        key=lambda row: (
            row["selection_qualified"],
            row["robust_preholdout_score_bps"],
            row["calibration_quality"]["brier_skill_vs_discovery_base"],
        ),
    )
    forward_candidates = [
        {
            "name": candidate["name"],
            "threshold": candidate["selected_threshold"],
            "verdict": candidate["verdict"],
        }
        for candidate in eligible
        if candidate["verdict"] == "historical_audit_pass_forward_required"
    ]

    return {
        "study": "side_neutral_squeeze_ensemble_calibration_lab",
        "symbol": symbol,
        "parameters": {
            "round_trip_cost_bps": cost_bps,
            "sample_seconds": sample_seconds,
            "max_gap_seconds": max_gap_seconds,
            "discovery_fraction": discovery_fraction,
            "calibration_end_fraction": calibration_end_fraction,
            "horizon_seconds": horizon_seconds,
            "account_exposure_multiple": account_exposure,
            "matched_coin_seeds": matched_coin_seeds,
            "bootstrap_samples": bootstrap_samples,
            "probability_thresholds": list(PROBABILITY_THRESHOLDS),
            "entry_rule": "existing side-symmetric volatility_squeeze_breakout",
            "exit_policy": REFERENCE_POLICY.name,
            "target": "predict whether the already-chosen squeeze trade has positive after-cost expectancy",
        },
        "data": {
            **mark_stats,
            **feature_stats,
            "calibration_start": _iso(calibration_start_ms),
            "holdout_start": _iso(holdout_start_ms),
            "holdout_status": "audit_only_because_prior_proposals_already_opened_it",
        },
        "entry_cohorts": cohort_stats,
        "baseline": baseline,
        "frozen_shortlist": [
            {
                "feature": transform.feature,
                "orientation": transform.orientation,
                "orientation_label": (
                    "higher_raw_is_better" if transform.orientation == 1 else "lower_raw_is_better"
                ),
                "discovery_median_oriented": transform.median,
                "discovery_robust_scale": transform.scale,
                "discovery_coverage": transform.coverage,
                "proposal_5_fdr_supported": False,
            }
            for transform in frozen_transform
        ],
        "validation_contract": {
            "discovery_use": "fit transforms, candidate weights, and expanding-validation probabilities",
            "calibration_use": "select one frozen acceptance threshold and demand independent reliability",
            "historical_holdout_use": "audit only; cannot tune weights, thresholds, or feature membership",
            "fresh_forward_use": "mandatory before any strategy-code or live-paper promotion",
            "expanding_discovery_splits": [
                {
                    "train": [int(train[0]), int(train[-1])],
                    "validate": [int(validate[0]), int(validate[-1])],
                }
                for train, validate in _expanding_splits(len(discovery_records))
            ],
        },
        "candidates": candidates,
        "selected_preholdout_candidate": {
            "name": selected["name"],
            "selection_qualified": selected["selection_qualified"],
            "selected_threshold": selected["selected_threshold"],
            "robust_preholdout_score_bps": selected["robust_preholdout_score_bps"],
            "historical_holdout_verdict": selected["verdict"],
        },
        "forward_shadow_candidates": forward_candidates,
        "promotion_candidates": [],
        "interpretation_notes": [
            "All directional inputs were mirrored by the squeeze side in Proposal 5; this model gates trade quality and never chooses a long-only or short-only policy.",
            "The four feature names and orientations were frozen before Proposal 6. None passed Proposal 5 FDR, so every ensemble remains exploratory.",
            "Discovery uses expanding chronological validation. Calibration is independent of model fitting and is the only period allowed to choose the acceptance threshold.",
            "Probability quality must beat the discovery base-rate forecast on calibration; a high-ranked trade subset alone is not enough.",
            "The historical holdout has already informed earlier research and is audit-only. Even a strong audit cannot promote an ensemble without fresh forward shadow confirmation.",
            "The single-feature candidate is a benchmark and cannot qualify. Feed-lag fields and the incomplete BTC hypothesis are excluded from the ensemble.",
        ],
    }


def _fit_candidate(
    timeline: PriceTimeline,
    period_records: dict[str, list[FeatureRecord]],
    spec: EnsembleSpec,
    *,
    cost_bps: float,
    reference_win_rate: float,
    account_exposure: float,
    matched_coin_seeds: int,
    bootstrap_samples: int,
    namespace: str,
    coin_namespace: str,
) -> dict[str, Any]:
    discovery = period_records["discovery"]
    calibration = period_records["calibration"]
    holdout = period_records["holdout"]
    discovery_oof_indices, discovery_probabilities, model_details = _fit_oof_candidate(discovery, spec)
    discovery_oof = [discovery[index] for index in discovery_oof_indices]

    transform = _fit_feature_transform(discovery, spec.features)
    discovery_matrix = _transform_records(discovery, transform)
    calibration_matrix = _transform_records(calibration, transform)
    holdout_matrix = _transform_records(holdout, transform)
    discovery_labels = np.asarray([_record_label(record) for record in discovery], dtype=np.int64)
    if spec.kind == "ridge":
        selected_c = float(model_details["selected_c"])
        model = _fit_ridge(discovery_matrix, discovery_labels, selected_c)
        calibration_probabilities = _ridge_predict(model, calibration_matrix)
        holdout_probabilities = _ridge_predict(model, holdout_matrix)
        coefficients = [
            {"feature": feature, "coefficient": float(value)}
            for feature, value in zip(spec.features, model.coef_[0], strict=True)
        ]
        final_details = {
            **model_details,
            "intercept": float(model.intercept_[0]),
            "standardized_coefficients": coefficients,
        }
    else:
        raw_discovery = _raw_candidate_score(discovery_matrix, spec.kind)
        calibrator = _fit_monotonic_calibrator(raw_discovery, discovery_labels)
        calibration_probabilities = calibrator.predict(
            _raw_candidate_score(calibration_matrix, spec.kind)
        )
        holdout_probabilities = calibrator.predict(_raw_candidate_score(holdout_matrix, spec.kind))
        final_details = {
            **model_details,
            "final_score_calibrator": {
                "intercept": calibrator.intercept,
                "nonnegative_slope": calibrator.slope,
            },
        }

    discovery_quality = _probability_metrics(
        discovery_oof,
        discovery_probabilities,
        reference_win_rate=reference_win_rate,
    )
    calibration_quality = _probability_metrics(
        calibration,
        calibration_probabilities,
        reference_win_rate=reference_win_rate,
    )
    holdout_quality = _probability_metrics(
        holdout,
        holdout_probabilities,
        reference_win_rate=reference_win_rate,
    )

    threshold_candidates = []
    for threshold in PROBABILITY_THRESHOLDS:
        threshold_candidates.append(
            _threshold_candidate(
                discovery_oof,
                discovery_probabilities,
                calibration,
                calibration_probabilities,
                threshold=threshold,
                account_exposure=account_exposure,
                bootstrap_samples=bootstrap_samples,
                seed=_stable_seed(f"{namespace}:{threshold:.2f}"),
            )
        )
    basic_qualified = [row for row in threshold_candidates if row["basic_qualified"]]
    selected_threshold = max(
        basic_qualified or threshold_candidates,
        key=lambda row: (
            row["basic_qualified"],
            row["robust_preholdout_score_bps"],
            row["calibration"]["accepted"]["trades"],
        ),
    )
    selection_reasons = list(selected_threshold["qualification_reasons"])
    if not spec.selection_eligible:
        selection_reasons.append("benchmark_not_selection_eligible")
    if calibration_quality["brier_skill_vs_discovery_base"] <= 0:
        selection_reasons.append("calibration_brier_skill_not_positive")
    if calibration_quality["auc"] <= 0.50:
        selection_reasons.append("calibration_auc_not_above_random")
    if selected_threshold["discovery"]["bootstrap_accepted_minus_rejected"]["probability_positive"] < 0.80:
        selection_reasons.append("discovery_bootstrap_support_below_80pct")
    if selected_threshold["calibration"]["bootstrap_accepted_minus_rejected"]["probability_positive"] < 0.80:
        selection_reasons.append("calibration_bootstrap_support_below_80pct")
    selection_qualified = not selection_reasons

    threshold = float(selected_threshold["threshold"])
    holdout_gate = _gate_metrics(
        holdout,
        holdout_probabilities,
        threshold=threshold,
        account_exposure=account_exposure,
        bootstrap_samples=bootstrap_samples,
        seed=_stable_seed(f"{namespace}:holdout:{threshold:.2f}"),
    )
    accepted_holdout = [
        record
        for record, probability in zip(holdout, holdout_probabilities, strict=True)
        if probability >= threshold
    ]
    matched_coin = _matched_coin_control(
        timeline,
        accepted_holdout,
        cost_bps=cost_bps,
        account_exposure=account_exposure,
        seeds=matched_coin_seeds,
        namespace=coin_namespace,
    )
    verdict = _ensemble_verdict(
        selection_qualified,
        holdout_gate,
        holdout_quality,
        matched_coin,
    )
    return {
        "name": spec.name,
        "family": spec.family,
        "description": spec.description,
        "kind": spec.kind,
        "features": list(spec.features),
        "selection_eligible": spec.selection_eligible,
        "model": final_details,
        "discovery_oof_rows": len(discovery_oof),
        "discovery_quality": discovery_quality,
        "calibration_quality": calibration_quality,
        "holdout_quality": holdout_quality,
        "threshold_candidates": threshold_candidates,
        "selected_threshold": threshold,
        "robust_preholdout_score_bps": selected_threshold["robust_preholdout_score_bps"],
        "selection_qualified": selection_qualified,
        "selection_blockers": selection_reasons,
        "discovery_selected_gate": selected_threshold["discovery"],
        "calibration_selected_gate": selected_threshold["calibration"],
        "holdout_selected_gate": holdout_gate,
        "matched_coin_control": matched_coin,
        "verdict": verdict,
        "holdout_accepted_records": [
            {
                "entry": _iso(record.opportunity.timestamp_ms),
                "side": "long" if record.opportunity.side == 1 else "short",
                "predicted_probability": float(probability),
                "net_bps": record.trade.net_bps,
                "exit_reason": record.trade.exit_reason,
            }
            for record, probability in zip(holdout, holdout_probabilities, strict=True)
            if probability >= threshold
        ],
    }


def _fit_oof_candidate(
    records: list[FeatureRecord],
    spec: EnsembleSpec,
) -> tuple[list[int], np.ndarray, dict[str, Any]]:
    if spec.kind == "ridge":
        losses = []
        cache: dict[float, tuple[list[int], np.ndarray]] = {}
        for c_value in RIDGE_C_GRID:
            indices, probabilities = _ridge_oof(records, spec.features, c_value)
            labels = np.asarray([_record_label(records[index]) for index in indices], dtype=np.int64)
            loss = _binary_log_loss(labels, probabilities)
            losses.append({"c": c_value, "oof_log_loss": loss})
            cache[c_value] = (indices, probabilities)
        selected = min(losses, key=lambda row: (row["oof_log_loss"], row["c"]))
        indices, probabilities = cache[float(selected["c"])]
        return indices, probabilities, {
            "regularization_grid": losses,
            "selected_c": float(selected["c"]),
            "selection_rule": "minimum expanding-discovery log loss; ties favor stronger shrinkage",
        }

    indices = []
    probabilities = []
    fold_calibrators = []
    for train_indices, validate_indices in _expanding_splits(len(records)):
        train = [records[int(index)] for index in train_indices]
        validate = [records[int(index)] for index in validate_indices]
        transform = _fit_feature_transform(train, spec.features)
        train_matrix = _transform_records(train, transform)
        validate_matrix = _transform_records(validate, transform)
        train_scores = _raw_candidate_score(train_matrix, spec.kind)
        calibrator = _fit_monotonic_calibrator(
            train_scores,
            np.asarray([_record_label(record) for record in train], dtype=np.int64),
        )
        probabilities.extend(
            calibrator.predict(_raw_candidate_score(validate_matrix, spec.kind)).tolist()
        )
        indices.extend(int(index) for index in validate_indices)
        fold_calibrators.append(
            {
                "train_end_index": int(train_indices[-1]),
                "validation_start_index": int(validate_indices[0]),
                "intercept": calibrator.intercept,
                "nonnegative_slope": calibrator.slope,
            }
        )
    return indices, np.asarray(probabilities, dtype=np.float64), {
        "expanding_fold_score_calibrators": fold_calibrators,
        "orientation_constraint": "nonnegative score slope",
    }


def _ridge_oof(
    records: list[FeatureRecord],
    features: tuple[str, ...],
    c_value: float,
) -> tuple[list[int], np.ndarray]:
    indices = []
    probabilities = []
    for train_indices, validate_indices in _expanding_splits(len(records)):
        train = [records[int(index)] for index in train_indices]
        validate = [records[int(index)] for index in validate_indices]
        transform = _fit_feature_transform(train, features)
        train_matrix = _transform_records(train, transform)
        validate_matrix = _transform_records(validate, transform)
        labels = np.asarray([_record_label(record) for record in train], dtype=np.int64)
        if len(np.unique(labels)) < 2:
            fold_probabilities = np.full(len(validate), float(np.mean(labels)), dtype=np.float64)
        else:
            model = _fit_ridge(train_matrix, labels, c_value)
            fold_probabilities = _ridge_predict(model, validate_matrix)
        probabilities.extend(fold_probabilities.tolist())
        indices.extend(int(index) for index in validate_indices)
    return indices, np.asarray(probabilities, dtype=np.float64)


def _fit_ridge(matrix: np.ndarray, labels: np.ndarray, c_value: float) -> LogisticRegression:
    model = LogisticRegression(
        C=c_value,
        solver="lbfgs",
        max_iter=2_000,
        random_state=29,
    )
    model.fit(matrix, labels)
    return model


def _ridge_predict(model: LogisticRegression, matrix: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict_proba(matrix)[:, 1], dtype=np.float64)


def _fit_feature_transform(
    records: list[FeatureRecord],
    features: tuple[str, ...],
) -> tuple[FeatureTransform, ...]:
    output = []
    for feature in features:
        orientation = FROZEN_ORIENTATIONS[feature]
        values = [
            float(value) * orientation
            for record in records
            if (value := record.values.get(feature)) is not None and np.isfinite(value)
        ]
        if not values:
            output.append(FeatureTransform(feature, orientation, 0.0, 1.0, 0.0))
            continue
        array = np.asarray(values, dtype=np.float64)
        center = float(np.median(array))
        q25, q75 = np.quantile(array, [0.25, 0.75])
        scale = float((q75 - q25) / 1.349)
        if scale <= 1e-12:
            scale = float(np.std(array))
        if scale <= 1e-12:
            scale = 1.0
        output.append(
            FeatureTransform(
                feature=feature,
                orientation=orientation,
                median=center,
                scale=scale,
                coverage=len(values) / len(records) if records else 0.0,
            )
        )
    return tuple(output)


def _transform_records(
    records: list[FeatureRecord],
    transforms: tuple[FeatureTransform, ...],
) -> np.ndarray:
    matrix = np.zeros((len(records), len(transforms)), dtype=np.float64)
    for row_index, record in enumerate(records):
        for column_index, transform in enumerate(transforms):
            raw = record.values.get(transform.feature)
            if raw is None or not np.isfinite(raw):
                oriented = transform.median
            else:
                oriented = float(raw) * transform.orientation
            matrix[row_index, column_index] = np.clip(
                (oriented - transform.median) / transform.scale,
                -3.0,
                3.0,
            )
    return matrix


def _raw_candidate_score(matrix: np.ndarray, kind: str) -> np.ndarray:
    if kind == "single":
        return matrix[:, 0]
    if kind == "equal":
        return np.mean(matrix, axis=1)
    if kind == "vote":
        return np.mean(matrix > 0.0, axis=1) - 0.5
    raise ValueError(f"Unsupported score candidate kind: {kind}")


def _fit_monotonic_calibrator(scores: np.ndarray, labels: np.ndarray) -> MonotonicCalibrator:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if len(scores) != len(labels) or not len(scores):
        raise ValueError("Score calibration requires aligned non-empty scores and labels")
    best: tuple[float, MonotonicCalibrator] | None = None
    for slope in (0.0, 0.10, 0.25, 0.50, 1.0, 2.0, 4.0):
        intercept = _balanced_intercept(scores, labels, slope)
        calibrator = MonotonicCalibrator(intercept, slope)
        loss = _binary_log_loss(labels.astype(np.int64), calibrator.predict(scores))
        candidate = (loss, calibrator)
        if best is None or candidate[0] < best[0] - 1e-12:
            best = candidate
    assert best is not None
    return best[1]


def _balanced_intercept(scores: np.ndarray, labels: np.ndarray, slope: float) -> float:
    target = float(np.clip(np.mean(labels), 1e-6, 1.0 - 1e-6))
    low = -20.0
    high = 20.0
    for _ in range(100):
        middle = (low + high) / 2.0
        predicted = float(np.mean(_sigmoid(middle + slope * scores)))
        if predicted < target:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def _threshold_candidate(
    discovery_records: list[FeatureRecord],
    discovery_probabilities: np.ndarray,
    calibration_records: list[FeatureRecord],
    calibration_probabilities: np.ndarray,
    *,
    threshold: float,
    account_exposure: float,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    discovery = _gate_metrics(
        discovery_records,
        discovery_probabilities,
        threshold=threshold,
        account_exposure=account_exposure,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    calibration = _gate_metrics(
        calibration_records,
        calibration_probabilities,
        threshold=threshold,
        account_exposure=account_exposure,
        bootstrap_samples=bootstrap_samples,
        seed=seed ^ 0x5A5A5A5A,
    )
    accepted_preholdout = [
        record
        for records, probabilities in (
            (discovery_records, discovery_probabilities),
            (calibration_records, calibration_probabilities),
        )
        for record, probability in zip(records, probabilities, strict=True)
        if probability >= threshold
    ]
    long_count = sum(record.opportunity.side == 1 for record in accepted_preholdout)
    short_count = sum(record.opportunity.side == -1 for record in accepted_preholdout)
    reasons = []
    if discovery["accepted"]["trades"] < 6:
        reasons.append("discovery_accepted_trades_below_6")
    if calibration["accepted"]["trades"] < 4:
        reasons.append("calibration_accepted_trades_below_4")
    if discovery["accepted_delta_vs_all_bps"] <= 0:
        reasons.append("discovery_expectancy_not_improved")
    if calibration["accepted_delta_vs_all_bps"] <= 0:
        reasons.append("calibration_expectancy_not_improved")
    if discovery["accepted"]["average_net_bps_per_trade"] <= 0:
        reasons.append("discovery_accepted_expectancy_not_positive")
    if calibration["accepted"]["average_net_bps_per_trade"] <= 0:
        reasons.append("calibration_accepted_expectancy_not_positive")
    if long_count < 2:
        reasons.append("accepted_long_trades_below_2")
    if short_count < 2:
        reasons.append("accepted_short_trades_below_2")
    return {
        "threshold": threshold,
        "discovery": discovery,
        "calibration": calibration,
        "accepted_preholdout_long_trades": long_count,
        "accepted_preholdout_short_trades": short_count,
        "robust_preholdout_score_bps": min(
            discovery["accepted_delta_vs_all_bps"],
            calibration["accepted_delta_vs_all_bps"],
        ),
        "basic_qualified": not reasons,
        "qualification_reasons": reasons,
    }


def _gate_metrics(
    records: list[FeatureRecord],
    probabilities: np.ndarray,
    *,
    threshold: float,
    account_exposure: float,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    if len(records) != len(probabilities):
        raise ValueError("Gate metrics require one probability per record")
    accepted = [
        record
        for record, probability in zip(records, probabilities, strict=True)
        if probability >= threshold
    ]
    rejected = [
        record
        for record, probability in zip(records, probabilities, strict=True)
        if probability < threshold
    ]
    all_metrics = _compact_trade_metrics(
        summarize_exit_trades([record.trade for record in records], account_exposure=account_exposure)
    )
    accepted_metrics = _compact_trade_metrics(
        summarize_exit_trades([record.trade for record in accepted], account_exposure=account_exposure)
    )
    rejected_metrics = _compact_trade_metrics(
        summarize_exit_trades([record.trade for record in rejected], account_exposure=account_exposure)
    )
    accepted_probabilities = [
        float(probability)
        for probability in probabilities
        if probability >= threshold
    ]
    return {
        "all": all_metrics,
        "accepted": accepted_metrics,
        "rejected": rejected_metrics,
        "accepted_fraction": len(accepted) / len(records) if records else 0.0,
        "accepted_average_probability": mean(accepted_probabilities) if accepted_probabilities else 0.0,
        "accepted_delta_vs_all_bps": (
            accepted_metrics["average_net_bps_per_trade"]
            - all_metrics["average_net_bps_per_trade"]
        ),
        "accepted_minus_rejected_bps": (
            accepted_metrics["average_net_bps_per_trade"]
            - rejected_metrics["average_net_bps_per_trade"]
            if accepted and rejected
            else 0.0
        ),
        "bootstrap_accepted_minus_rejected": _bootstrap_gate_delta(
            records,
            probabilities,
            threshold=threshold,
            samples=bootstrap_samples,
            seed=seed,
        ),
    }


def _probability_metrics(
    records: list[FeatureRecord],
    probabilities: np.ndarray,
    *,
    reference_win_rate: float,
) -> dict[str, Any]:
    if len(records) != len(probabilities):
        raise ValueError("Probability metrics require one probability per record")
    labels = np.asarray([_record_label(record) for record in records], dtype=np.int64)
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    if not len(labels):
        return {
            "rows": 0,
            "positive_rate": 0.0,
            "average_probability": 0.0,
            "brier_score": 0.0,
            "base_rate_brier_score": 0.0,
            "brier_skill_vs_discovery_base": 0.0,
            "log_loss": 0.0,
            "auc": 0.5,
            "expected_calibration_error": 0.0,
            "calibration_bins": [],
        }
    brier = float(np.mean((probabilities - labels) ** 2))
    base_brier = float(np.mean((reference_win_rate - labels) ** 2))
    bins = _calibration_bins(labels, probabilities)
    return {
        "rows": len(labels),
        "positive_rate": float(np.mean(labels)),
        "average_probability": float(np.mean(probabilities)),
        "brier_score": brier,
        "base_rate_brier_score": base_brier,
        "brier_skill_vs_discovery_base": 1.0 - brier / base_brier if base_brier > 0 else 0.0,
        "log_loss": _binary_log_loss(labels, probabilities),
        "auc": _auc(probabilities.tolist(), labels.astype(bool).tolist()),
        "expected_calibration_error": sum(
            row["fraction"] * abs(row["average_probability"] - row["positive_rate"])
            for row in bins
        ),
        "calibration_bins": bins,
    }


def _calibration_bins(labels: np.ndarray, probabilities: np.ndarray) -> list[dict[str, Any]]:
    boundaries = (0.0, 0.50, 0.55, 0.60, 0.65, 0.70, 1.000001)
    output = []
    for low, high in zip(boundaries[:-1], boundaries[1:], strict=True):
        mask = (probabilities >= low) & (probabilities < high)
        if not np.any(mask):
            continue
        output.append(
            {
                "low": low,
                "high": min(1.0, high),
                "rows": int(np.sum(mask)),
                "fraction": float(np.mean(mask)),
                "average_probability": float(np.mean(probabilities[mask])),
                "positive_rate": float(np.mean(labels[mask])),
            }
        )
    return output


def _bootstrap_gate_delta(
    records: list[FeatureRecord],
    probabilities: np.ndarray,
    *,
    threshold: float,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if samples <= 0 or len(records) < 4:
        return {
            "samples": 0,
            "mean_delta_bps": 0.0,
            "p025_bps": 0.0,
            "p975_bps": 0.0,
            "probability_positive": 0.0,
        }
    returns = np.asarray([record.trade.net_bps for record in records], dtype=np.float64)
    accepted = np.asarray(probabilities >= threshold, dtype=bool)
    if not np.any(accepted) or np.all(accepted):
        return {
            "samples": 0,
            "mean_delta_bps": 0.0,
            "p025_bps": 0.0,
            "p975_bps": 0.0,
            "probability_positive": 0.0,
        }
    rng = np.random.default_rng(seed)
    n = len(records)
    block_size = min(3, n)
    values = []
    attempts = 0
    while len(values) < samples and attempts < samples * 5:
        attempts += 1
        indices = []
        while len(indices) < n:
            start = int(rng.integers(0, n))
            indices.extend((start + offset) % n for offset in range(block_size))
        chosen = np.asarray(indices[:n], dtype=np.int64)
        flags = accepted[chosen]
        if not np.any(flags) or np.all(flags):
            continue
        outcomes = returns[chosen]
        values.append(float(np.mean(outcomes[flags]) - np.mean(outcomes[~flags])))
    return {
        "samples": len(values),
        "mean_delta_bps": mean(values) if values else 0.0,
        "p025_bps": float(np.quantile(values, 0.025)) if values else 0.0,
        "p975_bps": float(np.quantile(values, 0.975)) if values else 0.0,
        "probability_positive": mean(value > 0 for value in values) if values else 0.0,
    }


def _matched_coin_control(
    timeline: PriceTimeline,
    records: list[FeatureRecord],
    *,
    cost_bps: float,
    account_exposure: float,
    seeds: int,
    namespace: str,
) -> dict[str, Any]:
    strategy_value = mean(record.trade.net_bps for record in records) if records else 0.0
    values = []
    for seed in range(seeds):
        cohort = [
            replace(
                record.opportunity,
                side=_coin_side(namespace, seed, record.opportunity.timestamp_ms),
            )
            for record in records
        ]
        trades = evaluate_policy(timeline, cohort, REFERENCE_POLICY, cost_bps=cost_bps)
        metrics = summarize_exit_trades(trades, account_exposure=account_exposure)
        values.append(metrics["average_net_bps_per_trade"])
    values.sort()
    return {
        "seeds": seeds,
        "accepted_timestamps": len(records),
        "median_average_net_bps": float(np.median(values)) if values else 0.0,
        "p05_average_net_bps": float(np.quantile(values, 0.05)) if values else 0.0,
        "p95_average_net_bps": float(np.quantile(values, 0.95)) if values else 0.0,
        "positive_seed_fraction": mean(value > 0 for value in values) if values else 0.0,
        "seed_fraction_at_or_above_strategy": (
            mean(value >= strategy_value for value in values) if values else 0.0
        ),
    }


def _ensemble_verdict(
    selection_qualified: bool,
    holdout_gate: dict[str, Any],
    holdout_quality: dict[str, Any],
    matched_coin: dict[str, Any],
) -> str:
    if not selection_qualified:
        return "selection_failed"
    accepted = holdout_gate["accepted"]
    if accepted["trades"] < 6:
        return "insufficient_historical_audit"
    if accepted["average_net_bps_per_trade"] <= 0 or holdout_gate["accepted_delta_vs_all_bps"] <= 0:
        return "failed_historical_audit"
    if (
        accepted["long"]["trades"] == 0
        or accepted["short"]["trades"] == 0
        or accepted["long"]["average_net_bps_per_trade"] <= 0
        or accepted["short"]["average_net_bps_per_trade"] <= 0
    ):
        return "not_side_general"
    if holdout_quality["brier_skill_vs_discovery_base"] <= 0:
        return "probability_calibration_failed_audit"
    if accepted["average_net_bps_per_trade"] <= matched_coin["p95_average_net_bps"]:
        return "matched_coin_not_cleared"
    if holdout_gate["bootstrap_accepted_minus_rejected"]["probability_positive"] < 0.80:
        return "audit_uncertainty_too_wide"
    return "historical_audit_pass_forward_required"


def _expanding_splits(
    total: int,
    *,
    minimum_train: int = 16,
    validation_size: int = 8,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if total < minimum_train + validation_size:
        raise ValueError(
            f"Need at least {minimum_train + validation_size} rows for expanding validation"
        )
    output = []
    start = minimum_train
    while start < total:
        end = min(total, start + validation_size)
        output.append(
            (
                np.arange(0, start, dtype=np.int64),
                np.arange(start, end, dtype=np.int64),
            )
        )
        start = end
    return output


def _record_label(record: FeatureRecord) -> int:
    return 1 if record.trade.net_bps > 0 else 0


def _binary_log_loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return float(
        -np.mean(labels * np.log(probabilities) + (1.0 - labels) * np.log(1.0 - probabilities))
    )


def _sigmoid(values: np.ndarray | float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    output = np.empty_like(array)
    positive = array >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    exponent = np.exp(array[~positive])
    output[~positive] = exponent / (1.0 + exponent)
    return output


def _stable_seed(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")
