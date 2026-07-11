from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import numpy as np

from futures_lab.exit_laboratory import (
    REFERENCE_POLICY,
    ExitOpportunity,
    ExitTrade,
    evaluate_policy,
    lock_reference_cohort,
    summarize_exit_trades,
)
from futures_lab.first_touch import (
    PriceTimeline,
    _mean_htf,
    _micro_signal,
    load_feature_samples,
    load_mark_price_timeline,
)
from futures_lab.strategy_tournament import _period_boundaries


@dataclass(frozen=True)
class RegimeDefinition:
    name: str
    description: str
    categories: tuple[str, ...]
    gate_eligible: bool = True


@dataclass(frozen=True)
class RegimeRecord:
    opportunity: ExitOpportunity
    trade: ExitTrade
    regimes: dict[str, str]


REGIME_DEFINITIONS = (
    RegimeDefinition(
        "volatility",
        "Entry-time 180-second realized volatility, split by discovery-period market quantiles.",
        ("low", "normal", "high"),
    ),
    RegimeDefinition(
        "spread",
        "Top-of-book spread, split by discovery-period market quantiles.",
        ("tight", "normal", "wide"),
    ),
    RegimeDefinition(
        "htf_alignment",
        "The squeeze direction compared with the mean 15m/30m/1h trend score.",
        ("opposed", "neutral", "aligned"),
    ),
    RegimeDefinition(
        "flow_strength",
        "Side-relative microstructure strength within valid squeeze signals.",
        ("low", "medium", "high"),
    ),
    RegimeDefinition(
        "breakout_extension",
        "Side-relative 180-second price extension at entry.",
        ("early", "developing", "extended"),
    ),
    RegimeDefinition(
        "utc_session",
        "Fixed eight-hour UTC session bucket; no outcome-derived clock boundaries.",
        ("asia_00_08", "europe_08_16", "americas_16_24"),
    ),
    RegimeDefinition(
        "liquidity_volatility",
        "Joint state from discovery-frozen high-volatility and wide-spread thresholds.",
        ("orderly", "volatile", "wide_spread", "stressed"),
    ),
    RegimeDefinition(
        "entry_side",
        "Long/short audit only. This dimension is forbidden from permission-gate selection.",
        ("long", "short"),
        gate_eligible=False,
    ),
)


def run_regime_atlas(
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
    minimum_discovery_trades: int = 10,
    minimum_calibration_trades: int = 5,
    minimum_gate_coverage: float = 0.25,
    maximum_gate_coverage: float = 0.85,
    matched_coin_seeds: int = 32,
    bootstrap_samples: int = 2_000,
) -> dict[str, Any]:
    if cost_bps < 0 or sample_seconds <= 0 or horizon_seconds <= 0:
        raise ValueError("Regime-atlas cost, sampling, and horizon settings must be valid")
    if not 0.0 < minimum_gate_coverage < maximum_gate_coverage < 1.0:
        raise ValueError("Regime gate coverage bounds are invalid")
    if matched_coin_seeds < 8 or bootstrap_samples < 100:
        raise ValueError("Regime controls require at least 8 coin seeds and 100 bootstrap samples")

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

    cohorts: dict[str, list[ExitOpportunity]] = {}
    cohort_stats: dict[str, dict[str, Any]] = {}
    for period, rows in period_features.items():
        cohorts[period], cohort_stats[period] = lock_reference_cohort(
            timeline,
            rows,
            cost_bps=cost_bps,
            horizon_seconds=horizon_seconds,
        )
    thresholds = derive_regime_thresholds(period_features["discovery"], cohorts["discovery"])

    period_records: dict[str, list[RegimeRecord]] = {}
    baseline_metrics: dict[str, dict[str, Any]] = {}
    for period, cohort in cohorts.items():
        trades = evaluate_policy(
            timeline,
            cohort,
            REFERENCE_POLICY,
            cost_bps=cost_bps,
        )
        period_records[period] = [
            RegimeRecord(
                opportunity=opportunity,
                trade=trade,
                regimes=classify_opportunity(opportunity, thresholds),
            )
            for opportunity, trade in zip(cohort, trades, strict=True)
        ]
        baseline_metrics[period] = _summarize_records(
            period_records[period],
            account_exposure=account_exposure,
            bootstrap_samples=bootstrap_samples,
            seed=_stable_seed(f"{symbol}:{period}:baseline"),
        )

    cell_results = []
    for definition in REGIME_DEFINITIONS:
        for category in definition.categories:
            for period in ("discovery", "calibration", "holdout"):
                records = [
                    record for record in period_records[period] if record.regimes[definition.name] == category
                ]
                metrics = _summarize_records(
                    records,
                    account_exposure=account_exposure,
                    bootstrap_samples=bootstrap_samples,
                    seed=_stable_seed(f"{symbol}:{definition.name}:{category}:{period}"),
                )
                cell_results.append(
                    {
                        "dimension": definition.name,
                        "category": category,
                        "period": period,
                        "coverage": len(records) / len(period_records[period]) if period_records[period] else 0.0,
                        "delta_vs_period_all_bps": (
                            metrics["average_net_bps_per_trade"]
                            - baseline_metrics[period]["average_net_bps_per_trade"]
                        ),
                        **metrics,
                    }
                )

    selection_candidates = []
    for definition in REGIME_DEFINITIONS:
        if not definition.gate_eligible:
            continue
        for accepted_categories in candidate_category_sets(definition):
            selection_candidates.append(
                _gate_candidate(
                    definition,
                    accepted_categories,
                    period_records,
                    baseline_metrics,
                    minimum_discovery_trades=minimum_discovery_trades,
                    minimum_calibration_trades=minimum_calibration_trades,
                    minimum_gate_coverage=minimum_gate_coverage,
                    maximum_gate_coverage=maximum_gate_coverage,
                    account_exposure=account_exposure,
                )
            )
    selected_by_dimension = _select_gate_candidates(selection_candidates)

    gate_results = []
    for dimension, selection in sorted(selected_by_dimension.items()):
        accepted_categories = frozenset(selection["accepted_categories"])
        holdout_accepted, holdout_rejected = _partition_records(
            period_records["holdout"],
            dimension,
            accepted_categories,
        )
        accepted_metrics = _summarize_records(
            holdout_accepted,
            account_exposure=account_exposure,
            bootstrap_samples=bootstrap_samples,
            seed=_stable_seed(f"{symbol}:{dimension}:holdout:accepted"),
        )
        rejected_metrics = _summarize_records(
            holdout_rejected,
            account_exposure=account_exposure,
            bootstrap_samples=bootstrap_samples,
            seed=_stable_seed(f"{symbol}:{dimension}:holdout:rejected"),
        )
        group_delta = _bootstrap_group_delta(
            period_records["holdout"],
            dimension=dimension,
            accepted_categories=accepted_categories,
            samples=bootstrap_samples,
            seed=_stable_seed(f"{symbol}:{dimension}:holdout:group-delta"),
        )
        coin_control = _matched_coin_control(
            timeline,
            holdout_accepted,
            cost_bps=cost_bps,
            account_exposure=account_exposure,
            seeds=matched_coin_seeds,
            namespace=f"{symbol}:{dimension}",
        )
        verdict = _gate_verdict(
            selection,
            accepted_metrics,
            rejected_metrics,
            baseline_metrics["holdout"],
            group_delta,
            coin_control,
        )
        baseline_total = baseline_metrics["holdout"]["total_net_bps"]
        gate_results.append(
            {
                **selection,
                "holdout": {
                    "all": baseline_metrics["holdout"],
                    "accepted": accepted_metrics,
                    "rejected": rejected_metrics,
                    "accepted_delta_vs_all_bps": (
                        accepted_metrics["average_net_bps_per_trade"]
                        - baseline_metrics["holdout"]["average_net_bps_per_trade"]
                    ),
                    "accepted_minus_rejected_bps": (
                        accepted_metrics["average_net_bps_per_trade"]
                        - rejected_metrics["average_net_bps_per_trade"]
                    ),
                    "retained_trade_fraction": (
                        len(holdout_accepted) / len(period_records["holdout"])
                        if period_records["holdout"]
                        else 0.0
                    ),
                    "retained_total_net_fraction": (
                        accepted_metrics["total_net_bps"] / baseline_total if baseline_total else None
                    ),
                    "avoided_losing_trades": sum(record.trade.net_bps < 0 for record in holdout_rejected),
                    "missed_winning_trades": sum(record.trade.net_bps > 0 for record in holdout_rejected),
                },
                "bootstrap_accepted_minus_rejected": group_delta,
                "matched_coin_control": coin_control,
                "verdict": verdict,
                "holdout_records": [
                    {
                        "entry": _iso(record.opportunity.timestamp_ms),
                        "side": "long" if record.opportunity.side == 1 else "short",
                        "category": record.regimes[dimension],
                        "accepted": record in holdout_accepted,
                        "net_bps": record.trade.net_bps,
                        "exit_reason": record.trade.exit_reason,
                    }
                    for record in period_records["holdout"]
                ],
            }
        )

    gate_results.sort(
        key=lambda row: (
            _verdict_rank(row["verdict"]),
            row["robust_gate_score_bps"],
            row["holdout"]["accepted_delta_vs_all_bps"],
        ),
        reverse=True,
    )
    qualified = [row for row in gate_results if row["selection_qualified"]]
    overall = max(
        qualified or gate_results,
        key=lambda row: (
            row["robust_gate_score_bps"],
            row["calibration"]["accepted"]["average_net_bps_per_trade"],
        ),
    )

    return {
        "study": "nested_side_neutral_regime_atlas",
        "symbol": symbol,
        "parameters": {
            "round_trip_cost_bps": cost_bps,
            "sample_seconds": sample_seconds,
            "max_gap_seconds": max_gap_seconds,
            "discovery_fraction": discovery_fraction,
            "calibration_end_fraction": calibration_end_fraction,
            "horizon_seconds": horizon_seconds,
            "account_exposure_multiple": account_exposure,
            "minimum_discovery_trades": minimum_discovery_trades,
            "minimum_calibration_trades": minimum_calibration_trades,
            "minimum_gate_coverage": minimum_gate_coverage,
            "maximum_gate_coverage": maximum_gate_coverage,
            "matched_coin_seeds": matched_coin_seeds,
            "bootstrap_samples": bootstrap_samples,
            "entry_rule": "existing side-symmetric volatility_squeeze_breakout",
            "exit_policy": REFERENCE_POLICY.name,
        },
        "data": {
            **mark_stats,
            **feature_stats,
            "calibration_start": _iso(calibration_start_ms),
            "holdout_start": _iso(holdout_start_ms),
        },
        "entry_cohorts": cohort_stats,
        "thresholds_frozen_from_discovery": thresholds,
        "dimensions": [
            {
                "name": definition.name,
                "description": definition.description,
                "categories": list(definition.categories),
                "gate_eligible": definition.gate_eligible,
            }
            for definition in REGIME_DEFINITIONS
        ],
        "baseline": baseline_metrics,
        "cell_results": cell_results,
        "selection_candidates": selection_candidates,
        "gate_results": gate_results,
        "overall_pre_holdout_selection": {
            "dimension": overall["dimension"],
            "accepted_categories": overall["accepted_categories"],
            "selection_qualified": overall["selection_qualified"],
            "robust_gate_score_bps": overall["robust_gate_score_bps"],
            "holdout_verdict": overall["verdict"],
        },
        "promotion_candidates": [
            {
                "dimension": row["dimension"],
                "accepted_categories": row["accepted_categories"],
                "verdict": row["verdict"],
            }
            for row in gate_results
            if row["verdict"] == "robust_permission_candidate"
        ],
        "interpretation_notes": [
            "Regime thresholds are frozen from discovery-period features or fixed clock/HTF definitions before calibration and holdout outcomes are evaluated.",
            "All regimes use the same non-overlapping squeeze entries and fixed 100/150/6h exit established by the prior laboratories.",
            "Permission candidates are side-neutral category sets. Entry side is reported only as an audit and cannot be selected as a gate.",
            "A gate must improve both discovery and calibration expectancy versus trading every locked entry, retain both long and short trades, and preserve chronological robustness.",
            "Holdout promotion additionally requires accepted trades to beat rejected trades with a positive block-bootstrap lower bound and to clear matched direction luck.",
            "The archive has already informed previous studies, so even a holdout winner remains retrospective development evidence requiring forward shadow confirmation.",
            "Small regime cells have wide uncertainty; the atlas reports them rather than silently pooling or deleting them.",
        ],
    }


def derive_regime_thresholds(
    discovery_features: list[tuple[int, dict[str, Any]]],
    discovery_cohort: list[ExitOpportunity],
) -> dict[str, Any]:
    volatility = [
        value * 10_000.0
        for _, row in discovery_features
        if (value := _finite_float(row.get("realized_vol_180s_pct"))) is not None
    ]
    spreads = [
        value
        for _, row in discovery_features
        if (value := _finite_float(row.get("spread_bps"))) is not None
    ]
    flow_strength = [opportunity.side * _micro_signal(opportunity.row) for opportunity in discovery_cohort]
    extension = [
        opportunity.side * (_finite_float(opportunity.row.get("return_180s_pct")) or 0.0) * 10_000.0
        for opportunity in discovery_cohort
    ]
    return {
        "volatility_bps": _tercile_thresholds(volatility),
        "spread_bps": _tercile_thresholds(spreads),
        "flow_strength": _tercile_thresholds(flow_strength),
        "breakout_extension_bps": _tercile_thresholds(extension),
        "htf_alignment_threshold": 0.20,
        "utc_session_boundaries": [0, 8, 16, 24],
    }


def classify_opportunity(opportunity: ExitOpportunity, thresholds: dict[str, Any]) -> dict[str, str]:
    row = opportunity.row
    volatility = (_finite_float(row.get("realized_vol_180s_pct")) or 0.0) * 10_000.0
    spread = _finite_float(row.get("spread_bps"))
    signed_htf = opportunity.side * _mean_htf(row, "trend_score")
    signed_flow = opportunity.side * _micro_signal(row)
    signed_extension = opportunity.side * (_finite_float(row.get("return_180s_pct")) or 0.0) * 10_000.0
    hour = datetime.fromtimestamp(opportunity.timestamp_ms / 1_000, tz=timezone.utc).hour

    volatility_label = _tercile_label(volatility, thresholds["volatility_bps"], ("low", "normal", "high"))
    spread_label = (
        _tercile_label(spread, thresholds["spread_bps"], ("tight", "normal", "wide"))
        if spread is not None
        else "wide"
    )
    htf_threshold = thresholds["htf_alignment_threshold"]
    htf_label = "opposed" if signed_htf < -htf_threshold else "aligned" if signed_htf > htf_threshold else "neutral"
    flow_label = _tercile_label(signed_flow, thresholds["flow_strength"], ("low", "medium", "high"))
    extension_label = _tercile_label(
        signed_extension,
        thresholds["breakout_extension_bps"],
        ("early", "developing", "extended"),
    )
    session = "asia_00_08" if hour < 8 else "europe_08_16" if hour < 16 else "americas_16_24"
    is_volatile = volatility > thresholds["volatility_bps"]["high_min"]
    is_wide = spread is None or spread > thresholds["spread_bps"]["high_min"]
    joint = (
        "stressed"
        if is_volatile and is_wide
        else "volatile"
        if is_volatile
        else "wide_spread"
        if is_wide
        else "orderly"
    )
    return {
        "volatility": volatility_label,
        "spread": spread_label,
        "htf_alignment": htf_label,
        "flow_strength": flow_label,
        "breakout_extension": extension_label,
        "utc_session": session,
        "liquidity_volatility": joint,
        "entry_side": "long" if opportunity.side == 1 else "short",
    }


def candidate_category_sets(definition: RegimeDefinition) -> list[frozenset[str]]:
    categories = definition.categories
    if definition.name == "utc_session":
        return [
            *(frozenset((category,)) for category in categories),
            *(frozenset(values) for values in combinations(categories, 2)),
        ]
    if len(categories) == 3:
        return [
            frozenset((categories[0],)),
            frozenset((categories[1],)),
            frozenset((categories[2],)),
            frozenset((categories[0], categories[1])),
            frozenset((categories[1], categories[2])),
        ]
    if definition.name == "liquidity_volatility":
        output = [frozenset((category,)) for category in categories]
        output.extend(frozenset(value for value in categories if value != blocked) for blocked in categories)
        return output
    output = []
    for size in range(1, len(categories)):
        output.extend(frozenset(values) for values in combinations(categories, size))
    return output


def _gate_candidate(
    definition: RegimeDefinition,
    accepted_categories: frozenset[str],
    period_records: dict[str, list[RegimeRecord]],
    baseline_metrics: dict[str, dict[str, Any]],
    *,
    minimum_discovery_trades: int,
    minimum_calibration_trades: int,
    minimum_gate_coverage: float,
    maximum_gate_coverage: float,
    account_exposure: float,
) -> dict[str, Any]:
    periods = {}
    improvement = []
    separation = []
    coverages = []
    side_balances = []
    for period in ("discovery", "calibration"):
        accepted, rejected = _partition_records(period_records[period], definition.name, accepted_categories)
        accepted_metrics = _summarize_records(
            accepted,
            account_exposure=account_exposure,
            bootstrap_samples=0,
            seed=0,
        )
        rejected_metrics = _summarize_records(
            rejected,
            account_exposure=account_exposure,
            bootstrap_samples=0,
            seed=0,
        )
        coverage = len(accepted) / len(period_records[period]) if period_records[period] else 0.0
        improvement.append(
            accepted_metrics["average_net_bps_per_trade"]
            - baseline_metrics[period]["average_net_bps_per_trade"]
        )
        separation.append(
            accepted_metrics["average_net_bps_per_trade"]
            - rejected_metrics["average_net_bps_per_trade"]
        )
        coverages.append(coverage)
        side_balances.append(_side_balance(accepted_metrics))
        periods[period] = {
            "coverage": coverage,
            "accepted": accepted_metrics,
            "rejected": rejected_metrics,
            "accepted_delta_vs_all_bps": improvement[-1],
            "accepted_minus_rejected_bps": separation[-1],
        }
    robust_improvement = min(improvement)
    robust_separation = min(separation)
    robust_score = min(robust_improvement, robust_separation)
    qualifies = bool(
        periods["discovery"]["accepted"]["trades"] >= minimum_discovery_trades
        and periods["calibration"]["accepted"]["trades"] >= minimum_calibration_trades
        and robust_improvement > 0
        and robust_separation > 0
        and min(coverages) >= minimum_gate_coverage
        and max(coverages) <= maximum_gate_coverage
        and min(side_balances) >= 0.15
        and periods["discovery"]["accepted"]["average_net_bps_per_trade"] > 0
        and periods["calibration"]["accepted"]["average_net_bps_per_trade"] > 0
        and periods["discovery"]["accepted"]["positive_chronological_block_fraction"] >= 0.50
        and periods["calibration"]["accepted"]["positive_chronological_block_fraction"] >= 0.50
    )
    return {
        "dimension": definition.name,
        "accepted_categories": sorted(accepted_categories),
        "qualifies": qualifies,
        "robust_improvement_vs_all_bps": robust_improvement,
        "robust_accepted_minus_rejected_bps": robust_separation,
        "robust_gate_score_bps": robust_score,
        **periods,
    }


def _select_gate_candidates(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate["dimension"], []).append(candidate)
    output = {}
    for dimension, rows in grouped.items():
        qualified = [row for row in rows if row["qualifies"]]
        selected = max(
            qualified or rows,
            key=lambda row: (
                row["robust_gate_score_bps"],
                row["calibration"]["accepted"]["average_net_bps_per_trade"],
                row["calibration"]["accepted"]["trades"],
            ),
        )
        output[dimension] = {
            **selected,
            "dimension_candidate_count": len(rows),
            "qualified_candidate_count": len(qualified),
            "selection_qualified": bool(qualified),
        }
    return output


def _partition_records(
    records: list[RegimeRecord],
    dimension: str,
    accepted_categories: frozenset[str],
) -> tuple[list[RegimeRecord], list[RegimeRecord]]:
    accepted = [record for record in records if record.regimes[dimension] in accepted_categories]
    rejected = [record for record in records if record.regimes[dimension] not in accepted_categories]
    return accepted, rejected


def _summarize_records(
    records: list[RegimeRecord],
    *,
    account_exposure: float,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    raw = summarize_exit_trades([record.trade for record in records], account_exposure=account_exposure)
    interval = _bootstrap_mean_interval(
        [record.trade.net_bps for record in records],
        samples=bootstrap_samples,
        seed=seed,
    )
    return {
        "trades": raw["trades"],
        "wins": raw["wins"],
        "losses": raw["losses"],
        "positive_net_trade_rate": raw["positive_net_trade_rate"],
        "average_net_bps_per_trade": raw["average_net_bps_per_trade"],
        "mean_net_bps_bootstrap_95": interval,
        "total_net_bps": raw["total_net_bps"],
        "profit_factor": raw["profit_factor"],
        "average_duration_seconds": raw["average_duration_seconds"],
        "average_mfe_bps": raw["average_mfe_bps"],
        "average_mae_bps": raw["average_mae_bps"],
        "max_additive_account_drawdown_pct": raw["max_additive_account_drawdown_pct"],
        "max_losing_streak": raw["max_losing_streak"],
        "positive_chronological_block_fraction": raw["positive_chronological_block_fraction"],
        "exit_reasons": raw["exit_reasons"],
        "long": raw["by_side"]["long"],
        "short": raw["by_side"]["short"],
        "chronological_trade_blocks": raw["chronological_trade_blocks"],
    }


def _bootstrap_mean_interval(values: list[float], *, samples: int, seed: int) -> list[float] | None:
    if not values or samples <= 0:
        return None
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(array)
    block_size = min(3, n)
    boot = np.empty(samples, dtype=np.float64)
    for sample_index in range(samples):
        collected = []
        while len(collected) < n:
            start = int(rng.integers(0, n))
            collected.extend(array[(start + offset) % n] for offset in range(block_size))
        boot[sample_index] = float(np.mean(collected[:n]))
    return [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]


def _bootstrap_group_delta(
    records: list[RegimeRecord],
    *,
    dimension: str,
    accepted_categories: frozenset[str],
    samples: int,
    seed: int,
    block_size: int = 5,
) -> dict[str, Any]:
    accepted_flags = np.asarray(
        [record.regimes[dimension] in accepted_categories for record in records],
        dtype=bool,
    )
    returns = np.asarray([record.trade.net_bps for record in records], dtype=np.float64)
    if not len(records) or np.all(accepted_flags) or not np.any(accepted_flags):
        return {
            "mean_delta_bps": 0.0,
            "bootstrap_p025_bps": 0.0,
            "bootstrap_p975_bps": 0.0,
            "bootstrap_probability_positive": 0.0,
            "samples_used": 0,
        }
    observed = float(np.mean(returns[accepted_flags]) - np.mean(returns[~accepted_flags]))
    rng = np.random.default_rng(seed)
    n = len(records)
    size = min(block_size, n)
    boot = []
    attempts = 0
    while len(boot) < samples and attempts < samples * 5:
        attempts += 1
        indices = []
        while len(indices) < n:
            start = int(rng.integers(0, n))
            indices.extend((start + offset) % n for offset in range(size))
        chosen = np.asarray(indices[:n], dtype=np.int64)
        flags = accepted_flags[chosen]
        if np.all(flags) or not np.any(flags):
            continue
        values = returns[chosen]
        boot.append(float(np.mean(values[flags]) - np.mean(values[~flags])))
    if not boot:
        return {
            "mean_delta_bps": observed,
            "bootstrap_p025_bps": 0.0,
            "bootstrap_p975_bps": 0.0,
            "bootstrap_probability_positive": 0.0,
            "samples_used": 0,
        }
    return {
        "mean_delta_bps": observed,
        "bootstrap_p025_bps": float(np.quantile(boot, 0.025)),
        "bootstrap_p975_bps": float(np.quantile(boot, 0.975)),
        "bootstrap_probability_positive": float(np.mean(np.asarray(boot) > 0)),
        "samples_used": len(boot),
        "block_size": size,
    }


def _matched_coin_control(
    timeline: PriceTimeline,
    accepted_records: list[RegimeRecord],
    *,
    cost_bps: float,
    account_exposure: float,
    seeds: int,
    namespace: str,
) -> dict[str, Any]:
    strategy_value = (
        mean(record.trade.net_bps for record in accepted_records) if accepted_records else 0.0
    )
    values = []
    for seed in range(seeds):
        cohort = [
            replace(
                record.opportunity,
                side=_coin_side(namespace, seed, record.opportunity.timestamp_ms),
            )
            for record in accepted_records
        ]
        trades = evaluate_policy(timeline, cohort, REFERENCE_POLICY, cost_bps=cost_bps)
        metrics = summarize_exit_trades(trades, account_exposure=account_exposure)
        values.append(metrics["average_net_bps_per_trade"])
    values.sort()
    return {
        "seeds": seeds,
        "median_average_net_bps": median(values) if values else 0.0,
        "p05_average_net_bps": float(np.quantile(values, 0.05)) if values else 0.0,
        "p95_average_net_bps": float(np.quantile(values, 0.95)) if values else 0.0,
        "positive_seed_fraction": mean(value > 0 for value in values) if values else 0.0,
        "seed_fraction_at_or_above_strategy": (
            mean(value >= strategy_value for value in values) if values else 0.0
        ),
    }


def _gate_verdict(
    selection: dict[str, Any],
    accepted: dict[str, Any],
    rejected: dict[str, Any],
    baseline: dict[str, Any],
    group_delta: dict[str, Any],
    coin_control: dict[str, Any],
) -> str:
    if not selection["selection_qualified"]:
        return "selection_failed"
    both_sides_positive = bool(
        accepted["long"]["trades"] > 0
        and accepted["short"]["trades"] > 0
        and accepted["long"]["average_net_bps_per_trade"] > 0
        and accepted["short"]["average_net_bps_per_trade"] > 0
    )
    accepted_average = accepted["average_net_bps_per_trade"]
    rejected_average = rejected["average_net_bps_per_trade"]
    if accepted_average <= baseline["average_net_bps_per_trade"] or accepted_average <= rejected_average:
        return "failed_holdout"
    if (
        group_delta["bootstrap_p025_bps"] > 0
        and rejected_average <= 0
        and accepted_average > coin_control["p95_average_net_bps"]
        and both_sides_positive
    ):
        return "robust_permission_candidate"
    if both_sides_positive:
        return "promising_but_unconfirmed"
    return "not_side_general"


def _side_balance(metrics: dict[str, Any]) -> float:
    trades = metrics["trades"]
    return min(metrics["long"]["trades"], metrics["short"]["trades"]) / trades if trades else 0.0


def _tercile_thresholds(values: Iterable[float]) -> dict[str, float]:
    finite = [float(value) for value in values if np.isfinite(value)]
    if len(finite) < 3:
        raise ValueError("At least three finite discovery values are required for regime thresholds")
    return {
        "low_max": float(np.quantile(finite, 1 / 3)),
        "high_min": float(np.quantile(finite, 2 / 3)),
    }


def _tercile_label(value: float, thresholds: dict[str, float], labels: tuple[str, str, str]) -> str:
    if value <= thresholds["low_max"]:
        return labels[0]
    if value >= thresholds["high_min"]:
        return labels[2]
    return labels[1]


def _coin_side(namespace: str, seed: int, timestamp_ms: int) -> int:
    digest = hashlib.sha256(f"{namespace}:{seed}:{timestamp_ms}".encode("ascii")).digest()
    return 1 if digest[0] & 1 else -1


def _stable_seed(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _iso(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1_000, tz=timezone.utc).isoformat()


def _verdict_rank(verdict: str) -> int:
    return {
        "robust_permission_candidate": 4,
        "promising_but_unconfirmed": 3,
        "not_side_general": 2,
        "failed_holdout": 1,
        "selection_failed": 0,
    }[verdict]
