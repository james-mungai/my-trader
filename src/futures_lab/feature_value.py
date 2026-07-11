from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

from futures_lab.exit_laboratory import (
    REFERENCE_POLICY,
    ExitOpportunity,
    ExitTrade,
    evaluate_policy,
    lock_reference_cohort,
    summarize_exit_trades,
)
from futures_lab.first_touch import PriceTimeline, load_feature_samples, load_mark_price_timeline
from futures_lab.strategy_tournament import _period_boundaries


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    group: str
    description: str
    unit: str
    alpha_eligible: bool = True


@dataclass(frozen=True)
class FeatureRecord:
    opportunity: ExitOpportunity
    trade: ExitTrade
    values: dict[str, float | None]


FEATURE_DEFINITIONS = (
    FeatureDefinition("signed_ofi_1s", "local_order_flow", "One-second order-flow imbalance aligned to the trade side.", "score"),
    FeatureDefinition("signed_ofi_5s", "local_order_flow", "Five-second order-flow imbalance aligned to the trade side.", "score"),
    FeatureDefinition("signed_aggression_1s", "local_order_flow", "One-second taker aggression aligned to the trade side.", "score"),
    FeatureDefinition("signed_aggression_5s", "local_order_flow", "Five-second taker aggression aligned to the trade side.", "score"),
    FeatureDefinition("signed_aggression_15s", "local_order_flow", "Fifteen-second taker aggression aligned to the trade side.", "score"),
    FeatureDefinition("signed_taker_ratio_10s", "local_order_flow", "Ten-second taker-buy ratio mirrored around 50% by trade side.", "score"),
    FeatureDefinition("signed_microprice_bps", "book_pressure", "Microprice displacement from mid aligned to the trade side.", "bps"),
    FeatureDefinition("signed_vamp_bps", "book_pressure", "VAMP displacement from mid aligned to the trade side.", "bps"),
    FeatureDefinition("signed_weighted_depth_bps", "book_pressure", "Weighted-depth price displacement aligned to the trade side.", "bps"),
    FeatureDefinition("signed_depth_imbalance", "book_pressure", "Top-five depth imbalance aligned to the trade side.", "score"),
    FeatureDefinition("signed_book_imbalance", "book_pressure", "Top-of-book quantity imbalance aligned to the trade side.", "score"),
    FeatureDefinition("side_refill_advantage_5s", "book_pressure", "Supporting-side refill minus opposing-side refill.", "rate"),
    FeatureDefinition("side_wall_advantage", "book_pressure", "Supporting-side wall ratio minus opposing-side wall ratio.", "score"),
    FeatureDefinition("signed_return_15s_bps", "price_action", "Fifteen-second return aligned to the trade side.", "bps"),
    FeatureDefinition("signed_return_60s_bps", "price_action", "Sixty-second return aligned to the trade side.", "bps"),
    FeatureDefinition("signed_return_180s_bps", "price_action", "Three-minute return aligned to the trade side.", "bps"),
    FeatureDefinition("breakout_edge_position", "price_action", "Local range position mirrored so one means the chosen-side edge.", "fraction"),
    FeatureDefinition("realized_vol_60s_bps", "volatility_liquidity", "One-minute realized volatility.", "bps"),
    FeatureDefinition("realized_vol_180s_bps", "volatility_liquidity", "Three-minute realized volatility.", "bps"),
    FeatureDefinition("range_180s_bps", "volatility_liquidity", "Three-minute high-low range.", "bps"),
    FeatureDefinition("spread_bps", "volatility_liquidity", "Current top-of-book spread.", "bps"),
    FeatureDefinition("spread_max_5s_bps", "volatility_liquidity", "Maximum spread over five seconds.", "bps"),
    FeatureDefinition("spread_std_5s_bps", "volatility_liquidity", "Spread instability over five seconds.", "bps"),
    FeatureDefinition("log_depth_qty_top5", "volatility_liquidity", "Log total top-five bid and ask quantity.", "log_qty"),
    FeatureDefinition("htf_trend_alignment", "higher_timeframe", "Mean 15m/30m/1h trend score aligned to trade side.", "score"),
    FeatureDefinition("htf_return_alignment_bps", "higher_timeframe", "Mean 15m/30m/1h return aligned to trade side.", "bps"),
    FeatureDefinition("htf_edge_position", "higher_timeframe", "Mean HTF range position mirrored toward the chosen side.", "fraction"),
    FeatureDefinition("htf_target_room_bps", "higher_timeframe", "Mean chosen-side room to HTF resistance/support.", "bps"),
    FeatureDefinition("htf_taker_alignment", "higher_timeframe", "Mean HTF taker-buy ratio mirrored by trade side.", "score"),
    FeatureDefinition("signed_btc_ofi_1s", "cross_market", "BTC one-second OFI aligned to the ETH trade side.", "score"),
    FeatureDefinition("signed_btc_aggression_1s", "cross_market", "BTC taker aggression aligned to the ETH trade side.", "score"),
    FeatureDefinition("signed_btc_microprice_bps", "cross_market", "BTC microprice displacement aligned to the ETH trade side.", "bps"),
    FeatureDefinition("signed_btc_return_15s_bps", "cross_market", "BTC 15-second return aligned to the ETH trade side.", "bps"),
    FeatureDefinition("signed_eth_btc_relative_15s_bps", "cross_market", "ETH-minus-BTC 15-second relative return aligned to trade side.", "bps"),
    FeatureDefinition("open_interest_change_5m_bps", "derivatives", "Five-minute open-interest change.", "bps"),
    FeatureDefinition("favorable_funding_carry_bps", "derivatives", "Funding carry signed positive when favorable to the trade side.", "bps"),
    FeatureDefinition("signed_liquidation_imbalance", "derivatives", "Censored liquidation buy ratio aligned to trade side.", "score"),
    FeatureDefinition("avg_book_lag_30s_ms", "data_quality", "Average accepted book-event lag over 30 seconds.", "ms", alpha_eligible=False),
    FeatureDefinition("avg_trade_lag_30s_ms", "data_quality", "Average accepted trade-event lag over 30 seconds.", "ms", alpha_eligible=False),
)


def run_feature_value_analysis(
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
    minimum_coverage: float = 0.80,
    minimum_discovery_tail: int = 6,
    minimum_calibration_tail: int = 3,
    matched_coin_seeds: int = 32,
    permutation_samples: int = 2_000,
    bootstrap_samples: int = 2_000,
) -> dict[str, Any]:
    if cost_bps < 0 or sample_seconds <= 0 or horizon_seconds <= 0:
        raise ValueError("Feature-value cost, sampling, and horizon settings must be valid")
    if not 0.5 <= minimum_coverage <= 1.0:
        raise ValueError("Feature coverage threshold must be between 0.5 and 1.0")
    if matched_coin_seeds < 8 or permutation_samples < 100 or bootstrap_samples < 100:
        raise ValueError("Feature controls require at least 8 coin seeds and 100 resamples")

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
    period_records: dict[str, list[FeatureRecord]] = {}
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
        cohorts[period] = cohort
        cohort_stats[period] = stats
        period_records[period] = records
        baseline[period] = _compact_trade_metrics(
            summarize_exit_trades(trades, account_exposure=account_exposure)
        )

    feature_rows = []
    for definition in FEATURE_DEFINITIONS:
        orientation, raw_discovery_effect = _discovery_orientation(
            period_records["discovery"],
            definition.name,
        )
        discovery_oriented = [
            value * orientation
            for record in period_records["discovery"]
            if (value := record.values[definition.name]) is not None
        ]
        thresholds = _tercile_thresholds(discovery_oriented)
        periods = {}
        for period in ("discovery", "calibration", "holdout"):
            periods[period] = _feature_period_metrics(
                period_records[period],
                feature_name=definition.name,
                orientation=orientation,
                thresholds=thresholds,
                baseline_average_bps=baseline[period]["average_net_bps_per_trade"],
                account_exposure=account_exposure,
                permutation_samples=permutation_samples,
                bootstrap_samples=bootstrap_samples,
                seed=_stable_seed(f"{symbol}:{definition.name}:{period}"),
            )
        combined_side = _combined_preholdout_side_effect(
            [*period_records["discovery"], *period_records["calibration"]],
            feature_name=definition.name,
            orientation=orientation,
            thresholds=thresholds,
        )
        feature_rows.append(
            {
                "feature": definition.name,
                "group": definition.group,
                "description": definition.description,
                "unit": definition.unit,
                "alpha_eligible": definition.alpha_eligible,
                "orientation": orientation,
                "orientation_label": "higher_raw_is_better" if orientation == 1 else "lower_raw_is_better",
                "raw_discovery_high_minus_low_bps": raw_discovery_effect,
                "oriented_discovery_thresholds": thresholds,
                "preholdout_side_effect": combined_side,
                **periods,
            }
        )

    for period in ("discovery", "calibration", "holdout"):
        q_values = _benjamini_hochberg([row[period]["permutation_p_value"] for row in feature_rows])
        for row, q_value in zip(feature_rows, q_values, strict=True):
            row[period]["fdr_q_value"] = q_value

    for row in feature_rows:
        discovery = row["discovery"]
        calibration = row["calibration"]
        side = row["preholdout_side_effect"]
        robust_score = min(
            discovery["top_minus_bottom_bps"],
            calibration["top_minus_bottom_bps"],
            discovery["top_delta_vs_all_bps"],
            calibration["top_delta_vs_all_bps"],
        )
        stable = bool(
            row["alpha_eligible"]
            and discovery["coverage"] >= minimum_coverage
            and calibration["coverage"] >= minimum_coverage
            and discovery["top"]["trades"] >= minimum_discovery_tail
            and discovery["bottom"]["trades"] >= minimum_discovery_tail
            and calibration["top"]["trades"] >= minimum_calibration_tail
            and calibration["bottom"]["trades"] >= minimum_calibration_tail
            and discovery["spearman_rho"] > 0
            and calibration["spearman_rho"] > 0
            and robust_score > 0
            and discovery["top"]["positive_chronological_block_fraction"] >= 0.50
            and calibration["top"]["positive_chronological_block_fraction"] >= 0.50
            and side["long"]["top_trades"] >= 2
            and side["long"]["bottom_trades"] >= 2
            and side["short"]["top_trades"] >= 2
            and side["short"]["bottom_trades"] >= 2
            and side["long"]["top_minus_bottom_bps"] > 0
            and side["short"]["top_minus_bottom_bps"] > 0
        )
        row["robust_preholdout_score_bps"] = robust_score
        row["stable_preholdout"] = stable
        row["preholdout_fdr_supported"] = bool(
            discovery["fdr_q_value"] <= 0.20 and calibration["fdr_q_value"] <= 0.20
        )

    selected_by_group = _select_group_features(feature_rows)
    group_results = []
    for group, selection in sorted(selected_by_group.items()):
        row = next(item for item in feature_rows if item["feature"] == selection["feature"])
        top_records = _tail_records(
            period_records["holdout"],
            feature_name=row["feature"],
            orientation=row["orientation"],
            thresholds=row["oriented_discovery_thresholds"],
            want_top=True,
        )
        coin_control = _matched_coin_control(
            timeline,
            top_records,
            cost_bps=cost_bps,
            account_exposure=account_exposure,
            seeds=matched_coin_seeds,
            namespace=f"{symbol}:{row['feature']}",
        )
        verdict = _feature_verdict(selection, row["holdout"], coin_control)
        group_results.append(
            {
                **selection,
                "description": row["description"],
                "unit": row["unit"],
                "orientation": row["orientation"],
                "orientation_label": row["orientation_label"],
                "thresholds": row["oriented_discovery_thresholds"],
                "discovery": row["discovery"],
                "calibration": row["calibration"],
                "holdout": row["holdout"],
                "preholdout_side_effect": row["preholdout_side_effect"],
                "matched_coin_control": coin_control,
                "verdict": verdict,
                "holdout_top_records": [
                    {
                        "entry": _iso(record.opportunity.timestamp_ms),
                        "side": "long" if record.opportunity.side == 1 else "short",
                        "feature_value": record.values[row["feature"]],
                        "net_bps": record.trade.net_bps,
                        "exit_reason": record.trade.exit_reason,
                    }
                    for record in top_records
                ],
            }
        )
    group_results.sort(
        key=lambda row: (
            _verdict_rank(row["verdict"]),
            row["robust_preholdout_score_bps"],
            row["holdout"]["top_minus_bottom_bps"],
        ),
        reverse=True,
    )
    qualified = [row for row in group_results if row["selection_qualified"]]
    overall = max(
        qualified or group_results,
        key=lambda row: (
            row["robust_preholdout_score_bps"],
            row["calibration"]["top_minus_bottom_bps"],
        ),
    )
    correlation = _feature_correlation_report(
        period_records["discovery"],
        feature_rows,
        selected_features={row["feature"] for row in group_results},
        maximum_features=14,
    )

    return {
        "study": "nested_side_neutral_feature_value_analysis",
        "symbol": symbol,
        "parameters": {
            "round_trip_cost_bps": cost_bps,
            "sample_seconds": sample_seconds,
            "max_gap_seconds": max_gap_seconds,
            "discovery_fraction": discovery_fraction,
            "calibration_end_fraction": calibration_end_fraction,
            "horizon_seconds": horizon_seconds,
            "account_exposure_multiple": account_exposure,
            "minimum_coverage": minimum_coverage,
            "minimum_discovery_tail": minimum_discovery_tail,
            "minimum_calibration_tail": minimum_calibration_tail,
            "matched_coin_seeds": matched_coin_seeds,
            "permutation_samples": permutation_samples,
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
        "baseline": baseline,
        "features": [
            {
                "name": definition.name,
                "group": definition.group,
                "description": definition.description,
                "unit": definition.unit,
                "alpha_eligible": definition.alpha_eligible,
            }
            for definition in FEATURE_DEFINITIONS
        ],
        "feature_results": feature_rows,
        "group_results": group_results,
        "correlation": correlation,
        "overall_preholdout_selection": {
            "group": overall["group"],
            "feature": overall["feature"],
            "selection_qualified": overall["selection_qualified"],
            "robust_preholdout_score_bps": overall["robust_preholdout_score_bps"],
            "holdout_verdict": overall["verdict"],
        },
        "promotion_candidates": [
            {
                "group": row["group"],
                "feature": row["feature"],
                "verdict": row["verdict"],
            }
            for row in group_results
            if row["verdict"] == "robust_feature_candidate"
        ],
        "interpretation_notes": [
            "Directional features are multiplied by the squeeze side so higher values always mean stronger support for the chosen direction before feature orientation is learned.",
            "Raw feature orientation and top/bottom tercile cut points are frozen from discovery only; calibration must confirm the same sign before holdout can matter.",
            "One feature per economic group is selected pre-holdout. Holdout diagnostics for unselected features are exploratory and cannot promote them retrospectively.",
            "Stable selection requires coverage, positive rank and tail effects in discovery and calibration, positive chronological blocks, and positive long and short tail separation.",
            "Permutation p-values are corrected across all tested features with Benjamini-Hochberg FDR for each period.",
            "Feed-lag features are diagnostic-only and cannot be selected as alpha even if they correlate with outcomes.",
            "The sample remains small and the archive has informed prior work, so any candidate still requires fresh forward shadow confirmation.",
        ],
    }


def extract_feature_values(opportunity: ExitOpportunity) -> dict[str, float | None]:
    row = opportunity.row
    side = opportunity.side
    bid_refill = _finite_float(row.get("bid_depth_refill_rate_5s"))
    ask_refill = _finite_float(row.get("ask_depth_refill_rate_5s"))
    bid_wall = _finite_float(row.get("depth_bid_wall_ratio_top5"))
    ask_wall = _finite_float(row.get("depth_ask_wall_ratio_top5"))
    bid_depth = _finite_float(row.get("depth_bid_qty_top5"))
    ask_depth = _finite_float(row.get("depth_ask_qty_top5"))
    range_position = _finite_float(row.get("range_position_180s"))
    liquidation_ratio = _finite_float(row.get("liquidation_buy_ratio_30s"))
    taker_ratio = _finite_float(row.get("taker_buy_ratio_10s"))
    frames = _htf_frames(row)

    support_refill = bid_refill if side == 1 else ask_refill
    opposing_refill = ask_refill if side == 1 else bid_refill
    support_wall = bid_wall if side == 1 else ask_wall
    opposing_wall = ask_wall if side == 1 else bid_wall
    depth_total = bid_depth + ask_depth if bid_depth is not None and ask_depth is not None else None
    return {
        "signed_ofi_1s": _signed(row, "order_flow_imbalance_1s", side),
        "signed_ofi_5s": _signed(row, "order_flow_imbalance_5s", side),
        "signed_aggression_1s": _signed(row, "taker_aggression_imbalance_1s", side),
        "signed_aggression_5s": _signed(row, "taker_aggression_imbalance_5s", side),
        "signed_aggression_15s": _signed(row, "taker_aggression_imbalance_15s", side),
        "signed_taker_ratio_10s": side * (2.0 * taker_ratio - 1.0) if taker_ratio is not None else None,
        "signed_microprice_bps": _signed(row, "microprice_mid_bps", side),
        "signed_vamp_bps": _signed(row, "vamp_mid_bps", side),
        "signed_weighted_depth_bps": _signed(row, "weighted_depth_mid_bps", side),
        "signed_depth_imbalance": _signed(row, "depth_imbalance_top5", side),
        "signed_book_imbalance": _signed(row, "book_imbalance_top", side),
        "side_refill_advantage_5s": (
            support_refill - opposing_refill
            if support_refill is not None and opposing_refill is not None
            else None
        ),
        "side_wall_advantage": (
            support_wall - opposing_wall
            if support_wall is not None and opposing_wall is not None
            else None
        ),
        "signed_return_15s_bps": _signed(row, "return_15s_pct", side, multiplier=10_000.0),
        "signed_return_60s_bps": _signed(row, "return_60s_pct", side, multiplier=10_000.0),
        "signed_return_180s_bps": _signed(row, "return_180s_pct", side, multiplier=10_000.0),
        "breakout_edge_position": (
            (range_position if side == 1 else 1.0 - range_position)
            if range_position is not None
            else None
        ),
        "realized_vol_60s_bps": _scaled(row, "realized_vol_60s_pct", 10_000.0),
        "realized_vol_180s_bps": _scaled(row, "realized_vol_180s_pct", 10_000.0),
        "range_180s_bps": _scaled(row, "range_180s_pct", 10_000.0),
        "spread_bps": _finite_float(row.get("spread_bps")),
        "spread_max_5s_bps": _finite_float(row.get("spread_bps_max_5s")),
        "spread_std_5s_bps": _finite_float(row.get("spread_bps_std_5s")),
        "log_depth_qty_top5": math.log1p(depth_total) if depth_total is not None and depth_total >= 0 else None,
        "htf_trend_alignment": _htf_mean(frames, "trend_score", side=side),
        "htf_return_alignment_bps": _htf_mean(frames, "return_pct", side=side, multiplier=10_000.0),
        "htf_edge_position": _htf_edge_position(frames, side),
        "htf_target_room_bps": _htf_target_room(frames, side),
        "htf_taker_alignment": _htf_taker_alignment(frames, side),
        "signed_btc_ofi_1s": _signed(row, "btc_order_flow_imbalance_1s", side),
        "signed_btc_aggression_1s": _signed(row, "btc_taker_aggression_imbalance_1s", side),
        "signed_btc_microprice_bps": _signed(row, "btc_microprice_mid_bps", side),
        "signed_btc_return_15s_bps": _signed(row, "btc_return_15s_pct", side, multiplier=10_000.0),
        "signed_eth_btc_relative_15s_bps": _signed(
            row, "eth_btc_relative_return_15s_pct", side, multiplier=10_000.0
        ),
        "open_interest_change_5m_bps": _scaled(row, "open_interest_change_5m_pct", 10_000.0),
        "favorable_funding_carry_bps": _signed(row, "funding_rate", -side, multiplier=10_000.0),
        "signed_liquidation_imbalance": (
            side * (2.0 * liquidation_ratio - 1.0) if liquidation_ratio is not None else None
        ),
        "avg_book_lag_30s_ms": _finite_float(row.get("avg_book_event_lag_30s_ms")),
        "avg_trade_lag_30s_ms": _finite_float(row.get("avg_trade_event_lag_30s_ms")),
    }


def _discovery_orientation(records: list[FeatureRecord], feature_name: str) -> tuple[int, float]:
    pairs = [
        (value, record.trade.net_bps)
        for record in records
        if (value := record.values[feature_name]) is not None
    ]
    if len(pairs) < 3:
        return 1, 0.0
    values = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    returns = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
    low, high = np.quantile(values, [1 / 3, 2 / 3])
    low_returns = returns[values <= low]
    high_returns = returns[values >= high]
    effect = float(np.mean(high_returns) - np.mean(low_returns)) if len(low_returns) and len(high_returns) else 0.0
    return (1 if effect >= 0 else -1), effect


def _feature_period_metrics(
    records: list[FeatureRecord],
    *,
    feature_name: str,
    orientation: int,
    thresholds: dict[str, float],
    baseline_average_bps: float,
    account_exposure: float,
    permutation_samples: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    valid = [
        (record, value * orientation)
        for record in records
        if (value := record.values[feature_name]) is not None
    ]
    top = [record for record, value in valid if value >= thresholds["high_min"]]
    bottom = [record for record, value in valid if value <= thresholds["low_max"]]
    middle = [
        record
        for record, value in valid
        if thresholds["low_max"] < value < thresholds["high_min"]
    ]
    values = [value for _, value in valid]
    returns = [record.trade.net_bps for record, _ in valid]
    labels = [record.trade.net_bps > 0 for record, _ in valid]
    top_metrics = _compact_trade_metrics(
        summarize_exit_trades([record.trade for record in top], account_exposure=account_exposure)
    )
    bottom_metrics = _compact_trade_metrics(
        summarize_exit_trades([record.trade for record in bottom], account_exposure=account_exposure)
    )
    middle_metrics = _compact_trade_metrics(
        summarize_exit_trades([record.trade for record in middle], account_exposure=account_exposure)
    )
    effect = top_metrics["average_net_bps_per_trade"] - bottom_metrics["average_net_bps_per_trade"]
    return {
        "coverage": len(valid) / len(records) if records else 0.0,
        "valid_rows": len(valid),
        "spearman_rho": _spearman(values, returns),
        "positive_trade_auc": _auc(values, labels),
        "top_minus_bottom_bps": effect,
        "top_minus_bottom_bootstrap_95": _bootstrap_tail_effect(
            valid,
            thresholds=thresholds,
            samples=bootstrap_samples,
            seed=seed,
        ),
        "permutation_p_value": _permutation_tail_p_value(
            valid,
            thresholds=thresholds,
            samples=permutation_samples,
            seed=seed ^ 0x5A5A5A5A,
        ),
        "top_delta_vs_all_bps": top_metrics["average_net_bps_per_trade"] - baseline_average_bps,
        "top": top_metrics,
        "middle": middle_metrics,
        "bottom": bottom_metrics,
        "long_top_minus_bottom_bps": _side_tail_effect(top, bottom, 1),
        "short_top_minus_bottom_bps": _side_tail_effect(top, bottom, -1),
    }


def _combined_preholdout_side_effect(
    records: list[FeatureRecord],
    *,
    feature_name: str,
    orientation: int,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    valid = [
        (record, value * orientation)
        for record in records
        if (value := record.values[feature_name]) is not None
    ]
    top = [record for record, value in valid if value >= thresholds["high_min"]]
    bottom = [record for record, value in valid if value <= thresholds["low_max"]]
    output = {}
    for side, label in ((1, "long"), (-1, "short")):
        top_side = [record.trade.net_bps for record in top if record.opportunity.side == side]
        bottom_side = [record.trade.net_bps for record in bottom if record.opportunity.side == side]
        output[label] = {
            "top_trades": len(top_side),
            "bottom_trades": len(bottom_side),
            "top_average_net_bps": mean(top_side) if top_side else 0.0,
            "bottom_average_net_bps": mean(bottom_side) if bottom_side else 0.0,
            "top_minus_bottom_bps": (
                mean(top_side) - mean(bottom_side) if top_side and bottom_side else 0.0
            ),
        }
    return output


def _select_group_features(feature_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in feature_rows:
        if row["alpha_eligible"]:
            grouped.setdefault(row["group"], []).append(row)
    output = {}
    for group, rows in grouped.items():
        stable = [row for row in rows if row["stable_preholdout"]]
        qualified = [row for row in stable if row["preholdout_fdr_supported"]]
        selected = max(
            qualified or stable or rows,
            key=lambda row: (
                row["robust_preholdout_score_bps"],
                row["calibration"]["top_minus_bottom_bps"],
                row["calibration"]["spearman_rho"],
            ),
        )
        output[group] = {
            "group": group,
            "feature": selected["feature"],
            "robust_preholdout_score_bps": selected["robust_preholdout_score_bps"],
            "stable_preholdout": selected["stable_preholdout"],
            "preholdout_fdr_supported": selected["preholdout_fdr_supported"],
            "group_feature_count": len(rows),
            "stable_group_feature_count": len(stable),
            "fdr_supported_group_feature_count": len(qualified),
            "selection_qualified": bool(qualified),
        }
    return output


def _feature_verdict(
    selection: dict[str, Any],
    holdout: dict[str, Any],
    coin_control: dict[str, Any],
) -> str:
    if not selection["selection_qualified"]:
        return "selection_failed"
    if holdout["top_minus_bottom_bps"] <= 0 or holdout["top_delta_vs_all_bps"] <= 0:
        return "failed_holdout"
    both_sides_positive = bool(
        holdout["top"]["long"]["trades"] > 0
        and holdout["top"]["short"]["trades"] > 0
        and holdout["top"]["long"]["average_net_bps_per_trade"] > 0
        and holdout["top"]["short"]["average_net_bps_per_trade"] > 0
    )
    if not both_sides_positive:
        return "not_side_general"
    interval = holdout["top_minus_bottom_bootstrap_95"]
    if (
        interval is not None
        and interval[0] > 0
        and holdout["fdr_q_value"] <= 0.20
        and holdout["top"]["average_net_bps_per_trade"] > coin_control["p95_average_net_bps"]
    ):
        return "robust_feature_candidate"
    return "promising_but_unconfirmed"


def _tail_records(
    records: list[FeatureRecord],
    *,
    feature_name: str,
    orientation: int,
    thresholds: dict[str, float],
    want_top: bool,
) -> list[FeatureRecord]:
    output = []
    for record in records:
        value = record.values[feature_name]
        if value is None:
            continue
        oriented = value * orientation
        if want_top and oriented >= thresholds["high_min"]:
            output.append(record)
        elif not want_top and oriented <= thresholds["low_max"]:
            output.append(record)
    return output


def _matched_coin_control(
    timeline: PriceTimeline,
    top_records: list[FeatureRecord],
    *,
    cost_bps: float,
    account_exposure: float,
    seeds: int,
    namespace: str,
) -> dict[str, Any]:
    strategy_value = mean(record.trade.net_bps for record in top_records) if top_records else 0.0
    values = []
    for seed in range(seeds):
        cohort = [
            replace(
                record.opportunity,
                side=_coin_side(namespace, seed, record.opportunity.timestamp_ms),
            )
            for record in top_records
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


def _feature_correlation_report(
    discovery_records: list[FeatureRecord],
    feature_rows: list[dict[str, Any]],
    *,
    selected_features: set[str],
    maximum_features: int,
) -> dict[str, Any]:
    ordered = sorted(
        feature_rows,
        key=lambda row: (
            row["feature"] in selected_features,
            row["stable_preholdout"],
            row["robust_preholdout_score_bps"],
        ),
        reverse=True,
    )
    chosen = []
    for row in ordered:
        if row["feature"] not in chosen:
            chosen.append(row["feature"])
        if len(chosen) >= maximum_features:
            break
    matrix = []
    strong_pairs = []
    for first in chosen:
        matrix_row = []
        for second in chosen:
            pairs = [
                (record.values[first], record.values[second])
                for record in discovery_records
                if record.values[first] is not None and record.values[second] is not None
            ]
            correlation = (
                _spearman([pair[0] for pair in pairs], [pair[1] for pair in pairs])
                if len(pairs) >= 8
                else 0.0
            )
            matrix_row.append(correlation)
        matrix.append(matrix_row)
    for first_index, first in enumerate(chosen):
        for second_index in range(first_index + 1, len(chosen)):
            correlation = matrix[first_index][second_index]
            if abs(correlation) >= 0.80:
                strong_pairs.append(
                    {"first": first, "second": chosen[second_index], "spearman_rho": correlation}
                )
    return {
        "features": chosen,
        "spearman_matrix": matrix,
        "strong_pairs_abs_rho_gte_0_80": strong_pairs,
    }


def _compact_trade_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "trades": metrics["trades"],
        "wins": metrics["wins"],
        "losses": metrics["losses"],
        "positive_net_trade_rate": metrics["positive_net_trade_rate"],
        "average_net_bps_per_trade": metrics["average_net_bps_per_trade"],
        "total_net_bps": metrics["total_net_bps"],
        "max_additive_account_drawdown_pct": metrics["max_additive_account_drawdown_pct"],
        "positive_chronological_block_fraction": metrics["positive_chronological_block_fraction"],
        "long": metrics["by_side"]["long"],
        "short": metrics["by_side"]["short"],
    }


def _bootstrap_tail_effect(
    valid: list[tuple[FeatureRecord, float]],
    *,
    thresholds: dict[str, float],
    samples: int,
    seed: int,
) -> list[float] | None:
    if samples <= 0 or len(valid) < 4:
        return None
    flags_top = np.asarray([value >= thresholds["high_min"] for _, value in valid], dtype=bool)
    flags_bottom = np.asarray([value <= thresholds["low_max"] for _, value in valid], dtype=bool)
    returns = np.asarray([record.trade.net_bps for record, _ in valid], dtype=np.float64)
    if not np.any(flags_top) or not np.any(flags_bottom):
        return None
    rng = np.random.default_rng(seed)
    n = len(valid)
    block_size = min(5, n)
    boot = []
    attempts = 0
    while len(boot) < samples and attempts < samples * 5:
        attempts += 1
        indices = []
        while len(indices) < n:
            start = int(rng.integers(0, n))
            indices.extend((start + offset) % n for offset in range(block_size))
        chosen = np.asarray(indices[:n], dtype=np.int64)
        top = flags_top[chosen]
        bottom = flags_bottom[chosen]
        if not np.any(top) or not np.any(bottom):
            continue
        values = returns[chosen]
        boot.append(float(np.mean(values[top]) - np.mean(values[bottom])))
    return [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))] if boot else None


def _permutation_tail_p_value(
    valid: list[tuple[FeatureRecord, float]],
    *,
    thresholds: dict[str, float],
    samples: int,
    seed: int,
) -> float:
    if samples <= 0 or len(valid) < 4:
        return 1.0
    values = np.asarray([value for _, value in valid], dtype=np.float64)
    returns = np.asarray([record.trade.net_bps for record, _ in valid], dtype=np.float64)
    top = values >= thresholds["high_min"]
    bottom = values <= thresholds["low_max"]
    if not np.any(top) or not np.any(bottom):
        return 1.0
    observed = float(np.mean(returns[top]) - np.mean(returns[bottom]))
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(samples):
        shuffled = rng.permutation(returns)
        effect = float(np.mean(shuffled[top]) - np.mean(shuffled[bottom]))
        extreme += abs(effect) >= abs(observed)
    return (extreme + 1) / (samples + 1)


def _benjamini_hochberg(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    order = np.argsort(np.asarray(p_values, dtype=np.float64))
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running = 1.0
    for reverse_rank in range(len(order) - 1, -1, -1):
        index = int(order[reverse_rank])
        rank = reverse_rank + 1
        running = min(running, float(p_values[index]) * len(p_values) / rank)
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def _spearman(values: list[float], outcomes: list[float]) -> float:
    if len(values) < 3 or len(values) != len(outcomes):
        return 0.0
    first = _rankdata(np.asarray(values, dtype=np.float64))
    second = _rankdata(np.asarray(outcomes, dtype=np.float64))
    if np.std(first) <= 1e-12 or np.std(second) <= 1e-12:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0 + 1.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def _auc(values: list[float], labels: list[bool]) -> float:
    if len(values) < 2 or len(values) != len(labels):
        return 0.5
    ranks = _rankdata(np.asarray(values, dtype=np.float64))
    positive = np.asarray(labels, dtype=bool)
    positive_count = int(np.sum(positive))
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        return 0.5
    rank_sum = float(np.sum(ranks[positive]))
    return (rank_sum - positive_count * (positive_count + 1) / 2.0) / (positive_count * negative_count)


def _side_tail_effect(top: list[FeatureRecord], bottom: list[FeatureRecord], side: int) -> float:
    top_values = [record.trade.net_bps for record in top if record.opportunity.side == side]
    bottom_values = [record.trade.net_bps for record in bottom if record.opportunity.side == side]
    return mean(top_values) - mean(bottom_values) if top_values and bottom_values else 0.0


def _tercile_thresholds(values: list[float]) -> dict[str, float]:
    finite = [float(value) for value in values if np.isfinite(value)]
    if len(finite) < 3:
        return {"low_max": 0.0, "high_min": 0.0}
    return {
        "low_max": float(np.quantile(finite, 1 / 3)),
        "high_min": float(np.quantile(finite, 2 / 3)),
    }


def _htf_frames(row: dict[str, Any]) -> list[dict[str, Any]]:
    timeframes = ((row.get("higher_timeframe_context") or {}).get("timeframes") or {})
    return [timeframes.get(interval) or {} for interval in ("15m", "30m", "1h")]


def _htf_mean(
    frames: list[dict[str, Any]],
    key: str,
    *,
    side: int = 1,
    multiplier: float = 1.0,
) -> float | None:
    values = [value for frame in frames if (value := _finite_float(frame.get(key))) is not None]
    return side * mean(values) * multiplier if values else None


def _htf_edge_position(frames: list[dict[str, Any]], side: int) -> float | None:
    values = [value for frame in frames if (value := _finite_float(frame.get("range_position"))) is not None]
    if not values:
        return None
    return mean(values) if side == 1 else mean(1.0 - value for value in values)


def _htf_target_room(frames: list[dict[str, Any]], side: int) -> float | None:
    key = "resistance_distance_pct" if side == 1 else "support_distance_pct"
    values = [value for frame in frames if (value := _finite_float(frame.get(key))) is not None]
    return mean(values) * 10_000.0 if values else None


def _htf_taker_alignment(frames: list[dict[str, Any]], side: int) -> float | None:
    values = [value for frame in frames if (value := _finite_float(frame.get("taker_buy_ratio"))) is not None]
    return side * (2.0 * mean(values) - 1.0) if values else None


def _signed(row: dict[str, Any], key: str, side: int, multiplier: float = 1.0) -> float | None:
    value = _finite_float(row.get(key))
    return side * value * multiplier if value is not None else None


def _scaled(row: dict[str, Any], key: str, multiplier: float) -> float | None:
    value = _finite_float(row.get(key))
    return value * multiplier if value is not None else None


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
    from datetime import datetime, timezone

    return datetime.fromtimestamp(timestamp_ms / 1_000, tz=timezone.utc).isoformat()


def _verdict_rank(verdict: str) -> int:
    return {
        "robust_feature_candidate": 4,
        "promising_but_unconfirmed": 3,
        "not_side_general": 2,
        "failed_holdout": 1,
        "selection_failed": 0,
    }[verdict]
