from __future__ import annotations

import hashlib
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Sequence

import numpy as np

from futures_lab.first_touch import (
    PriceTimeline,
    _micro_signal,
    _side,
    _squeeze_breakout_signal,
    load_feature_samples,
    load_mark_price_timeline,
)
from futures_lab.strategy_tournament import _period_boundaries


@dataclass(frozen=True)
class ExitPolicy:
    family: str
    name: str
    params: dict[str, Any]


@dataclass(frozen=True)
class ExitOpportunity:
    timestamp_ms: int
    side: int
    row: dict[str, Any]
    entry_index: int
    end_index: int


@dataclass(frozen=True)
class ExitTrade:
    timestamp_ms: int
    side: int
    exit_ms: int
    exit_reason: str
    gross_bps: float
    net_bps: float
    mfe_bps: float
    mae_bps: float
    duration_seconds: float
    partial_taken: bool = False


REFERENCE_POLICY = ExitPolicy(
    family="fixed_barrier",
    name="fixed_100_150_6h",
    params={"target_bps": 100.0, "stop_bps": 150.0, "horizon_seconds": 21_600},
)


def default_exit_policies() -> tuple[ExitPolicy, ...]:
    policies = [
        _policy("fixed_barrier", "fixed_60_60_6h", target_bps=60.0, stop_bps=60.0),
        _policy("fixed_barrier", "fixed_60_80_6h", target_bps=60.0, stop_bps=80.0),
        _policy("fixed_barrier", "fixed_80_150_6h", target_bps=80.0, stop_bps=150.0),
        REFERENCE_POLICY,
    ]
    for decay_seconds, minimum_mfe_bps in ((900, 20.0), (1_800, 20.0), (1_800, 30.0), (3_600, 30.0)):
        policies.append(
            _policy(
                "conditional_time_decay",
                f"time_decay_{decay_seconds}s_mfe{minimum_mfe_bps:.0f}",
                target_bps=100.0,
                stop_bps=150.0,
                decay_seconds=decay_seconds,
                minimum_mfe_bps=minimum_mfe_bps,
            )
        )
    for activation_bps, lock_gross_bps in ((25.0, 10.0), (40.0, 10.0), (40.0, 15.0), (60.0, 15.0)):
        policies.append(
            _policy(
                "fee_breakeven_trail",
                f"fee_trail_activate{activation_bps:.0f}_lock{lock_gross_bps:.0f}",
                target_bps=100.0,
                stop_bps=150.0,
                activation_bps=activation_bps,
                lock_gross_bps=lock_gross_bps,
            )
        )
    for activation_bps, retrace_bps in ((40.0, 20.0), (40.0, 30.0), (60.0, 20.0), (60.0, 30.0)):
        policies.append(
            _policy(
                "mfe_trailing",
                f"mfe_trail_activate{activation_bps:.0f}_retrace{retrace_bps:.0f}",
                target_bps=100.0,
                stop_bps=150.0,
                activation_bps=activation_bps,
                retrace_bps=retrace_bps,
            )
        )
    for minimum_hold_seconds, invalidation_threshold in ((60, 0.0), (60, -0.05), (180, 0.0), (180, -0.05)):
        threshold_token = f"m{abs(invalidation_threshold):.2f}" if invalidation_threshold < 0 else "zero"
        policies.append(
            _policy(
                "signal_invalidation",
                f"micro_invalidation_{minimum_hold_seconds}s_{threshold_token}",
                target_bps=100.0,
                stop_bps=150.0,
                minimum_hold_seconds=minimum_hold_seconds,
                invalidation_threshold=invalidation_threshold,
            )
        )
    for volatility_multiplier, stop_ratio in ((1.5, 1.25), (1.5, 1.5), (2.0, 1.25), (2.0, 1.5)):
        policies.append(
            _policy(
                "volatility_scaled",
                f"vol_scaled_{volatility_multiplier:.1f}x_stop{stop_ratio:.2f}",
                volatility_multiplier=volatility_multiplier,
                stop_ratio=stop_ratio,
                minimum_target_bps=40.0,
                maximum_target_bps=120.0,
                minimum_stop_bps=50.0,
                maximum_stop_bps=180.0,
            )
        )
    for partial_target_bps, move_to_breakeven in ((40.0, False), (40.0, True), (60.0, False), (60.0, True)):
        suffix = "be" if move_to_breakeven else "original_stop"
        policies.append(
            _policy(
                "partial_profit",
                f"partial50_at{partial_target_bps:.0f}_{suffix}",
                target_bps=100.0,
                stop_bps=150.0,
                partial_target_bps=partial_target_bps,
                partial_fraction=0.50,
                move_to_breakeven=move_to_breakeven,
            )
        )
    return tuple(policies)


def run_exit_laboratory(
    runs_root: Path,
    *,
    symbol: str = "ETHUSDT",
    policies: Iterable[ExitPolicy] | None = None,
    cost_bps: float = 10.0,
    sample_seconds: int = 60,
    max_gap_seconds: int = 5,
    discovery_fraction: float = 0.50,
    calibration_end_fraction: float = 0.70,
    horizon_seconds: int = 21_600,
    account_exposure: float = 4.95,
    minimum_discovery_trades: int = 20,
    minimum_calibration_trades: int = 10,
    matched_coin_seeds: int = 32,
    bootstrap_samples: int = 2_000,
) -> dict[str, Any]:
    selected_policies = tuple(policies or default_exit_policies())
    if not selected_policies:
        raise ValueError("At least one exit policy is required")
    if REFERENCE_POLICY.name not in {policy.name for policy in selected_policies}:
        raise ValueError(f"The locked reference policy {REFERENCE_POLICY.name} must be included")
    if cost_bps < 0 or horizon_seconds <= 0 or sample_seconds <= 0:
        raise ValueError("Exit-lab cost, horizon, and sampling settings must be valid")
    if matched_coin_seeds < 8 or bootstrap_samples < 100:
        raise ValueError("Exit-lab controls require at least 8 coin seeds and 100 bootstrap samples")

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
    feature_times = [timestamp for timestamp, _ in features]
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

    trade_cache: dict[tuple[str, str], list[ExitTrade]] = {}
    metrics_cache: dict[tuple[str, str], dict[str, Any]] = {}
    for period, cohort in cohorts.items():
        for policy in selected_policies:
            trades = evaluate_policy(
                timeline,
                cohort,
                policy,
                features=features,
                feature_times=feature_times,
                cost_bps=cost_bps,
            )
            trade_cache[(period, policy.name)] = trades
            metrics_cache[(period, policy.name)] = summarize_exit_trades(
                trades,
                account_exposure=account_exposure,
            )

    candidates = []
    for policy in selected_policies:
        discovery = metrics_cache[("discovery", policy.name)]
        calibration = metrics_cache[("calibration", policy.name)]
        candidates.append(
            _selection_candidate(
                policy,
                discovery,
                calibration,
                minimum_discovery_trades=minimum_discovery_trades,
                minimum_calibration_trades=minimum_calibration_trades,
            )
        )
    selected_by_family = _select_exit_candidates(candidates)

    baseline_holdout_trades = trade_cache[("holdout", REFERENCE_POLICY.name)]
    baseline_holdout = metrics_cache[("holdout", REFERENCE_POLICY.name)]
    family_results = []
    for family, selection in sorted(selected_by_family.items()):
        policy = next(policy for policy in selected_policies if policy.name == selection["policy"])
        holdout_trades = trade_cache[("holdout", policy.name)]
        holdout = metrics_cache[("holdout", policy.name)]
        paired = _paired_bootstrap_delta(
            holdout_trades,
            baseline_holdout_trades,
            samples=bootstrap_samples,
            seed=_stable_seed(f"{symbol}:{policy.name}:paired"),
        )
        controls = _matched_coin_controls(
            timeline,
            cohorts["holdout"],
            policy,
            features=features,
            feature_times=feature_times,
            cost_bps=cost_bps,
            account_exposure=account_exposure,
            seeds=matched_coin_seeds,
            namespace=symbol,
        )
        verdict = _exit_verdict(selection, holdout, paired, controls)
        family_results.append(
            {
                **selection,
                "holdout": _compact_metrics(holdout),
                "paired_vs_reference": paired,
                "matched_coin_control": controls,
                "verdict": verdict,
                "holdout_trades": [_trade_row(trade) for trade in holdout_trades],
            }
        )

    family_results.sort(
        key=lambda row: (
            _verdict_rank(row["verdict"]),
            row["robust_floor_bps"],
            row["holdout"]["average_net_bps_per_trade"],
        ),
        reverse=True,
    )
    qualified = [row for row in family_results if row["selection_qualified"]]
    overall = max(
        qualified or family_results,
        key=lambda row: (row["robust_floor_bps"], row["calibration"]["average_net_bps_per_trade"]),
    )

    return {
        "study": "locked_entry_exit_laboratory",
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
            "matched_coin_seeds": matched_coin_seeds,
            "bootstrap_samples": bootstrap_samples,
            "entry_rule": "existing side-symmetric volatility_squeeze_breakout",
            "reference_policy": REFERENCE_POLICY.name,
        },
        "data": {
            **mark_stats,
            **feature_stats,
            "calibration_start": _iso(calibration_start_ms),
            "holdout_start": _iso(holdout_start_ms),
            "feature_exit_cadence_seconds": sample_seconds,
        },
        "entry_cohorts": cohort_stats,
        "policies": [
            {"family": policy.family, "name": policy.name, "params": policy.params}
            for policy in selected_policies
        ],
        "selection_candidates": candidates,
        "reference_baseline": {
            "policy": REFERENCE_POLICY.name,
            "discovery": _compact_metrics(metrics_cache[("discovery", REFERENCE_POLICY.name)]),
            "calibration": _compact_metrics(metrics_cache[("calibration", REFERENCE_POLICY.name)]),
            "holdout": _compact_metrics(baseline_holdout),
            "holdout_trades": [_trade_row(trade) for trade in baseline_holdout_trades],
        },
        "family_results": family_results,
        "overall_pre_holdout_selection": {
            "family": overall["family"],
            "policy": overall["policy"],
            "selection_qualified": overall["selection_qualified"],
            "robust_floor_bps": overall["robust_floor_bps"],
            "holdout_verdict": overall["verdict"],
        },
        "interpretation_notes": [
            "The known squeeze rule and fixed 100/150/6h ledger lock non-overlapping entry timestamps before exit policies are compared.",
            "Every primary policy metric therefore uses the same opportunities; faster exits are not rewarded with extra recycled entries.",
            "Price barriers and trails use dense mark-price data. Signal invalidation can react only at the stored feature cadence.",
            "Policy selection uses discovery plus calibration only. Holdout is evaluated once per selected family with a horizon purge.",
            "Matched coin controls randomize direction at the same timestamps; paired block bootstrap compares each policy with the reference trade by trade.",
            "This archive has already informed earlier research, so these are retrospective development results and require fresh forward confirmation.",
            "Mark-price labels and fixed costs do not reproduce live taker slippage, order latency, partial fills, funding, or liquidation mechanics.",
        ],
    }


def lock_reference_cohort(
    timeline: PriceTimeline,
    features: list[tuple[int, dict[str, Any]]],
    *,
    cost_bps: float,
    horizon_seconds: int,
) -> tuple[list[ExitOpportunity], dict[str, Any]]:
    signals = 0
    skipped_overlap = 0
    incomplete = 0
    no_price = 0
    next_free_ms = -1
    cohort: list[ExitOpportunity] = []
    for timestamp_ms, row in features:
        side = _side(_squeeze_breakout_signal(row))
        if side not in {-1, 1}:
            continue
        signals += 1
        if timestamp_ms < next_free_ms:
            skipped_overlap += 1
            continue
        entry_index = timeline.entry_index(timestamp_ms)
        if entry_index is None:
            no_price += 1
            continue
        end_index, complete = timeline.horizon_end(entry_index, horizon_seconds)
        if not complete or end_index <= entry_index:
            incomplete += 1
            continue
        opportunity = ExitOpportunity(
            timestamp_ms=timestamp_ms,
            side=side,
            row=row,
            entry_index=entry_index,
            end_index=end_index,
        )
        reference_trade = simulate_exit_policy(
            timeline,
            opportunity,
            REFERENCE_POLICY,
            features=(),
            feature_times=(),
            cost_bps=cost_bps,
        )
        cohort.append(opportunity)
        next_free_ms = reference_trade.exit_ms
    return cohort, {
        "signals": signals,
        "locked_entries": len(cohort),
        "skipped_overlapping_signals": skipped_overlap,
        "incomplete_horizons": incomplete,
        "missing_entry_prices": no_price,
        "long_entries": sum(row.side == 1 for row in cohort),
        "short_entries": sum(row.side == -1 for row in cohort),
        "start": _iso(cohort[0].timestamp_ms) if cohort else None,
        "end": _iso(cohort[-1].timestamp_ms) if cohort else None,
    }


def evaluate_policy(
    timeline: PriceTimeline,
    cohort: list[ExitOpportunity],
    policy: ExitPolicy,
    *,
    features: Sequence[tuple[int, dict[str, Any]]] = (),
    feature_times: Sequence[int] = (),
    cost_bps: float,
) -> list[ExitTrade]:
    return [
        simulate_exit_policy(
            timeline,
            opportunity,
            policy,
            features=features,
            feature_times=feature_times,
            cost_bps=cost_bps,
        )
        for opportunity in cohort
    ]


def simulate_exit_policy(
    timeline: PriceTimeline,
    opportunity: ExitOpportunity,
    policy: ExitPolicy,
    *,
    features: Sequence[tuple[int, dict[str, Any]]],
    feature_times: Sequence[int],
    cost_bps: float,
) -> ExitTrade:
    times = timeline.times_ms[opportunity.entry_index + 1 : opportunity.end_index + 1]
    prices = timeline.prices[opportunity.entry_index + 1 : opportunity.end_index + 1]
    entry = float(timeline.prices[opportunity.entry_index])
    moves = ((prices / entry) - 1.0) * opportunity.side * 10_000.0
    if len(moves) == 0:
        raise ValueError("Exit opportunity has no price path")

    family = policy.family
    params = policy.params
    if family == "fixed_barrier":
        return _fixed_trade(opportunity, times, moves, params["target_bps"], params["stop_bps"], cost_bps)
    if family == "conditional_time_decay":
        return _time_decay_trade(opportunity, times, moves, params, cost_bps)
    if family == "fee_breakeven_trail":
        return _fee_trail_trade(opportunity, times, moves, params, cost_bps)
    if family == "mfe_trailing":
        return _mfe_trail_trade(opportunity, times, moves, params, cost_bps)
    if family == "signal_invalidation":
        return _signal_invalidation_trade(
            timeline,
            opportunity,
            times,
            moves,
            params,
            features,
            feature_times,
            cost_bps,
        )
    if family == "volatility_scaled":
        target_bps, stop_bps = _volatility_barriers(opportunity.row, params)
        return _fixed_trade(opportunity, times, moves, target_bps, stop_bps, cost_bps)
    if family == "partial_profit":
        return _partial_profit_trade(opportunity, times, moves, params, cost_bps)
    raise ValueError(f"Unsupported exit policy family: {family}")


def summarize_exit_trades(trades: list[ExitTrade], *, account_exposure: float) -> dict[str, Any]:
    net = [trade.net_bps for trade in trades]
    gross = [trade.gross_bps for trade in trades]
    account_returns = [value / 100.0 * account_exposure for value in net]
    reasons = Counter(trade.exit_reason for trade in trades)
    blocks = []
    if trades:
        for index, chunk in enumerate(np.array_split(np.asarray(trades, dtype=object), min(4, len(trades))), start=1):
            rows = list(chunk.tolist())
            blocks.append(
                {
                    "block": index,
                    "start": _iso(rows[0].timestamp_ms),
                    "end": _iso(rows[-1].timestamp_ms),
                    **_trade_breakdown(rows),
                }
            )
    positive = [value for value in net if value > 0]
    negative = [value for value in net if value < 0]
    by_side = {
        "long": _trade_breakdown([trade for trade in trades if trade.side == 1]),
        "short": _trade_breakdown([trade for trade in trades if trade.side == -1]),
    }
    return {
        "trades": len(trades),
        "wins": len(positive),
        "losses": len(negative),
        "flat": len(trades) - len(positive) - len(negative),
        "positive_net_trade_rate": len(positive) / len(trades) if trades else 0.0,
        "average_gross_bps_per_trade": mean(gross) if gross else 0.0,
        "average_net_bps_per_trade": mean(net) if net else 0.0,
        "median_net_bps_per_trade": median(net) if net else 0.0,
        "total_net_bps": sum(net),
        "profit_factor": sum(positive) / abs(sum(negative)) if negative else None,
        "average_duration_seconds": mean(trade.duration_seconds for trade in trades) if trades else 0.0,
        "average_mfe_bps": mean(trade.mfe_bps for trade in trades) if trades else 0.0,
        "average_mae_bps": mean(trade.mae_bps for trade in trades) if trades else 0.0,
        "maximum_adverse_excursion_bps": min((trade.mae_bps for trade in trades), default=0.0),
        "mfe_capture_ratio": (
            sum(max(0.0, trade.gross_bps) for trade in trades) / sum(trade.mfe_bps for trade in trades)
            if sum(trade.mfe_bps for trade in trades) > 0
            else 0.0
        ),
        "partial_trade_count": sum(trade.partial_taken for trade in trades),
        "exit_reasons": dict(sorted(reasons.items())),
        "additive_account_return_pct": sum(account_returns),
        "max_additive_account_drawdown_pct": _max_drawdown(account_returns),
        "max_losing_streak": _max_losing_streak(net),
        "positive_chronological_block_fraction": (
            sum(block["average_net_bps_per_trade"] > 0 for block in blocks) / len(blocks) if blocks else 0.0
        ),
        "by_side": by_side,
        "chronological_trade_blocks": blocks,
    }


def _fixed_trade(
    opportunity: ExitOpportunity,
    times: np.ndarray,
    moves: np.ndarray,
    target_bps: float,
    stop_bps: float,
    cost_bps: float,
) -> ExitTrade:
    target_index = _first_true(moves >= target_bps)
    stop_index = _first_true(moves <= -stop_bps)
    event = _first_event((target_index, "take_profit", target_bps), (stop_index, "price_stop", -stop_bps))
    if event is None:
        return _make_trade(opportunity, times, moves, len(moves) - 1, "vertical_timeout", float(moves[-1]), cost_bps)
    index, reason, gross_bps = event
    return _make_trade(opportunity, times, moves, index, reason, gross_bps, cost_bps)


def _time_decay_trade(
    opportunity: ExitOpportunity,
    times: np.ndarray,
    moves: np.ndarray,
    params: dict[str, Any],
    cost_bps: float,
) -> ExitTrade:
    target_index = _first_true(moves >= params["target_bps"])
    stop_index = _first_true(moves <= -params["stop_bps"])
    decay_ms = opportunity.timestamp_ms + int(params["decay_seconds"] * 1_000)
    decay_index = bisect_left(times, decay_ms)
    decay_event = None
    if decay_index < len(moves) and float(np.max(moves[: decay_index + 1])) < params["minimum_mfe_bps"]:
        decay_event = (decay_index, "time_decay", float(moves[decay_index]))
    event = _first_event(
        (target_index, "take_profit", params["target_bps"]),
        (stop_index, "price_stop", -params["stop_bps"]),
        decay_event,
    )
    if event is None:
        return _make_trade(opportunity, times, moves, len(moves) - 1, "vertical_timeout", float(moves[-1]), cost_bps)
    index, reason, gross_bps = event
    return _make_trade(opportunity, times, moves, index, reason, gross_bps, cost_bps)


def _fee_trail_trade(
    opportunity: ExitOpportunity,
    times: np.ndarray,
    moves: np.ndarray,
    params: dict[str, Any],
    cost_bps: float,
) -> ExitTrade:
    target_index = _first_true(moves >= params["target_bps"])
    stop_index = _first_true(moves <= -params["stop_bps"])
    activation_index = _first_true(moves >= params["activation_bps"])
    trail_index = None
    if activation_index is not None and activation_index + 1 < len(moves):
        offset = _first_true(moves[activation_index + 1 :] <= params["lock_gross_bps"])
        trail_index = activation_index + 1 + offset if offset is not None else None
    event = _first_event(
        (target_index, "take_profit", params["target_bps"]),
        (trail_index, "fee_breakeven_trail", params["lock_gross_bps"]),
        (stop_index, "price_stop", -params["stop_bps"]),
    )
    if event is None:
        return _make_trade(opportunity, times, moves, len(moves) - 1, "vertical_timeout", float(moves[-1]), cost_bps)
    index, reason, gross_bps = event
    return _make_trade(opportunity, times, moves, index, reason, gross_bps, cost_bps)


def _mfe_trail_trade(
    opportunity: ExitOpportunity,
    times: np.ndarray,
    moves: np.ndarray,
    params: dict[str, Any],
    cost_bps: float,
) -> ExitTrade:
    target_index = _first_true(moves >= params["target_bps"])
    stop_index = _first_true(moves <= -params["stop_bps"])
    running_mfe = np.maximum.accumulate(moves)
    trail_mask = (running_mfe >= params["activation_bps"]) & (moves <= running_mfe - params["retrace_bps"])
    trail_index = _first_true(trail_mask)
    trail_gross = float(moves[trail_index]) if trail_index is not None else 0.0
    event = _first_event(
        (target_index, "take_profit", params["target_bps"]),
        (trail_index, "mfe_trailing_stop", trail_gross),
        (stop_index, "price_stop", -params["stop_bps"]),
    )
    if event is None:
        return _make_trade(opportunity, times, moves, len(moves) - 1, "vertical_timeout", float(moves[-1]), cost_bps)
    index, reason, gross_bps = event
    return _make_trade(opportunity, times, moves, index, reason, gross_bps, cost_bps)


def _signal_invalidation_trade(
    timeline: PriceTimeline,
    opportunity: ExitOpportunity,
    times: np.ndarray,
    moves: np.ndarray,
    params: dict[str, Any],
    features: Sequence[tuple[int, dict[str, Any]]],
    feature_times: Sequence[int],
    cost_bps: float,
) -> ExitTrade:
    target_index = _first_true(moves >= params["target_bps"])
    stop_index = _first_true(moves <= -params["stop_bps"])
    invalidation_index = None
    start_ms = opportunity.timestamp_ms + int(params["minimum_hold_seconds"] * 1_000)
    feature_index = bisect_left(feature_times, start_ms)
    while feature_index < len(features):
        timestamp_ms, row = features[feature_index]
        if timestamp_ms > int(times[-1]):
            break
        if opportunity.side * _micro_signal(row) <= params["invalidation_threshold"]:
            mark_index = timeline.entry_index(timestamp_ms)
            if mark_index is not None:
                relative_index = mark_index - opportunity.entry_index - 1
                if 0 <= relative_index < len(moves):
                    invalidation_index = relative_index
                    break
        feature_index += 1
    invalidation_gross = float(moves[invalidation_index]) if invalidation_index is not None else 0.0
    event = _first_event(
        (target_index, "take_profit", params["target_bps"]),
        (invalidation_index, "signal_invalidation", invalidation_gross),
        (stop_index, "price_stop", -params["stop_bps"]),
    )
    if event is None:
        return _make_trade(opportunity, times, moves, len(moves) - 1, "vertical_timeout", float(moves[-1]), cost_bps)
    index, reason, gross_bps = event
    return _make_trade(opportunity, times, moves, index, reason, gross_bps, cost_bps)


def _partial_profit_trade(
    opportunity: ExitOpportunity,
    times: np.ndarray,
    moves: np.ndarray,
    params: dict[str, Any],
    cost_bps: float,
) -> ExitTrade:
    stop_index = _first_true(moves <= -params["stop_bps"])
    partial_index = _first_true(moves >= params["partial_target_bps"])
    if partial_index is None or (stop_index is not None and stop_index <= partial_index):
        if stop_index is not None:
            return _make_trade(
                opportunity,
                times,
                moves,
                stop_index,
                "price_stop",
                -params["stop_bps"],
                cost_bps,
            )
        return _make_trade(opportunity, times, moves, len(moves) - 1, "vertical_timeout", float(moves[-1]), cost_bps)

    fraction = params["partial_fraction"]
    if moves[partial_index] >= params["target_bps"]:
        blended_gross = (
            fraction * params["partial_target_bps"]
            + (1.0 - fraction) * params["target_bps"]
        )
        return _make_trade(
            opportunity,
            times,
            moves,
            partial_index,
            "partial_then_target",
            blended_gross,
            cost_bps,
            partial_taken=True,
        )
    remainder_stop_bps = 0.0 if params["move_to_breakeven"] else -params["stop_bps"]
    target_offset = _first_true(moves[partial_index + 1 :] >= params["target_bps"])
    stop_offset = _first_true(moves[partial_index + 1 :] <= remainder_stop_bps)
    target_index = partial_index + 1 + target_offset if target_offset is not None else None
    remainder_stop_index = partial_index + 1 + stop_offset if stop_offset is not None else None
    event = _first_event(
        (target_index, "partial_then_target", params["target_bps"]),
        (
            remainder_stop_index,
            "partial_then_breakeven" if params["move_to_breakeven"] else "partial_then_stop",
            remainder_stop_bps,
        ),
    )
    if event is None:
        exit_index = len(moves) - 1
        remainder_gross = float(moves[exit_index])
        reason = "partial_then_timeout"
    else:
        exit_index, reason, remainder_gross = event
    blended_gross = fraction * params["partial_target_bps"] + (1.0 - fraction) * remainder_gross
    return _make_trade(
        opportunity,
        times,
        moves,
        exit_index,
        reason,
        blended_gross,
        cost_bps,
        partial_taken=True,
    )


def _volatility_barriers(row: dict[str, Any], params: dict[str, Any]) -> tuple[float, float]:
    realized = _finite_float(row.get("realized_vol_180s_pct"))
    range_pct = _finite_float(row.get("range_180s_pct"))
    volatility_bps = max(abs(realized or 0.0), abs(range_pct or 0.0) / 3.0) * 10_000.0
    target_bps = min(
        params["maximum_target_bps"],
        max(params["minimum_target_bps"], volatility_bps * params["volatility_multiplier"]),
    )
    stop_bps = min(
        params["maximum_stop_bps"],
        max(params["minimum_stop_bps"], target_bps * params["stop_ratio"]),
    )
    return float(target_bps), float(stop_bps)


def _selection_candidate(
    policy: ExitPolicy,
    discovery: dict[str, Any],
    calibration: dict[str, Any],
    *,
    minimum_discovery_trades: int,
    minimum_calibration_trades: int,
) -> dict[str, Any]:
    discovery_compact = _compact_metrics(discovery)
    calibration_compact = _compact_metrics(calibration)
    robust_floor = min(
        discovery_compact["average_net_bps_per_trade"],
        calibration_compact["average_net_bps_per_trade"],
    )
    qualifies = bool(
        discovery_compact["trades"] >= minimum_discovery_trades
        and calibration_compact["trades"] >= minimum_calibration_trades
        and robust_floor > 0
        and discovery_compact["positive_chronological_block_fraction"] >= 0.50
        and calibration_compact["positive_chronological_block_fraction"] >= 0.50
    )
    return {
        "family": policy.family,
        "policy": policy.name,
        "params": policy.params,
        "qualifies": qualifies,
        "robust_floor_bps": robust_floor,
        "discovery": discovery_compact,
        "calibration": calibration_compact,
    }


def _select_exit_candidates(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate["family"]].append(candidate)
    output = {}
    for family, rows in grouped.items():
        qualified = [row for row in rows if row["qualifies"]]
        selected = max(
            qualified or rows,
            key=lambda row: (
                row["robust_floor_bps"],
                row["calibration"]["average_net_bps_per_trade"],
                -row["calibration"]["max_additive_account_drawdown_pct"],
            ),
        )
        output[family] = {
            **selected,
            "family_candidate_count": len(rows),
            "qualified_candidate_count": len(qualified),
            "selection_qualified": bool(qualified),
        }
    return output


def _matched_coin_controls(
    timeline: PriceTimeline,
    cohort: list[ExitOpportunity],
    policy: ExitPolicy,
    *,
    features: list[tuple[int, dict[str, Any]]],
    feature_times: list[int],
    cost_bps: float,
    account_exposure: float,
    seeds: int,
    namespace: str,
) -> dict[str, Any]:
    strategy_trades = evaluate_policy(
        timeline,
        cohort,
        policy,
        features=features,
        feature_times=feature_times,
        cost_bps=cost_bps,
    )
    strategy_value = summarize_exit_trades(strategy_trades, account_exposure=account_exposure)[
        "average_net_bps_per_trade"
    ]
    values = []
    for seed in range(seeds):
        coin_cohort = [
            replace(
                opportunity,
                side=_coin_side(namespace, seed, opportunity.timestamp_ms),
            )
            for opportunity in cohort
        ]
        trades = evaluate_policy(
            timeline,
            coin_cohort,
            policy,
            features=features,
            feature_times=feature_times,
            cost_bps=cost_bps,
        )
        values.append(summarize_exit_trades(trades, account_exposure=account_exposure)["average_net_bps_per_trade"])
    values.sort()
    return {
        "seeds": seeds,
        "median_average_net_bps": median(values),
        "p05_average_net_bps": float(np.quantile(values, 0.05)),
        "p95_average_net_bps": float(np.quantile(values, 0.95)),
        "positive_seed_fraction": mean(value > 0 for value in values),
        "seed_fraction_at_or_above_strategy": mean(value >= strategy_value for value in values),
    }


def _paired_bootstrap_delta(
    policy_trades: list[ExitTrade],
    baseline_trades: list[ExitTrade],
    *,
    samples: int,
    seed: int,
    block_size: int = 5,
) -> dict[str, Any]:
    if len(policy_trades) != len(baseline_trades):
        raise ValueError("Paired exit comparison requires identical entry cohorts")
    if not policy_trades:
        return {
            "mean_delta_bps": 0.0,
            "bootstrap_p025_bps": 0.0,
            "bootstrap_p975_bps": 0.0,
            "bootstrap_probability_positive": 0.0,
            "trades": 0,
        }
    deltas = np.asarray(
        [policy.net_bps - baseline.net_bps for policy, baseline in zip(policy_trades, baseline_trades, strict=True)],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    n = len(deltas)
    size = min(block_size, n)
    boot = np.empty(samples, dtype=np.float64)
    for sample_index in range(samples):
        collected = []
        while len(collected) < n:
            start = int(rng.integers(0, n))
            collected.extend(deltas[(start + offset) % n] for offset in range(size))
        boot[sample_index] = float(np.mean(collected[:n]))
    return {
        "mean_delta_bps": float(np.mean(deltas)),
        "median_delta_bps": float(np.median(deltas)),
        "bootstrap_p025_bps": float(np.quantile(boot, 0.025)),
        "bootstrap_p975_bps": float(np.quantile(boot, 0.975)),
        "bootstrap_probability_positive": float(np.mean(boot > 0)),
        "trades": n,
        "block_size": size,
        "policy_better_trade_fraction": float(np.mean(deltas > 0)),
    }


def _exit_verdict(
    selection: dict[str, Any],
    holdout: dict[str, Any],
    paired: dict[str, Any],
    controls: dict[str, Any],
) -> str:
    if not selection["selection_qualified"]:
        return "selection_failed"
    if holdout["average_net_bps_per_trade"] <= 0:
        return "failed_holdout"
    both_sides_positive = bool(
        holdout["by_side"]["long"]["trades"] > 0
        and holdout["by_side"]["short"]["trades"] > 0
        and holdout["by_side"]["long"]["average_net_bps_per_trade"] > 0
        and holdout["by_side"]["short"]["average_net_bps_per_trade"] > 0
    )
    if (
        paired["bootstrap_p025_bps"] > 0
        and holdout["average_net_bps_per_trade"] > controls["p95_average_net_bps"]
        and both_sides_positive
    ):
        return "robust_exit_improvement"
    if paired["mean_delta_bps"] > 0 and both_sides_positive:
        return "promising_but_unconfirmed"
    return "profitable_without_reference_improvement"


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "trades": metrics["trades"],
        "wins": metrics["wins"],
        "losses": metrics["losses"],
        "positive_net_trade_rate": metrics["positive_net_trade_rate"],
        "average_gross_bps_per_trade": metrics["average_gross_bps_per_trade"],
        "average_net_bps_per_trade": metrics["average_net_bps_per_trade"],
        "median_net_bps_per_trade": metrics["median_net_bps_per_trade"],
        "total_net_bps": metrics["total_net_bps"],
        "profit_factor": metrics["profit_factor"],
        "average_duration_seconds": metrics["average_duration_seconds"],
        "average_mfe_bps": metrics["average_mfe_bps"],
        "average_mae_bps": metrics["average_mae_bps"],
        "maximum_adverse_excursion_bps": metrics["maximum_adverse_excursion_bps"],
        "mfe_capture_ratio": metrics["mfe_capture_ratio"],
        "partial_trade_count": metrics["partial_trade_count"],
        "exit_reasons": metrics["exit_reasons"],
        "additive_account_return_pct": metrics["additive_account_return_pct"],
        "max_additive_account_drawdown_pct": metrics["max_additive_account_drawdown_pct"],
        "max_losing_streak": metrics["max_losing_streak"],
        "positive_chronological_block_fraction": metrics["positive_chronological_block_fraction"],
        "long": metrics["by_side"]["long"],
        "short": metrics["by_side"]["short"],
        "chronological_trade_blocks": metrics["chronological_trade_blocks"],
    }


def _trade_breakdown(trades: list[ExitTrade]) -> dict[str, Any]:
    net = [trade.net_bps for trade in trades]
    return {
        "trades": len(trades),
        "wins": sum(value > 0 for value in net),
        "losses": sum(value < 0 for value in net),
        "positive_net_trade_rate": sum(value > 0 for value in net) / len(net) if net else 0.0,
        "average_net_bps_per_trade": mean(net) if net else 0.0,
        "total_net_bps": sum(net),
        "average_duration_seconds": mean(trade.duration_seconds for trade in trades) if trades else 0.0,
    }


def _make_trade(
    opportunity: ExitOpportunity,
    times: np.ndarray,
    moves: np.ndarray,
    exit_index: int,
    reason: str,
    gross_bps: float,
    cost_bps: float,
    *,
    partial_taken: bool = False,
) -> ExitTrade:
    observed = moves[: exit_index + 1]
    return ExitTrade(
        timestamp_ms=opportunity.timestamp_ms,
        side=opportunity.side,
        exit_ms=int(times[exit_index]),
        exit_reason=reason,
        gross_bps=float(gross_bps),
        net_bps=float(gross_bps - cost_bps),
        mfe_bps=max(0.0, float(np.max(observed))),
        mae_bps=min(0.0, float(np.min(observed))),
        duration_seconds=max(0.0, (int(times[exit_index]) - opportunity.timestamp_ms) / 1_000.0),
        partial_taken=partial_taken,
    )


def _first_true(mask: np.ndarray) -> int | None:
    indices = np.flatnonzero(mask)
    return int(indices[0]) if len(indices) else None


def _first_event(*events: tuple[int | None, str, float] | None) -> tuple[int, str, float] | None:
    available = [event for event in events if event is not None and event[0] is not None]
    return min(available, key=lambda event: event[0]) if available else None


def _policy(family: str, name: str, **params: Any) -> ExitPolicy:
    params.setdefault("horizon_seconds", 21_600)
    return ExitPolicy(family=family, name=name, params=params)


def _coin_side(namespace: str, seed: int, timestamp_ms: int) -> int:
    digest = hashlib.sha256(f"{namespace}:{seed}:{timestamp_ms}".encode("ascii")).digest()
    return 1 if digest[0] & 1 else -1


def _stable_seed(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def _trade_row(trade: ExitTrade) -> dict[str, Any]:
    return {
        "entry": _iso(trade.timestamp_ms),
        "exit": _iso(trade.exit_ms),
        "side": "long" if trade.side == 1 else "short",
        "exit_reason": trade.exit_reason,
        "gross_bps": trade.gross_bps,
        "net_bps": trade.net_bps,
        "mfe_bps": trade.mfe_bps,
        "mae_bps": trade.mae_bps,
        "duration_seconds": trade.duration_seconds,
        "partial_taken": trade.partial_taken,
    }


def _max_drawdown(returns: list[float]) -> float:
    cumulative = 0.0
    peak = 0.0
    drawdown = 0.0
    for value in returns:
        cumulative += value
        peak = max(peak, cumulative)
        drawdown = max(drawdown, peak - cumulative)
    return drawdown


def _max_losing_streak(values: list[float]) -> int:
    longest = 0
    current = 0
    for value in values:
        current = current + 1 if value < 0 else 0
        longest = max(longest, current)
    return longest


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
        "robust_exit_improvement": 4,
        "promising_but_unconfirmed": 3,
        "profitable_without_reference_improvement": 2,
        "failed_holdout": 1,
        "selection_failed": 0,
    }[verdict]
