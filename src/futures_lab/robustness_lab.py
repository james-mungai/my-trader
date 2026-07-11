from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from futures_lab.exit_laboratory import (
    REFERENCE_POLICY,
    ExitTrade,
    evaluate_policy,
    lock_reference_cohort,
    summarize_exit_trades,
)
from futures_lab.first_touch import load_feature_samples, load_mark_price_timeline
from futures_lab.strategy_tournament import _period_boundaries


STRESS_COST_BPS = (8.0, 10.0, 12.0, 15.0, 20.0, 25.0)
MISSED_WINNER_RATES = (0.0, 0.05, 0.10, 0.20)
EXPOSURE_MULTIPLES = (1.0, 2.0, 4.95, 10.0, 15.0)
FORWARD_TRUE_WIN_RATES = (0.60, 0.62, 0.65, 0.70)


def run_robustness_promotion_lab(
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
    path_trades: int = 100,
    bootstrap_paths: int = 5_000,
    block_size: int = 5,
    random_seed: int = 73,
    forward_summary_path: Path | None = None,
) -> dict[str, Any]:
    if cost_bps < 0 or sample_seconds <= 0 or horizon_seconds <= 0:
        raise ValueError("Robustness cost, sampling, and horizon settings must be valid")
    if path_trades < 25 or bootstrap_paths < 500 or block_size < 2:
        raise ValueError("Robustness simulation needs at least 25 trades, 500 paths, and block size 2")

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
    period_trades: dict[str, list[ExitTrade]] = {}
    cohort_stats: dict[str, dict[str, Any]] = {}
    period_metrics: dict[str, dict[str, Any]] = {}
    for period, rows in period_features.items():
        cohort, stats = lock_reference_cohort(
            timeline,
            rows,
            cost_bps=cost_bps,
            horizon_seconds=horizon_seconds,
        )
        trades = evaluate_policy(timeline, cohort, REFERENCE_POLICY, cost_bps=cost_bps)
        period_trades[period] = trades
        cohort_stats[period] = stats
        period_metrics[period] = _historical_trade_summary(
            trades,
            reference_cost_bps=cost_bps,
            hostile_cost_bps=15.0,
            account_exposure=account_exposure,
        )

    trades = sorted(
        [trade for period in ("discovery", "calibration", "holdout") for trade in period_trades[period]],
        key=lambda trade: trade.timestamp_ms,
    )
    if len(trades) < 50:
        raise ValueError(f"Only {len(trades)} locked trades were found")
    gross_bps = np.asarray([trade.gross_bps for trade in trades], dtype=np.float64)
    scenario_costs = tuple(sorted({*STRESS_COST_BPS, float(cost_bps)}))

    path_indices = _moving_block_indices(
        len(trades),
        path_trades=path_trades,
        paths=bootstrap_paths,
        block_size=block_size,
        seed=random_seed,
    )
    rng = np.random.default_rng(random_seed ^ 0x5A5A5A5A)
    missed_uniforms = rng.random((bootstrap_paths, path_trades))
    gross_paths = gross_bps[path_indices]
    stress_grid = [
        _stress_scenario(
            gross_paths,
            missed_uniforms,
            total_cost_bps=scenario_cost,
            missed_winner_rate=missed_rate,
            exposure_multiples=EXPOSURE_MULTIPLES,
        )
        for scenario_cost in scenario_costs
        for missed_rate in MISSED_WINNER_RATES
    ]
    reference_stress = next(
        row
        for row in stress_grid
        if row["total_cost_bps"] == cost_bps and row["missed_winner_rate"] == 0.0
    )

    chronological_blocks = _chronological_blocks(
        trades,
        reference_cost_bps=cost_bps,
        hostile_cost_bps=15.0,
        blocks=8,
    )
    side_robustness = _side_robustness(
        trades,
        reference_cost_bps=cost_bps,
        hostile_cost_bps=15.0,
        account_exposure=account_exposure,
    )
    historical = _historical_trade_summary(
        trades,
        reference_cost_bps=cost_bps,
        hostile_cost_bps=15.0,
        account_exposure=account_exposure,
    )
    historical["bootstrap_100_trade_reference"] = reference_stress
    historical["average_gross_cost_capacity_bps"] = float(np.mean(gross_bps))
    historical["development_evidence_only"] = True

    forward_summary = _load_forward_summary(forward_summary_path)
    forward_scorecard = _forward_promotion_scorecard(forward_summary)
    power_plan = _forward_power_plan(target_bps=60.0, stop_bps=60.0, cost_bps=10.0)

    return {
        "study": "squeeze_robustness_and_promotion_laboratory",
        "symbol": symbol,
        "parameters": {
            "historical_entry_rule": "existing side-symmetric volatility_squeeze_breakout",
            "historical_exit_policy": REFERENCE_POLICY.name,
            "historical_round_trip_cost_bps": cost_bps,
            "historical_horizon_seconds": horizon_seconds,
            "historical_account_exposure_multiple": account_exposure,
            "stress_total_cost_bps": list(scenario_costs),
            "stress_missed_winner_rates": list(MISSED_WINNER_RATES),
            "stress_exposure_multiples": list(EXPOSURE_MULTIPLES),
            "bootstrap_path_trades": path_trades,
            "bootstrap_paths": bootstrap_paths,
            "moving_block_size": block_size,
            "random_seed": random_seed,
            "forward_contract": "symmetric_60_60_6h_with_10bps_cost",
        },
        "data": {
            **mark_stats,
            **feature_stats,
            "calibration_start": _iso(calibration_start_ms),
            "holdout_start": _iso(holdout_start_ms),
            "historical_locked_trades": len(trades),
            "historical_status": "development_data_already_used_by_prior_proposals",
            "forward_summary_path": str(forward_summary_path) if forward_summary_path else None,
        },
        "entry_cohorts": cohort_stats,
        "historical_periods": period_metrics,
        "historical_combined": historical,
        "chronological_blocks": chronological_blocks,
        "side_robustness": side_robustness,
        "stress_grid": stress_grid,
        "forward_power_plan": power_plan,
        "forward_snapshot": forward_summary,
        "forward_promotion_scorecard": forward_scorecard,
        "historical_promotion_status": "not_eligible_because_development_archive_is_not_independent",
        "promotion_candidates": (
            ["squeeze_breakout_forward"]
            if forward_scorecard.get("overall_status") == "scorecard_pass"
            else []
        ),
        "interpretation_notes": [
            "Historical simulations use paired moving-block bootstrap paths so fee, missed-winner, and exposure comparisons differ only by the applied stress.",
            "A missed winner is modeled as zero return for that opportunity while adverse trades remain executable. This intentionally represents toxic execution asymmetry.",
            "Exposure is not leverage itself: it is position notional divided by account equity. The same signal path can be tolerable at 4.95x and destructive at 15x.",
            "The 94 historical trades were used throughout prior research. Their bootstrap intervals measure path fragility, not independent proof of edge.",
            "The AWS shadow arena uses a different symmetric 60/60 contract and is evaluated only by its own frozen forward scorecard.",
            "No historical result can trigger live trading. Only a completed forward scorecard can nominate a separately authorized tiny canary.",
        ],
    }


def _historical_trade_summary(
    trades: list[ExitTrade],
    *,
    reference_cost_bps: float,
    hostile_cost_bps: float,
    account_exposure: float,
) -> dict[str, Any]:
    reference = _reprice_trades(trades, reference_cost_bps)
    hostile = _reprice_trades(trades, hostile_cost_bps)
    reference_metrics = summarize_exit_trades(reference, account_exposure=account_exposure)
    hostile_metrics = summarize_exit_trades(hostile, account_exposure=account_exposure)
    positives = sum(trade.net_bps > 0 for trade in reference)
    return {
        "trades": len(trades),
        "reference_cost_bps": reference_cost_bps,
        "hostile_cost_bps": hostile_cost_bps,
        "reference": _compact_metrics(reference_metrics),
        "hostile": _compact_metrics(hostile_metrics),
        "positive_trade_rate_wilson_95": list(_wilson_interval(positives, len(reference))),
        "gross_average_bps": mean(trade.gross_bps for trade in trades) if trades else 0.0,
    }


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "trades": metrics["trades"],
        "wins": metrics["wins"],
        "losses": metrics["losses"],
        "positive_net_trade_rate": metrics["positive_net_trade_rate"],
        "average_net_bps_per_trade": metrics["average_net_bps_per_trade"],
        "total_net_bps": metrics["total_net_bps"],
        "max_additive_account_drawdown_pct": metrics["max_additive_account_drawdown_pct"],
        "max_losing_streak": metrics["max_losing_streak"],
        "positive_chronological_block_fraction": metrics["positive_chronological_block_fraction"],
        "long": metrics["by_side"]["long"],
        "short": metrics["by_side"]["short"],
    }


def _reprice_trades(trades: list[ExitTrade], total_cost_bps: float) -> list[ExitTrade]:
    return [
        ExitTrade(
            timestamp_ms=trade.timestamp_ms,
            side=trade.side,
            exit_ms=trade.exit_ms,
            exit_reason=trade.exit_reason,
            gross_bps=trade.gross_bps,
            net_bps=trade.gross_bps - total_cost_bps,
            mfe_bps=trade.mfe_bps,
            mae_bps=trade.mae_bps,
            duration_seconds=trade.duration_seconds,
            partial_taken=trade.partial_taken,
        )
        for trade in trades
    ]


def _moving_block_indices(
    observations: int,
    *,
    path_trades: int,
    paths: int,
    block_size: int,
    seed: int,
) -> np.ndarray:
    if observations <= 0 or path_trades <= 0 or paths <= 0 or block_size <= 0:
        raise ValueError("Moving-block bootstrap dimensions must be positive")
    rng = np.random.default_rng(seed)
    blocks_per_path = math.ceil(path_trades / block_size)
    starts = rng.integers(0, observations, size=(paths, blocks_per_path), dtype=np.int64)
    offsets = np.arange(block_size, dtype=np.int64)
    indices = (starts[:, :, None] + offsets[None, None, :]) % observations
    return indices.reshape(paths, -1)[:, :path_trades]


def _stress_scenario(
    gross_paths: np.ndarray,
    missed_uniforms: np.ndarray,
    *,
    total_cost_bps: float,
    missed_winner_rate: float,
    exposure_multiples: tuple[float, ...],
) -> dict[str, Any]:
    if gross_paths.shape != missed_uniforms.shape:
        raise ValueError("Stress paths and missed-fill uniforms must have identical shapes")
    net = gross_paths - total_cost_bps
    missed = (net > 0) & (missed_uniforms < missed_winner_rate)
    stressed = np.where(missed, 0.0, net)
    path_totals = np.sum(stressed, axis=1)
    path_averages = np.mean(stressed, axis=1)
    losing_streaks = _path_losing_streaks(stressed)
    return {
        "total_cost_bps": total_cost_bps,
        "missed_winner_rate": missed_winner_rate,
        "opportunity_execution_rate": float(1.0 - np.mean(missed)),
        "positive_path_fraction": float(np.mean(path_totals > 0)),
        "path_average_net_bps": _distribution(path_averages),
        "path_total_net_bps": _distribution(path_totals),
        "max_losing_streak": _distribution(losing_streaks.astype(np.float64)),
        "exposures": [
            _exposure_path_metrics(stressed, exposure) for exposure in exposure_multiples
        ],
    }


def _exposure_path_metrics(net_bps: np.ndarray, exposure: float) -> dict[str, Any]:
    returns = net_bps * exposure / 10_000.0
    factors = 1.0 + returns
    ruined = np.any(factors <= 0, axis=1)
    safe_factors = np.maximum(factors, 1e-12)
    equity = np.cumprod(safe_factors, axis=1)
    equity = np.concatenate([np.ones((len(equity), 1)), equity], axis=1)
    peaks = np.maximum.accumulate(equity, axis=1)
    drawdowns = 1.0 - equity / peaks
    max_drawdown_pct = np.max(drawdowns, axis=1) * 100.0
    terminal_return_pct = (equity[:, -1] - 1.0) * 100.0
    additive_return_pct = np.sum(returns, axis=1) * 100.0
    return {
        "exposure_multiple": exposure,
        "terminal_compounded_return_pct": _distribution(terminal_return_pct),
        "terminal_additive_return_pct": _distribution(additive_return_pct),
        "max_drawdown_pct": _distribution(max_drawdown_pct),
        "probability_drawdown_gte_30pct": float(np.mean(max_drawdown_pct >= 30.0)),
        "probability_drawdown_gte_50pct": float(np.mean(max_drawdown_pct >= 50.0)),
        "ruin_probability": float(np.mean(ruined)),
    }


def _path_losing_streaks(net_bps: np.ndarray) -> np.ndarray:
    output = np.zeros(len(net_bps), dtype=np.int64)
    for path_index, path in enumerate(net_bps):
        current = 0
        maximum = 0
        for value in path:
            if value < 0:
                current += 1
                maximum = max(maximum, current)
            else:
                current = 0
        output[path_index] = maximum
    return output


def _distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
    }


def _chronological_blocks(
    trades: list[ExitTrade],
    *,
    reference_cost_bps: float,
    hostile_cost_bps: float,
    blocks: int,
) -> list[dict[str, Any]]:
    output = []
    chunks = np.array_split(np.asarray(trades, dtype=object), min(blocks, len(trades)))
    for index, chunk in enumerate(chunks, start=1):
        rows = list(chunk.tolist())
        reference = [trade.gross_bps - reference_cost_bps for trade in rows]
        hostile = [trade.gross_bps - hostile_cost_bps for trade in rows]
        output.append(
            {
                "block": index,
                "start": _iso(rows[0].timestamp_ms),
                "end": _iso(rows[-1].timestamp_ms),
                "trades": len(rows),
                "long_trades": sum(trade.side == 1 for trade in rows),
                "short_trades": sum(trade.side == -1 for trade in rows),
                "reference_average_net_bps": mean(reference),
                "hostile_average_net_bps": mean(hostile),
                "reference_positive_trade_rate": mean(value > 0 for value in reference),
            }
        )
    return output


def _side_robustness(
    trades: list[ExitTrade],
    *,
    reference_cost_bps: float,
    hostile_cost_bps: float,
    account_exposure: float,
) -> dict[str, Any]:
    output = {}
    for side, label in ((1, "long"), (-1, "short")):
        rows = [trade for trade in trades if trade.side == side]
        output[label] = _historical_trade_summary(
            rows,
            reference_cost_bps=reference_cost_bps,
            hostile_cost_bps=hostile_cost_bps,
            account_exposure=account_exposure,
        )
    return output


def _forward_power_plan(
    *,
    target_bps: float,
    stop_bps: float,
    cost_bps: float,
) -> dict[str, Any]:
    break_even = (stop_bps + cost_bps) / (target_bps + stop_bps)
    return {
        "target_bps": target_bps,
        "stop_bps": stop_bps,
        "cost_bps": cost_bps,
        "net_win_bps": target_bps - cost_bps,
        "net_loss_bps": -(stop_bps + cost_bps),
        "break_even_win_rate": break_even,
        "power_assumption": "one-sided alpha 5%, 80% power, normal approximation",
        "true_rate_scenarios": [
            {
                "true_win_rate": rate,
                "expected_net_bps_per_trade": (
                    rate * (target_bps - cost_bps)
                    + (1.0 - rate) * (-(stop_bps + cost_bps))
                ),
                "approximate_required_resolved_trades": _approximate_proportion_sample_size(
                    break_even,
                    rate,
                ),
            }
            for rate in FORWARD_TRUE_WIN_RATES
        ],
        "confidence_checkpoints": [
            {
                "resolved_trades": count,
                "minimum_targets_for_one_sided_95pct_lower_bound_above_break_even": (
                    _required_successes_for_wilson(count, break_even)
                ),
                "minimum_observed_win_rate": (
                    _required_successes_for_wilson(count, break_even) / count
                ),
            }
            for count in (100, 200, 300, 500, 1_000)
        ],
    }


def _approximate_proportion_sample_size(null_rate: float, true_rate: float) -> int | None:
    if true_rate <= null_rate:
        return None
    z_alpha = 1.6448536269514722
    z_power = 0.8416212335729143
    numerator = (
        z_alpha * math.sqrt(null_rate * (1.0 - null_rate))
        + z_power * math.sqrt(true_rate * (1.0 - true_rate))
    )
    return int(math.ceil((numerator / (true_rate - null_rate)) ** 2))


def _required_successes_for_wilson(trials: int, break_even: float) -> int:
    for successes in range(trials + 1):
        lower, _ = _wilson_interval(successes, trials, z=1.6448536269514722)
        if lower > break_even:
            return successes
    return trials + 1


def _wilson_interval(
    successes: int,
    trials: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if trials <= 0:
        return 0.0, 0.0
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _load_forward_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        raise ValueError(f"Forward summary does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or "arms" not in payload:
        raise ValueError("Forward summary must be a shadow-arena-summary JSON object")
    return payload


def _forward_promotion_scorecard(summary: dict[str, Any] | None) -> dict[str, Any]:
    requirements = {
        "minimum_resolved_trades": 100,
        "minimum_resolved_trades_per_side": 20,
        "minimum_full_blocks_25": 4,
        "minimum_positive_block_fraction": 0.75,
        "maximum_data_age_seconds": 2.0,
        "maximum_average_feed_lag_ms": 2_000.0,
        "confidence_rule": "two-sided Wilson 95% lower bound above fee-adjusted break-even",
    }
    if summary is None:
        return {
            "overall_status": "forward_snapshot_not_available",
            "requirements": requirements,
            "gates": [],
        }
    arms = summary.get("arms") or {}
    strategy = arms.get("squeeze_breakout") or {}
    control = arms.get("squeeze_matched_coin") or {}
    targets = int(strategy.get("targets") or 0)
    stops = int(strategy.get("stops") or 0)
    resolved = targets + stops
    break_even = float(summary.get("break_even_win_rate") or 0.0)
    interval = _wilson_interval(targets, resolved) if resolved else (0.0, 0.0)
    gates = []
    gates.append(
        _scorecard_gate(
            "minimum_sample",
            resolved >= requirements["minimum_resolved_trades"],
            pending=resolved < requirements["minimum_resolved_trades"],
            observed=resolved,
            required=requirements["minimum_resolved_trades"],
        )
    )
    confidence_ready = resolved >= requirements["minimum_resolved_trades"]
    gates.append(
        _scorecard_gate(
            "win_rate_confidence",
            interval[0] > break_even,
            pending=not confidence_ready,
            observed={"win_rate": targets / resolved if resolved else None, "wilson_95": list(interval)},
            required={"lower_bound_above": break_even},
        )
    )
    gates.append(
        _scorecard_gate(
            "positive_after_cost_expectancy",
            float(strategy.get("total_net_bps") or 0.0) > 0,
            pending=int(strategy.get("closed") or 0) < 25,
            observed={
                "closed": int(strategy.get("closed") or 0),
                "average_net_bps": strategy.get("average_net_bps"),
                "total_net_bps": strategy.get("total_net_bps"),
            },
            required={"minimum_closed_before_judgment": 25, "total_net_bps_above": 0},
        )
    )
    side_status = {}
    for side in ("long", "short"):
        row = ((strategy.get("by_side") or {}).get(side) or {})
        side_status[side] = {
            "trades": int(row.get("trades") or 0),
            "net_bps": float(row.get("net_bps") or 0.0),
        }
    side_ready = all(
        row["trades"] >= requirements["minimum_resolved_trades_per_side"]
        for row in side_status.values()
    )
    gates.append(
        _scorecard_gate(
            "side_neutrality",
            all(row["net_bps"] > 0 for row in side_status.values()),
            pending=not side_ready,
            observed=side_status,
            required={
                "minimum_trades_each_side": requirements["minimum_resolved_trades_per_side"],
                "net_bps_each_side_above": 0,
            },
        )
    )
    blocks = list(strategy.get("chronological_blocks_25") or [])
    full_blocks = [row for row in blocks if _block_trade_count(row.get("trades")) >= 25]
    positive_fraction = (
        mean(float(row.get("net_bps") or 0.0) > 0 for row in full_blocks)
        if full_blocks
        else 0.0
    )
    blocks_ready = len(full_blocks) >= requirements["minimum_full_blocks_25"]
    gates.append(
        _scorecard_gate(
            "chronological_stability",
            positive_fraction >= requirements["minimum_positive_block_fraction"],
            pending=not blocks_ready,
            observed={"full_blocks": len(full_blocks), "positive_block_fraction": positive_fraction},
            required={
                "minimum_full_blocks": requirements["minimum_full_blocks_25"],
                "minimum_positive_fraction": requirements["minimum_positive_block_fraction"],
            },
        )
    )
    strategy_closed = int(strategy.get("closed") or 0)
    control_closed = int(control.get("closed") or 0)
    control_ready = min(strategy_closed, control_closed) >= 25
    strategy_average = float(strategy.get("average_net_bps") or 0.0)
    control_average = float(control.get("average_net_bps") or 0.0)
    gates.append(
        _scorecard_gate(
            "matched_direction_control",
            strategy_average > control_average and strategy_average > 0,
            pending=not control_ready,
            observed={
                "strategy_closed": strategy_closed,
                "strategy_average_net_bps": strategy_average,
                "control_closed": control_closed,
                "control_average_net_bps": control_average,
            },
            required={"minimum_each": 25, "strategy_average_above_control_and_zero": True},
        )
    )
    health = summary.get("last_health") or {}
    book_lag = _float_or_inf(health.get("avg_book_event_lag_30s_ms"))
    trade_lag = _float_or_inf(health.get("avg_trade_event_lag_30s_ms"))
    data_age = _float_or_inf(health.get("data_age_seconds"))
    health_present = bool(health)
    gates.append(
        _scorecard_gate(
            "feed_health",
            bool(health.get("connected"))
            and data_age <= requirements["maximum_data_age_seconds"]
            and max(book_lag, trade_lag) <= requirements["maximum_average_feed_lag_ms"],
            pending=not health_present,
            observed={
                "connected": health.get("connected"),
                "data_age_seconds": None if not np.isfinite(data_age) else data_age,
                "book_lag_ms": None if not np.isfinite(book_lag) else book_lag,
                "trade_lag_ms": None if not np.isfinite(trade_lag) else trade_lag,
                "snapshot_at": health.get("ts"),
            },
            required={
                "connected": True,
                "maximum_data_age_seconds": requirements["maximum_data_age_seconds"],
                "maximum_average_lag_ms": requirements["maximum_average_feed_lag_ms"],
            },
        )
    )
    statuses = {row["status"] for row in gates}
    if "fail" in statuses:
        overall = "scorecard_fail"
    elif statuses == {"pass"}:
        overall = "scorecard_pass"
    else:
        overall = "scorecard_pending"
    return {
        "overall_status": overall,
        "snapshot_at": (summary.get("last_health") or {}).get("ts"),
        "strategy_arm": strategy,
        "matched_control_arm": control,
        "break_even_win_rate": break_even,
        "observed_wilson_95": list(interval) if resolved else None,
        "requirements": requirements,
        "gates": gates,
    }


def _scorecard_gate(
    name: str,
    passed: bool,
    *,
    pending: bool,
    observed: Any,
    required: Any,
) -> dict[str, Any]:
    return {
        "gate": name,
        "status": "pending" if pending else "pass" if passed else "fail",
        "observed": observed,
        "required": required,
    }


def _block_trade_count(label: Any) -> int:
    try:
        start, end = str(label).split("-", maxsplit=1)
        return int(end) - int(start) + 1
    except (TypeError, ValueError):
        return 0


def _float_or_inf(value: Any) -> float:
    if value is None:
        return math.inf
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.inf


def _iso(timestamp_ms: int) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(timestamp_ms / 1_000, tz=timezone.utc).isoformat()
