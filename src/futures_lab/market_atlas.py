from __future__ import annotations

import hashlib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import numpy as np

from futures_lab.first_touch import (
    LabeledSample,
    _deterministic_predictions,
    _evaluate_predictions,
    _iso,
    break_even_win_rate,
    label_samples,
    load_feature_samples,
    load_mark_price_timeline,
)


DEFAULT_NATURAL_BARRIERS_BPS = (10.0, 20.0, 30.0, 40.0, 60.0, 80.0, 100.0, 150.0)
DEFAULT_NATURAL_HORIZONS_SECONDS = (60, 300, 900, 3_600, 21_600)
DEFAULT_STRATEGY_TARGETS_BPS = (20.0, 40.0, 60.0, 80.0, 100.0, 150.0)
DEFAULT_STRATEGY_STOPS_BPS = (20.0, 40.0, 60.0, 80.0, 100.0, 150.0)
DEFAULT_STRATEGY_HORIZONS_SECONDS = (300, 900, 3_600, 21_600)
DEFAULT_STRATEGIES = (
    "fair_coin_control",
    "range_reversion",
    "micro_momentum",
    "micro_persistence",
    "impulse_pullback",
    "flow_exhaustion_reversal",
    "volatility_squeeze_breakout",
    "strategy_family_router",
    "htf_momentum",
    "regime_router",
)


def run_market_atlas(
    runs_root: Path,
    *,
    symbol: str = "ETHUSDT",
    natural_barriers_bps: Iterable[float] = DEFAULT_NATURAL_BARRIERS_BPS,
    natural_horizons_seconds: Iterable[int] = DEFAULT_NATURAL_HORIZONS_SECONDS,
    strategy_targets_bps: Iterable[float] = DEFAULT_STRATEGY_TARGETS_BPS,
    strategy_stops_bps: Iterable[float] = DEFAULT_STRATEGY_STOPS_BPS,
    strategy_horizons_seconds: Iterable[int] = DEFAULT_STRATEGY_HORIZONS_SECONDS,
    strategies: Iterable[str] = DEFAULT_STRATEGIES,
    cost_bps: float = 10.0,
    sample_seconds: int = 60,
    max_gap_seconds: int = 5,
    discovery_fraction: float = 0.70,
    account_exposure: float = 4.95,
    minimum_robust_trades: int = 20,
) -> dict[str, Any]:
    natural_barriers = _positive_floats(natural_barriers_bps, "natural barriers")
    natural_horizons = _positive_ints(natural_horizons_seconds, "natural horizons")
    strategy_targets = _positive_floats(strategy_targets_bps, "strategy targets")
    strategy_stops = _positive_floats(strategy_stops_bps, "strategy stops")
    strategy_horizons = _positive_ints(strategy_horizons_seconds, "strategy horizons")
    selected_strategies = tuple(dict.fromkeys(strategies))
    if not 0.55 <= discovery_fraction <= 0.80:
        raise ValueError("discovery_fraction must be between 0.55 and 0.80")
    if cost_bps < 0 or sample_seconds <= 0 or max_gap_seconds <= 0:
        raise ValueError("Cost and sampling parameters are invalid")

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
    if len(features) < 100:
        raise ValueError(f"Only {len(features)} sampled feature rows were found")

    split_index = max(1, min(len(features) - 1, int(len(features) * discovery_fraction)))
    split_ms = features[split_index][0]
    regime_thresholds = _regime_thresholds(features)

    natural_rows: list[dict[str, Any]] = []
    regime_rows: list[dict[str, Any]] = []
    for barrier in natural_barriers:
        for horizon in natural_horizons:
            labeled = label_samples(
                timeline,
                features,
                target_bps=barrier,
                stop_bps_values=[barrier],
                cost_bps=cost_bps,
                horizon_seconds=horizon,
            )
            natural_rows.append(
                {
                    "barrier_bps": barrier,
                    "horizon_seconds": horizon,
                    **_touch_summary(labeled),
                }
            )
            regime_rows.extend(
                _regime_touch_rows(
                    labeled,
                    barrier_bps=barrier,
                    horizon_seconds=horizon,
                    thresholds=regime_thresholds,
                )
            )

    strategy_rows: list[dict[str, Any]] = []
    requested_strategy_set = set(selected_strategies)
    unknown_strategies: set[str] = set()
    for target in strategy_targets:
        for horizon in strategy_horizons:
            labeled = label_samples(
                timeline,
                features,
                target_bps=target,
                stop_bps_values=strategy_stops,
                cost_bps=cost_bps,
                horizon_seconds=horizon,
            )
            discovery = [row for row in labeled if row.timestamp_ms < split_ms - horizon * 1_000]
            validation = [row for row in labeled if row.timestamp_ms >= split_ms]
            for period, period_rows in (("discovery", discovery), ("validation", validation)):
                predictions = _deterministic_predictions(period_rows)
                predictions["fair_coin_control"] = _coin_predictions(period_rows, namespace=symbol)
                unknown_strategies.update(requested_strategy_set - set(predictions))
                for strategy in selected_strategies:
                    if strategy not in predictions:
                        continue
                    for stop in strategy_stops:
                        metrics = _evaluate_predictions(
                            period_rows,
                            predictions[strategy],
                            stop,
                            target_bps=target,
                            cost_bps=cost_bps,
                            account_exposure=account_exposure,
                        )
                        strategy_rows.append(
                            _flatten_strategy_metrics(
                                metrics,
                                strategy=strategy,
                                target_bps=target,
                                stop_bps=stop,
                                horizon_seconds=horizon,
                                period=period,
                            )
                        )
    if unknown_strategies:
        raise ValueError(f"Unknown strategies: {sorted(unknown_strategies)}")

    _add_neighbor_stability(strategy_rows, strategy_targets, strategy_stops)
    robust = _robust_candidates(strategy_rows, minimum_trades=minimum_robust_trades)
    economics = _economics_surface(
        sorted(set(natural_barriers) | set(strategy_targets)),
        sorted(set(natural_barriers) | set(strategy_stops)),
        costs=(4.0, 8.0, cost_bps, 12.0, 15.0),
    )
    return {
        "study": "market_behavior_atlas",
        "symbol": symbol,
        "parameters": {
            "natural_barriers_bps": natural_barriers,
            "natural_horizons_seconds": natural_horizons,
            "strategy_targets_bps": strategy_targets,
            "strategy_stops_bps": strategy_stops,
            "strategy_horizons_seconds": strategy_horizons,
            "strategies": list(selected_strategies),
            "round_trip_cost_bps": cost_bps,
            "sample_seconds": sample_seconds,
            "max_gap_seconds": max_gap_seconds,
            "discovery_fraction": discovery_fraction,
            "purge_seconds_by_surface": "equal to each scenario horizon",
            "account_exposure_multiple": account_exposure,
            "minimum_robust_trades": minimum_robust_trades,
        },
        "data": {
            **mark_stats,
            **feature_stats,
            "split_at": _iso(split_ms),
            "regime_thresholds": regime_thresholds,
        },
        "natural_first_passage": natural_rows,
        "conditional_first_passage": regime_rows,
        "strategy_surface": strategy_rows,
        "robust_candidate_shortlist": robust,
        "economics_surface": economics,
        "interpretation_notes": [
            "The natural first-passage atlas describes ETH movement without choosing a trading side.",
            "Strategy cells use sequential non-overlapping trades; repeated feature snapshots are not counted as independent trades.",
            "Discovery and validation are chronological and separated by a purge equal to each scenario horizon.",
            "The shortlist requires positive average net bps in both periods and local neighbor stability; it is hypothesis generation, not final proof.",
            "Every result includes modeled round-trip cost but not a full queue, spread-crossing, or market-impact simulation.",
            "This archive covers one month and a limited set of regimes; future forward data remains the final judge.",
        ],
    }


def _touch_summary(samples: list[LabeledSample]) -> dict[str, Any]:
    resolved = [row for row in samples if row.symmetric_direction in {-1, 1}]
    times = [
        (row.symmetric_exit_ms - row.timestamp_ms) / 1_000
        for row in resolved
        if row.symmetric_exit_ms is not None
    ]
    up = sum(row.symmetric_direction == 1 for row in resolved)
    return {
        "samples": len(samples),
        "resolved": len(resolved),
        "unresolved": len(samples) - len(resolved),
        "resolution_rate": len(resolved) / len(samples) if samples else 0.0,
        "up_first": up,
        "down_first": len(resolved) - up,
        "up_first_rate": up / len(resolved) if resolved else None,
        "directional_skew_from_half": up / len(resolved) - 0.5 if resolved else None,
        "median_touch_seconds": median(times) if times else None,
        "p90_touch_seconds": float(np.quantile(times, 0.90)) if times else None,
    }


def _regime_touch_rows(
    samples: list[LabeledSample],
    *,
    barrier_bps: float,
    horizon_seconds: int,
    thresholds: dict[str, list[float]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[LabeledSample]] = defaultdict(list)
    for sample in samples:
        for dimension, bucket in _regime_labels(sample, thresholds).items():
            if bucket is not None:
                grouped[(dimension, bucket)].append(sample)
    return [
        {
            "barrier_bps": barrier_bps,
            "horizon_seconds": horizon_seconds,
            "dimension": dimension,
            "bucket": bucket,
            **_touch_summary(rows),
        }
        for (dimension, bucket), rows in sorted(grouped.items())
    ]


def _regime_thresholds(features: list[tuple[int, dict[str, Any]]]) -> dict[str, list[float]]:
    mapping = {
        "volatility": "realized_vol_180s_pct",
        "trend": "return_180s_pct",
        "spread": "spread_bps",
    }
    output: dict[str, list[float]] = {}
    for dimension, key in mapping.items():
        values = [value for _, row in features if (value := _finite_float(row.get(key))) is not None]
        output[dimension] = [float(np.quantile(values, 1 / 3)), float(np.quantile(values, 2 / 3))] if values else []
    return output


def _regime_labels(sample: LabeledSample, thresholds: dict[str, list[float]]) -> dict[str, str | None]:
    timestamp = datetime.fromtimestamp(sample.timestamp_ms / 1_000, tz=timezone.utc)
    hour = timestamp.hour
    labels: dict[str, str | None] = {
        "session_utc": "00-08 Asia" if hour < 8 else "08-16 Europe" if hour < 16 else "16-24 Americas",
    }
    specs = {
        "volatility": ("realized_vol_180s_pct", ("low", "middle", "high")),
        "trend": ("return_180s_pct", ("down", "flat", "up")),
        "spread": ("spread_bps", ("tight", "normal", "wide")),
    }
    for dimension, (key, names) in specs.items():
        value = _finite_float(sample.row.get(key))
        cuts = thresholds.get(dimension, [])
        if value is None or len(cuts) != 2:
            labels[dimension] = None
        elif value <= cuts[0]:
            labels[dimension] = names[0]
        elif value <= cuts[1]:
            labels[dimension] = names[1]
        else:
            labels[dimension] = names[2]
    return labels


def _coin_predictions(samples: list[LabeledSample], *, namespace: str = "ETH") -> list[int]:
    output = []
    for sample in samples:
        digest = hashlib.sha256(f"{namespace}:first-passage:{sample.timestamp_ms}".encode("ascii")).digest()
        output.append(1 if digest[0] & 1 else -1)
    return output


def _flatten_strategy_metrics(
    metrics: dict[str, Any],
    *,
    strategy: str,
    target_bps: float,
    stop_bps: float,
    horizon_seconds: int,
    period: str,
) -> dict[str, Any]:
    blocks = metrics["chronological_trade_blocks"]
    positive_blocks = sum(block["average_net_bps_per_trade"] > 0 for block in blocks)
    return {
        "strategy": strategy,
        "target_bps": target_bps,
        "stop_bps": stop_bps,
        "horizon_seconds": horizon_seconds,
        "period": period,
        "signals": metrics["signals"],
        "sequential_trades": metrics["sequential_trades"],
        "targets": metrics["targets"],
        "stops": metrics["stops"],
        "timeouts": metrics["timeouts"],
        "barrier_win_rate": metrics["barrier_win_rate"],
        "barrier_win_rate_wilson_95": metrics["barrier_win_rate_wilson_95"],
        "break_even_win_rate": metrics["break_even_win_rate"],
        "win_rate_edge": metrics["barrier_win_rate"] - metrics["break_even_win_rate"],
        "symmetric_direction_accuracy": metrics["symmetric_direction_accuracy"],
        "positive_net_trade_rate": metrics["positive_net_trade_rate"],
        "average_net_bps_per_trade": metrics["average_net_bps_per_trade"],
        "total_net_bps": metrics["total_net_bps"],
        "max_additive_account_drawdown_pct": metrics["max_additive_account_drawdown_pct"],
        "max_losing_streak": metrics["max_losing_streak"],
        "positive_chronological_block_fraction": positive_blocks / len(blocks) if blocks else 0.0,
        "long_trades": metrics["by_side"]["long"]["trades"],
        "long_average_net_bps": metrics["by_side"]["long"]["average_net_bps_per_trade"],
        "short_trades": metrics["by_side"]["short"]["trades"],
        "short_average_net_bps": metrics["by_side"]["short"]["average_net_bps_per_trade"],
        "neighbor_positive_fraction": None,
    }


def _add_neighbor_stability(
    rows: list[dict[str, Any]],
    targets: list[float],
    stops: list[float],
) -> None:
    target_index = {value: index for index, value in enumerate(targets)}
    stop_index = {value: index for index, value in enumerate(stops)}
    lookup = {
        (row["strategy"], row["horizon_seconds"], row["period"], row["target_bps"], row["stop_bps"]): row
        for row in rows
    }
    for row in rows:
        ti = target_index[row["target_bps"]]
        si = stop_index[row["stop_bps"]]
        neighbors = []
        for target_position, stop_position in ((ti, si), (ti - 1, si), (ti + 1, si), (ti, si - 1), (ti, si + 1)):
            if not (0 <= target_position < len(targets) and 0 <= stop_position < len(stops)):
                continue
            neighbor = lookup.get(
                (
                    row["strategy"],
                    row["horizon_seconds"],
                    row["period"],
                    targets[target_position],
                    stops[stop_position],
                )
            )
            if neighbor is not None:
                neighbors.append(neighbor)
        row["neighbor_positive_fraction"] = (
            sum(neighbor["average_net_bps_per_trade"] > 0 for neighbor in neighbors) / len(neighbors)
            if neighbors
            else 0.0
        )


def _robust_candidates(rows: list[dict[str, Any]], *, minimum_trades: int) -> list[dict[str, Any]]:
    periods: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (row["strategy"], row["target_bps"], row["stop_bps"], row["horizon_seconds"])
        periods[key][row["period"]] = row
    output = []
    for key, values in periods.items():
        discovery = values.get("discovery")
        validation = values.get("validation")
        if discovery is None or validation is None:
            continue
        robust_floor = min(
            discovery["average_net_bps_per_trade"],
            validation["average_net_bps_per_trade"],
        )
        qualifies = bool(
            discovery["sequential_trades"] >= minimum_trades
            and validation["sequential_trades"] >= minimum_trades
            and robust_floor > 0
            and validation["neighbor_positive_fraction"] >= 0.50
            and validation["positive_chronological_block_fraction"] >= 0.50
        )
        output.append(
            {
                "strategy": key[0],
                "target_bps": key[1],
                "stop_bps": key[2],
                "horizon_seconds": key[3],
                "qualifies": qualifies,
                "robust_floor_bps": robust_floor,
                "discovery_trades": discovery["sequential_trades"],
                "discovery_average_net_bps": discovery["average_net_bps_per_trade"],
                "validation_trades": validation["sequential_trades"],
                "validation_average_net_bps": validation["average_net_bps_per_trade"],
                "validation_win_rate": validation["barrier_win_rate"],
                "validation_break_even_win_rate": validation["break_even_win_rate"],
                "validation_neighbor_positive_fraction": validation["neighbor_positive_fraction"],
                "validation_positive_block_fraction": validation["positive_chronological_block_fraction"],
                "validation_max_drawdown_pct": validation["max_additive_account_drawdown_pct"],
            }
        )
    return sorted(
        output,
        key=lambda row: (row["qualifies"], row["robust_floor_bps"], row["validation_trades"]),
        reverse=True,
    )


def _economics_surface(targets: list[float], stops: list[float], *, costs: Iterable[float]) -> list[dict[str, Any]]:
    return [
        {
            "target_bps": target,
            "stop_bps": stop,
            "cost_bps": cost,
            "net_win_bps": target - cost,
            "net_loss_bps": -(stop + cost),
            "break_even_win_rate": break_even_win_rate(target, stop, cost),
        }
        for cost in sorted(set(float(value) for value in costs))
        for target in targets
        for stop in stops
    ]


def _positive_floats(values: Iterable[float], name: str) -> list[float]:
    output = sorted(set(float(value) for value in values if float(value) > 0))
    if not output:
        raise ValueError(f"At least one positive {name} value is required")
    return output


def _positive_ints(values: Iterable[int], name: str) -> list[int]:
    output = sorted(set(int(value) for value in values if int(value) > 0))
    if not output:
        raise ValueError(f"At least one positive {name} value is required")
    return output


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None
