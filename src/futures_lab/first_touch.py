from __future__ import annotations

import gzip
import json
import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np


MODEL_FEATURES = (
    "range_position_180s",
    "range_180s_pct",
    "return_15s_pct",
    "return_60s_pct",
    "return_180s_pct",
    "realized_vol_60s_pct",
    "realized_vol_180s_pct",
    "order_flow_imbalance_1s",
    "order_flow_imbalance_5s",
    "taker_aggression_imbalance_1s",
    "taker_aggression_imbalance_5s",
    "taker_aggression_imbalance_15s",
    "taker_buy_ratio_10s",
    "book_imbalance_top",
    "depth_imbalance_top5",
    "microprice_mid_bps",
    "vamp_mid_bps",
    "mark_last_basis_bps",
    "open_interest_change_5m_pct",
    "liquidation_buy_ratio_30s",
    "btc_order_flow_imbalance_1s",
    "btc_taker_aggression_imbalance_1s",
    "btc_microprice_mid_bps",
    "btc_return_15s_pct",
    "btc_return_60s_pct",
    "eth_btc_relative_return_15s_pct",
    "eth_btc_relative_return_60s_pct",
    "htf_15m_range_position",
    "htf_15m_trend_score",
    "htf_30m_range_position",
    "htf_30m_trend_score",
    "htf_1h_range_position",
    "htf_1h_trend_score",
)


@dataclass(frozen=True)
class BarrierResult:
    outcome: str
    exit_ms: int | None
    net_bps: float | None


@dataclass(frozen=True)
class LabeledSample:
    timestamp_ms: int
    row: dict[str, Any]
    symmetric_direction: int | None
    symmetric_exit_ms: int | None
    scenarios: dict[float, dict[int, BarrierResult]]


class PriceTimeline:
    def __init__(self, times_ms: np.ndarray, prices: np.ndarray, max_gap_ms: int) -> None:
        if len(times_ms) != len(prices) or len(times_ms) == 0:
            raise ValueError("Price timeline requires equally sized non-empty arrays")
        self.times_ms = times_ms.astype(np.int64, copy=False)
        self.prices = prices.astype(np.float64, copy=False)
        self.max_gap_ms = max_gap_ms
        self.session_ends = np.empty(len(times_ms), dtype=np.int64)
        self.session_ends[-1] = len(times_ms) - 1
        for index in range(len(times_ms) - 2, -1, -1):
            if int(times_ms[index + 1] - times_ms[index]) > max_gap_ms:
                self.session_ends[index] = index
            else:
                self.session_ends[index] = self.session_ends[index + 1]

        size = 1
        while size < len(prices):
            size *= 2
        self._tree_size = size
        self._max_tree = np.full(size * 2, -np.inf, dtype=np.float64)
        self._min_tree = np.full(size * 2, np.inf, dtype=np.float64)
        self._max_tree[size : size + len(prices)] = prices
        self._min_tree[size : size + len(prices)] = prices
        for index in range(size - 1, 0, -1):
            self._max_tree[index] = max(self._max_tree[index * 2], self._max_tree[index * 2 + 1])
            self._min_tree[index] = min(self._min_tree[index * 2], self._min_tree[index * 2 + 1])

    def entry_index(self, timestamp_ms: int) -> int | None:
        index = bisect_left(self.times_ms, timestamp_ms)
        if index >= len(self.times_ms):
            return None
        if int(self.times_ms[index] - timestamp_ms) > self.max_gap_ms:
            return None
        return index

    def horizon_end(self, entry_index: int, horizon_seconds: int) -> tuple[int, bool]:
        deadline = int(self.times_ms[entry_index]) + horizon_seconds * 1000
        time_end = bisect_right(self.times_ms, deadline) - 1
        end = min(time_end, int(self.session_ends[entry_index]))
        complete = int(self.times_ms[end]) >= deadline - self.max_gap_ms
        return end, complete

    def first_ge(self, left: int, right: int, threshold: float) -> int | None:
        return self._first(left, right, threshold, want_max=True, node=1, node_left=0, node_right=self._tree_size - 1)

    def first_le(self, left: int, right: int, threshold: float) -> int | None:
        return self._first(left, right, threshold, want_max=False, node=1, node_left=0, node_right=self._tree_size - 1)

    def _first(
        self,
        left: int,
        right: int,
        threshold: float,
        *,
        want_max: bool,
        node: int,
        node_left: int,
        node_right: int,
    ) -> int | None:
        if node_right < left or node_left > right:
            return None
        value = self._max_tree[node] if want_max else self._min_tree[node]
        if (want_max and value < threshold) or (not want_max and value > threshold):
            return None
        if node_left == node_right:
            return node_left if node_left < len(self.prices) else None
        middle = (node_left + node_right) // 2
        first = self._first(
            left,
            right,
            threshold,
            want_max=want_max,
            node=node * 2,
            node_left=node_left,
            node_right=middle,
        )
        if first is not None:
            return first
        return self._first(
            left,
            right,
            threshold,
            want_max=want_max,
            node=node * 2 + 1,
            node_left=middle + 1,
            node_right=node_right,
        )


def run_first_touch_study(
    runs_root: Path,
    *,
    symbol: str = "ETHUSDT",
    target_bps: float = 60.0,
    stop_bps_values: Iterable[float] = (40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 120.0, 150.0, 450.0),
    cost_bps: float = 10.0,
    horizon_seconds: int = 21_600,
    sample_seconds: int = 60,
    max_gap_seconds: int = 5,
    train_fraction: float = 0.70,
    account_exposure: float = 20.0,
    max_selected_stop_bps: float = 150.0,
) -> dict[str, Any]:
    if target_bps <= cost_bps:
        raise ValueError("Target must exceed round-trip cost")
    if not 0.5 <= train_fraction < 0.9:
        raise ValueError("train_fraction must be between 0.5 and 0.9")
    stops = sorted({float(value) for value in stop_bps_values if float(value) > 0})
    if not stops:
        raise ValueError("At least one positive stop is required")

    timeline, mark_stats = load_mark_price_timeline(runs_root, symbol=symbol, max_gap_seconds=max_gap_seconds)
    features, feature_stats = load_feature_samples(runs_root, symbol=symbol, sample_seconds=sample_seconds)
    labeled = label_samples(
        timeline,
        features,
        target_bps=target_bps,
        stop_bps_values=stops,
        cost_bps=cost_bps,
        horizon_seconds=horizon_seconds,
    )
    if len(labeled) < 20:
        raise ValueError(f"Only {len(labeled)} labelable samples were found")

    split_index = max(1, min(len(labeled) - 1, int(len(labeled) * train_fraction)))
    split_ms = labeled[split_index].timestamp_ms
    purge_ms = horizon_seconds * 1000
    train = [row for row in labeled if row.timestamp_ms < split_ms - purge_ms]
    test = [row for row in labeled if row.timestamp_ms >= split_ms]
    if len(train) < 10 or len(test) < 10:
        raise ValueError(f"Insufficient purged split: train={len(train)} test={len(test)}")

    strategies = _deterministic_predictions(test)
    train_strategies = _deterministic_predictions(train)
    strategy_rows: dict[str, Any] = {}
    train_selected_rows: dict[str, Any] = {}
    for name, predictions in strategies.items():
        strategy_rows[name] = _scenario_metrics(
            test,
            predictions,
            stops,
            target_bps=target_bps,
            cost_bps=cost_bps,
            account_exposure=account_exposure,
        )
        train_metrics = _scenario_metrics(
            train,
            train_strategies[name],
            stops,
            target_bps=target_bps,
            cost_bps=cost_bps,
            account_exposure=account_exposure,
        )
        eligible = [
            stop
            for stop in stops
            if stop <= max_selected_stop_bps
            and train_metrics[str(stop)]["sequential_trades"] >= 30
            and train_metrics[str(stop)]["average_net_bps_per_trade"] > 0
        ]
        if eligible:
            selected_stop = max(
                eligible,
                key=lambda stop: train_metrics[str(stop)]["average_net_bps_per_trade"],
            )
            train_selected_rows[name] = {
                "selected_stop_bps": selected_stop,
                "selection_rule": "highest training average net bps with >=30 sequential trades",
                "training_metrics": train_metrics[str(selected_stop)],
                "holdout_metrics": strategy_rows[name][str(selected_stop)],
            }

    model = _fit_logistic(train)
    probabilities = _predict_logistic(model, test)
    model_rows: dict[str, Any] = {}
    for threshold in (0.50, 0.55, 0.60, 0.65):
        predictions = [1 if value >= threshold else -1 if value <= 1.0 - threshold else None for value in probabilities]
        model_rows[f"confidence_{threshold:.2f}"] = _scenario_metrics(
            test,
            predictions,
            stops,
            target_bps=target_bps,
            cost_bps=cost_bps,
            account_exposure=account_exposure,
        )

    resolved_test = [(row, probability) for row, probability in zip(test, probabilities) if row.symmetric_direction is not None]
    direction_accuracy = (
        mean((probability >= 0.5) == (row.symmetric_direction == 1) for row, probability in resolved_test)
        if resolved_test
        else 0.0
    )
    brier = (
        mean((probability - (1.0 if row.symmetric_direction == 1 else 0.0)) ** 2 for row, probability in resolved_test)
        if resolved_test
        else 0.0
    )
    calibrated = _calibration_rows(resolved_test)
    top_weights = sorted(
        zip(MODEL_FEATURES, model["weights"], strict=True),
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:12]

    symmetric_resolved = sum(row.symmetric_direction is not None for row in test)
    return {
        "study": "purged_chronological_first_touch",
        "symbol": symbol,
        "parameters": {
            "target_bps": target_bps,
            "stop_bps_values": stops,
            "round_trip_cost_bps": cost_bps,
            "horizon_seconds": horizon_seconds,
            "sample_seconds": sample_seconds,
            "max_gap_seconds": max_gap_seconds,
            "train_fraction": train_fraction,
            "purge_seconds": horizon_seconds,
            "account_exposure_multiple": account_exposure,
            "max_selected_stop_bps": max_selected_stop_bps,
        },
        "economics": {
            str(stop): {
                "net_win_bps": target_bps - cost_bps,
                "net_loss_bps": -(stop + cost_bps),
                "break_even_win_rate": break_even_win_rate(target_bps, stop, cost_bps),
            }
            for stop in stops
        },
        "data": {
            **mark_stats,
            **feature_stats,
            "labelable_samples": len(labeled),
            "train_samples_after_purge": len(train),
            "test_samples": len(test),
            "test_symmetric_resolved": symmetric_resolved,
            "test_symmetric_unresolved": len(test) - symmetric_resolved,
            "split_at": _iso(split_ms),
            "first_sample_at": _iso(labeled[0].timestamp_ms),
            "last_sample_at": _iso(labeled[-1].timestamp_ms),
        },
        "holdout_deterministic_strategies": strategy_rows,
        "training_selected_stop_holdout": train_selected_rows,
        "holdout_logistic_model": {
            "training_rows": model["training_rows"],
            "test_resolved_rows": len(resolved_test),
            "symmetric_direction_accuracy": direction_accuracy,
            "brier_score": brier,
            "calibration": calibrated,
            "top_standardized_weights": [
                {"feature": feature, "weight": weight} for feature, weight in top_weights
            ],
            "thresholds": model_rows,
        },
        "interpretation_notes": [
            "Primary metrics use the last chronological holdout only; the horizon is purged from training.",
            "Trades are sequential and non-overlapping, so repeated snapshots do not masquerade as independent trades.",
            "Mark price determines barrier order; round-trip fees/slippage are represented by round_trip_cost_bps.",
            "A high hit rate with a wide stop can still have negative expectancy; compare it with break_even_win_rate.",
            "This study is evidence for strategy selection, not a guarantee that live fill quality will match mark-price labels.",
        ],
    }


def load_mark_price_timeline(
    runs_root: Path,
    *,
    symbol: str,
    max_gap_seconds: int,
) -> tuple[PriceTimeline, dict[str, Any]]:
    paths = _run_files(runs_root, f"raw_ws/{symbol}_markPriceUpdate_*.jsonl*")
    points: dict[int, float] = {}
    rows_read = 0
    for path in paths:
        for row in _jsonl_rows(path):
            rows_read += 1
            message = row.get("message") or row
            data = message.get("data") or message
            try:
                timestamp_ms = int(data["E"])
                price = float(data["p"])
            except (KeyError, TypeError, ValueError):
                continue
            if price > 0:
                points[timestamp_ms] = price
    if not points:
        raise ValueError(f"No {symbol} mark-price points found under {runs_root}")
    ordered = sorted(points.items())
    times = np.fromiter((item[0] for item in ordered), dtype=np.int64)
    prices = np.fromiter((item[1] for item in ordered), dtype=np.float64)
    gaps = np.diff(times)
    return PriceTimeline(times, prices, max_gap_seconds * 1000), {
        "mark_price_files": len(paths),
        "mark_price_rows_read": rows_read,
        "unique_mark_price_points": len(points),
        "contiguous_segments": int(np.sum(gaps > max_gap_seconds * 1000)) + 1,
        "mark_price_start": _iso(int(times[0])),
        "mark_price_end": _iso(int(times[-1])),
    }


def load_feature_samples(
    runs_root: Path,
    *,
    symbol: str,
    sample_seconds: int,
) -> tuple[list[tuple[int, dict[str, Any]]], dict[str, Any]]:
    paths = _run_files(runs_root, f"features/{symbol}_features_*.jsonl*")
    buckets: dict[int, tuple[int, dict[str, Any], int]] = {}
    rows_read = 0
    bucket_ms = max(1, sample_seconds) * 1000
    for path in paths:
        for row in _jsonl_rows(path):
            rows_read += 1
            timestamp_ms = _timestamp_ms(row.get("ts"))
            if timestamp_ms is None:
                continue
            bucket = timestamp_ms // bucket_ms
            completeness = sum(row.get(key) is not None for key in MODEL_FEATURES[:27])
            current = buckets.get(bucket)
            if current is None or completeness > current[2]:
                buckets[bucket] = (timestamp_ms, row, completeness)
    samples = sorted((timestamp, row) for timestamp, row, _ in buckets.values())
    return samples, {
        "feature_files": len(paths),
        "feature_rows_read": rows_read,
        "sampled_feature_rows": len(samples),
    }


def label_samples(
    timeline: PriceTimeline,
    features: list[tuple[int, dict[str, Any]]],
    *,
    target_bps: float,
    stop_bps_values: list[float],
    cost_bps: float,
    horizon_seconds: int,
) -> list[LabeledSample]:
    rows = []
    for timestamp_ms, feature in features:
        entry_index = timeline.entry_index(timestamp_ms)
        if entry_index is None:
            continue
        end_index, complete = timeline.horizon_end(entry_index, horizon_seconds)
        if end_index <= entry_index:
            continue
        symmetric_direction, symmetric_exit = _symmetric_label(
            timeline,
            entry_index,
            end_index,
            target_bps,
        )
        scenarios = {}
        for stop_bps in stop_bps_values:
            scenarios[stop_bps] = {
                1: _side_barrier_result(
                    timeline,
                    entry_index,
                    end_index,
                    complete,
                    side=1,
                    target_bps=target_bps,
                    stop_bps=stop_bps,
                    cost_bps=cost_bps,
                ),
                -1: _side_barrier_result(
                    timeline,
                    entry_index,
                    end_index,
                    complete,
                    side=-1,
                    target_bps=target_bps,
                    stop_bps=stop_bps,
                    cost_bps=cost_bps,
                ),
            }
        if not complete and symmetric_direction is None and all(
            result.outcome == "incomplete" for scenario in scenarios.values() for result in scenario.values()
        ):
            continue
        rows.append(
            LabeledSample(
                timestamp_ms=timestamp_ms,
                row=feature,
                symmetric_direction=symmetric_direction,
                symmetric_exit_ms=symmetric_exit,
                scenarios=scenarios,
            )
        )
    return rows


def break_even_win_rate(target_bps: float, stop_bps: float, cost_bps: float) -> float:
    return (stop_bps + cost_bps) / (target_bps + stop_bps)


def _symmetric_label(
    timeline: PriceTimeline,
    entry_index: int,
    end_index: int,
    barrier_bps: float,
) -> tuple[int | None, int | None]:
    entry = float(timeline.prices[entry_index])
    upper = entry * (1.0 + barrier_bps / 10_000)
    lower = entry * (1.0 - barrier_bps / 10_000)
    upper_index = timeline.first_ge(entry_index + 1, end_index, upper)
    lower_index = timeline.first_le(entry_index + 1, end_index, lower)
    if upper_index is None and lower_index is None:
        return None, None
    if lower_index is None or (upper_index is not None and upper_index <= lower_index):
        return 1, int(timeline.times_ms[upper_index])
    return -1, int(timeline.times_ms[lower_index])


def _side_barrier_result(
    timeline: PriceTimeline,
    entry_index: int,
    end_index: int,
    complete: bool,
    *,
    side: int,
    target_bps: float,
    stop_bps: float,
    cost_bps: float,
) -> BarrierResult:
    entry = float(timeline.prices[entry_index])
    if side == 1:
        target_index = timeline.first_ge(entry_index + 1, end_index, entry * (1.0 + target_bps / 10_000))
        stop_index = timeline.first_le(entry_index + 1, end_index, entry * (1.0 - stop_bps / 10_000))
    else:
        target_index = timeline.first_le(entry_index + 1, end_index, entry * (1.0 - target_bps / 10_000))
        stop_index = timeline.first_ge(entry_index + 1, end_index, entry * (1.0 + stop_bps / 10_000))
    if target_index is not None and (stop_index is None or target_index <= stop_index):
        return BarrierResult("target", int(timeline.times_ms[target_index]), target_bps - cost_bps)
    if stop_index is not None:
        return BarrierResult("stop", int(timeline.times_ms[stop_index]), -(stop_bps + cost_bps))
    if not complete:
        return BarrierResult("incomplete", None, None)
    signed_move_bps = ((float(timeline.prices[end_index]) - entry) / entry) * side * 10_000
    return BarrierResult("timeout", int(timeline.times_ms[end_index]), signed_move_bps - cost_bps)


def _deterministic_predictions(samples: list[LabeledSample]) -> dict[str, list[int | None]]:
    rows: dict[str, list[int | None]] = {
        "always_long": [],
        "always_short": [],
        "range_reversion": [],
        "range_reversion_micro_confirmed": [],
        "micro_momentum": [],
        "micro_persistence": [],
        "impulse_pullback": [],
        "flow_exhaustion_reversal": [],
        "volatility_squeeze_breakout": [],
        "strategy_family_router": [],
        "htf_momentum": [],
        "regime_router": [],
    }
    for sample in samples:
        row = sample.row
        range_signal = _range_signal(row)
        micro_signal = _micro_signal(row)
        htf_signal = _htf_signal(row)
        range_side = _side(range_signal)
        micro_side = _side(micro_signal)
        htf_side = _side(htf_signal)
        trend_strength = abs(_mean_htf(row, "trend_score"))
        router_signal = htf_signal + 0.30 * micro_signal if trend_strength >= 0.25 else range_signal + 0.20 * micro_signal
        rows["always_long"].append(1)
        rows["always_short"].append(-1)
        rows["range_reversion"].append(range_side)
        rows["range_reversion_micro_confirmed"].append(range_side if range_side == micro_side else None)
        rows["micro_momentum"].append(micro_side)
        rows["micro_persistence"].append(_side(_persistent_micro_signal(row)))
        rows["impulse_pullback"].append(_side(_impulse_pullback_signal(row)))
        rows["flow_exhaustion_reversal"].append(_side(_flow_exhaustion_signal(row)))
        rows["volatility_squeeze_breakout"].append(_side(_squeeze_breakout_signal(row)))
        rows["strategy_family_router"].append(_side(_strategy_family_signal(row)))
        rows["htf_momentum"].append(htf_side)
        rows["regime_router"].append(_side(router_signal))
    return rows


def _scenario_metrics(
    samples: list[LabeledSample],
    predictions: list[int | None],
    stops: list[float],
    *,
    target_bps: float,
    cost_bps: float,
    account_exposure: float,
) -> dict[str, Any]:
    output = {}
    for stop in stops:
        output[str(stop)] = _evaluate_predictions(
            samples,
            predictions,
            stop,
            target_bps=target_bps,
            cost_bps=cost_bps,
            account_exposure=account_exposure,
        )
    return output


def _evaluate_predictions(
    samples: list[LabeledSample],
    predictions: list[int | None],
    stop_bps: float,
    *,
    target_bps: float,
    cost_bps: float,
    account_exposure: float,
) -> dict[str, Any]:
    next_free_ms = -1
    trades: list[tuple[LabeledSample, int, BarrierResult]] = []
    signals = 0
    skipped_overlap = 0
    incomplete = 0
    for sample, prediction in zip(samples, predictions, strict=True):
        if prediction not in {-1, 1}:
            continue
        signals += 1
        if sample.timestamp_ms < next_free_ms:
            skipped_overlap += 1
            continue
        result = sample.scenarios[stop_bps][prediction]
        if result.outcome == "incomplete" or result.exit_ms is None or result.net_bps is None:
            incomplete += 1
            continue
        trades.append((sample, prediction, result))
        next_free_ms = result.exit_ms
    targets = sum(result.outcome == "target" for _, _, result in trades)
    stops = sum(result.outcome == "stop" for _, _, result in trades)
    timeouts = sum(result.outcome == "timeout" for _, _, result in trades)
    net_bps = [float(result.net_bps) for _, _, result in trades]
    account_returns = [value / 100.0 * account_exposure for value in net_bps]
    barrier_count = targets + stops
    lower, upper = _wilson_interval(targets, barrier_count)
    symmetric = [
        prediction == sample.symmetric_direction
        for sample, prediction, _ in trades
        if sample.symmetric_direction is not None
    ]
    by_side = {
        "long": _trade_breakdown([trade for trade in trades if trade[1] == 1]),
        "short": _trade_breakdown([trade for trade in trades if trade[1] == -1]),
    }
    quartiles = []
    if trades:
        for index, chunk in enumerate(np.array_split(np.asarray(trades, dtype=object), min(4, len(trades))), start=1):
            chunk_rows = [tuple(row) for row in chunk.tolist()]
            quartiles.append(
                {
                    "block": index,
                    "start": _iso(chunk_rows[0][0].timestamp_ms),
                    "end": _iso(chunk_rows[-1][0].timestamp_ms),
                    **_trade_breakdown(chunk_rows),
                }
            )
    return {
        "signals": signals,
        "sequential_trades": len(trades),
        "skipped_overlapping_signals": skipped_overlap,
        "incomplete_signals": incomplete,
        "targets": targets,
        "stops": stops,
        "timeouts": timeouts,
        "barrier_win_rate": targets / barrier_count if barrier_count else 0.0,
        "barrier_win_rate_wilson_95": [lower, upper],
        "break_even_win_rate": break_even_win_rate(target_bps, stop_bps, cost_bps),
        "positive_net_trade_rate": sum(value > 0 for value in net_bps) / len(net_bps) if net_bps else 0.0,
        "symmetric_direction_accuracy": mean(symmetric) if symmetric else 0.0,
        "average_net_bps_per_trade": mean(net_bps) if net_bps else 0.0,
        "total_net_bps": sum(net_bps),
        "additive_account_return_pct": sum(account_returns),
        "max_additive_account_drawdown_pct": _max_drawdown(account_returns),
        "max_losing_streak": _max_losing_streak(net_bps),
        "by_side": by_side,
        "chronological_trade_blocks": quartiles,
    }


def _trade_breakdown(trades: list[tuple[LabeledSample, int, BarrierResult]]) -> dict[str, Any]:
    targets = sum(result.outcome == "target" for _, _, result in trades)
    stops = sum(result.outcome == "stop" for _, _, result in trades)
    timeouts = sum(result.outcome == "timeout" for _, _, result in trades)
    barrier_count = targets + stops
    net_bps = [float(result.net_bps) for _, _, result in trades if result.net_bps is not None]
    lower, upper = _wilson_interval(targets, barrier_count)
    return {
        "trades": len(trades),
        "targets": targets,
        "stops": stops,
        "timeouts": timeouts,
        "barrier_win_rate": targets / barrier_count if barrier_count else 0.0,
        "barrier_win_rate_wilson_95": [lower, upper],
        "average_net_bps_per_trade": mean(net_bps) if net_bps else 0.0,
    }


def _fit_logistic(samples: list[LabeledSample]) -> dict[str, Any]:
    usable = [row for row in samples if row.symmetric_direction is not None]
    if len(usable) < 10:
        raise ValueError("Not enough resolved training rows for logistic model")
    matrix = np.asarray([_feature_vector(row.row) for row in usable], dtype=np.float64)
    labels = np.asarray([1.0 if row.symmetric_direction == 1 else 0.0 for row in usable], dtype=np.float64)
    valid_columns = np.any(np.isfinite(matrix), axis=0)
    medians = np.zeros(matrix.shape[1], dtype=np.float64)
    medians[valid_columns] = np.nanmedian(matrix[:, valid_columns], axis=0)
    matrix = np.where(np.isnan(matrix), medians, matrix)
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales = np.where(scales < 1e-9, 1.0, scales)
    normalized = np.clip((matrix - means) / scales, -8.0, 8.0)
    weights = np.zeros(normalized.shape[1], dtype=np.float64)
    bias = 0.0
    learning_rate = 0.08
    l2 = 0.02
    for _ in range(800):
        logits = np.clip(normalized @ weights + bias, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        error = probabilities - labels
        weights -= learning_rate * ((normalized.T @ error) / len(labels) + l2 * weights)
        bias -= learning_rate * float(error.mean())
    return {
        "medians": medians,
        "means": means,
        "scales": scales,
        "weights": weights,
        "bias": bias,
        "training_rows": len(usable),
    }


def _predict_logistic(model: dict[str, Any], samples: list[LabeledSample]) -> list[float]:
    matrix = np.asarray([_feature_vector(row.row) for row in samples], dtype=np.float64)
    matrix = np.where(np.isnan(matrix), model["medians"], matrix)
    normalized = np.clip((matrix - model["means"]) / model["scales"], -8.0, 8.0)
    logits = np.clip(normalized @ model["weights"] + model["bias"], -30.0, 30.0)
    return (1.0 / (1.0 + np.exp(-logits))).tolist()


def _feature_vector(row: dict[str, Any]) -> list[float]:
    flattened = dict(row)
    timeframes = ((row.get("higher_timeframe_context") or {}).get("timeframes") or {})
    for interval in ("15m", "30m", "1h"):
        frame = timeframes.get(interval) or {}
        flattened[f"htf_{interval}_range_position"] = frame.get("range_position")
        flattened[f"htf_{interval}_trend_score"] = frame.get("trend_score")
    return [_float_or_nan(flattened.get(name)) for name in MODEL_FEATURES]


def _calibration_rows(rows: list[tuple[LabeledSample, float]]) -> list[dict[str, Any]]:
    output = []
    for lower in np.arange(0.0, 1.0, 0.1):
        upper = lower + 0.1
        bucket = [(row, value) for row, value in rows if lower <= value < upper or (upper >= 1.0 and value == 1.0)]
        if not bucket:
            continue
        output.append(
            {
                "lower": round(float(lower), 2),
                "upper": round(float(upper), 2),
                "count": len(bucket),
                "average_predicted_up_probability": mean(value for _, value in bucket),
                "actual_up_first_rate": mean(row.symmetric_direction == 1 for row, _ in bucket),
            }
        )
    return output


def _range_signal(row: dict[str, Any]) -> float:
    positions = _htf_values(row, "range_position")
    local = _float_or_none(row.get("range_position_180s"))
    if local is not None:
        positions.append(local)
    return mean(0.5 - value for value in positions) if positions else 0.0


def _micro_signal(row: dict[str, Any]) -> float:
    return (
        0.24 * _value(row, "order_flow_imbalance_1s")
        + 0.14 * _value(row, "order_flow_imbalance_5s")
        + 0.20 * _value(row, "taker_aggression_imbalance_1s")
        + 0.10 * _value(row, "taker_aggression_imbalance_5s")
        + 0.08 * math.tanh(_value(row, "microprice_mid_bps"))
        + 0.06 * math.tanh(_value(row, "vamp_mid_bps"))
        + 0.10 * _value(row, "depth_imbalance_top5")
        + 0.08 * _value(row, "book_imbalance_top")
    )


def _persistent_micro_signal(row: dict[str, Any]) -> float:
    signal = _micro_signal(row)
    side = _side(signal)
    if side is None or abs(signal) < 0.05:
        return 0.0

    components = (
        _value(row, "order_flow_imbalance_1s"),
        _value(row, "order_flow_imbalance_5s"),
        _value(row, "taker_aggression_imbalance_1s"),
        _value(row, "taker_aggression_imbalance_5s"),
        _value(row, "taker_aggression_imbalance_15s"),
        math.tanh(_value(row, "microprice_mid_bps")),
        math.tanh(_value(row, "vamp_mid_bps")),
        _value(row, "depth_imbalance_top5"),
        _value(row, "book_imbalance_top"),
    )
    confirmations = sum(side * value > 0.0 for value in components)
    strong_confirmations = sum(side * value >= 0.05 for value in components)
    return signal if confirmations >= 6 and strong_confirmations >= 3 else 0.0


def _impulse_pullback_signal(row: dict[str, Any]) -> float:
    impulse = _value(row, "return_180s_pct")
    side = _side(impulse)
    if side is None or abs(impulse) < 0.0006:
        return 0.0

    persistent = _persistent_micro_signal(row)
    if _side(persistent) != side:
        return 0.0
    signed_return_15s = side * _value(row, "return_15s_pct")
    signed_return_60s = side * _value(row, "return_60s_pct")
    if -0.0006 <= signed_return_15s <= 0.00015 and signed_return_60s >= -0.0004:
        return side * (abs(impulse) + abs(persistent))
    return 0.0


def _flow_exhaustion_signal(row: dict[str, Any]) -> float:
    impulse = _value(row, "return_180s_pct")
    impulse_side = _side(impulse)
    if impulse_side is None or abs(impulse) < 0.0010:
        return 0.0

    reverse_side = -impulse_side
    flow = 0.45 * _value(row, "taker_aggression_imbalance_5s") + 0.55 * _value(
        row, "taker_aggression_imbalance_15s"
    )
    book_pressure = (
        0.30 * math.tanh(_value(row, "microprice_mid_bps"))
        + 0.20 * math.tanh(_value(row, "vamp_mid_bps"))
        + 0.25 * _value(row, "depth_imbalance_top5")
        + 0.25 * _value(row, "book_imbalance_top")
    )
    stalled = impulse_side * _value(row, "return_15s_pct") <= 0.00005
    if stalled and impulse_side * flow >= 0.15 and reverse_side * book_pressure >= 0.05:
        return reverse_side * (abs(flow) + abs(book_pressure))
    return 0.0


def _squeeze_breakout_signal(row: dict[str, Any]) -> float:
    range_pct = _value(row, "range_180s_pct")
    range_position = _float_or_none(row.get("range_position_180s"))
    if range_position is None or range_pct <= 0.0 or range_pct > 0.0015:
        return 0.0

    side = 1 if range_position >= 0.80 else -1 if range_position <= 0.20 else 0
    if side == 0 or side * _value(row, "return_15s_pct") <= 0.0:
        return 0.0
    persistent = _persistent_micro_signal(row)
    return persistent if _side(persistent) == side else 0.0


def _strategy_family_signal(row: dict[str, Any]) -> float:
    for signal in (
        _flow_exhaustion_signal(row),
        _impulse_pullback_signal(row),
        _squeeze_breakout_signal(row),
        _persistent_micro_signal(row),
    ):
        if _side(signal) is not None:
            return signal
    return 0.0


def _htf_signal(row: dict[str, Any]) -> float:
    trend = _mean_htf(row, "trend_score")
    bias_side = str(row.get("higher_timeframe_bias_side") or "neutral")
    bias = _value(row, "higher_timeframe_bias_strength") * (1 if bias_side == "long" else -1 if bias_side == "short" else 0)
    return trend + 0.20 * math.tanh(_value(row, "return_180s_pct") / 0.002) + 0.20 * bias


def _mean_htf(row: dict[str, Any], key: str) -> float:
    values = _htf_values(row, key)
    return mean(values) if values else 0.0


def _htf_values(row: dict[str, Any], key: str) -> list[float]:
    timeframes = ((row.get("higher_timeframe_context") or {}).get("timeframes") or {})
    values = []
    for interval in ("15m", "30m", "1h"):
        value = _float_or_none((timeframes.get(interval) or {}).get(key))
        if value is not None:
            values.append(value)
    return values


def _side(signal: float) -> int | None:
    if signal > 1e-12:
        return 1
    if signal < -1e-12:
        return -1
    return None


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
    current = 0
    maximum = 0
    for value in values:
        current = current + 1 if value <= 0 else 0
        maximum = max(maximum, current)
    return maximum


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(probability * (1.0 - probability) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def _jsonl_rows(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(row, dict):
                yield row


def _run_files(runs_root: Path, relative_pattern: str) -> list[Path]:
    if (runs_root / "raw_ws").is_dir() or (runs_root / "features").is_dir():
        return sorted(runs_root.glob(relative_pattern))
    return sorted(runs_root.glob(f"*/{relative_pattern}"))


def _timestamp_ms(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value if value > 10_000_000_000 else value * 1000)
    try:
        return int(datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp() * 1000)
    except ValueError:
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _float_or_nan(value: Any) -> float:
    number = _float_or_none(value)
    return number if number is not None else math.nan


def _value(row: dict[str, Any], key: str) -> float:
    return _float_or_none(row.get(key)) or 0.0


def _iso(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat()
