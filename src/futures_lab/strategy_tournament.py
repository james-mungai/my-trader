from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import numpy as np

from futures_lab.first_touch import (
    LabeledSample,
    _evaluate_predictions,
    _flow_exhaustion_signal,
    _htf_signal,
    _impulse_pullback_signal,
    _micro_signal,
    _range_signal,
    _side,
    _squeeze_breakout_signal,
    label_samples,
    load_feature_samples,
    load_mark_price_timeline,
)


@dataclass(frozen=True)
class ExitProfile:
    name: str
    target_bps: float
    stop_bps: float
    horizon_seconds: int


@dataclass(frozen=True)
class SignalVariant:
    family: str
    name: str
    params: dict[str, Any]


DEFAULT_EXIT_PROFILES = (
    ExitProfile("20_20_5m", 20.0, 20.0, 300),
    ExitProfile("20_40_15m", 20.0, 40.0, 900),
    ExitProfile("40_40_15m", 40.0, 40.0, 900),
    ExitProfile("40_60_1h", 40.0, 60.0, 3_600),
    ExitProfile("60_60_1h", 60.0, 60.0, 3_600),
    ExitProfile("60_80_6h", 60.0, 80.0, 21_600),
    ExitProfile("60_100_6h", 60.0, 100.0, 21_600),
    ExitProfile("80_100_6h", 80.0, 100.0, 21_600),
    ExitProfile("80_150_6h", 80.0, 150.0, 21_600),
    ExitProfile("100_150_6h", 100.0, 150.0, 21_600),
    ExitProfile("150_150_6h", 150.0, 150.0, 21_600),
)


FAMILY_DESCRIPTIONS = {
    "volatility_squeeze_breakout": "Compressed 180-second range breaks at an edge with aligned flow.",
    "micro_impulse_continuation": "Multi-component order flow, aggression, and book pressure point together.",
    "range_edge_reversion": "Price reaches a multi-timeframe range edge while trend remains limited.",
    "trend_pullback_continuation": "A directional impulse pauses while microstructure still supports continuation.",
    "flow_exhaustion_reversal": "Aggressive flow persists but opposite-side book pressure absorbs it.",
    "book_pressure_continuation": "Microprice, VAMP, and depth pressure agree with trade aggression.",
    "htf_momentum": "15m/30m/1h trend direction is confirmed rather than contradicted by local flow.",
    "oi_price_expansion": "Open interest expands while price and local flow move together.",
    "btc_flow_lead": "BTC taker aggression leads ETH while ETH local flow does not strongly oppose it.",
}

BASELINE_SIGNAL_FUNCTIONS = {
    "volatility_squeeze_breakout": _squeeze_breakout_signal,
    "micro_impulse_continuation": _micro_signal,
    "range_edge_reversion": _range_signal,
    "trend_pullback_continuation": _impulse_pullback_signal,
    "flow_exhaustion_reversal": _flow_exhaustion_signal,
    "htf_momentum": _htf_signal,
}


def run_strategy_tournament(
    runs_root: Path,
    *,
    symbol: str = "ETHUSDT",
    profiles: Iterable[ExitProfile] = DEFAULT_EXIT_PROFILES,
    cost_bps: float = 10.0,
    sample_seconds: int = 60,
    max_gap_seconds: int = 5,
    discovery_fraction: float = 0.50,
    calibration_end_fraction: float = 0.70,
    account_exposure: float = 4.95,
    minimum_discovery_trades: int = 25,
    minimum_calibration_trades: int = 12,
    matched_coin_seeds: int = 32,
) -> dict[str, Any]:
    selected_profiles = tuple(profiles)
    if not selected_profiles:
        raise ValueError("At least one exit profile is required")
    if not 0.40 <= discovery_fraction < calibration_end_fraction <= 0.80:
        raise ValueError("Chronological split fractions are invalid")
    if cost_bps < 0 or sample_seconds <= 0 or matched_coin_seeds < 8:
        raise ValueError("Tournament cost, sampling, or control settings are invalid")

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
    discovery_features = [row for timestamp, row in features if timestamp < calibration_start_ms]
    discovery_thresholds = _discovery_thresholds(discovery_features)
    variants = build_signal_variants(discovery_thresholds)

    grouped_profiles: dict[tuple[float, int], list[ExitProfile]] = defaultdict(list)
    for profile in selected_profiles:
        grouped_profiles[(profile.target_bps, profile.horizon_seconds)].append(profile)

    period_samples: dict[tuple[float, int, str], list[LabeledSample]] = {}
    for (target_bps, horizon_seconds), profile_rows in grouped_profiles.items():
        stops = sorted({profile.stop_bps for profile in profile_rows})
        labeled = label_samples(
            timeline,
            features,
            target_bps=target_bps,
            stop_bps_values=stops,
            cost_bps=cost_bps,
            horizon_seconds=horizon_seconds,
        )
        period_samples[(target_bps, horizon_seconds, "discovery")] = [
            sample for sample in labeled if sample.timestamp_ms < calibration_start_ms - horizon_seconds * 1_000
        ]
        period_samples[(target_bps, horizon_seconds, "calibration")] = [
            sample
            for sample in labeled
            if calibration_start_ms <= sample.timestamp_ms < holdout_start_ms - horizon_seconds * 1_000
        ]
        period_samples[(target_bps, horizon_seconds, "holdout")] = [
            sample for sample in labeled if sample.timestamp_ms >= holdout_start_ms
        ]

    candidates: list[dict[str, Any]] = []
    predictions_cache: dict[tuple[str, float, int, str], list[int | None]] = {}
    for variant in variants:
        for profile in selected_profiles:
            period_metrics: dict[str, dict[str, Any]] = {}
            for period in ("discovery", "calibration"):
                samples = period_samples[(profile.target_bps, profile.horizon_seconds, period)]
                cache_key = (variant.name, profile.target_bps, profile.horizon_seconds, period)
                predictions = predictions_cache.get(cache_key)
                if predictions is None:
                    predictions = [predict_variant(variant, sample.row) for sample in samples]
                    predictions_cache[cache_key] = predictions
                period_metrics[period] = _evaluate_predictions(
                    samples,
                    predictions,
                    profile.stop_bps,
                    target_bps=profile.target_bps,
                    cost_bps=cost_bps,
                    account_exposure=account_exposure,
                )
            candidates.append(
                _candidate_row(
                    variant,
                    profile,
                    period_metrics["discovery"],
                    period_metrics["calibration"],
                    minimum_discovery_trades=minimum_discovery_trades,
                    minimum_calibration_trades=minimum_calibration_trades,
                )
            )

    selected_by_family = _select_family_candidates(candidates)
    family_results = []
    for family, selection in sorted(selected_by_family.items()):
        variant = next(row for row in variants if row.name == selection["variant"])
        profile = next(row for row in selected_profiles if row.name == selection["profile"])
        samples = period_samples[(profile.target_bps, profile.horizon_seconds, "holdout")]
        predictions = [predict_variant(variant, sample.row) for sample in samples]
        holdout = _evaluate_predictions(
            samples,
            predictions,
            profile.stop_bps,
            target_bps=profile.target_bps,
            cost_bps=cost_bps,
            account_exposure=account_exposure,
        )
        controls = []
        for seed in range(matched_coin_seeds):
            coin_predictions = _matched_coin_predictions(samples, predictions, seed=seed, namespace=symbol)
            controls.append(
                _evaluate_predictions(
                    samples,
                    coin_predictions,
                    profile.stop_bps,
                    target_bps=profile.target_bps,
                    cost_bps=cost_bps,
                    account_exposure=account_exposure,
                )
            )
        long_control = _evaluate_predictions(
            samples,
            [1 if prediction in {-1, 1} else None for prediction in predictions],
            profile.stop_bps,
            target_bps=profile.target_bps,
            cost_bps=cost_bps,
            account_exposure=account_exposure,
        )
        short_control = _evaluate_predictions(
            samples,
            [-1 if prediction in {-1, 1} else None for prediction in predictions],
            profile.stop_bps,
            target_bps=profile.target_bps,
            cost_bps=cost_bps,
            account_exposure=account_exposure,
        )
        control_summary = _matched_control_summary(controls, holdout)
        family_results.append(
            {
                **selection,
                "description": FAMILY_DESCRIPTIONS[family],
                "holdout": _compact_metrics(holdout),
                "matched_coin_control": control_summary,
                "always_long_at_signal_times": _compact_metrics(long_control),
                "always_short_at_signal_times": _compact_metrics(short_control),
                "verdict": _family_verdict(selection, holdout, control_summary),
            }
        )

    family_results.sort(
        key=lambda row: (
            _verdict_rank(row["verdict"]),
            row["holdout"]["average_net_bps_per_trade"],
        ),
        reverse=True,
    )
    return {
        "study": "nested_strategy_family_tournament",
        "symbol": symbol,
        "parameters": {
            "round_trip_cost_bps": cost_bps,
            "sample_seconds": sample_seconds,
            "max_gap_seconds": max_gap_seconds,
            "discovery_fraction": discovery_fraction,
            "calibration_end_fraction": calibration_end_fraction,
            "account_exposure_multiple": account_exposure,
            "minimum_discovery_trades": minimum_discovery_trades,
            "minimum_calibration_trades": minimum_calibration_trades,
            "matched_coin_seeds": matched_coin_seeds,
            "exit_profiles": [profile.__dict__ for profile in selected_profiles],
        },
        "data": {
            **mark_stats,
            **feature_stats,
            "calibration_start": _iso_ms(calibration_start_ms),
            "holdout_start": _iso_ms(holdout_start_ms),
            "discovery_thresholds": discovery_thresholds,
            "signal_variants": len(variants),
            "candidate_configurations": len(candidates),
        },
        "families": [
            {
                "family": family,
                "description": description,
                "variants": sum(variant.family == family for variant in variants),
            }
            for family, description in FAMILY_DESCRIPTIONS.items()
        ],
        "selection_candidates": candidates,
        "family_results": family_results,
        "interpretation_notes": [
            "Variant and exit-profile selection uses discovery plus calibration only; holdout is evaluated once per family.",
            "Every family is side-symmetric: the same rule emits long or short from the sign of its evidence.",
            "Matched coin controls preserve each selected strategy's signal timestamps but randomize direction.",
            "A family can fail selection before holdout; a positive holdout alone does not repair unstable earlier periods.",
            "The tournament searches many deterministic variants, so even holdout winners still require forward shadow confirmation.",
            "Liquidation-pulse strategies are excluded because stable liquidation coverage is too low in this archive.",
        ],
    }


def build_signal_variants(thresholds: dict[str, list[float]]) -> list[SignalVariant]:
    variants: list[SignalVariant] = [
        _variant("volatility_squeeze_breakout", baseline="existing"),
        _variant("micro_impulse_continuation", baseline="existing_raw_micro"),
        _variant("range_edge_reversion", baseline="existing"),
        _variant("trend_pullback_continuation", baseline="existing"),
        _variant("flow_exhaustion_reversal", baseline="existing"),
        _variant("htf_momentum", baseline="existing"),
    ]
    for range_max_bps in (10.0, 15.0, 25.0):
        for edge in (0.75, 0.85):
            for micro_min in (0.03, 0.08):
                variants.append(
                    _variant(
                        "volatility_squeeze_breakout",
                        range_max_bps=range_max_bps,
                        edge=edge,
                        micro_min=micro_min,
                    )
                )
    for micro_min in (0.03, 0.06, 0.10, 0.15):
        for confirmations in (3, 5):
            variants.append(
                _variant("micro_impulse_continuation", micro_min=micro_min, confirmations=confirmations)
            )
    for edge in (0.15, 0.25, 0.35):
        for trend_cap in (0.20, 0.40):
            for micro_rule in ("not_opposed", "confirm"):
                variants.append(
                    _variant("range_edge_reversion", edge=edge, trend_cap=trend_cap, micro_rule=micro_rule)
                )
    for impulse_bps in (5.0, 10.0, 15.0):
        for pullback_bps in (2.0, 4.0):
            for micro_min in (0.03, 0.08):
                variants.append(
                    _variant(
                        "trend_pullback_continuation",
                        impulse_bps=impulse_bps,
                        pullback_bps=pullback_bps,
                        micro_min=micro_min,
                    )
                )
    for impulse_bps in (8.0, 12.0):
        for aggression_min in (0.10, 0.20):
            for pressure_min in (0.03, 0.08):
                variants.append(
                    _variant(
                        "flow_exhaustion_reversal",
                        impulse_bps=impulse_bps,
                        aggression_min=aggression_min,
                        pressure_min=pressure_min,
                    )
                )
    for pressure_min in (0.05, 0.10, 0.20, 0.30):
        for aggression_min in (0.0, 0.10):
            variants.append(
                _variant(
                    "book_pressure_continuation",
                    pressure_min=pressure_min,
                    aggression_min=aggression_min,
                )
            )
    for trend_min in (0.10, 0.20, 0.30, 0.40):
        for micro_min in (0.0, 0.05):
            variants.append(_variant("htf_momentum", trend_min=trend_min, micro_min=micro_min))
    for oi_min in thresholds.get("positive_oi_change", []):
        for micro_rule in ("not_opposed", "confirm"):
            variants.append(_variant("oi_price_expansion", oi_min=oi_min, micro_rule=micro_rule))
    for btc_min in thresholds.get("absolute_btc_aggression", []):
        for micro_rule in ("not_opposed", "confirm"):
            variants.append(_variant("btc_flow_lead", btc_min=btc_min, micro_rule=micro_rule))
    return variants


def predict_variant(variant: SignalVariant, row: dict[str, Any]) -> int | None:
    family = variant.family
    params = variant.params
    micro = _micro_signal(row)
    if "baseline" in params:
        return _side(BASELINE_SIGNAL_FUNCTIONS[family](row))
    if family == "volatility_squeeze_breakout":
        range_bps = _value(row, "range_180s_pct") * 10_000
        position = _optional_value(row.get("range_position_180s"))
        if position is None or range_bps <= 0 or range_bps > params["range_max_bps"]:
            return None
        side = 1 if position >= params["edge"] else -1 if position <= 1 - params["edge"] else None
        if side is None:
            return None
        return side if side * _value(row, "return_15s_pct") > 0 and side * micro >= params["micro_min"] else None
    if family == "micro_impulse_continuation":
        side = _side(micro)
        if side is None or abs(micro) < params["micro_min"]:
            return None
        components = _micro_components(row)
        confirmations = sum(side * value > 0 for value in components)
        return side if confirmations >= params["confirmations"] else None
    if family == "range_edge_reversion":
        position = _mean_range_position(row)
        if position is None or abs(_mean_htf_trend(row)) > params["trend_cap"]:
            return None
        side = 1 if position <= params["edge"] else -1 if position >= 1 - params["edge"] else None
        if side is None:
            return None
        threshold = 0.0 if params["micro_rule"] == "confirm" else -0.05
        return side if side * micro >= threshold else None
    if family == "trend_pullback_continuation":
        impulse = _value(row, "return_180s_pct") * 10_000
        side = _side(impulse)
        if side is None or abs(impulse) < params["impulse_bps"]:
            return None
        signed_pullback = side * _value(row, "return_15s_pct") * 10_000
        htf = _mean_htf_trend(row)
        if not (-params["pullback_bps"] <= signed_pullback <= 1.0) or side * htf < -0.10:
            return None
        return side if side * micro >= params["micro_min"] else None
    if family == "flow_exhaustion_reversal":
        impulse = _value(row, "return_180s_pct") * 10_000
        impulse_side = _side(impulse)
        if impulse_side is None or abs(impulse) < params["impulse_bps"]:
            return None
        reverse = -impulse_side
        aggression = 0.45 * _value(row, "taker_aggression_imbalance_5s") + 0.55 * _value(
            row, "taker_aggression_imbalance_15s"
        )
        pressure = _book_pressure(row)
        stalled = impulse_side * _value(row, "return_15s_pct") <= 0.00005
        return (
            reverse
            if stalled
            and impulse_side * aggression >= params["aggression_min"]
            and reverse * pressure >= params["pressure_min"]
            else None
        )
    if family == "book_pressure_continuation":
        pressure = _book_pressure(row)
        side = _side(pressure)
        aggression = 0.5 * _value(row, "taker_aggression_imbalance_1s") + 0.5 * _value(
            row, "taker_aggression_imbalance_5s"
        )
        return (
            side
            if side is not None
            and abs(pressure) >= params["pressure_min"]
            and side * aggression >= params["aggression_min"]
            else None
        )
    if family == "htf_momentum":
        trend = _mean_htf_trend(row)
        side = _side(trend)
        return side if side is not None and abs(trend) >= params["trend_min"] and side * micro >= params["micro_min"] else None
    if family == "oi_price_expansion":
        oi_change = _value(row, "open_interest_change_5m_pct")
        price_move = _value(row, "return_180s_pct")
        side = _side(price_move)
        if side is None or oi_change < params["oi_min"] or abs(price_move) < 0.0005:
            return None
        threshold = 0.0 if params["micro_rule"] == "confirm" else -0.05
        return side if side * micro >= threshold else None
    if family == "btc_flow_lead":
        btc = _optional_value(row.get("btc_taker_aggression_imbalance_1s"))
        side = _side(btc) if btc is not None else None
        if side is None or abs(btc) < params["btc_min"]:
            return None
        threshold = 0.0 if params["micro_rule"] == "confirm" else -0.05
        return side if side * micro >= threshold else None
    raise ValueError(f"Unsupported signal family: {family}")


def _candidate_row(
    variant: SignalVariant,
    profile: ExitProfile,
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
    side_balance = min(
        _side_balance(discovery_compact),
        _side_balance(calibration_compact),
    )
    qualifies = bool(
        discovery_compact["sequential_trades"] >= minimum_discovery_trades
        and calibration_compact["sequential_trades"] >= minimum_calibration_trades
        and robust_floor > 0
        and side_balance >= 0.15
        and discovery_compact["positive_chronological_block_fraction"] >= 0.50
        and calibration_compact["positive_chronological_block_fraction"] >= 0.50
    )
    return {
        "family": variant.family,
        "variant": variant.name,
        "variant_params": variant.params,
        "profile": profile.name,
        "target_bps": profile.target_bps,
        "stop_bps": profile.stop_bps,
        "horizon_seconds": profile.horizon_seconds,
        "qualifies": qualifies,
        "robust_floor_bps": robust_floor,
        "minimum_side_fraction": side_balance,
        "discovery": discovery_compact,
        "calibration": calibration_compact,
    }


def _select_family_candidates(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate["family"]].append(candidate)
    output = {}
    for family, rows in grouped.items():
        qualified = [row for row in rows if row["qualifies"]]
        pool = qualified or [
            row
            for row in rows
            if row["discovery"]["sequential_trades"] > 0 and row["calibration"]["sequential_trades"] > 0
        ]
        pool = pool or rows
        selected = max(
            pool,
            key=lambda row: (
                row["robust_floor_bps"],
                row["calibration"]["sequential_trades"],
                row["minimum_side_fraction"],
            ),
        )
        output[family] = {
            **selected,
            "family_candidate_count": len(rows),
            "qualified_candidate_count": len(qualified),
            "selection_qualified": bool(qualified),
        }
    return output


def _matched_coin_predictions(
    samples: list[LabeledSample],
    signal_predictions: list[int | None],
    *,
    seed: int,
    namespace: str,
) -> list[int | None]:
    output = []
    for sample, prediction in zip(samples, signal_predictions, strict=True):
        if prediction not in {-1, 1}:
            output.append(None)
            continue
        digest = hashlib.sha256(f"{namespace}:{seed}:{sample.timestamp_ms}".encode("ascii")).digest()
        output.append(1 if digest[0] & 1 else -1)
    return output


def _matched_control_summary(controls: list[dict[str, Any]], holdout: dict[str, Any]) -> dict[str, Any]:
    values = sorted(control["average_net_bps_per_trade"] for control in controls)
    strategy_value = holdout["average_net_bps_per_trade"]
    return {
        "seeds": len(controls),
        "median_average_net_bps": median(values),
        "p05_average_net_bps": float(np.quantile(values, 0.05)),
        "p95_average_net_bps": float(np.quantile(values, 0.95)),
        "positive_seed_fraction": mean(value > 0 for value in values),
        "seed_fraction_at_or_above_strategy": mean(value >= strategy_value for value in values),
        "average_sequential_trades": mean(control["sequential_trades"] for control in controls),
    }


def _family_verdict(selection: dict[str, Any], holdout: dict[str, Any], control: dict[str, Any]) -> str:
    if not selection["selection_qualified"]:
        return "selection_failed"
    holdout_net = holdout["average_net_bps_per_trade"]
    if holdout_net <= 0:
        return "failed_holdout"
    both_sides_positive = bool(
        holdout["by_side"]["long"]["trades"] > 0
        and holdout["by_side"]["short"]["trades"] > 0
        and holdout["by_side"]["long"]["average_net_bps_per_trade"] > 0
        and holdout["by_side"]["short"]["average_net_bps_per_trade"] > 0
    )
    if holdout_net > control["p95_average_net_bps"] and both_sides_positive:
        return "generalized_candidate"
    if holdout_net > control["median_average_net_bps"]:
        return "promising_but_unconfirmed"
    return "indistinguishable_from_direction_luck"


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    blocks = metrics["chronological_trade_blocks"]
    return {
        "signals": metrics["signals"],
        "sequential_trades": metrics["sequential_trades"],
        "targets": metrics["targets"],
        "stops": metrics["stops"],
        "timeouts": metrics["timeouts"],
        "barrier_win_rate": metrics["barrier_win_rate"],
        "break_even_win_rate": metrics["break_even_win_rate"],
        "average_net_bps_per_trade": metrics["average_net_bps_per_trade"],
        "total_net_bps": metrics["total_net_bps"],
        "max_additive_account_drawdown_pct": metrics["max_additive_account_drawdown_pct"],
        "max_losing_streak": metrics["max_losing_streak"],
        "positive_chronological_block_fraction": (
            sum(block["average_net_bps_per_trade"] > 0 for block in blocks) / len(blocks) if blocks else 0.0
        ),
        "long": metrics["by_side"]["long"],
        "short": metrics["by_side"]["short"],
    }


def _period_boundaries(
    features: list[tuple[int, dict[str, Any]]],
    *,
    discovery_fraction: float,
    calibration_end_fraction: float,
) -> tuple[int, int]:
    calibration_index = max(1, min(len(features) - 2, int(len(features) * discovery_fraction)))
    holdout_index = max(calibration_index + 1, min(len(features) - 1, int(len(features) * calibration_end_fraction)))
    return features[calibration_index][0], features[holdout_index][0]


def _discovery_thresholds(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    positive_oi = sorted(
        value
        for row in rows
        if (value := _optional_value(row.get("open_interest_change_5m_pct"))) is not None and value > 0
    )
    absolute_btc = sorted(
        abs(value)
        for row in rows
        if (value := _optional_value(row.get("btc_taker_aggression_imbalance_1s"))) is not None
    )
    return {
        "positive_oi_change": _unique_quantiles(positive_oi, (0.50, 0.70, 0.85)),
        "absolute_btc_aggression": _unique_quantiles(absolute_btc, (0.50, 0.70, 0.85)),
    }


def _unique_quantiles(values: list[float], quantiles: Iterable[float]) -> list[float]:
    if not values:
        return []
    return sorted({float(np.quantile(values, quantile)) for quantile in quantiles})


def _variant(family: str, **params: Any) -> SignalVariant:
    suffix = "_".join(f"{key}-{_param_token(value)}" for key, value in params.items())
    return SignalVariant(family=family, name=f"{family}__{suffix}", params=params)


def _param_token(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}".replace(".", "p")
    return str(value)


def _micro_components(row: dict[str, Any]) -> tuple[float, ...]:
    return (
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


def _book_pressure(row: dict[str, Any]) -> float:
    return (
        0.25 * math.tanh(_value(row, "microprice_mid_bps"))
        + 0.20 * math.tanh(_value(row, "vamp_mid_bps"))
        + 0.30 * _value(row, "depth_imbalance_top5")
        + 0.25 * _value(row, "book_imbalance_top")
    )


def _mean_range_position(row: dict[str, Any]) -> float | None:
    values = []
    local = _optional_value(row.get("range_position_180s"))
    if local is not None:
        values.append(local)
    for frame in _htf_frames(row):
        value = _optional_value(frame.get("range_position"))
        if value is not None:
            values.append(value)
    return mean(values) if values else None


def _mean_htf_trend(row: dict[str, Any]) -> float:
    values = [value for frame in _htf_frames(row) if (value := _optional_value(frame.get("trend_score"))) is not None]
    return mean(values) if values else 0.0


def _htf_frames(row: dict[str, Any]) -> list[dict[str, Any]]:
    timeframes = ((row.get("higher_timeframe_context") or {}).get("timeframes") or {})
    return [timeframes.get(interval) or {} for interval in ("15m", "30m", "1h")]


def _side_balance(metrics: dict[str, Any]) -> float:
    trades = metrics["sequential_trades"]
    return min(metrics["long"]["trades"], metrics["short"]["trades"]) / trades if trades else 0.0


def _value(row: dict[str, Any], key: str) -> float:
    value = _optional_value(row.get(key))
    return value if value is not None else 0.0


def _optional_value(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _iso_ms(timestamp_ms: int) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(timestamp_ms / 1_000, tz=timezone.utc).isoformat()


def _verdict_rank(verdict: str) -> int:
    return {
        "generalized_candidate": 4,
        "promising_but_unconfirmed": 3,
        "indistinguishable_from_direction_luck": 2,
        "failed_holdout": 1,
        "selection_failed": 0,
    }[verdict]
