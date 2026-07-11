from datetime import datetime, timezone

from futures_lab.first_touch import BarrierResult, LabeledSample
from futures_lab.market_atlas import (
    _add_neighbor_stability,
    _coin_predictions,
    _regime_labels,
    _robust_candidates,
    _touch_summary,
)


def test_touch_summary_reports_direction_resolution_and_time() -> None:
    rows = [
        _sample(0, direction=1, exit_ms=20_000),
        _sample(60_000, direction=-1, exit_ms=100_000),
        _sample(120_000, direction=None, exit_ms=None),
    ]

    summary = _touch_summary(rows)

    assert summary["samples"] == 3
    assert summary["resolved"] == 2
    assert summary["resolution_rate"] == 2 / 3
    assert summary["up_first_rate"] == 0.5
    assert summary["median_touch_seconds"] == 30


def test_coin_control_is_deterministic_and_two_sided() -> None:
    rows = [_sample(index * 60_000, direction=1, exit_ms=index * 60_000 + 1_000) for index in range(20)]

    first = _coin_predictions(rows)
    second = _coin_predictions(rows)

    assert first == second
    assert set(first) == {-1, 1}


def test_regime_labels_use_terciles_and_utc_windows() -> None:
    timestamp = int(datetime(2026, 7, 11, 10, tzinfo=timezone.utc).timestamp() * 1_000)
    row = _sample(timestamp, direction=1, exit_ms=timestamp + 1_000)
    row.row.update({"realized_vol_180s_pct": 0.9, "return_180s_pct": -0.5, "spread_bps": 0.2})

    labels = _regime_labels(
        row,
        {"volatility": [0.2, 0.6], "trend": [-0.2, 0.2], "spread": [0.1, 0.4]},
    )

    assert labels == {
        "session_utc": "08-16 Europe",
        "volatility": "high",
        "trend": "down",
        "spread": "normal",
    }


def test_robust_shortlist_requires_both_periods_and_neighbor_support() -> None:
    rows = []
    for period, average in (("discovery", 3.0), ("validation", 2.0)):
        for target, stop in ((40.0, 40.0), (40.0, 60.0), (60.0, 40.0)):
            rows.append(_strategy_row(period, average, target, stop))
    _add_neighbor_stability(rows, [40.0, 60.0], [40.0, 60.0])

    shortlist = _robust_candidates(rows, minimum_trades=20)
    selected = next(row for row in shortlist if row["target_bps"] == 40 and row["stop_bps"] == 40)

    assert selected["qualifies"] is True
    assert selected["robust_floor_bps"] == 2.0
    assert selected["validation_neighbor_positive_fraction"] == 1.0


def _sample(timestamp_ms: int, *, direction: int | None, exit_ms: int | None) -> LabeledSample:
    return LabeledSample(
        timestamp_ms=timestamp_ms,
        row={},
        symmetric_direction=direction,
        symmetric_exit_ms=exit_ms,
        scenarios={60.0: {1: BarrierResult("timeout", exit_ms, 0), -1: BarrierResult("timeout", exit_ms, 0)}},
    )


def _strategy_row(period: str, average: float, target: float, stop: float) -> dict:
    return {
        "strategy": "test",
        "target_bps": target,
        "stop_bps": stop,
        "horizon_seconds": 900,
        "period": period,
        "sequential_trades": 30,
        "average_net_bps_per_trade": average,
        "barrier_win_rate": 0.7,
        "break_even_win_rate": 0.6,
        "positive_chronological_block_fraction": 0.75,
        "max_additive_account_drawdown_pct": 4.0,
        "neighbor_positive_fraction": None,
    }
