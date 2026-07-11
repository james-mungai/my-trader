import json

import numpy as np

from futures_lab.robustness_lab import (
    _approximate_proportion_sample_size,
    _exposure_path_metrics,
    _forward_promotion_scorecard,
    _load_forward_summary,
    _moving_block_indices,
    _required_successes_for_wilson,
    _stress_scenario,
)


def test_moving_block_paths_preserve_contiguous_rows_inside_each_block() -> None:
    indices = _moving_block_indices(
        20,
        path_trades=12,
        paths=3,
        block_size=4,
        seed=5,
    )

    assert indices.shape == (3, 12)
    for path in indices:
        for start in range(0, 12, 4):
            block = path[start : start + 4]
            assert np.array_equal((block[1:] - block[:-1]) % 20, np.ones(3, dtype=np.int64))


def test_cost_and_missed_winner_stress_cannot_improve_paired_path_returns() -> None:
    gross = np.asarray(
        [
            [100.0, -150.0, 100.0, 50.0],
            [100.0, 100.0, -150.0, 20.0],
        ]
    )
    uniforms = np.asarray(
        [
            [0.01, 0.50, 0.20, 0.05],
            [0.02, 0.03, 0.50, 0.80],
        ]
    )

    reference = _stress_scenario(
        gross,
        uniforms,
        total_cost_bps=10.0,
        missed_winner_rate=0.0,
        exposure_multiples=(1.0,),
    )
    hostile = _stress_scenario(
        gross,
        uniforms,
        total_cost_bps=20.0,
        missed_winner_rate=0.20,
        exposure_multiples=(1.0,),
    )

    assert hostile["path_total_net_bps"]["mean"] <= reference["path_total_net_bps"]["mean"]
    assert hostile["opportunity_execution_rate"] < reference["opportunity_execution_rate"]


def test_higher_exposure_increases_drawdown_on_the_same_path() -> None:
    paths = np.asarray(
        [
            [50.0, -160.0, -160.0, 90.0],
            [-160.0, 90.0, 90.0, -160.0],
        ]
    )

    low = _exposure_path_metrics(paths, 2.0)
    high = _exposure_path_metrics(paths, 15.0)

    assert high["max_drawdown_pct"]["median"] > low["max_drawdown_pct"]["median"]
    assert high["terminal_compounded_return_pct"]["median"] < low["terminal_compounded_return_pct"]["median"]


def test_power_requirement_falls_as_true_edge_grows() -> None:
    break_even = 70.0 / 120.0

    sample_62 = _approximate_proportion_sample_size(break_even, 0.62)
    sample_65 = _approximate_proportion_sample_size(break_even, 0.65)
    sample_70 = _approximate_proportion_sample_size(break_even, 0.70)

    assert sample_62 is not None and sample_65 is not None and sample_70 is not None
    assert sample_62 > sample_65 > sample_70


def test_required_successes_clear_the_wilson_break_even_boundary() -> None:
    break_even = 70.0 / 120.0
    required = _required_successes_for_wilson(100, break_even)

    assert 58 < required <= 100


def test_incomplete_forward_sample_remains_pending() -> None:
    summary = _forward_summary(closed=8, targets=5, stops=3)

    scorecard = _forward_promotion_scorecard(summary)

    assert scorecard["overall_status"] == "scorecard_pending"
    assert next(row for row in scorecard["gates"] if row["gate"] == "minimum_sample")["status"] == "pending"
    assert next(row for row in scorecard["gates"] if row["gate"] == "feed_health")["status"] == "pass"


def test_unhealthy_feed_fails_forward_scorecard_immediately() -> None:
    summary = _forward_summary(closed=8, targets=5, stops=3)
    summary["last_health"]["connected"] = False

    scorecard = _forward_promotion_scorecard(summary)

    assert scorecard["overall_status"] == "scorecard_fail"
    assert next(row for row in scorecard["gates"] if row["gate"] == "feed_health")["status"] == "fail"


def test_zero_feed_lag_is_valid_health_data() -> None:
    summary = _forward_summary(closed=8, targets=5, stops=3)
    summary["last_health"]["data_age_seconds"] = 0.0
    summary["last_health"]["avg_book_event_lag_30s_ms"] = 0.0
    summary["last_health"]["avg_trade_event_lag_30s_ms"] = 0.0

    scorecard = _forward_promotion_scorecard(summary)

    assert next(row for row in scorecard["gates"] if row["gate"] == "feed_health")["status"] == "pass"


def test_forward_summary_loader_accepts_utf8_bom(tmp_path) -> None:
    path = tmp_path / "forward.json"
    path.write_text(json.dumps(_forward_summary(closed=2, targets=1, stops=1)), encoding="utf-8-sig")

    loaded = _load_forward_summary(path)

    assert loaded is not None
    assert loaded["arms"]["squeeze_breakout"]["closed"] == 2


def _forward_summary(*, closed: int, targets: int, stops: int) -> dict:
    strategy = {
        "closed": closed,
        "targets": targets,
        "stops": stops,
        "total_net_bps": 50.0,
        "average_net_bps": 50.0 / closed,
        "by_side": {
            "long": {"trades": closed // 2, "net_bps": 25.0},
            "short": {"trades": closed - closed // 2, "net_bps": 25.0},
        },
        "chronological_blocks_25": [{"trades": f"1-{closed}", "net_bps": 50.0}],
    }
    control = {
        "closed": closed,
        "average_net_bps": -5.0,
    }
    return {
        "break_even_win_rate": 70.0 / 120.0,
        "last_health": {
            "ts": "2026-07-11T15:00:00+00:00",
            "connected": True,
            "data_age_seconds": 0.2,
            "avg_book_event_lag_30s_ms": 3.0,
            "avg_trade_event_lag_30s_ms": 4.0,
        },
        "arms": {
            "squeeze_breakout": strategy,
            "squeeze_matched_coin": control,
        },
    }
