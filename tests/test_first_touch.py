import json
from datetime import datetime, timezone

import numpy as np

from futures_lab.first_touch import (
    PriceTimeline,
    _evaluate_predictions,
    break_even_win_rate,
    label_samples,
    load_feature_samples,
    load_mark_price_timeline,
)


def test_price_timeline_finds_first_barrier_and_respects_gap() -> None:
    timeline = PriceTimeline(
        np.asarray([0, 1_000, 2_000, 20_000, 21_000]),
        np.asarray([100.0, 100.2, 100.7, 100.0, 99.0]),
        max_gap_ms=2_000,
    )

    assert timeline.first_ge(1, 2, 100.6) == 2
    assert timeline.first_le(1, 2, 99.9) is None
    assert timeline.horizon_end(0, 10) == (2, False)


def test_label_and_sequential_economics_use_real_barriers() -> None:
    timeline = PriceTimeline(
        np.arange(0, 14_000, 1_000),
        np.asarray([100.0, 100.1, 100.61, 100.2, 100.0, 99.4, 99.3, 99.9, 100.0, 100.7, 100.8, 100.0, 99.4, 99.3]),
        max_gap_ms=2_000,
    )
    features = [
        (0, {"ts": "1970-01-01T00:00:00+00:00"}),
        (4_000, {"ts": "1970-01-01T00:00:04+00:00"}),
        (8_000, {"ts": "1970-01-01T00:00:08+00:00"}),
    ]

    samples = label_samples(
        timeline,
        features,
        target_bps=60,
        stop_bps_values=[50],
        cost_bps=10,
        horizon_seconds=5,
    )
    metrics = _evaluate_predictions(
        samples,
        [1, -1, 1],
        50,
        target_bps=60,
        cost_bps=10,
        account_exposure=20,
    )

    assert metrics["sequential_trades"] == 3
    assert metrics["targets"] == 3
    assert metrics["average_net_bps_per_trade"] == 50
    assert metrics["additive_account_return_pct"] == 30
    assert metrics["by_side"]["long"]["targets"] == 2
    assert metrics["by_side"]["short"]["targets"] == 1
    assert len(metrics["chronological_trade_blocks"]) == 3
    assert break_even_win_rate(60, 50, 10) == 60 / 110


def test_loaders_deduplicate_overlapping_run_data(tmp_path) -> None:
    runs = tmp_path / "runs"
    feature_row = {
        "ts": datetime.fromtimestamp(0, tz=timezone.utc).isoformat(),
        "symbol": "ETHUSDT",
        "order_flow_imbalance_1s": 0.2,
    }
    mark_rows = [
        {"message": {"data": {"E": index * 1_000, "p": str(100 + index / 10)}}}
        for index in range(3)
    ]
    for name in ("one", "two"):
        raw_dir = runs / name / "raw_ws"
        feature_dir = runs / name / "features"
        raw_dir.mkdir(parents=True)
        feature_dir.mkdir(parents=True)
        (raw_dir / "ETHUSDT_markPriceUpdate_test.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in mark_rows),
            encoding="utf-8",
        )
        (feature_dir / "ETHUSDT_features_test.jsonl").write_text(json.dumps(feature_row) + "\n", encoding="utf-8")

    timeline, mark_stats = load_mark_price_timeline(runs, symbol="ETHUSDT", max_gap_seconds=2)
    features, feature_stats = load_feature_samples(runs, symbol="ETHUSDT", sample_seconds=60)

    assert len(timeline.times_ms) == 3
    assert mark_stats["mark_price_rows_read"] == 6
    assert mark_stats["unique_mark_price_points"] == 3
    assert len(features) == 1
    assert feature_stats["feature_rows_read"] == 2
