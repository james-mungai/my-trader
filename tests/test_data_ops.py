import os
from datetime import datetime, timedelta, timezone

from futures_lab.config import Settings
from futures_lab.data_ops import compress_raw, prune_raw, summarize_data, summarize_regime_outcomes


def test_data_summary_compresses_and_prunes_raw_files(tmp_path):
    raw_dir = tmp_path / "raw_ws"
    raw_dir.mkdir(parents=True)
    old_path = raw_dir / "BTCUSDT_bookTicker_2026-05-11T0700Z.jsonl"
    old_path.write_text('{"ok":true}\n' * 10, encoding="utf-8")
    old_time = (datetime.now(timezone.utc) - timedelta(hours=3)).timestamp()
    os.utime(old_path, (old_time, old_time))

    settings = Settings(DATA_DIR=str(tmp_path))
    summary = summarize_data(settings)
    assert summary.total_bytes > 0
    assert summary.largest_files[0].path.endswith(".jsonl")

    compressed = compress_raw(settings, older_than_minutes=5)
    gz_path = old_path.with_suffix(old_path.suffix + ".gz")
    assert compressed.changed == 1
    assert gz_path.exists()
    assert not old_path.exists()

    pruned_preview = prune_raw(settings, older_than_hours=1, dry_run=True)
    assert pruned_preview.matched == 1
    assert gz_path.exists()

    pruned = prune_raw(settings, older_than_hours=1)
    assert pruned.changed == 1
    assert not gz_path.exists()


def test_regime_outcome_summary_counts_targets_and_horizons(tmp_path):
    outcome_dir = tmp_path / "regime_outcomes"
    outcome_dir.mkdir(parents=True)
    path = outcome_dir / "BTCUSDT_regime_outcomes_2026-05-16.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"event":"open","side":"short","state":"short_continuation_confirmed","quality_score":0.81}',
                '{"event":"close","side":"short","close_reason":"max_horizon_elapsed","target_hits":{"0.001":{"hit":true},"0.002":{"hit":false}},"stop_hits":{"0.001":{"hit":false}},"target_before_stop":{"0.001":{"0.001":true}},"horizons":{"15":{"direction_correct":true},"60":{"direction_correct":false}},"early_follow_through":{"qualified":true}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = summarize_regime_outcomes(Settings(DATA_DIR=str(tmp_path)))

    assert summary.opens == 1
    assert summary.closes == 1
    assert summary.sides == {"short": 1}
    assert summary.target_hits == {"0.001": 1}
    assert summary.target_before_stop == {"0.001": {"0.001": 1}}
    assert summary.horizon_direction_correct["15"] == {"correct": 1}
    assert summary.horizon_direction_correct["60"] == {"wrong": 1}
    assert summary.early_follow_through == {"qualified": 1}
    assert summary.side_early_follow_through == {"short": {"qualified": 1}}
    assert summary.average_quality_score == 0.81
