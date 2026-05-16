import os
from datetime import datetime, timedelta, timezone

from futures_lab.config import Settings
from futures_lab.data_ops import compress_raw, prune_raw, summarize_data


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
