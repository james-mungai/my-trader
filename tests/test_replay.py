import json
from datetime import datetime, timedelta, timezone

from futures_lab.config import Settings
from futures_lab.replay import discover_raw_files, replay_files


def _write_row(handle, received_at: datetime, payload: dict) -> None:
    handle.write(
        json.dumps(
            {
                "received_at": received_at.isoformat(),
                "message": {"stream": "btcusdt@test", "data": payload},
            }
        )
    )
    handle.write("\n")


def test_replay_loads_recorded_jsonl_and_runs_decisions(tmp_path):
    raw_dir = tmp_path / "raw_ws"
    raw_dir.mkdir(parents=True)
    path = raw_dir / "BTCUSDT_fixture_2026-05-03.jsonl"
    start = datetime(2026, 5, 3, 8, 0, tzinfo=timezone.utc)
    with path.open("w", encoding="utf-8") as handle:
        for idx in range(20):
            ts = start + timedelta(seconds=idx)
            price = 100.0 + (idx * 0.01)
            _write_row(
                handle,
                ts,
                {
                    "e": "bookTicker",
                    "E": int(ts.timestamp() * 1000),
                    "b": str(price - 0.005),
                    "a": str(price + 0.005),
                    "B": "12",
                    "A": "8",
                },
            )
            _write_row(
                handle,
                ts + timedelta(milliseconds=10),
                {
                    "e": "aggTrade",
                    "E": int(ts.timestamp() * 1000),
                    "p": str(price),
                    "q": "1.0",
                    "m": False,
                },
            )

    settings = Settings(
        DATA_DIR=str(tmp_path),
        MIN_WARMUP_SECONDS=1,
        STALE_AFTER_SECONDS=999,
        STATE_WINDOW_SECONDS=60,
    )
    files = discover_raw_files(settings, pattern="BTCUSDT_fixture_*.jsonl")
    summary = replay_files(settings, files, decision_interval_ms=1000)

    assert files == [path]
    assert summary.messages == 40
    assert summary.decisions > 0
    assert summary.model_dump()["messages"] == 40

