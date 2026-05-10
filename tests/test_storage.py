import gzip
import json

from futures_lab.audit import AuditLog
from futures_lab.binance_streams import BinanceStreamRecorder
from futures_lab.config import Settings
from futures_lab.market_state import MarketStateBook


def _book_ticker(price: str = "100.0") -> dict:
    return {
        "stream": "btcusdt@bookTicker",
        "data": {
            "e": "bookTicker",
            "E": 1,
            "b": price,
            "a": str(float(price) + 0.01),
            "B": "1",
            "A": "1",
        },
    }


def test_raw_book_ticker_downsampling(tmp_path):
    settings = Settings(
        DATA_DIR=str(tmp_path),
        RECORD_BOOK_TICKER_MIN_INTERVAL_MS=10_000,
        COMPRESS_ROTATED_RAW=False,
    )
    recorder = BinanceStreamRecorder(settings, MarketStateBook(settings), AuditLog(settings))

    recorder._record_raw(_book_ticker("100"))
    recorder._record_raw(_book_ticker("101"))

    files = list((tmp_path / "raw_ws").glob("*.jsonl"))
    assert len(files) == 1
    rows = files[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 1


def test_compresses_previous_rotation_bucket(tmp_path):
    settings = Settings(
        DATA_DIR=str(tmp_path),
        RECORD_BOOK_TICKER_MIN_INTERVAL_MS=0,
        RAW_ROTATION_MINUTES=60,
        COMPRESS_ROTATED_RAW=True,
    )
    recorder = BinanceStreamRecorder(settings, MarketStateBook(settings), AuditLog(settings))
    previous = "2026-05-10T1700Z"
    recorder._current_buckets["bookTicker"] = previous
    raw_path = tmp_path / "raw_ws" / f"BTCUSDT_bookTicker_{previous}.jsonl"
    raw_path.write_text(json.dumps({"ok": True}) + "\n", encoding="utf-8")

    recorder._compress_previous_bucket("bookTicker", "2026-05-10T1800Z")

    gz_path = raw_path.with_suffix(raw_path.suffix + ".gz")
    assert not raw_path.exists()
    assert gz_path.exists()
    with gzip.open(gz_path, "rt", encoding="utf-8") as handle:
        assert json.loads(handle.readline()) == {"ok": True}
