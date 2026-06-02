import gzip
import json
from datetime import datetime, timedelta, timezone

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
        SYMBOL="BTCUSDT",
        RECORD_BOOK_TICKER_MIN_INTERVAL_MS=0,
        RAW_ROTATION_MINUTES=60,
        COMPRESS_ROTATED_RAW=True,
    )
    recorder = BinanceStreamRecorder(settings, MarketStateBook(settings), AuditLog(settings))
    previous = "2026-05-10T1700Z"
    recorder._current_buckets["BTCUSDT:bookTicker"] = previous
    raw_path = tmp_path / "raw_ws" / f"BTCUSDT_bookTicker_{previous}.jsonl"
    raw_path.write_text(json.dumps({"ok": True}) + "\n", encoding="utf-8")

    recorder._compress_previous_bucket("BTCUSDT", "bookTicker", "2026-05-10T1800Z")

    gz_path = raw_path.with_suffix(raw_path.suffix + ".gz")
    assert not raw_path.exists()
    assert gz_path.exists()
    with gzip.open(gz_path, "rt", encoding="utf-8") as handle:
        assert json.loads(handle.readline()) == {"ok": True}


def test_current_stream_profile_keeps_existing_combined_layout(tmp_path):
    settings = Settings(
        DATA_DIR=str(tmp_path),
        SYMBOL="ETHUSDT",
        CROSS_MARKET_ENABLED=True,
        CROSS_MARKET_ANCHOR_SYMBOL="BTCUSDT",
        BINANCE_STREAM_PROFILE="current",
    )
    recorder = BinanceStreamRecorder(settings, MarketStateBook(settings), AuditLog(settings))

    specs = recorder._stream_specs()

    assert [spec.name for spec in specs] == ["public-current", "market-current"]
    assert specs[0].streams == ("ethusdt@bookTicker", "ethusdt@depth5@100ms", "btcusdt@bookTicker")
    assert specs[1].streams == (
        "ethusdt@aggTrade",
        "ethusdt@markPrice@1s",
        "ethusdt@kline_1m",
        "ethusdt@forceOrder",
        "btcusdt@aggTrade",
        "btcusdt@markPrice@1s",
    )


def test_hot_split_stream_profile_isolates_primary_hot_feeds(tmp_path):
    settings = Settings(
        DATA_DIR=str(tmp_path),
        SYMBOL="ETHUSDT",
        CROSS_MARKET_ENABLED=True,
        CROSS_MARKET_ANCHOR_SYMBOL="BTCUSDT",
        BINANCE_STREAM_PROFILE="hot-split",
    )
    recorder = BinanceStreamRecorder(settings, MarketStateBook(settings), AuditLog(settings))

    specs = recorder._stream_specs()

    assert [spec.name for spec in specs] == ["bookticker", "depth", "aggtrade", "public-context", "market-context"]
    assert specs[0].streams == ("ethusdt@bookTicker",)
    assert specs[1].streams == ("ethusdt@depth5@100ms",)
    assert specs[2].streams == ("ethusdt@aggTrade",)
    assert specs[3].streams == ("btcusdt@bookTicker",)
    assert specs[4].streams == (
        "ethusdt@markPrice@1s",
        "ethusdt@kline_1m",
        "ethusdt@forceOrder",
        "btcusdt@aggTrade",
        "btcusdt@markPrice@1s",
    )


def test_hot_split_stream_profile_can_omit_primary_book_ticker(tmp_path):
    settings = Settings(
        DATA_DIR=str(tmp_path),
        SYMBOL="ETHUSDT",
        CROSS_MARKET_ENABLED=True,
        CROSS_MARKET_ANCHOR_SYMBOL="BTCUSDT",
        BINANCE_STREAM_PROFILE="hot-split",
        CONSUME_BOOK_TICKER_STREAM=False,
    )
    recorder = BinanceStreamRecorder(settings, MarketStateBook(settings), AuditLog(settings))

    specs = recorder._stream_specs()

    assert [spec.name for spec in specs] == ["depth", "aggtrade", "market-context"]
    assert specs[0].streams == ("ethusdt@depth5@100ms",)
    assert all("ethusdt@bookTicker" not in spec.streams for spec in specs)
    assert all("btcusdt@bookTicker" not in spec.streams for spec in specs)


def test_book_ticker_ingestion_can_be_throttled(tmp_path):
    settings = Settings(
        DATA_DIR=str(tmp_path),
        SYMBOL="ETHUSDT",
        CONSUME_BOOK_TICKER_MIN_INTERVAL_MS=50,
    )
    recorder = BinanceStreamRecorder(settings, MarketStateBook(settings), AuditLog(settings))
    first = datetime(2026, 5, 31, 12, 0, tzinfo=timezone.utc)
    envelope = {
        "stream": "ethusdt@bookTicker",
        "data": {
            "e": "bookTicker",
            "E": int(first.timestamp() * 1000),
            "s": "ETHUSDT",
            "b": "100.00",
            "a": "100.01",
            "B": "1",
            "A": "1",
        },
    }

    assert recorder._should_ingest_payload(envelope, envelope["data"], first)
    assert not recorder._should_ingest_payload(envelope, envelope["data"], first + timedelta(milliseconds=10))
    assert recorder._should_ingest_payload(envelope, envelope["data"], first + timedelta(milliseconds=50))


def test_partial_depth_stream_replaces_book_even_with_depth_update_event(tmp_path):
    settings = Settings(
        DATA_DIR=str(tmp_path),
        SYMBOL="ETHUSDT",
        CONSUME_DEPTH_TOP_BOOK_MIN_INTERVAL_MS=0,
        MAX_EXCHANGE_EVENT_LAG_MS=10_000,
    )
    state = MarketStateBook(settings)
    recorder = BinanceStreamRecorder(settings, state, AuditLog(settings))
    first = datetime(2026, 6, 2, 9, 0, tzinfo=timezone.utc)

    recorder._ingest_payload(
        {
            "stream": "ethusdt@depth5@100ms",
            "data": {
                "e": "depthUpdate",
                "E": int(first.timestamp() * 1000),
                "s": "ETHUSDT",
                "b": [["100.00", "3"], ["99.99", "2"]],
                "a": [["100.02", "4"], ["100.03", "2"]],
            },
        },
        {
            "e": "depthUpdate",
            "E": int(first.timestamp() * 1000),
            "s": "ETHUSDT",
            "b": [["100.00", "3"], ["99.99", "2"]],
            "a": [["100.02", "4"], ["100.03", "2"]],
        },
        received_at=first,
    )
    recorder._ingest_payload(
        {
            "stream": "ethusdt@depth5@100ms",
            "data": {
                "e": "depthUpdate",
                "E": int((first + timedelta(milliseconds=100)).timestamp() * 1000),
                "s": "ETHUSDT",
                "b": [["99.98", "5"], ["99.97", "2"]],
                "a": [["100.01", "6"], ["100.02", "2"]],
            },
        },
        {
            "e": "depthUpdate",
            "E": int((first + timedelta(milliseconds=100)).timestamp() * 1000),
            "s": "ETHUSDT",
            "b": [["99.98", "5"], ["99.97", "2"]],
            "a": [["100.01", "6"], ["100.02", "2"]],
        },
        received_at=first + timedelta(milliseconds=100),
    )

    snapshot = state.snapshot(first + timedelta(milliseconds=100))

    assert snapshot.best_bid == 99.98
    assert snapshot.best_ask == 100.01
    assert snapshot.spread_bps is not None
    assert snapshot.spread_bps > 0
