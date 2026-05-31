from datetime import datetime, timezone

from futures_lab.latency_probe import (
    build_probe_streams,
    summarize_values,
    _event_time_iso,
    _lag_ms,
)


def test_build_current_probe_profile_mirrors_split_routes() -> None:
    specs = build_probe_streams(profile="current", symbol="ETHUSDT", anchor_symbol="BTCUSDT", depth_levels=5)

    assert len(specs) == 2
    assert specs[0].name == "public-current"
    assert "ethusdt@bookTicker" in specs[0].streams
    assert "ethusdt@depth5@100ms" in specs[0].streams
    assert "btcusdt@bookTicker" in specs[0].streams
    assert specs[1].name == "market-current"
    assert "ethusdt@aggTrade" in specs[1].streams
    assert "ethusdt@forceOrder" in specs[1].streams
    assert "btcusdt@aggTrade" in specs[1].streams


def test_build_hot_split_probe_profile_uses_separate_connections() -> None:
    specs = build_probe_streams(profile="hot-split", symbol="ETHUSDT", depth_levels=10)

    assert [spec.name for spec in specs] == ["bookticker", "depth", "aggtrade"]
    assert specs[1].streams == ("ethusdt@depth10@100ms",)


def test_summarize_values_reports_percentiles() -> None:
    summary = summarize_values([1, 2, 3, 4])

    assert summary["count"] == 4
    assert summary["min"] == 1
    assert summary["p50"] == 2
    assert summary["p75"] == 3
    assert summary["p99"] == 4
    assert summary["max"] == 4


def test_event_lag_helpers_tolerate_missing_or_invalid_values() -> None:
    received_at = datetime(2026, 5, 31, 12, 0, 1, tzinfo=timezone.utc)
    event_ms = 1780228800000

    assert _lag_ms(received_at, event_ms) == 1000
    assert _event_time_iso(event_ms) == "2026-05-31T12:00:00+00:00"
    assert _lag_ms(received_at, None) is None
    assert _event_time_iso("not-a-timestamp") is None
