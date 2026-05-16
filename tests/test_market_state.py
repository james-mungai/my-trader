from datetime import datetime, timedelta, timezone

from futures_lab.config import Settings
from futures_lab.market_state import MarketStateBook


def test_market_state_rolls_depth_liquidation_and_latency_features():
    settings = Settings(MIN_WARMUP_SECONDS=1, STALE_AFTER_SECONDS=999)
    book = MarketStateBook(settings)
    book.set_connected(True)
    received_at = datetime(2026, 5, 13, 9, 0, 1, tzinfo=timezone.utc)
    event_ms = int((received_at - timedelta(milliseconds=25)).timestamp() * 1000)

    book.ingest(
        {
            "lastUpdateId": 1,
            "E": event_ms,
            "bids": [["100.00", "5"], ["99.99", "3"], ["99.98", "2"]],
            "asks": [["100.01", "2"], ["100.02", "2"], ["100.03", "1"]],
        },
        received_at=received_at,
    )
    book.ingest(
        {
            "e": "bookTicker",
            "E": event_ms,
            "b": "100.00",
            "a": "100.01",
            "B": "5",
            "A": "2",
        },
        received_at=received_at,
    )
    book.ingest(
        {
            "e": "forceOrder",
            "E": event_ms,
            "o": {"S": "BUY", "p": "100.00", "q": "0.5"},
        },
        received_at=received_at,
    )

    snapshot = book.snapshot(current=received_at + timedelta(seconds=2))

    assert snapshot.depth_bid_qty_top5 == 10
    assert snapshot.depth_ask_qty_top5 == 5
    assert snapshot.depth_imbalance_top5 == 1 / 3
    assert snapshot.short_liquidation_notional_30s == 50
    assert snapshot.liquidation_buy_ratio_30s == 1.0
    assert snapshot.exchange_event_lag_ms == 25
    assert snapshot.avg_event_lag_30s_ms == 25


def test_market_state_rolls_open_interest_change():
    book = MarketStateBook(Settings())
    start = datetime(2026, 5, 13, 9, 0, tzinfo=timezone.utc)

    book.set_open_interest(1000, updated_at=start)
    book.set_open_interest(1025, updated_at=start + timedelta(minutes=2))

    snapshot = book.snapshot(current=start + timedelta(minutes=2))

    assert snapshot.open_interest == 1025
    assert snapshot.open_interest_change_5m_pct == 0.025
