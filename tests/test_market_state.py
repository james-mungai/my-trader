from datetime import datetime, timedelta, timezone

import pytest

from futures_lab.config import Settings
from futures_lab.market_state import MarketStateBook
from futures_lab.models import Regime


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
    assert snapshot.hot_event_lag_ms == 25
    assert snapshot.avg_hot_event_lag_30s_ms == 25


def test_depth_snapshot_updates_top_of_book_without_book_ticker():
    settings = Settings(
        MIN_WARMUP_SECONDS=1,
        STALE_AFTER_SECONDS=999,
        DEPTH_LEVELS=5,
        CONSUME_DEPTH_TOP_BOOK_MIN_INTERVAL_MS=0,
    )
    book = MarketStateBook(settings)
    book.set_connected(True)
    start = datetime(2026, 6, 1, 9, 0, tzinfo=timezone.utc)

    book.ingest(
        {
            "lastUpdateId": 1,
            "E": int(start.timestamp() * 1000),
            "bids": [["100.00", "4"], ["99.99", "2"]],
            "asks": [["100.02", "3"], ["100.03", "1"]],
        },
        received_at=start,
    )
    book.ingest(
        {
            "lastUpdateId": 2,
            "E": int((start + timedelta(milliseconds=100)).timestamp() * 1000),
            "bids": [["100.01", "5"], ["100.00", "2"]],
            "asks": [["100.02", "2"], ["100.03", "1"]],
        },
        received_at=start + timedelta(milliseconds=100),
    )

    snapshot = book.snapshot(current=start + timedelta(milliseconds=100))

    assert snapshot.best_bid == 100.01
    assert snapshot.best_ask == 100.02
    assert snapshot.mid_price == pytest.approx(100.015)
    assert snapshot.spread_bps == pytest.approx(((100.02 - 100.01) / 100.015) * 10_000)
    assert snapshot.book_imbalance_top == pytest.approx((5 - 2) / (5 + 2))
    assert snapshot.order_flow_imbalance_1s is not None


def test_market_state_rolls_open_interest_change():
    book = MarketStateBook(Settings())
    start = datetime(2026, 5, 13, 9, 0, tzinfo=timezone.utc)

    book.set_open_interest(1000, updated_at=start)
    book.set_open_interest(1025, updated_at=start + timedelta(minutes=2))

    snapshot = book.snapshot(current=start + timedelta(minutes=2))

    assert snapshot.open_interest == 1025
    assert snapshot.open_interest_change_5m_pct == 0.025


def test_market_state_drops_exchange_events_that_arrive_too_late():
    settings = Settings(MAX_EXCHANGE_EVENT_LAG_MS=1_000, MIN_WARMUP_SECONDS=1, STALE_AFTER_SECONDS=999)
    book = MarketStateBook(settings)
    book.set_connected(True)
    received_at = datetime(2026, 5, 30, 9, 0, 10, tzinfo=timezone.utc)
    stale_event_ms = int((received_at - timedelta(seconds=10)).timestamp() * 1000)

    accepted = book.ingest(
        {
            "e": "aggTrade",
            "E": stale_event_ms,
            "p": "100.00",
            "q": "1",
            "m": False,
        },
        received_at=received_at,
    )
    snapshot = book.snapshot(current=received_at)

    assert accepted is False
    assert snapshot.last_trade_price is None
    assert snapshot.last_received_at is None
    assert snapshot.exchange_event_lag_ms is None
    assert snapshot.hot_event_lag_ms is None
    assert snapshot.trade_event_lag_ms is None
    assert snapshot.regime.value == "stale"


def test_market_state_rejected_trade_lag_does_not_poison_fresh_book_latency():
    settings = Settings(MAX_EXCHANGE_EVENT_LAG_MS=1_000, MIN_WARMUP_SECONDS=1, STALE_AFTER_SECONDS=999)
    book = MarketStateBook(settings)
    book.set_connected(True)
    start = datetime(2026, 6, 8, 9, 0, tzinfo=timezone.utc)

    assert book.ingest(
        {
            "lastUpdateId": 1,
            "E": int(start.timestamp() * 1000),
            "bids": [["100.00", "4"], ["99.99", "2"]],
            "asks": [["100.01", "3"], ["100.02", "1"]],
        },
        received_at=start,
    )
    accepted = book.ingest(
        {
            "e": "aggTrade",
            "E": int((start - timedelta(seconds=10)).timestamp() * 1000),
            "p": "100.00",
            "q": "1",
            "m": False,
        },
        received_at=start + timedelta(seconds=2),
    )

    snapshot = book.snapshot(current=start + timedelta(seconds=2))

    assert accepted is False
    assert snapshot.book_event_lag_ms == 0
    assert snapshot.avg_book_event_lag_30s_ms == 0
    assert snapshot.trade_event_lag_ms is None
    assert snapshot.hot_event_lag_ms == 0
    assert snapshot.avg_hot_event_lag_30s_ms == 0
    assert snapshot.regime != Regime.stale


def test_market_state_context_lag_does_not_stale_fresh_hot_book():
    settings = Settings(MAX_EXCHANGE_EVENT_LAG_MS=1_000, MIN_WARMUP_SECONDS=1, STALE_AFTER_SECONDS=999)
    book = MarketStateBook(settings)
    book.set_connected(True)
    start = datetime(2026, 6, 6, 9, 0, tzinfo=timezone.utc)

    assert book.ingest(
        {
            "lastUpdateId": 1,
            "E": int(start.timestamp() * 1000),
            "bids": [["100.00", "4"], ["99.99", "2"]],
            "asks": [["100.01", "3"], ["100.02", "1"]],
        },
        received_at=start,
    )
    assert book.ingest(
        {
            "lastUpdateId": 2,
            "E": int((start + timedelta(seconds=2)).timestamp() * 1000),
            "bids": [["100.05", "4"], ["100.04", "2"]],
            "asks": [["100.06", "3"], ["100.07", "1"]],
        },
        received_at=start + timedelta(seconds=2),
    )
    accepted_context = book.ingest(
        {
            "e": "markPriceUpdate",
            "E": int((start - timedelta(seconds=10)).timestamp() * 1000),
            "p": "100.05",
            "r": "0",
        },
        received_at=start + timedelta(seconds=2, milliseconds=100),
    )

    snapshot = book.snapshot(current=start + timedelta(seconds=2, milliseconds=100))

    assert accepted_context is True
    assert snapshot.context_event_lag_ms == 12_100
    assert snapshot.book_event_lag_ms == 0
    assert snapshot.avg_book_event_lag_30s_ms == 0
    assert snapshot.hot_event_lag_ms == 0
    assert snapshot.avg_hot_event_lag_30s_ms == 0
    assert snapshot.regime != Regime.stale


def test_market_state_rolls_microstructure_features():
    settings = Settings(MIN_WARMUP_SECONDS=1, STALE_AFTER_SECONDS=999, DEPTH_LEVELS=5)
    book = MarketStateBook(settings)
    book.set_connected(True)
    start = datetime(2026, 5, 27, 9, 0, tzinfo=timezone.utc)

    book.ingest(
        {"bids": [["100.00", "5"], ["99.90", "5"]], "asks": [["100.10", "15"], ["100.20", "5"]]},
        received_at=start,
    )
    book.ingest(
        {"e": "bookTicker", "b": "100.00", "a": "100.10", "B": "10", "A": "10"},
        received_at=start,
    )
    book.ingest(
        {"e": "bookTicker", "b": "100.00", "a": "100.10", "B": "12", "A": "8"},
        received_at=start + timedelta(milliseconds=100),
    )
    book.ingest(
        {"e": "bookTicker", "b": "99.99", "a": "100.10", "B": "9", "A": "8"},
        received_at=start + timedelta(milliseconds=200),
    )
    book.ingest(
        {"e": "aggTrade", "p": "100.00", "q": "3", "m": False},
        received_at=start + timedelta(milliseconds=250),
    )
    book.ingest(
        {"e": "aggTrade", "p": "100.00", "q": "1", "m": True},
        received_at=start + timedelta(milliseconds=300),
    )
    book.ingest(
        {"e": "markPriceUpdate", "p": "100.20", "r": "0"},
        received_at=start + timedelta(milliseconds=350),
    )
    micro_snapshot = book.snapshot(current=start + timedelta(milliseconds=350))
    book.ingest(
        {"bids": [["100.00", "10"], ["99.90", "5"]], "asks": [["100.10", "5"], ["100.20", "5"]]},
        received_at=start + timedelta(seconds=5),
    )

    snapshot = book.snapshot(current=start + timedelta(seconds=5))

    assert micro_snapshot.order_flow_imbalance_250ms == pytest.approx(-0.5)
    assert micro_snapshot.order_flow_imbalance_1s == pytest.approx(-0.5)
    assert micro_snapshot.order_flow_imbalance_5s == pytest.approx(-0.5)
    assert micro_snapshot.taker_aggression_imbalance_1s == pytest.approx(0.5)
    assert micro_snapshot.taker_aggression_imbalance_5s == pytest.approx(0.5)
    assert micro_snapshot.microprice == pytest.approx(((99.99 * 8) + (100.10 * 9)) / 17)
    assert micro_snapshot.microprice_mid_bps == pytest.approx(
        ((micro_snapshot.microprice - micro_snapshot.mid_price) / micro_snapshot.mid_price) * 10_000
    )
    assert micro_snapshot.vamp_price_top == pytest.approx(100.03333333333333)
    assert micro_snapshot.vamp_mid_bps == pytest.approx(
        ((micro_snapshot.vamp_price_top - micro_snapshot.mid_price) / micro_snapshot.mid_price) * 10_000
    )
    assert micro_snapshot.weighted_depth_price_top == pytest.approx(100.06666666666666)
    assert micro_snapshot.spread_bps_avg_5s is not None
    assert micro_snapshot.spread_bps_std_5s is not None
    assert micro_snapshot.spread_bps_max_5s is not None
    assert snapshot.bid_depth_refill_rate_5s == pytest.approx(0.1)
    assert snapshot.ask_depth_evaporation_rate_5s == pytest.approx(0.1)
    assert micro_snapshot.mark_last_basis_bps == pytest.approx(20.0)
