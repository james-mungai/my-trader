from datetime import datetime, timedelta, timezone

from futures_lab.audit import AuditLog
from futures_lab.config import Settings
from futures_lab.context_polling import BinanceContextPoller
from futures_lab.market_state import MarketStateBook
from futures_lab.models import MarketState, Regime
from futures_lab.strategy import HitAndRunStrategy


def _poller(tmp_path):
    settings = Settings(DATA_DIR=str(tmp_path), HIGHER_TIMEFRAME_MIN_CLOSED_CANDLES=8)
    state = MarketStateBook(settings)
    return BinanceContextPoller(settings=settings, state=state, audit=AuditLog(settings))


def _candles(start: float, step: float, count: int = 40):
    rows = []
    ts = datetime(2026, 5, 17, tzinfo=timezone.utc)
    price = start
    for idx in range(count):
        opened = ts + timedelta(hours=idx)
        closed = opened + timedelta(hours=1) - timedelta(milliseconds=1)
        close = price + step
        high = max(price, close) + abs(step) * 0.7 + 1
        low = min(price, close) - abs(step) * 0.4 - 1
        volume = 100 + idx
        taker_buy = volume * (0.62 if step > 0 else 0.38)
        rows.append(
            {
                "open_time": int(opened.timestamp() * 1000),
                "open": price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "close_time": int(closed.timestamp() * 1000),
                "quote_volume": volume * close,
                "trades": 1000 + idx,
                "taker_buy_base": taker_buy,
                "taker_buy_quote": taker_buy * close,
            }
        )
        price = close
    return rows


def test_higher_timeframe_summary_detects_direction(tmp_path):
    poller = _poller(tmp_path)

    up = poller._summarize_candles("1h", _candles(100.0, 0.4))
    down = poller._summarize_candles("1h", _candles(100.0, -0.4))

    assert up["trend_score"] > 0
    assert up["structure"].startswith("uptrend")
    assert down["trend_score"] < 0
    assert down["structure"].startswith("downtrend")


def test_higher_timeframe_aggregate_creates_signed_bias(tmp_path):
    poller = _poller(tmp_path)
    frames = {
        "1h": poller._summarize_candles("1h", _candles(100.0, -0.5)),
        "4h": poller._summarize_candles("4h", _candles(100.0, -0.7)),
        "1d": poller._summarize_candles("1d", _candles(100.0, -1.0)),
    }

    context = poller._aggregate_higher_timeframes(frames)

    assert context["bias"]["side"] == "short"
    assert context["bias"]["strength"] > 0
    assert "1h:" in context["bias"]["reason"]


def test_market_snapshot_exposes_higher_timeframe_context(tmp_path):
    settings = Settings(DATA_DIR=str(tmp_path))
    book = MarketStateBook(settings)
    now = datetime(2026, 5, 17, 12, tzinfo=timezone.utc)
    book.set_higher_timeframe_context(
        {"bias": {"side": "long", "strength": 0.42, "reason": "1h:uptrend"}},
        updated_at=now,
    )

    snapshot = book.snapshot(current=now + timedelta(seconds=30))

    assert snapshot.higher_timeframe_bias_side == "long"
    assert snapshot.higher_timeframe_bias_strength == 0.42
    assert snapshot.higher_timeframe_context_age_seconds == 30


def test_strategy_records_higher_timeframe_context_in_evidence(tmp_path):
    strategy = HitAndRunStrategy(Settings(DATA_DIR=str(tmp_path), MIN_CONFIDENCE=0.70))
    market = {
        "symbol": "BTCUSDT",
        "connected": True,
        "data_age_seconds": 0.1,
        "observed_seconds": 300,
        "best_bid": 100.0,
        "best_ask": 100.01,
        "best_bid_qty": 12.0,
        "best_ask_qty": 8.0,
        "mid_price": 100.005,
        "spread_bps": 0.5,
        "last_trade_price": 100.0,
        "mark_price": 100.0,
        "funding_rate": 0.0,
        "return_15s_pct": 0.0001,
        "return_60s_pct": -0.0005,
        "return_180s_pct": 0.0002,
        "realized_vol_60s_pct": 0.00004,
        "realized_vol_180s_pct": 0.00006,
        "range_180s_pct": 0.007,
        "range_high_180s": 101.0,
        "range_low_180s": 100.0,
        "range_position_180s": 0.05,
        "taker_buy_ratio_10s": 0.78,
        "taker_buy_ratio_30s": 0.66,
        "book_imbalance_top": 0.2,
        "higher_timeframe_context": {"bias": {"side": "long", "strength": 0.5, "reason": "1h:uptrend"}},
        "higher_timeframe_context_age_seconds": 10,
        "higher_timeframe_bias_side": "long",
        "higher_timeframe_bias_strength": 0.5,
        "higher_timeframe_bias_reason": "1h:uptrend",
        "regime": Regime.sideways,
    }

    strategy_market = MarketState(**market)
    decision = strategy.decide(strategy_market)

    assert decision.evidence["higher_timeframe_context"]["bias_side"] == "long"
    assert strategy_market.higher_timeframe_bias_strength == 0.5
