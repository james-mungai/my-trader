from datetime import datetime, timedelta, timezone

from futures_lab.config import Settings
from futures_lab.exit_shadow import ExitShadowEvaluator
from futures_lab.models import MarketState, Regime, Side


def _market(price: float, **overrides) -> MarketState:
    base = {
        "symbol": "ETHUSDT",
        "connected": True,
        "data_age_seconds": 0.1,
        "observed_seconds": 300,
        "best_bid": price - 0.005,
        "best_ask": price + 0.005,
        "best_bid_qty": 12.0,
        "best_ask_qty": 8.0,
        "mid_price": price,
        "spread_bps": 0.5,
        "last_trade_price": price,
        "mark_price": price,
        "funding_rate": 0.0,
        "return_15s_pct": 0.0001,
        "return_60s_pct": 0.0005,
        "realized_vol_60s_pct": 0.00004,
        "range_180s_pct": 0.007,
        "range_position_180s": 0.50,
        "taker_buy_ratio_10s": 0.70,
        "taker_buy_ratio_30s": 0.60,
        "book_imbalance_top": 0.2,
        "regime": Regime.sideways,
    }
    base.update(overrides)
    return MarketState(**base)


def test_exit_shadow_tracks_fee_trail_and_partial_outcomes():
    settings = Settings(
        TAKER_FEE_BPS=4.0,
        EXIT_SHADOW_FEE_TRAIL_ACTIVATION_FEE_MULTIPLE=1.25,
        EXIT_SHADOW_FEE_TRAIL_FLOOR_FEE_MULTIPLE=1.0,
        EXIT_SHADOW_PARTIAL_TAKE_PROFIT_PCT=0.001,
    )
    opened_at = datetime(2026, 5, 26, 10, 0, tzinfo=timezone.utc)
    shadow = ExitShadowEvaluator(
        settings=settings,
        symbol="ETHUSDT",
        side=Side.long,
        entry_price=100.0,
        notional_usd=30_000.0,
        opened_at=opened_at,
    )

    shadow.mark(_market(100.12), opened_at + timedelta(seconds=30))
    shadow.mark(_market(100.08), opened_at + timedelta(seconds=60))
    result = shadow.close_at_actual(99.85, "stop_loss", opened_at + timedelta(seconds=90))

    policies = result["policies"]
    assert policies["fixed_tp_stop"]["exit_reason"] == "actual_stop_loss"
    assert policies["fee_breakeven_trail"]["exit_reason"] == "fee_breakeven_trail"
    assert policies["partial_tp"]["partial_taken"] is True
    assert policies["fee_breakeven_trail"]["net_pnl_usd"] > policies["fixed_tp_stop"]["net_pnl_usd"]


def test_exit_shadow_time_decay_cuts_stale_non_mover():
    settings = Settings(TAKER_FEE_BPS=4.0, EXIT_SHADOW_TIME_DECAY_SECONDS=180)
    opened_at = datetime(2026, 5, 26, 10, 0, tzinfo=timezone.utc)
    shadow = ExitShadowEvaluator(
        settings=settings,
        symbol="ETHUSDT",
        side=Side.long,
        entry_price=100.0,
        notional_usd=30_000.0,
        opened_at=opened_at,
    )

    shadow.mark(
        _market(99.99, return_60s_pct=-0.0002, taker_buy_ratio_10s=0.45),
        opened_at + timedelta(seconds=181),
    )
    result = shadow.close_at_actual(99.80, "stop_loss", opened_at + timedelta(seconds=300))

    assert result["policies"]["time_decay"]["exit_reason"] == "time_decay"
    assert result["policies"]["time_decay"]["net_pnl_usd"] > result["policies"]["fixed_tp_stop"]["net_pnl_usd"]
