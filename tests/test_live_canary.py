import pytest

from futures_lab.binance_private import BinancePrivateError
from futures_lab.config import Settings
from futures_lab.live_canary import (
    LiveProfitProtection,
    _filled_notional_usd,
    _live_paper_reconciliation,
    _live_profit_protection_check,
    live_order_side_from_decision,
    validate_live_canary_settings,
)
from futures_lab.models import Decision, DecisionAction, PaperTrade, Side, TradeMode


def _settings(**overrides):
    values = {
        "SYMBOL": "ETHUSDT",
        "BINANCE_API_KEY": "live-key",
        "BINANCE_API_SECRET": "live-secret",
        "LIVE_TRADING_ENABLED": True,
        "LIVE_DRY_RUN": False,
        "LIVE_MAX_NOTIONAL_USD": 25,
        "LIVE_MAX_OPEN_POSITIONS": 1,
        "LIVE_MAX_TRADES_PER_DAY": 2,
        "LIVE_DAILY_MAX_LOSS_USD": 2,
    }
    values.update(overrides)
    return Settings(**values)


def test_live_order_side_from_decision_maps_trade_actions() -> None:
    assert live_order_side_from_decision(
        Decision(symbol="ETHUSDT", action=DecisionAction.propose_long, confidence=0.9, reason="test")
    ) == "BUY"
    assert live_order_side_from_decision(
        Decision(symbol="ETHUSDT", action=DecisionAction.propose_short, confidence=0.9, reason="test")
    ) == "SELL"
    assert live_order_side_from_decision(
        Decision(symbol="ETHUSDT", action=DecisionAction.wait, confidence=0.0, reason="test")
    ) is None


def test_validate_live_canary_settings_accepts_tiny_live_config() -> None:
    validate_live_canary_settings(_settings(), max_trades=1)


def test_validate_live_canary_settings_accepts_raised_canary_limits() -> None:
    validate_live_canary_settings(
        _settings(
            LIVE_MAX_NOTIONAL_USD=100,
            LIVE_DUST_TEST_NOTIONAL_USD=95,
            LIVE_DAILY_MAX_LOSS_USD=8,
            LIVE_CANARY_MAX_NOTIONAL_USD=100,
            LIVE_CANARY_MAX_LOSS_USD=8,
            LIVE_CANARY_INTRATRADE_MAX_LOSS_USD=6,
        ),
        max_trades=2,
    )


def test_validate_live_canary_settings_rejects_unsafe_live_config() -> None:
    with pytest.raises(BinancePrivateError, match="LIVE_TRADING_ENABLED"):
        validate_live_canary_settings(_settings(LIVE_TRADING_ENABLED=False), max_trades=1)
    with pytest.raises(BinancePrivateError, match="LIVE_DRY_RUN"):
        validate_live_canary_settings(_settings(LIVE_DRY_RUN=True), max_trades=1)
    with pytest.raises(BinancePrivateError, match="LIVE_MAX_NOTIONAL_USD"):
        validate_live_canary_settings(_settings(LIVE_MAX_NOTIONAL_USD=26), max_trades=1)
    with pytest.raises(BinancePrivateError, match="LIVE_DAILY_MAX_LOSS_USD"):
        validate_live_canary_settings(_settings(LIVE_DAILY_MAX_LOSS_USD=3), max_trades=1)
    with pytest.raises(BinancePrivateError, match="LIVE_CANARY_POSITION_CHECK_SECONDS"):
        validate_live_canary_settings(_settings(LIVE_CANARY_POSITION_CHECK_SECONDS=0), max_trades=1)
    with pytest.raises(BinancePrivateError, match="LIVE_CANARY_MAX_CONSECUTIVE_LOSSES"):
        validate_live_canary_settings(_settings(LIVE_CANARY_MAX_CONSECUTIVE_LOSSES=-1), max_trades=1)
    with pytest.raises(BinancePrivateError, match="LIVE_CANARY_MAX_PROFIT_GIVEBACK_FRACTION"):
        validate_live_canary_settings(_settings(LIVE_CANARY_MAX_PROFIT_GIVEBACK_FRACTION=1.1), max_trades=1)
    with pytest.raises(BinancePrivateError, match="LIVE_MAX_TRADES_PER_DAY"):
        validate_live_canary_settings(_settings(LIVE_MAX_TRADES_PER_DAY=1), max_trades=2)


def test_filled_notional_prefers_actual_cum_quote() -> None:
    assert _filled_notional_usd(
        {
            "opened_execution": {"quote_qty": "94.25"},
            "opened_order": {"cumQuote": "93.51"},
            "order_template": {"estimated_notional_usd": "95.00"},
        }
    ) == pytest.approx(94.25)


def test_filled_notional_falls_back_to_order_cum_quote() -> None:
    assert _filled_notional_usd(
        {
            "opened_order": {"cumQuote": "93.51"},
            "order_template": {"estimated_notional_usd": "95.00"},
        }
    ) == pytest.approx(93.51)


def test_filled_notional_falls_back_to_template() -> None:
    assert _filled_notional_usd(
        {
            "opened_order": {},
            "order_template": {"estimated_notional_usd": "95.00"},
        }
    ) == pytest.approx(95.0)


def test_live_paper_reconciliation_records_wallet_delta_gap() -> None:
    trade = PaperTrade(
        symbol="ETHUSDT",
        side=Side.short,
        mode=TradeMode.fast,
        trade_profile="range_bound_support_resistance",
        entry_price=1570.82,
        exit_price=1570.96,
        quantity=0.633,
        stake_usd=100,
        notional_usd=994.31,
        leverage=200,
        gross_pnl_usd=-0.09,
        fees_usd=0.99,
        net_pnl_usd=-1.08,
        exit_reason="time_decay",
        opened_at="2026-06-29T05:34:10+00:00",
        closed_at="2026-06-29T05:37:14+00:00",
    )

    payload = _live_paper_reconciliation(
        trade,
        {
            "wallet_balance_delta_usd": "-0.497",
            "opened_execution": {"effective_avg_price": "1570.82", "quote_qty": "994.31"},
        },
        {
            "wallet_balance_delta_usd": "-2.619",
            "wallet_balance_after": "104.474",
            "close_execution": {"effective_avg_price": "1573.78", "quote_qty": "996.18"},
        },
    )

    assert payload["live_total_wallet_delta_usd"] == "-3.116"
    assert payload["live_close_wallet_after"] == "104.474"
    assert payload["live_vs_paper_delta_usd"] == "-2.036"
    assert payload["live_exit_price"] == "1573.78"


def test_live_profit_protection_stops_after_consecutive_losses() -> None:
    settings = _settings(
        LIVE_CANARY_PROFIT_PROTECTION_ENABLED=True,
        LIVE_CANARY_MAX_CONSECUTIVE_LOSSES=2,
        LIVE_CANARY_PROFIT_LOCK_MIN_PROFIT_USD=0.25,
        LIVE_CANARY_MAX_PROFIT_GIVEBACK_FRACTION=0.5,
    )
    state = LiveProfitProtection(previous_close_wallet=100)

    first = _live_profit_protection_check(settings, start_wallet=100, current_wallet=99.8, state=state)
    second = _live_profit_protection_check(settings, start_wallet=100, current_wallet=99.7, state=first.state)

    assert first.stop is False
    assert first.state.consecutive_losses == 1
    assert second.stop is True
    assert second.reason == "live_consecutive_losses_hit"


def test_live_profit_protection_stops_after_peak_giveback() -> None:
    settings = _settings(
        LIVE_CANARY_PROFIT_PROTECTION_ENABLED=True,
        LIVE_CANARY_MAX_CONSECUTIVE_LOSSES=0,
        LIVE_CANARY_PROFIT_LOCK_MIN_PROFIT_USD=0.25,
        LIVE_CANARY_MAX_PROFIT_GIVEBACK_FRACTION=0.5,
    )
    state = LiveProfitProtection(previous_close_wallet=100)

    winner = _live_profit_protection_check(settings, start_wallet=100, current_wallet=101.0, state=state)
    giveback = _live_profit_protection_check(settings, start_wallet=100, current_wallet=100.45, state=winner.state)

    assert winner.stop is False
    assert winner.state.peak_profit_usd == 1
    assert giveback.stop is True
    assert giveback.reason == "live_profit_giveback_hit"


def test_live_profit_protection_disabled_tracks_without_stopping() -> None:
    settings = _settings(
        LIVE_CANARY_PROFIT_PROTECTION_ENABLED=False,
        LIVE_CANARY_MAX_CONSECUTIVE_LOSSES=1,
        LIVE_CANARY_PROFIT_LOCK_MIN_PROFIT_USD=0.25,
        LIVE_CANARY_MAX_PROFIT_GIVEBACK_FRACTION=0.5,
    )
    state = LiveProfitProtection(previous_close_wallet=100)

    verdict = _live_profit_protection_check(settings, start_wallet=100, current_wallet=99.0, state=state)

    assert verdict.stop is False
    assert verdict.state.consecutive_losses == 1
