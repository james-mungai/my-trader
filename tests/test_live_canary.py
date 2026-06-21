import pytest

from futures_lab.binance_private import BinancePrivateError
from futures_lab.config import Settings
from futures_lab.live_canary import _filled_notional_usd, live_order_side_from_decision, validate_live_canary_settings
from futures_lab.models import Decision, DecisionAction


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
    with pytest.raises(BinancePrivateError, match="LIVE_MAX_TRADES_PER_DAY"):
        validate_live_canary_settings(_settings(LIVE_MAX_TRADES_PER_DAY=1), max_trades=2)


def test_filled_notional_prefers_actual_cum_quote() -> None:
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
