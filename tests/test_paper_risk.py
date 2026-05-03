import pytest

from futures_lab.config import Settings
from futures_lab.models import DecisionAction, MarketState, Regime
from futures_lab.paper import PaperBroker
from futures_lab.risk import RiskEngine
from futures_lab.strategy import HitAndRunStrategy


def _market(price: float = 100.0, **overrides) -> MarketState:
    base = {
        "symbol": "BTCUSDT",
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
        "regime": Regime.sideways,
    }
    base.update(overrides)
    return MarketState(**base)


def test_risk_blocks_non_trade_decision():
    settings = Settings()
    broker = PaperBroker(settings)
    risk = RiskEngine(settings)
    decision = HitAndRunStrategy(settings).decide(_market(regime=Regime.stale, data_age_seconds=5.0))

    verdict = risk.evaluate(decision, _market(), broker.state())

    assert not verdict.allowed
    assert "wait/close" in verdict.blockers[0]


def test_paper_fast_trade_hits_target_after_fees():
    settings = Settings(MIN_CONFIDENCE=0.70)
    market = _market(100.0)
    decision = HitAndRunStrategy(settings).decide(market)
    broker = PaperBroker(settings)
    verdict = RiskEngine(settings).evaluate(decision, market, broker.state())

    assert verdict.allowed
    position = broker.open_from_decision(decision)
    assert position is not None
    assert position.stake_usd == 150.0
    assert position.notional_usd == 30000.0

    trade = broker.mark(_market(position.take_profit_price))

    assert trade is not None
    assert trade.exit_reason == "take_profit"
    assert trade.gross_pnl_usd == pytest.approx(150.0)
    assert trade.net_pnl_usd < 150.0
    assert broker.state().trades_today == 1


def test_risk_blocks_after_daily_target_hit():
    settings = Settings(MIN_CONFIDENCE=0.70, DAILY_TARGET_USD=1)
    broker = PaperBroker(settings)
    broker.realized_pnl_usd = 2
    market = _market()
    decision = HitAndRunStrategy(settings).decide(market)

    verdict = RiskEngine(settings).evaluate(decision, market, broker.state())

    assert not verdict.allowed
    assert "daily target" in " ".join(verdict.blockers)
