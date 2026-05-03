from futures_lab.config import Settings
from futures_lab.models import DecisionAction, MarketState, Regime
from futures_lab.strategy import HitAndRunStrategy


def _market(**overrides) -> MarketState:
    base = {
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
        "regime": Regime.sideways,
    }
    base.update(overrides)
    return MarketState(**base)


def test_strategy_waits_on_stale_regime():
    strategy = HitAndRunStrategy(Settings())

    decision = strategy.decide(_market(regime=Regime.stale, data_age_seconds=5.0))

    assert decision.action == DecisionAction.wait
    assert "regime=stale" in decision.reason


def test_strategy_proposes_fast_long_near_range_low():
    strategy = HitAndRunStrategy(Settings(MIN_CONFIDENCE=0.70))

    decision = strategy.decide(_market())

    assert decision.action == DecisionAction.propose_long
    assert decision.mode is not None
    assert decision.leverage == 200
    assert decision.take_profit_price is not None
    assert decision.take_profit_price > decision.entry_price


def test_strategy_proposes_short_near_range_high():
    strategy = HitAndRunStrategy(Settings(MIN_CONFIDENCE=0.70))

    decision = strategy.decide(
        _market(
            range_position_180s=0.96,
            taker_buy_ratio_10s=0.18,
            taker_buy_ratio_30s=0.30,
            book_imbalance_top=-0.25,
            return_15s_pct=-0.0001,
        )
    )

    assert decision.action == DecisionAction.propose_short
    assert decision.stop_loss_price is not None
    assert decision.stop_loss_price > decision.entry_price

