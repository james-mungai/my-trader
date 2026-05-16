from futures_lab.markov import MarketSequenceState, MarketStateMachine
from futures_lab.models import MarketState, Regime


def _market(**overrides) -> MarketState:
    base = {
        "symbol": "BTCUSDT",
        "connected": True,
        "data_age_seconds": 0.1,
        "observed_seconds": 300,
        "mid_price": 100.0,
        "spread_bps": 0.5,
        "return_15s_pct": 0.0001,
        "return_60s_pct": 0.0008,
        "return_180s_pct": 0.0015,
        "realized_vol_60s_pct": 0.00004,
        "realized_vol_180s_pct": 0.00006,
        "range_180s_pct": 0.004,
        "range_position_180s": 0.5,
        "taker_buy_ratio_10s": 0.72,
        "taker_buy_ratio_30s": 0.62,
        "book_imbalance_top": 0.35,
        "depth_imbalance_top5": 0.35,
        "regime": Regime.sideways,
    }
    base.update(overrides)
    return MarketState(**base)


def test_market_state_machine_confirms_long_sequence_and_logs_transitions():
    machine = MarketStateMachine(history_size=6)

    assert machine.update(_market()).state == MarketSequenceState.bullish_pressure
    assert machine.update(_market(return_15s_pct=-0.0003, taker_buy_ratio_10s=0.55)).state == MarketSequenceState.pullback
    assert machine.update(_market(return_15s_pct=0.00035, taker_buy_ratio_10s=0.72)).state == MarketSequenceState.reclaim
    confirmed = machine.update(_market(return_15s_pct=0.0004, taker_buy_ratio_10s=0.74))

    assert confirmed.state == MarketSequenceState.long_continuation_confirmed
    assert confirmed.allows_long
    assert confirmed.transition_count == 1
    dump = machine.model_dump()
    assert dump["transitions"]["reclaim->long_continuation_confirmed"]["count"] == 1


def test_market_state_machine_confirms_short_sequence_and_blocks_stale_data():
    machine = MarketStateMachine(history_size=6)
    bearish = _market(
        return_15s_pct=-0.0001,
        return_60s_pct=-0.0008,
        return_180s_pct=-0.0015,
        taker_buy_ratio_10s=0.25,
        book_imbalance_top=-0.35,
        depth_imbalance_top5=-0.35,
    )

    assert machine.update(bearish).state == MarketSequenceState.bearish_pressure
    assert machine.update(bearish.model_copy(update={"return_15s_pct": 0.0003})).state == MarketSequenceState.bounce
    assert machine.update(bearish.model_copy(update={"return_15s_pct": -0.00035})).state == MarketSequenceState.rejection
    confirmed = machine.update(bearish.model_copy(update={"return_15s_pct": -0.0004}))

    assert confirmed.state == MarketSequenceState.short_continuation_confirmed
    assert confirmed.allows_short

    blocked = machine.update(bearish.model_copy(update={"regime": Regime.stale, "data_age_seconds": 5.0}))
    assert blocked.state == MarketSequenceState.data_blocked
    assert "regime=stale" in blocked.blockers
