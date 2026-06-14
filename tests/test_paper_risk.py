import pytest
from datetime import datetime, timedelta, timezone

from futures_lab.config import Settings
from futures_lab.models import Decision, DecisionAction, MarketState, Regime, TradeMode
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
    settings = Settings(MIN_CONFIDENCE=0.70, PAPER_LIVE_EDGE_GATE_ENABLED=False)
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
    assert trade.gross_pnl_usd == pytest.approx(position.notional_usd * (decision.target_move_pct or 0.0))
    assert trade.net_pnl_usd < trade.gross_pnl_usd
    assert broker.state().trades_today == 1
    assert trade.exit_shadow["policies"]["fixed_tp_stop"]["exit_reason"] == "actual_take_profit"


def test_risk_blocks_target_that_does_not_clear_round_trip_fees():
    settings = Settings(MIN_CONFIDENCE=0.70, FAST_TARGET_MOVE_PCT=0.001, MIN_GROSS_TARGET_FEE_MULTIPLE=2.0)
    market = _market(100.0)
    decision = HitAndRunStrategy(settings).decide(market)
    broker = PaperBroker(settings)

    verdict = RiskEngine(settings).evaluate(decision, market, broker.state())

    assert not verdict.allowed
    assert "target does not clear effective costs" in " ".join(verdict.blockers)


def test_risk_allows_larger_target_that_clears_effective_costs():
    settings = Settings(
        MIN_CONFIDENCE=0.70,
        FAST_TARGET_MOVE_PCT=0.0025,
        EXPECTED_SLIPPAGE_BPS=0.5,
        LATENCY_ADVERSE_SELECTION_BPS=0.5,
        MIN_GROSS_TARGET_FEE_MULTIPLE=2.0,
        PAPER_LIVE_EDGE_GATE_ENABLED=False,
    )
    market = _market(100.0, spread_bps=0.5)
    decision = HitAndRunStrategy(settings).decide(market)
    broker = PaperBroker(settings)

    verdict = RiskEngine(settings).evaluate(decision, market, broker.state())

    assert verdict.allowed


def test_paper_broker_closes_open_position_at_session_end():
    settings = Settings(MIN_CONFIDENCE=0.70)
    decision = HitAndRunStrategy(settings).decide(_market())
    broker = PaperBroker(settings)
    broker.open_from_decision(decision, opened_at=datetime(2026, 1, 1, tzinfo=timezone.utc))

    closed = broker.close_open_position(
        _market(100.8, last_received_at=datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc)),
        reason="session_end",
    )

    assert closed is not None
    assert closed.exit_reason == "session_end"
    assert closed.closed_at == datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc)
    assert broker.open_position is None
    assert broker.trades_today == 1


def _edge_decision(target_move_pct: float, selected: dict) -> Decision:
    price = 100.0
    return Decision(
        symbol="BTCUSDT",
        action=DecisionAction.propose_long,
        mode=TradeMode.fast,
        confidence=0.95,
        reason="test edge",
        entry_price=price,
        take_profit_price=price * (1 + target_move_pct),
        stop_loss_price=price * (1 - 0.001),
        target_move_pct=target_move_pct,
        stop_move_pct=0.001,
        leverage=200,
        stake_usd=150,
        notional_usd=30_000,
        evidence={"edge_router": {"selected": selected}},
    )


def test_risk_blocks_live_paper_when_edge_after_costs_is_too_thin():
    settings = Settings(
        MIN_CONFIDENCE=0.70,
        PAPER_LIVE_EDGE_GATE_ENABLED=True,
        PAPER_LIVE_MIN_TARGET_COST_MULTIPLE=2.5,
        PAPER_LIVE_MIN_EXPECTED_EV_BPS=4.0,
        PAPER_LIVE_MIN_SCORE=0.82,
        PAPER_LIVE_MIN_TP_PROBABILITY=0.82,
    )
    decision = _edge_decision(
        0.002,
        {
            "strategy": "stateful_momentum_baseline",
            "score": 0.78,
            "p_hit_tp_before_sl": 0.78,
            "expected_ev_bps": 2.0,
        },
    )

    verdict = RiskEngine(settings).evaluate(decision, _market(spread_bps=0.5), PaperBroker(settings).state())

    assert not verdict.allowed
    joined = " ".join(verdict.blockers)
    assert "paper live target does not clear edge costs" in joined
    assert "paper live expected EV too low" in joined
    assert "paper live score too low" in joined
    assert "paper live TP probability too low" in joined


def test_risk_allows_live_paper_when_selected_edge_is_strong_after_costs():
    settings = Settings(
        MIN_CONFIDENCE=0.70,
        PAPER_LIVE_EDGE_GATE_ENABLED=True,
        PAPER_LIVE_MIN_TARGET_COST_MULTIPLE=2.5,
        PAPER_LIVE_MIN_EXPECTED_EV_BPS=4.0,
        PAPER_LIVE_MIN_SCORE=0.82,
        PAPER_LIVE_MIN_TP_PROBABILITY=0.82,
    )
    decision = _edge_decision(
        0.003,
        {
            "strategy": "taker_impulse_short",
            "score": 0.88,
            "p_hit_tp_before_sl": 0.88,
            "expected_ev_bps": 6.0,
        },
    )

    verdict = RiskEngine(settings).evaluate(decision, _market(spread_bps=0.5), PaperBroker(settings).state())

    assert verdict.allowed


def test_risk_blocks_live_paper_when_rolling_candidate_quality_is_weak():
    settings = Settings(
        MIN_CONFIDENCE=0.70,
        PAPER_LIVE_EDGE_GATE_ENABLED=True,
        PAPER_LIVE_MIN_TARGET_COST_MULTIPLE=2.5,
        PAPER_LIVE_MIN_EXPECTED_EV_BPS=4.0,
        PAPER_LIVE_MIN_SCORE=0.82,
        PAPER_LIVE_MIN_TP_PROBABILITY=0.82,
        PAPER_LIVE_ROLLING_EDGE_MONITOR_ENABLED=True,
    )
    decision = _edge_decision(
        0.003,
        {
            "strategy": "taker_impulse_short",
            "score": 0.88,
            "p_hit_tp_before_sl": 0.88,
            "expected_ev_bps": 6.0,
        },
    )
    candidate_quality = {
        "enabled": True,
        "ready": True,
        "block": True,
        "reason": "accepted candidates are not outperforming rejected candidates",
    }

    verdict = RiskEngine(settings).evaluate(
        decision,
        _market(spread_bps=0.5),
        PaperBroker(settings).state(),
        candidate_quality=candidate_quality,
    )

    assert not verdict.allowed
    assert "paper live rolling edge monitor blocked opens" in " ".join(verdict.blockers)


def test_risk_blocks_after_daily_target_hit():
    settings = Settings(MIN_CONFIDENCE=0.70, DAILY_TARGET_USD=1)
    broker = PaperBroker(settings)
    broker.realized_pnl_usd = 2
    market = _market()
    decision = HitAndRunStrategy(settings).decide(market)

    verdict = RiskEngine(settings).evaluate(decision, market, broker.state())

    assert not verdict.allowed
    assert "daily target" in " ".join(verdict.blockers)


def test_risk_blocks_during_trade_cooldown():
    settings = Settings(MIN_CONFIDENCE=0.70, TRADE_COOLDOWN_SECONDS=1800, PAPER_LIVE_EDGE_GATE_ENABLED=False)
    broker = PaperBroker(settings)
    market = _market()
    decision = HitAndRunStrategy(settings).decide(market)
    opened_at = datetime(2026, 5, 11, 12, 0, tzinfo=timezone.utc)
    position = broker.open_from_decision(decision, opened_at=opened_at)
    assert position is not None
    broker.close(position.take_profit_price, "take_profit", closed_at=opened_at + timedelta(minutes=1))

    next_market = _market(last_received_at=opened_at + timedelta(minutes=2))
    next_decision = HitAndRunStrategy(settings).decide(next_market)
    verdict = RiskEngine(settings).evaluate(next_decision, next_market, broker.state())

    assert not verdict.allowed
    assert "cooldown" in " ".join(verdict.blockers)


def test_risk_allows_more_than_one_trade_when_daily_trade_limit_disabled():
    settings = Settings(
        MIN_CONFIDENCE=0.70,
        MAX_TRADES_PER_DAY=0,
        TRADE_COOLDOWN_SECONDS=0,
        PAPER_LIVE_EDGE_GATE_ENABLED=False,
    )
    broker = PaperBroker(settings)
    broker.trades_today = 5
    market = _market()
    decision = HitAndRunStrategy(settings).decide(market)

    verdict = RiskEngine(settings).evaluate(decision, market, broker.state())

    assert verdict.allowed


def test_paper_fast_failure_exits_when_trade_does_not_move_enough():
    settings = Settings(
        MIN_CONFIDENCE=0.70,
        PAPER_LIVE_EDGE_GATE_ENABLED=False,
        ENABLE_FAST_FAILURE_EXIT=True,
        FAST_FAILURE_SECONDS=180,
        FAST_FAILURE_MIN_FAVORABLE_MOVE_PCT=0.0005,
    )
    market = _market(100.0)
    decision = HitAndRunStrategy(settings).decide(market)
    broker = PaperBroker(settings)
    opened_at = datetime(2026, 5, 11, 12, 0, tzinfo=timezone.utc)
    position = broker.open_from_decision(decision, opened_at=opened_at)

    assert position is not None
    assert broker.mark(_market(100.02), timestamp=opened_at + timedelta(seconds=120)) is None

    trade = broker.mark(_market(100.03), timestamp=opened_at + timedelta(seconds=181))

    assert trade is not None
    assert trade.exit_reason == "fast_failure"
    assert position.max_favorable_move_pct < settings.fast_failure_min_favorable_move_pct


def test_paper_does_not_close_from_stale_or_disconnected_market():
    settings = Settings(
        MIN_CONFIDENCE=0.70,
        PAPER_LIVE_EDGE_GATE_ENABLED=False,
        ENABLE_FAST_FAILURE_EXIT=True,
        FAST_FAILURE_SECONDS=180,
    )
    market = _market(100.0)
    decision = HitAndRunStrategy(settings).decide(market)
    broker = PaperBroker(settings)
    opened_at = datetime(2026, 5, 11, 12, 0, tzinfo=timezone.utc)
    position = broker.open_from_decision(decision, opened_at=opened_at)

    assert position is not None
    disconnected = _market(
        position.stop_loss_price,
        connected=False,
        data_age_seconds=999,
    )
    stale = _market(
        position.take_profit_price,
        connected=True,
        data_age_seconds=settings.stale_after_seconds + 1,
    )

    assert broker.mark(disconnected, timestamp=opened_at + timedelta(minutes=10)) is None
    assert broker.mark(stale, timestamp=opened_at + timedelta(minutes=11)) is None
    assert broker.state().open_position is not None


def test_paper_can_disable_normal_price_stop_but_keeps_emergency_guard():
    settings = Settings(
        MIN_CONFIDENCE=0.70,
        ENABLE_PRICE_STOP=False,
        ENABLE_FAST_FAILURE_EXIT=False,
        EMERGENCY_MAX_ADVERSE_MOVE_PCT=0.004,
    )
    market = _market(100.0)
    decision = HitAndRunStrategy(settings).decide(market)
    broker = PaperBroker(settings)
    opened_at = datetime(2026, 5, 11, 12, 0, tzinfo=timezone.utc)
    position = broker.open_from_decision(decision, opened_at=opened_at)

    assert position is not None
    assert broker.mark(_market(position.stop_loss_price), timestamp=opened_at + timedelta(minutes=1)) is None

    emergency_price = position.entry_price * (1 - settings.emergency_max_adverse_move_pct)
    trade = broker.mark(_market(emergency_price), timestamp=opened_at + timedelta(minutes=2))

    assert trade is not None
    assert trade.exit_reason == "emergency_adverse_move"


def test_paper_max_hold_can_close_position():
    settings = Settings(
        MIN_CONFIDENCE=0.70,
        ENABLE_PRICE_STOP=False,
        ENABLE_FAST_FAILURE_EXIT=False,
        MAX_POSITION_SECONDS=300,
    )
    market = _market(100.0)
    decision = HitAndRunStrategy(settings).decide(market)
    broker = PaperBroker(settings)
    opened_at = datetime(2026, 5, 11, 12, 0, tzinfo=timezone.utc)
    position = broker.open_from_decision(decision, opened_at=opened_at)

    assert position is not None
    assert broker.mark(_market(100.01), timestamp=opened_at + timedelta(seconds=299)) is None
    trade = broker.mark(_market(100.01), timestamp=opened_at + timedelta(seconds=300))

    assert trade is not None
    assert trade.exit_reason == "max_hold"


def test_paper_trade_logs_exit_shadow_for_alternative_exit_policies():
    settings = Settings(
        MIN_CONFIDENCE=0.70,
        EXIT_SHADOW_TIME_DECAY_SECONDS=180,
        ENABLE_FAST_FAILURE_EXIT=False,
    )
    market = _market(100.0)
    decision = HitAndRunStrategy(settings).decide(market)
    broker = PaperBroker(settings)
    opened_at = datetime(2026, 5, 11, 12, 0, tzinfo=timezone.utc)
    position = broker.open_from_decision(decision, opened_at=opened_at)

    assert position is not None
    assert broker.mark(
        _market(99.99, return_60s_pct=-0.0002, taker_buy_ratio_10s=0.45),
        timestamp=opened_at + timedelta(seconds=181),
    ) is None
    trade = broker.mark(_market(position.stop_loss_price), timestamp=opened_at + timedelta(seconds=240))

    assert trade is not None
    assert trade.exit_shadow["policies"]["time_decay"]["exit_reason"] == "time_decay"
    assert trade.exit_shadow["policies"]["time_decay"]["net_pnl_usd"] > trade.exit_shadow["policies"]["fixed_tp_stop"]["net_pnl_usd"]


def test_paper_mfe_trailing_exit_can_be_enabled_for_selected_profiles():
    settings = Settings(
        MIN_CONFIDENCE=0.70,
        PAPER_EXIT_POLICY="mfe_trailing_stop",
        PAPER_MFE_TRAILING_PROFILES="htf_aligned_fast",
        PAPER_MFE_TRAIL_ACTIVATION_PCT=0.0015,
        PAPER_MFE_TRAIL_DISTANCE_PCT=0.0010,
        ENABLE_FAST_FAILURE_EXIT=False,
    )
    market = _market(100.0)
    decision = HitAndRunStrategy(settings).decide(market).model_copy(
        update={"evidence": {"trade_profile": "htf_aligned_fast"}}
    )
    broker = PaperBroker(settings)
    opened_at = datetime(2026, 5, 11, 12, 0, tzinfo=timezone.utc)
    position = broker.open_from_decision(decision, opened_at=opened_at)

    assert position is not None
    assert position.exit_policy == "mfe_trailing_stop"
    assert broker.mark(_market(100.17), timestamp=opened_at + timedelta(seconds=30)) is None
    trade = broker.mark(_market(100.06), timestamp=opened_at + timedelta(seconds=60))

    assert trade is not None
    assert trade.exit_reason == "mfe_trailing_stop"
    assert trade.trade_profile == "htf_aligned_fast"
    assert trade.exit_policy == "mfe_trailing_stop"
    assert trade.exit_shadow["policies"]["mfe_trailing_stop"]["exit_reason"] == "mfe_trailing_stop"


def test_paper_mfe_trailing_exit_ignores_profiles_outside_allowlist():
    settings = Settings(
        MIN_CONFIDENCE=0.70,
        PAPER_EXIT_POLICY="mfe_trailing_stop",
        PAPER_MFE_TRAILING_PROFILES="htf_aligned_fast",
        PAPER_MFE_TRAIL_ACTIVATION_PCT=0.0015,
        PAPER_MFE_TRAIL_DISTANCE_PCT=0.0010,
        ENABLE_FAST_FAILURE_EXIT=False,
    )
    market = _market(100.0)
    decision = HitAndRunStrategy(settings).decide(market).model_copy(
        update={"evidence": {"trade_profile": "weak_or_neutral_htf"}}
    )
    broker = PaperBroker(settings)
    opened_at = datetime(2026, 5, 11, 12, 0, tzinfo=timezone.utc)
    position = broker.open_from_decision(decision, opened_at=opened_at)

    assert position is not None
    assert position.exit_policy == "fixed_tp_stop"
    assert broker.mark(_market(100.17), timestamp=opened_at + timedelta(seconds=30)) is None
    assert broker.mark(_market(100.06), timestamp=opened_at + timedelta(seconds=60)) is None
