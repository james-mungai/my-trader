from datetime import datetime, timedelta, timezone

from futures_lab.config import Settings
from futures_lab.models import Decision, DecisionAction, MarketState, Regime
from futures_lab.regime_outcomes import RegimeOutcomeTracker


def _market(price: float, ts: datetime, **overrides) -> MarketState:
    base = {
        "symbol": "BTCUSDT",
        "connected": True,
        "last_received_at": ts,
        "data_age_seconds": 0.01,
        "observed_seconds": 600,
        "mid_price": price,
        "spread_bps": 0.5,
        "range_180s_pct": 0.003,
        "range_position_180s": 0.4,
        "return_15s_pct": -0.0003,
        "return_60s_pct": -0.001,
        "return_180s_pct": -0.002,
        "taker_buy_ratio_10s": 0.22,
        "taker_buy_ratio_30s": 0.30,
        "book_imbalance_top": -0.35,
        "depth_imbalance_top5": -0.45,
        "open_interest_change_5m_pct": 0.001,
        "regime": Regime.sideways,
    }
    base.update(overrides)
    return MarketState(**base)


def _decision() -> Decision:
    return Decision(
        symbol="BTCUSDT",
        action=DecisionAction.wait,
        confidence=0.76,
        reason="confirmed but measuring regime outcome",
        evidence={
            "stateful_momentum_filter": {
                "confirmed": True,
                "side": "short",
                "state": "short_continuation_confirmed",
                "score": 0.76,
                "sequence_confidence": 0.94,
                "adaptive_entry_allowed": False,
                "target_feasible": False,
                "blockers": ["target_feasibility"],
                "path": ["bearish_pressure", "bounce", "rejection", "short_continuation_confirmed"],
            }
        },
    )


def test_regime_outcome_tracker_dedupes_and_scores_horizons():
    settings = Settings(
        REGIME_OUTCOME_HORIZONS_SECONDS="15,60",
        REGIME_OUTCOME_TARGET_MOVES_PCT="0.001,0.002",
        REGIME_OUTCOME_STOP_MOVES_PCT="0.001",
        REGIME_OUTCOME_COOLDOWN_SECONDS=900,
    )
    tracker = RegimeOutcomeTracker(settings)
    start = datetime(2026, 5, 16, 8, 0, tzinfo=timezone.utc)
    decision = _decision()

    opened = tracker.open_from_decision(decision, _market(100.0, start), opened_at=start)
    duplicate = tracker.open_from_decision(decision, _market(99.98, start + timedelta(seconds=1)), opened_at=start + timedelta(seconds=1))

    assert opened is not None
    assert duplicate is None
    assert opened["quality_score"] > 0.7

    assert tracker.mark(_market(99.85, start + timedelta(seconds=15)), timestamp=start + timedelta(seconds=15)) == []
    closed = tracker.mark(_market(99.75, start + timedelta(seconds=60)), timestamp=start + timedelta(seconds=60))

    assert len(closed) == 1
    outcome = closed[0]
    assert outcome["event"] == "close"
    assert outcome["target_hits"]["0.001"]["hit"] is True
    assert outcome["target_hits"]["0.002"]["hit"] is True
    assert outcome["stop_hits"]["0.001"]["hit"] is False
    assert outcome["target_before_stop"]["0.002"]["0.001"] is True
    assert outcome["early_follow_through"]["qualified"] is True
    assert outcome["early_follow_through"]["direction_correct_count"] == 2
    assert outcome["horizons"]["15"]["direction_correct"] is True
    assert outcome["horizons"]["60"]["direction_correct"] is True


def test_regime_outcome_tracker_ignores_non_confirmed_decisions():
    tracker = RegimeOutcomeTracker(Settings())
    market = _market(100.0, datetime(2026, 5, 16, 8, 0, tzinfo=timezone.utc))
    decision = Decision(symbol="BTCUSDT", action=DecisionAction.wait, confidence=0.0, reason="none")

    assert tracker.open_from_decision(decision, market) is None
