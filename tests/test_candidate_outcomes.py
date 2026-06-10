from datetime import datetime, timedelta, timezone

from futures_lab.candidate_outcomes import CandidateOutcomeTracker
from futures_lab.config import Settings
from futures_lab.models import Decision, DecisionAction, MarketState, Regime


def _market(price: float, ts: datetime, **overrides) -> MarketState:
    base = {
        "symbol": "ETHUSDT",
        "connected": True,
        "last_received_at": ts,
        "data_age_seconds": 0.01,
        "observed_seconds": 600,
        "mid_price": price,
        "spread_bps": 0.5,
        "order_flow_imbalance_1s": 0.6,
        "microprice_mid_bps": 0.2,
        "regime": Regime.directional,
    }
    base.update(overrides)
    return MarketState(**base)


def _decision() -> Decision:
    accepted = {
        "strategy": "taker_impulse_long",
        "family": "taker_impulse",
        "side": "long",
        "target_bps": 15.0,
        "stop_bps": 8.0,
        "max_hold_ms": 20_000,
        "expected_cost_bps": 4.0,
        "score": 0.88,
        "expected_ev_bps": 4.5,
        "viable": True,
        "blockers": [],
        "reasons": ["test accepted"],
    }
    rejected = {
        "strategy": "taker_impulse_short",
        "family": "taker_impulse",
        "side": "short",
        "target_bps": 15.0,
        "stop_bps": 8.0,
        "max_hold_ms": 20_000,
        "expected_cost_bps": 4.0,
        "score": 0.42,
        "expected_ev_bps": -3.0,
        "viable": False,
        "blockers": ["score_below_min"],
        "reasons": ["test rejected"],
    }
    return Decision(
        symbol="ETHUSDT",
        action=DecisionAction.wait,
        confidence=0.0,
        reason="candidate outcome test",
        evidence={"edge_router": {"selected": accepted, "candidates": [accepted, rejected]}},
    )


def test_candidate_outcome_tracker_labels_target_first_and_horizons():
    settings = Settings(CANDIDATE_OUTCOME_HORIZONS_SECONDS="1,3", CANDIDATE_OUTCOME_COOLDOWN_SECONDS=30)
    tracker = CandidateOutcomeTracker(settings)
    start = datetime(2026, 5, 27, 12, 0, tzinfo=timezone.utc)

    opened = tracker.open_from_decision(_decision(), _market(100.0, start), opened_at=start)
    assert len(opened) == 2
    assert opened[0]["accepted"] is True
    assert opened[1]["accepted"] is False

    assert tracker.mark(_market(100.2, start + timedelta(seconds=1)), timestamp=start + timedelta(seconds=1)) == []
    closed = tracker.mark(_market(100.25, start + timedelta(seconds=3)), timestamp=start + timedelta(seconds=3))

    accepted = next(row for row in closed if row["accepted"] is True)
    rejected = next(row for row in closed if row["accepted"] is False)
    assert accepted["outcome_label"] == "target_first"
    assert accepted["target_hits"]["gross"]["hit"] is True
    assert accepted["target_hits"]["cost_adjusted"]["hit"] is True
    assert accepted["target_before_stop"]["cost_adjusted"]["gross"] is True
    assert accepted["mfe_mae_horizons"]["1"]["mfe_pct"] > 0
    assert rejected["outcome_label"] in {"stop_first", "soft_invalidation_first"}


def test_candidate_outcome_tracker_dedupes_by_strategy_side_and_status():
    tracker = CandidateOutcomeTracker(Settings(CANDIDATE_OUTCOME_COOLDOWN_SECONDS=30))
    start = datetime(2026, 5, 27, 12, 0, tzinfo=timezone.utc)
    decision = _decision()

    first = tracker.open_from_decision(decision, _market(100.0, start), opened_at=start)
    second = tracker.open_from_decision(decision, _market(100.0, start + timedelta(seconds=1)), opened_at=start + timedelta(seconds=1))

    assert len(first) == 2
    assert second == []


def test_candidate_outcome_tracker_does_not_open_on_lagged_exchange_events():
    tracker = CandidateOutcomeTracker(Settings(MAX_EXCHANGE_EVENT_LAG_MS=1_000))
    start = datetime(2026, 5, 27, 12, 0, tzinfo=timezone.utc)

    opened = tracker.open_from_decision(
        _decision(),
        _market(100.0, start, avg_hot_event_lag_30s_ms=5_000),
        opened_at=start,
    )

    assert opened == []


def test_candidate_outcome_tracker_blocks_when_accepted_lags_rejected_baseline():
    tracker = CandidateOutcomeTracker(
        Settings(
            PAPER_LIVE_ROLLING_WINDOW=10,
            PAPER_LIVE_ROLLING_MIN_ACCEPTED=2,
            PAPER_LIVE_ROLLING_MIN_REJECTED=2,
            PAPER_LIVE_ROLLING_MIN_TARGET_RATE_EDGE=0.10,
            PAPER_LIVE_ROLLING_MIN_MFE_EDGE_BPS=1.0,
        )
    )
    tracker.rolling_closed = [
        {"accepted": True, "fee_adjusted_target_before_stop": False, "mfe_after_cost_bps": -1.0},
        {"accepted": True, "fee_adjusted_target_before_stop": True, "mfe_after_cost_bps": 1.0},
        {"accepted": False, "fee_adjusted_target_before_stop": True, "mfe_after_cost_bps": 1.0},
        {"accepted": False, "fee_adjusted_target_before_stop": False, "mfe_after_cost_bps": -1.0},
    ]

    snapshot = tracker.live_edge_quality_snapshot()

    assert snapshot["ready"] is True
    assert snapshot["block"] is True
    assert snapshot["target_rate_edge"] == 0
    assert snapshot["mfe_edge_bps"] == 0


def test_candidate_outcome_tracker_allows_when_accepted_beats_rejected_baseline():
    tracker = CandidateOutcomeTracker(
        Settings(
            PAPER_LIVE_ROLLING_WINDOW=10,
            PAPER_LIVE_ROLLING_MIN_ACCEPTED=2,
            PAPER_LIVE_ROLLING_MIN_REJECTED=2,
            PAPER_LIVE_ROLLING_MIN_TARGET_RATE_EDGE=0.10,
            PAPER_LIVE_ROLLING_MIN_MFE_EDGE_BPS=1.0,
        )
    )
    tracker.rolling_closed = [
        {"accepted": True, "fee_adjusted_target_before_stop": True, "mfe_after_cost_bps": 3.0},
        {"accepted": True, "fee_adjusted_target_before_stop": True, "mfe_after_cost_bps": 5.0},
        {"accepted": False, "fee_adjusted_target_before_stop": False, "mfe_after_cost_bps": 0.0},
        {"accepted": False, "fee_adjusted_target_before_stop": True, "mfe_after_cost_bps": 1.0},
    ]

    snapshot = tracker.live_edge_quality_snapshot()

    assert snapshot["ready"] is True
    assert snapshot["block"] is False
    assert snapshot["target_rate_edge"] == 0.5
    assert snapshot["mfe_edge_bps"] == 3.5
