from datetime import datetime, timezone

from futures_lab.models import PaperTrade, Side, TradeMode
from futures_lab.readiness import _candidate_quality, _candidate_gate, _gates, _pnl_gate, _summary_row
from futures_lab.replay import ReplaySummary


def _trade(net: float) -> PaperTrade:
    return PaperTrade(
        symbol="ETHUSDT",
        side=Side.long,
        mode=TradeMode.fast,
        entry_price=100.0,
        exit_price=100.2,
        quantity=1.0,
        stake_usd=150.0,
        notional_usd=30_000.0,
        leverage=200,
        gross_pnl_usd=net + 24.0,
        fees_usd=24.0,
        net_pnl_usd=net,
        exit_reason="take_profit" if net > 0 else "stop_loss",
        opened_at=datetime(2026, 5, 27, tzinfo=timezone.utc),
        closed_at=datetime(2026, 5, 27, 0, 1, tzinfo=timezone.utc),
    )


def _summary(net: float, trades: int = 1, mode: str = "normal") -> ReplaySummary:
    rows = [_trade(net / trades) for _ in range(trades)] if trades else []
    return ReplaySummary(
        files=["fixture.jsonl"],
        replay_mode=mode,
        messages=100,
        decisions=10,
        proposals=trades,
        risk_allowed=trades,
        paper_opens=trades,
        paper_closes=trades,
        net_pnl_usd=net,
        trades=rows,
    )


def _candidate_event(accepted: bool, score: float, label: str, mfe: float) -> dict:
    return {
        "event": "close",
        "accepted": accepted,
        "score": score,
        "outcome_label": label,
        "max_favorable_move_pct": mfe,
    }


def test_pnl_gate_passes_fails_and_detects_insufficient_data():
    assert _pnl_gate("fee", _summary(5.0)).status == "pass"
    assert _pnl_gate("fee", _summary(-5.0)).status == "fail"
    assert _pnl_gate("fee", _summary(0.0, trades=0)).status == "insufficient_data"


def test_candidate_quality_compares_accepted_rejected_and_score_buckets():
    events = []
    events.extend(_candidate_event(True, 0.90, "target_first", 0.002) for _ in range(6))
    events.extend(_candidate_event(False, 0.40, "stop_first", 0.0002) for _ in range(6))

    quality = _candidate_quality(list(events))

    assert quality.accepted_samples == 6
    assert quality.rejected_samples == 6
    assert quality.accepted_target_first_rate == 1.0
    assert quality.rejected_target_first_rate == 0.0
    assert quality.high_score_target_first_rate == 1.0
    assert quality.low_score_target_first_rate == 0.0


def test_candidate_gate_requires_enough_samples_and_performance_edge():
    insufficient = _candidate_gate("quality", 4, 6, 1.0, 0.0, 0.002, 0.0002)
    passing = _candidate_gate("quality", 6, 6, 1.0, 0.0, 0.002, 0.0002)
    failing = _candidate_gate("quality", 6, 6, 0.1, 0.2, 0.0002, 0.0003)

    assert insufficient.status == "insufficient_data"
    assert passing.status == "pass"
    assert failing.status == "fail"


def test_readiness_gates_do_not_pass_without_hostile_and_sample_evidence():
    quality = _candidate_quality([])
    gates = _gates(_summary(20.0, trades=3), _summary(-5.0, trades=3, mode="hostile"), quality)

    statuses = {gate.name: gate.status for gate in gates}
    assert statuses["positive_ev_after_real_fee_tier"] == "pass"
    assert statuses["positive_ev_after_hostile_slippage_latency"] == "fail"
    assert statuses["rejected_trades_perform_worse_than_accepted"] == "insufficient_data"
    assert statuses["enough_closed_trades_for_readiness"] == "insufficient_data"


def test_summary_row_exposes_replay_metrics():
    row = _summary_row(_summary(12.0, trades=2))

    assert row["paper_closes"] == 2
    assert row["net_pnl_usd"] == 12.0
    assert row["wins"] == 2
    assert row["losses"] == 0


def test_report_ready_requires_all_gates_pass():
    normal = _summary(20.0, trades=10)
    hostile = _summary(10.0, trades=10, mode="hostile")
    events = []
    events.extend(_candidate_event(True, 0.90, "target_first", 0.002) for _ in range(6))
    events.extend(_candidate_event(False, 0.40, "stop_first", 0.0002) for _ in range(6))
    quality = _candidate_quality(list(events))

    gates = _gates(normal, hostile, quality)

    assert all(gate.status == "pass" for gate in gates)
