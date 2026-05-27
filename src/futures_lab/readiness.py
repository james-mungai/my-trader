from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

from futures_lab.config import Settings
from futures_lab.replay import ReplaySummary, discover_raw_files, replay_files


@dataclass
class ReadinessGate:
    name: str
    status: str
    reason: str
    metrics: dict = field(default_factory=dict)

    def model_dump(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
            "metrics": self.metrics,
        }


@dataclass
class CandidateQualitySummary:
    accepted_samples: int = 0
    rejected_samples: int = 0
    accepted_target_first_rate: float | None = None
    rejected_target_first_rate: float | None = None
    accepted_avg_mfe_pct: float | None = None
    rejected_avg_mfe_pct: float | None = None
    high_score_samples: int = 0
    low_score_samples: int = 0
    high_score_target_first_rate: float | None = None
    low_score_target_first_rate: float | None = None
    high_score_avg_mfe_pct: float | None = None
    low_score_avg_mfe_pct: float | None = None

    def model_dump(self) -> dict:
        return {
            "accepted_samples": self.accepted_samples,
            "rejected_samples": self.rejected_samples,
            "accepted_target_first_rate": self.accepted_target_first_rate,
            "rejected_target_first_rate": self.rejected_target_first_rate,
            "accepted_avg_mfe_pct": self.accepted_avg_mfe_pct,
            "rejected_avg_mfe_pct": self.rejected_avg_mfe_pct,
            "high_score_samples": self.high_score_samples,
            "low_score_samples": self.low_score_samples,
            "high_score_target_first_rate": self.high_score_target_first_rate,
            "low_score_target_first_rate": self.low_score_target_first_rate,
            "high_score_avg_mfe_pct": self.high_score_avg_mfe_pct,
            "low_score_avg_mfe_pct": self.low_score_avg_mfe_pct,
        }


@dataclass
class ReadinessReport:
    data_dir: str
    files: list[str]
    normal: dict
    hostile: dict
    candidate_quality: CandidateQualitySummary
    gates: list[ReadinessGate]

    @property
    def ready(self) -> bool:
        return bool(self.gates) and all(gate.status == "pass" for gate in self.gates)

    def model_dump(self) -> dict:
        return {
            "data_dir": self.data_dir,
            "files": self.files,
            "ready": self.ready,
            "normal": self.normal,
            "hostile": self.hostile,
            "candidate_quality": self.candidate_quality.model_dump(),
            "gates": [gate.model_dump() for gate in self.gates],
        }


def evaluate_readiness(
    settings: Settings,
    pattern: str | None = None,
    decision_interval_ms: int | None = None,
    include_depth: bool = False,
    book_ticker_min_interval_ms: int = 100,
    flatten_at_end: bool = True,
) -> ReadinessReport:
    files = discover_raw_files(settings, pattern=pattern)
    if not files:
        raise FileNotFoundError("No raw WebSocket JSONL files found for readiness evaluation.")
    normal = replay_files(
        settings=settings,
        paths=files,
        decision_interval_ms=decision_interval_ms,
        include_depth=include_depth,
        book_ticker_min_interval_ms=book_ticker_min_interval_ms,
        flatten_at_end=flatten_at_end,
        hostile=False,
    )
    hostile = replay_files(
        settings=settings,
        paths=files,
        decision_interval_ms=decision_interval_ms,
        include_depth=include_depth,
        book_ticker_min_interval_ms=book_ticker_min_interval_ms,
        flatten_at_end=flatten_at_end,
        hostile=True,
    )
    candidate_quality = _candidate_quality(normal.candidate_outcome_events)
    gates = _gates(normal, hostile, candidate_quality)
    return ReadinessReport(
        data_dir=str(settings.data_dir),
        files=[str(Path(path)) for path in files],
        normal=_summary_row(normal),
        hostile=_summary_row(hostile),
        candidate_quality=candidate_quality,
        gates=gates,
    )


def _summary_row(summary: ReplaySummary) -> dict:
    max_adverse_pnl = min([stats.max_adverse_pnl_usd for stats in summary.position_stats], default=0.0)
    return {
        "replay_mode": summary.replay_mode,
        "messages": summary.messages,
        "decisions": summary.decisions,
        "proposals": summary.proposals,
        "risk_allowed": summary.risk_allowed,
        "paper_opens": summary.paper_opens,
        "paper_closes": summary.paper_closes,
        "net_pnl_usd": summary.net_pnl_usd,
        "wins": summary.wins,
        "losses": summary.losses,
        "win_rate": summary.win_rate,
        "max_adverse_pnl_usd": max_adverse_pnl,
        "hostile_replay": summary.hostile_replay.model_dump() if summary.hostile_replay else None,
    }


def _candidate_quality(events: list[dict]) -> CandidateQualitySummary:
    closes = [event for event in events if event.get("event") == "close"]
    accepted = [row for row in closes if row.get("accepted")]
    rejected = [row for row in closes if not row.get("accepted")]
    high_score = [row for row in closes if float(row.get("score") or 0.0) >= 0.85]
    low_score = [row for row in closes if float(row.get("score") or 0.0) < 0.70]
    return CandidateQualitySummary(
        accepted_samples=len(accepted),
        rejected_samples=len(rejected),
        accepted_target_first_rate=_target_first_rate(accepted),
        rejected_target_first_rate=_target_first_rate(rejected),
        accepted_avg_mfe_pct=_avg_mfe(accepted),
        rejected_avg_mfe_pct=_avg_mfe(rejected),
        high_score_samples=len(high_score),
        low_score_samples=len(low_score),
        high_score_target_first_rate=_target_first_rate(high_score),
        low_score_target_first_rate=_target_first_rate(low_score),
        high_score_avg_mfe_pct=_avg_mfe(high_score),
        low_score_avg_mfe_pct=_avg_mfe(low_score),
    )


def _target_first_rate(rows: list[dict]) -> float | None:
    if not rows:
        return None
    return round(sum(1 for row in rows if row.get("outcome_label") == "target_first") / len(rows), 4)


def _avg_mfe(rows: list[dict]) -> float | None:
    if not rows:
        return None
    return round(mean(float(row.get("max_favorable_move_pct") or 0.0) for row in rows), 6)


def _gates(
    normal: ReplaySummary,
    hostile: ReplaySummary,
    candidate_quality: CandidateQualitySummary,
) -> list[ReadinessGate]:
    return [
        _pnl_gate("positive_ev_after_real_fee_tier", normal),
        _pnl_gate("positive_ev_after_hostile_slippage_latency", hostile),
        _candidate_gate(
            "rejected_trades_perform_worse_than_accepted",
            candidate_quality.accepted_samples,
            candidate_quality.rejected_samples,
            candidate_quality.accepted_target_first_rate,
            candidate_quality.rejected_target_first_rate,
            candidate_quality.accepted_avg_mfe_pct,
            candidate_quality.rejected_avg_mfe_pct,
        ),
        _candidate_gate(
            "high_score_trades_outperform_low_score_trades",
            candidate_quality.high_score_samples,
            candidate_quality.low_score_samples,
            candidate_quality.high_score_target_first_rate,
            candidate_quality.low_score_target_first_rate,
            candidate_quality.high_score_avg_mfe_pct,
            candidate_quality.low_score_avg_mfe_pct,
        ),
        _trade_count_gate(normal, hostile),
    ]


def _pnl_gate(name: str, summary: ReplaySummary) -> ReadinessGate:
    if summary.paper_closes <= 0:
        return ReadinessGate(
            name=name,
            status="insufficient_data",
            reason="No closed paper trades in replay.",
            metrics=_summary_row(summary),
        )
    return ReadinessGate(
        name=name,
        status="pass" if summary.net_pnl_usd > 0 else "fail",
        reason="Net replay PnL is positive." if summary.net_pnl_usd > 0 else "Net replay PnL is not positive.",
        metrics=_summary_row(summary),
    )


def _candidate_gate(
    name: str,
    left_samples: int,
    right_samples: int,
    left_target_rate: float | None,
    right_target_rate: float | None,
    left_mfe: float | None,
    right_mfe: float | None,
) -> ReadinessGate:
    metrics = {
        "left_samples": left_samples,
        "right_samples": right_samples,
        "left_target_first_rate": left_target_rate,
        "right_target_first_rate": right_target_rate,
        "left_avg_mfe_pct": left_mfe,
        "right_avg_mfe_pct": right_mfe,
    }
    if left_samples < 5 or right_samples < 5:
        return ReadinessGate(
            name=name,
            status="insufficient_data",
            reason="Need at least 5 samples on both sides of the comparison.",
            metrics=metrics,
        )
    target_edge = (left_target_rate or 0.0) > (right_target_rate or 0.0)
    mfe_edge = (left_mfe or 0.0) > (right_mfe or 0.0)
    return ReadinessGate(
        name=name,
        status="pass" if target_edge and mfe_edge else "fail",
        reason="Left cohort outperformed on target-first rate and MFE." if target_edge and mfe_edge else "Left cohort did not outperform on both target-first rate and MFE.",
        metrics=metrics,
    )


def _trade_count_gate(normal: ReplaySummary, hostile: ReplaySummary) -> ReadinessGate:
    metrics = {
        "normal_closed_trades": normal.paper_closes,
        "hostile_closed_trades": hostile.paper_closes,
    }
    if normal.paper_closes < 10 or hostile.paper_closes < 10:
        return ReadinessGate(
            name="enough_closed_trades_for_readiness",
            status="insufficient_data",
            reason="Need at least 10 closed trades in both normal and hostile replay before readiness can be considered.",
            metrics=metrics,
        )
    return ReadinessGate(
        name="enough_closed_trades_for_readiness",
        status="pass",
        reason="Both normal and hostile replay have enough closed trades for a first readiness read.",
        metrics=metrics,
    )
