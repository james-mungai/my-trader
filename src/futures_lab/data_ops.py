import gzip
import json
import shutil
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from futures_lab.config import Settings


@dataclass
class FileSummary:
    path: str
    bytes: int

    def model_dump(self) -> dict:
        return {"path": self.path, "bytes": self.bytes}


@dataclass
class DirectorySummary:
    path: str
    bytes: int
    files: int

    def model_dump(self) -> dict:
        return {"path": self.path, "bytes": self.bytes, "files": self.files}


@dataclass
class DataSummary:
    data_dir: str
    total_bytes: int
    directories: list[DirectorySummary] = field(default_factory=list)
    largest_files: list[FileSummary] = field(default_factory=list)

    def model_dump(self) -> dict:
        return {
            "data_dir": self.data_dir,
            "total_bytes": self.total_bytes,
            "directories": [directory.model_dump() for directory in self.directories],
            "largest_files": [file.model_dump() for file in self.largest_files],
        }


@dataclass
class FileOperationSummary:
    data_dir: str
    matched: int = 0
    changed: int = 0
    freed_bytes: int = 0
    files: list[str] = field(default_factory=list)

    def model_dump(self) -> dict:
        return {
            "data_dir": self.data_dir,
            "matched": self.matched,
            "changed": self.changed,
            "freed_bytes": self.freed_bytes,
            "files": self.files,
        }


def summarize_data(settings: Settings, largest: int = 20) -> DataSummary:
    data_dir = Path(settings.data_dir)
    files = [path for path in data_dir.rglob("*") if path.is_file()]
    total = sum(path.stat().st_size for path in files)
    directories: list[DirectorySummary] = []
    for directory in sorted([data_dir, *[path for path in data_dir.iterdir() if path.is_dir()]]):
        directory_files = [path for path in directory.rglob("*") if path.is_file()]
        directories.append(
            DirectorySummary(
                path=str(directory),
                bytes=sum(path.stat().st_size for path in directory_files),
                files=len(directory_files),
            )
        )
    largest_files = [
        FileSummary(path=str(path), bytes=path.stat().st_size)
        for path in sorted(files, key=lambda item: item.stat().st_size, reverse=True)[:largest]
    ]
    return DataSummary(data_dir=str(data_dir), total_bytes=total, directories=directories, largest_files=largest_files)


def compress_raw(settings: Settings, older_than_minutes: int = 5, include_current: bool = False) -> FileOperationSummary:
    raw_dir = Path(settings.data_dir) / "raw_ws"
    summary = FileOperationSummary(data_dir=str(settings.data_dir))
    if not raw_dir.exists():
        return summary
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=older_than_minutes)
    for path in sorted(raw_dir.glob("*.jsonl")):
        summary.matched += 1
        modified = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
        if not include_current and modified > cutoff:
            continue
        gz_path = path.with_suffix(path.suffix + ".gz")
        if gz_path.exists():
            continue
        before = path.stat().st_size
        with path.open("rb") as source, gzip.open(gz_path, "wb") as target:
            shutil.copyfileobj(source, target)
        shutil.copystat(path, gz_path)
        after = gz_path.stat().st_size
        path.unlink()
        summary.changed += 1
        summary.freed_bytes += max(0, before - after)
        summary.files.append(str(gz_path))
    return summary


@dataclass
class RegimeOutcomeSummary:
    data_dir: str
    files: int = 0
    opens: int = 0
    closes: int = 0
    sides: dict[str, int] = field(default_factory=dict)
    states: dict[str, int] = field(default_factory=dict)
    close_reasons: dict[str, int] = field(default_factory=dict)
    target_hits: dict[str, int] = field(default_factory=dict)
    stop_hits: dict[str, int] = field(default_factory=dict)
    target_before_stop: dict[str, dict[str, int]] = field(default_factory=dict)
    horizon_direction_correct: dict[str, dict[str, int]] = field(default_factory=dict)
    early_follow_through: dict[str, int] = field(default_factory=dict)
    side_early_follow_through: dict[str, dict[str, int]] = field(default_factory=dict)
    average_quality_score: float | None = None

    def model_dump(self) -> dict:
        return {
            "data_dir": self.data_dir,
            "files": self.files,
            "opens": self.opens,
            "closes": self.closes,
            "sides": self.sides,
            "states": self.states,
            "close_reasons": self.close_reasons,
            "target_hits": self.target_hits,
            "stop_hits": self.stop_hits,
            "target_before_stop": self.target_before_stop,
            "horizon_direction_correct": self.horizon_direction_correct,
            "early_follow_through": self.early_follow_through,
            "side_early_follow_through": self.side_early_follow_through,
            "average_quality_score": self.average_quality_score,
        }


def summarize_regime_outcomes(settings: Settings) -> RegimeOutcomeSummary:
    outcome_dir = Path(settings.data_dir) / "regime_outcomes"
    summary = RegimeOutcomeSummary(data_dir=str(settings.data_dir))
    if not outcome_dir.exists():
        return summary

    sides: Counter[str] = Counter()
    states: Counter[str] = Counter()
    close_reasons: Counter[str] = Counter()
    target_hits: Counter[str] = Counter()
    stop_hits: Counter[str] = Counter()
    target_before_stop: dict[str, Counter[str]] = {}
    horizon_direction_correct: dict[str, Counter[str]] = {}
    early_follow_through: Counter[str] = Counter()
    side_early_follow_through: dict[str, Counter[str]] = {}
    quality_scores: list[float] = []

    for path in sorted(outcome_dir.glob("*.jsonl")):
        summary.files += 1
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                event = row.get("event")
                if event == "open":
                    summary.opens += 1
                    sides[str(row.get("side"))] += 1
                    states[str(row.get("state"))] += 1
                    score = row.get("quality_score")
                    if isinstance(score, (int, float)):
                        quality_scores.append(float(score))
                if event != "close":
                    continue
                summary.closes += 1
                close_reasons[str(row.get("close_reason"))] += 1
                for key, value in (row.get("target_hits") or {}).items():
                    if value.get("hit"):
                        target_hits[key] += 1
                for key, value in (row.get("stop_hits") or {}).items():
                    if value.get("hit"):
                        stop_hits[key] += 1
                for target, stops in (row.get("target_before_stop") or {}).items():
                    target_counter = target_before_stop.setdefault(target, Counter())
                    for stop, result in stops.items():
                        if result is True:
                            target_counter[stop] += 1
                for horizon, value in (row.get("horizons") or {}).items():
                    if not value:
                        continue
                    counter = horizon_direction_correct.setdefault(horizon, Counter())
                    counter["correct" if value.get("direction_correct") else "wrong"] += 1
                follow = row.get("early_follow_through") or {}
                if follow:
                    label = "qualified" if follow.get("qualified") else "failed"
                    early_follow_through[label] += 1
                    side = str(row.get("side"))
                    side_counter = side_early_follow_through.setdefault(side, Counter())
                    side_counter[label] += 1

    summary.sides = dict(sides)
    summary.states = dict(states)
    summary.close_reasons = dict(close_reasons)
    summary.target_hits = dict(target_hits)
    summary.stop_hits = dict(stop_hits)
    summary.target_before_stop = {target: dict(counter) for target, counter in target_before_stop.items()}
    summary.horizon_direction_correct = {horizon: dict(counter) for horizon, counter in horizon_direction_correct.items()}
    summary.early_follow_through = dict(early_follow_through)
    summary.side_early_follow_through = {side: dict(counter) for side, counter in side_early_follow_through.items()}
    summary.average_quality_score = round(sum(quality_scores) / len(quality_scores), 4) if quality_scores else None
    return summary


@dataclass
class CandidateOutcomeBucket:
    samples: int = 0
    accepted: int = 0
    rejected: int = 0
    target_first: int = 0
    fee_adjusted_target_before_stop: int = 0
    stop_first: int = 0
    timeout: int = 0
    soft_invalidation_first: int = 0
    total_mfe_60s_pct: float = 0.0
    total_mae_60s_pct: float = 0.0
    total_mfe_after_cost_bps: float = 0.0

    def model_dump(self) -> dict:
        return {
            "samples": self.samples,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "target_first": self.target_first,
            "fee_adjusted_target_before_stop": self.fee_adjusted_target_before_stop,
            "fee_adjusted_target_before_stop_rate": (
                round(self.fee_adjusted_target_before_stop / self.samples, 6) if self.samples else None
            ),
            "stop_first": self.stop_first,
            "timeout": self.timeout,
            "soft_invalidation_first": self.soft_invalidation_first,
            "avg_mfe_60s_pct": round(self.total_mfe_60s_pct / self.samples, 6) if self.samples else None,
            "avg_mae_60s_pct": round(self.total_mae_60s_pct / self.samples, 6) if self.samples else None,
            "avg_mfe_after_cost_bps": round(self.total_mfe_after_cost_bps / self.samples, 3) if self.samples else None,
        }


@dataclass
class CandidateOutcomeSummary:
    data_dir: str
    files: int = 0
    opens: int = 0
    closes: int = 0
    accepted_vs_rejected: dict[str, CandidateOutcomeBucket] = field(default_factory=dict)
    families: dict[str, CandidateOutcomeBucket] = field(default_factory=dict)
    strategies: dict[str, CandidateOutcomeBucket] = field(default_factory=dict)
    score_buckets: dict[str, CandidateOutcomeBucket] = field(default_factory=dict)
    target_before_stop: dict[str, dict[str, int]] = field(default_factory=dict)
    outcome_labels: dict[str, int] = field(default_factory=dict)
    fee_adjusted_outcome_labels: dict[str, int] = field(default_factory=dict)

    def model_dump(self) -> dict:
        return {
            "data_dir": self.data_dir,
            "files": self.files,
            "opens": self.opens,
            "closes": self.closes,
            "accepted_vs_rejected": {name: bucket.model_dump() for name, bucket in self.accepted_vs_rejected.items()},
            "families": {name: bucket.model_dump() for name, bucket in self.families.items()},
            "strategies": {name: bucket.model_dump() for name, bucket in self.strategies.items()},
            "score_buckets": {name: bucket.model_dump() for name, bucket in self.score_buckets.items()},
            "target_before_stop": self.target_before_stop,
            "outcome_labels": self.outcome_labels,
            "fee_adjusted_outcome_labels": self.fee_adjusted_outcome_labels,
        }


def summarize_candidate_outcomes(settings: Settings) -> CandidateOutcomeSummary:
    outcome_dir = Path(settings.data_dir) / "candidate_outcomes"
    summary = CandidateOutcomeSummary(data_dir=str(settings.data_dir))
    if not outcome_dir.exists():
        return summary

    target_before_stop: dict[str, Counter[str]] = {}
    outcome_labels: Counter[str] = Counter()
    fee_adjusted_outcome_labels: Counter[str] = Counter()
    for path in sorted(outcome_dir.glob("*.jsonl")):
        summary.files += 1
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                event = row.get("event")
                if event == "open":
                    summary.opens += 1
                    continue
                if event != "close":
                    continue
                summary.closes += 1
                accepted_key = "accepted" if row.get("accepted") else "rejected"
                family_key = str(row.get("family") or "unknown")
                strategy_key = str(row.get("strategy") or "unknown")
                score_key = _score_bucket(float(row.get("score") or 0.0))
                for bucket in [
                    summary.accepted_vs_rejected.setdefault(accepted_key, CandidateOutcomeBucket()),
                    summary.families.setdefault(family_key, CandidateOutcomeBucket()),
                    summary.strategies.setdefault(strategy_key, CandidateOutcomeBucket()),
                    summary.score_buckets.setdefault(score_key, CandidateOutcomeBucket()),
                ]:
                    _update_candidate_bucket(bucket, row)
                outcome_labels[str(row.get("outcome_label") or "unknown")] += 1
                fee_adjusted_outcome_labels[str(row.get("fee_adjusted_outcome_label") or "unknown")] += 1
                for target, stops in (row.get("target_before_stop") or {}).items():
                    target_counter = target_before_stop.setdefault(target, Counter())
                    for stop, result in stops.items():
                        if result is True:
                            target_counter[stop] += 1

    summary.target_before_stop = {target: dict(counter) for target, counter in target_before_stop.items()}
    summary.outcome_labels = dict(outcome_labels)
    summary.fee_adjusted_outcome_labels = dict(fee_adjusted_outcome_labels)
    return summary


def _update_candidate_bucket(bucket: CandidateOutcomeBucket, row: dict) -> None:
    bucket.samples += 1
    if row.get("accepted"):
        bucket.accepted += 1
    else:
        bucket.rejected += 1
    label = str(row.get("outcome_label") or "")
    if label == "target_first":
        bucket.target_first += 1
    if row.get("fee_adjusted_target_before_stop") is True:
        bucket.fee_adjusted_target_before_stop += 1
    if label == "stop_first":
        bucket.stop_first += 1
    elif label == "timeout":
        bucket.timeout += 1
    elif label == "soft_invalidation_first":
        bucket.soft_invalidation_first += 1
    horizon = (row.get("mfe_mae_horizons") or {}).get("60") or {}
    bucket.total_mfe_60s_pct += float(horizon.get("mfe_pct") or row.get("max_favorable_move_pct") or 0.0)
    bucket.total_mae_60s_pct += float(horizon.get("mae_pct") or row.get("max_adverse_move_pct") or 0.0)
    bucket.total_mfe_after_cost_bps += float(row.get("mfe_after_cost_bps") or 0.0)


def _score_bucket(score: float) -> str:
    if score >= 0.85:
        return "score_0.85_plus"
    if score >= 0.70:
        return "score_0.70_to_0.85"
    return "score_below_0.70"


@dataclass
class ExitShadowPolicySummary:
    samples: int = 0
    wins: int = 0
    losses: int = 0
    gross_pnl_usd: float = 0.0
    fees_usd: float = 0.0
    net_pnl_usd: float = 0.0
    net_vs_fixed_usd: float = 0.0
    improved_vs_fixed: int = 0
    worsened_vs_fixed: int = 0
    exit_reasons: dict[str, int] = field(default_factory=dict)

    def model_dump(self) -> dict:
        return {
            "samples": self.samples,
            "wins": self.wins,
            "losses": self.losses,
            "gross_pnl_usd": round(self.gross_pnl_usd, 4),
            "fees_usd": round(self.fees_usd, 4),
            "net_pnl_usd": round(self.net_pnl_usd, 4),
            "net_vs_fixed_usd": round(self.net_vs_fixed_usd, 4),
            "improved_vs_fixed": self.improved_vs_fixed,
            "worsened_vs_fixed": self.worsened_vs_fixed,
            "exit_reasons": self.exit_reasons,
        }


@dataclass
class ExitShadowSummary:
    data_dir: str
    files: int = 0
    rows: int = 0
    rows_with_exit_shadow: int = 0
    best_policy: dict[str, int] = field(default_factory=dict)
    policies: dict[str, ExitShadowPolicySummary] = field(default_factory=dict)

    def model_dump(self) -> dict:
        return {
            "data_dir": self.data_dir,
            "files": self.files,
            "rows": self.rows,
            "rows_with_exit_shadow": self.rows_with_exit_shadow,
            "best_policy": self.best_policy,
            "policies": {name: summary.model_dump() for name, summary in self.policies.items()},
        }


def summarize_exit_shadow(settings: Settings) -> ExitShadowSummary:
    summary = ExitShadowSummary(data_dir=str(settings.data_dir))
    base = Path(settings.data_dir)
    policy_reason_counts: dict[str, Counter[str]] = {}
    best_policy: Counter[str] = Counter()

    for directory in [base / "paper_trades", base / "shadow_trades"]:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.jsonl")):
            summary.files += 1
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("event") == "open":
                        continue
                    summary.rows += 1
                    exit_shadow = row.get("exit_shadow") or {}
                    policies = exit_shadow.get("policies") or {}
                    if not policies:
                        continue
                    summary.rows_with_exit_shadow += 1
                    if exit_shadow.get("best_policy"):
                        best_policy[str(exit_shadow["best_policy"])] += 1
                    for name, policy in policies.items():
                        if not policy.get("closed"):
                            continue
                        policy_summary = summary.policies.setdefault(str(name), ExitShadowPolicySummary())
                        reason_counts = policy_reason_counts.setdefault(str(name), Counter())
                        net = float(policy.get("net_pnl_usd") or 0.0)
                        policy_summary.samples += 1
                        policy_summary.gross_pnl_usd += float(policy.get("gross_pnl_usd") or 0.0)
                        policy_summary.fees_usd += float(policy.get("fees_usd") or 0.0)
                        policy_summary.net_pnl_usd += net
                        policy_summary.net_vs_fixed_usd += float(policy.get("net_vs_fixed_usd") or 0.0)
                        if net > 0:
                            policy_summary.wins += 1
                        elif net < 0:
                            policy_summary.losses += 1
                        delta = float(policy.get("net_vs_fixed_usd") or 0.0)
                        if delta > 0:
                            policy_summary.improved_vs_fixed += 1
                        elif delta < 0:
                            policy_summary.worsened_vs_fixed += 1
                        reason_counts[str(policy.get("exit_reason"))] += 1

    summary.best_policy = dict(best_policy)
    for name, counter in policy_reason_counts.items():
        summary.policies[name].exit_reasons = dict(counter)
    return summary


def prune_raw(settings: Settings, older_than_hours: float, dry_run: bool = False) -> FileOperationSummary:
    raw_dir = Path(settings.data_dir) / "raw_ws"
    summary = FileOperationSummary(data_dir=str(settings.data_dir))
    if not raw_dir.exists():
        return summary
    cutoff = datetime.now(timezone.utc) - timedelta(hours=older_than_hours)
    for path in sorted([*raw_dir.glob("*.jsonl"), *raw_dir.glob("*.jsonl.gz")]):
        modified = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
        if modified > cutoff:
            continue
        summary.matched += 1
        size = path.stat().st_size
        summary.files.append(str(path))
        if dry_run:
            continue
        path.unlink()
        summary.changed += 1
        summary.freed_bytes += size
    return summary
