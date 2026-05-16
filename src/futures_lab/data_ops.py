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

    summary.sides = dict(sides)
    summary.states = dict(states)
    summary.close_reasons = dict(close_reasons)
    summary.target_hits = dict(target_hits)
    summary.stop_hits = dict(stop_hits)
    summary.target_before_stop = {target: dict(counter) for target, counter in target_before_stop.items()}
    summary.horizon_direction_correct = {horizon: dict(counter) for horizon, counter in horizon_direction_correct.items()}
    summary.average_quality_score = round(sum(quality_scores) / len(quality_scores), 4) if quality_scores else None
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
