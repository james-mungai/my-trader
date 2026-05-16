import gzip
import shutil
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
