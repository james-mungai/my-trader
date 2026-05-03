import json
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Any

from futures_lab.config import Settings


@dataclass
class AuditLog:
    settings: Settings

    def __post_init__(self) -> None:
        self._lock = Lock()
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.settings.data_dir / "audit.log"

    def write(self, event: str, payload: dict[str, Any]) -> None:
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "payload": payload,
        }
        encoded = json.dumps(row, separators=(",", ":"), default=str)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.write("\n")

