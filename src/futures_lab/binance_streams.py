import asyncio
import gzip
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import websockets

from futures_lab.audit import AuditLog
from futures_lab.config import Settings
from futures_lab.market_state import MarketStateBook

logger = logging.getLogger(__name__)


PUBLIC_WS_BASE = "wss://fstream.binance.com/public/stream?streams="
MARKET_WS_BASE = "wss://fstream.binance.com/market/stream?streams="


@dataclass
class BinanceStreamRecorder:
    settings: Settings
    state: MarketStateBook
    audit: AuditLog

    def __post_init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._stop: asyncio.Event | None = None
        self._last_recorded_at: dict[str, datetime] = {}
        self._current_buckets: dict[str, str] = {}
        self.raw_dir = Path(self.settings.data_dir) / "raw_ws"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        if self.is_running():
            return
        self._stop = asyncio.Event()
        self._task = asyncio.create_task(self._run(), name="binance-stream-recorder")
        self.audit.write("stream_start", {"symbol": self.settings.symbol.upper()})

    async def stop(self) -> None:
        if self._stop is not None:
            self._stop.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.state.set_connected(False)
        self.audit.write("stream_stop", {"symbol": self.settings.symbol.upper()})

    async def _run(self) -> None:
        symbol = self.settings.symbol_lower
        public_stream_names = [f"{symbol}@bookTicker"]
        if self.settings.record_depth_stream:
            public_stream_names.append(f"{symbol}@depth@100ms")
        public_streams = "/".join(public_stream_names)
        market_streams = "/".join(
            [
                f"{symbol}@aggTrade",
                f"{symbol}@markPrice@1s",
                f"{symbol}@kline_1m",
            ]
        )
        await asyncio.gather(
            self._consume(PUBLIC_WS_BASE + public_streams),
            self._consume(MARKET_WS_BASE + market_streams),
        )

    async def _consume(self, url: str) -> None:
        assert self._stop is not None
        while not self._stop.is_set():
            try:
                async with websockets.connect(url, ping_interval=150, ping_timeout=30) as ws:
                    self.state.set_connected(True)
                    async for raw in ws:
                        if self._stop.is_set():
                            break
                        envelope = json.loads(raw)
                        payload = envelope.get("data", envelope)
                        if self.settings.record_raw_ws:
                            self._record_raw(envelope)
                        self.state.ingest(payload)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.state.set_connected(False)
                logger.warning("Binance stream error: %s", exc)
                self.audit.write("stream_error", {"url": url, "error": str(exc)})
                await asyncio.sleep(2)

    def _record_raw(self, envelope: dict[str, Any]) -> None:
        payload = envelope.get("data", envelope)
        event = payload.get("e", "unknown")
        now = datetime.now(timezone.utc)
        if not self._should_record_event(event, now):
            return

        bucket = self._bucket(now)
        self._compress_previous_bucket(event, bucket)
        path = self.raw_dir / f"{self.settings.symbol.upper()}_{event}_{bucket}.jsonl"
        row = {
            "received_at": now.isoformat(),
            "message": envelope,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, separators=(",", ":"), default=str))
            handle.write("\n")

    def _should_record_event(self, event: str, now: datetime) -> bool:
        if event == "depthUpdate" and not self.settings.record_depth_stream:
            return False
        if event != "bookTicker":
            return True
        interval = max(0, self.settings.record_book_ticker_min_interval_ms)
        if interval <= 0:
            return True
        last = self._last_recorded_at.get(event)
        if last is not None and (now - last).total_seconds() * 1000 < interval:
            return False
        self._last_recorded_at[event] = now
        return True

    def _bucket(self, now: datetime) -> str:
        rotation = max(1, self.settings.raw_rotation_minutes)
        minute = (now.minute // rotation) * rotation if rotation < 60 else 0
        bucket = now.replace(minute=minute, second=0, microsecond=0)
        return bucket.strftime("%Y-%m-%dT%H%MZ")

    def _compress_previous_bucket(self, event: str, current_bucket: str) -> None:
        previous_bucket = self._current_buckets.get(event)
        if previous_bucket is None:
            self._current_buckets[event] = current_bucket
            return
        if previous_bucket == current_bucket:
            return
        self._current_buckets[event] = current_bucket
        if not self.settings.compress_rotated_raw:
            return
        raw_path = self.raw_dir / f"{self.settings.symbol.upper()}_{event}_{previous_bucket}.jsonl"
        gz_path = raw_path.with_suffix(raw_path.suffix + ".gz")
        if not raw_path.exists() or gz_path.exists():
            return
        with raw_path.open("rb") as source, gzip.open(gz_path, "wb") as target:
            target.writelines(source)
        raw_path.unlink()
