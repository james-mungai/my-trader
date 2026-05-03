import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
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
        public_streams = "/".join(
            [
                f"{symbol}@bookTicker",
                f"{symbol}@depth@100ms",
            ]
        )
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
        path = self.raw_dir / f"{self.settings.symbol.upper()}_{event}_{date.today().isoformat()}.jsonl"
        row = {
            "received_at": datetime.now(timezone.utc).isoformat(),
            "message": envelope,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, separators=(",", ":"), default=str))
            handle.write("\n")

