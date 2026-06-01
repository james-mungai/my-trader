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
from futures_lab.cross_market import build_cross_market_context
from futures_lab.market_state import MarketStateBook

logger = logging.getLogger(__name__)


PUBLIC_WS_BASE = "wss://fstream.binance.com/public/stream?streams="
MARKET_WS_BASE = "wss://fstream.binance.com/market/stream?streams="


@dataclass
class StreamConnectionSpec:
    name: str
    url: str
    streams: tuple[str, ...]


@dataclass
class BinanceStreamRecorder:
    settings: Settings
    state: MarketStateBook
    audit: AuditLog

    def __post_init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._stop: asyncio.Event | None = None
        self._last_recorded_at: dict[str, datetime] = {}
        self._last_ingested_at: dict[str, datetime] = {}
        self._last_stale_event_audit_at: dict[str, datetime] = {}
        self._stream_connected_at: dict[str, datetime] = {}
        self._current_buckets: dict[str, str] = {}
        self._anchor_symbol = self.settings.cross_market_anchor_symbol.upper()
        self._cross_market_state = self._build_cross_market_state()
        self.raw_dir = Path(self.settings.data_dir) / "raw_ws"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        if self.is_running():
            return
        self._stop = asyncio.Event()
        self._task = asyncio.create_task(self._run(), name="binance-stream-recorder")
        self.audit.write(
            "stream_start",
            {
                "symbol": self.settings.symbol.upper(),
                "profile": self._stream_profile(),
            },
        )

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
        await asyncio.gather(*[self._consume(spec) for spec in self._stream_specs()])

    def _stream_specs(self) -> list[StreamConnectionSpec]:
        symbol = self.settings.symbol_lower
        anchor = self._anchor_symbol.lower()
        depth_stream = None
        if self.settings.consume_depth_stream or self.settings.record_depth_stream:
            levels = self._supported_depth_levels(self.settings.depth_levels)
            depth_stream = f"{symbol}@depth{levels}@100ms"

        book_ticker_stream = f"{symbol}@bookTicker" if self.settings.consume_book_ticker_stream else None
        primary_public = (*((book_ticker_stream,) if book_ticker_stream else ()), *((depth_stream,) if depth_stream else ()))
        primary_market = (f"{symbol}@aggTrade",)
        public_context = (
            (f"{anchor}@bookTicker",)
            if self._cross_market_state is not None and self.settings.consume_book_ticker_stream
            else ()
        )
        market_context = (
            f"{symbol}@markPrice@1s",
            f"{symbol}@kline_1m",
            *((f"{symbol}@forceOrder",) if self.settings.consume_liquidation_stream else ()),
            *((f"{anchor}@aggTrade", f"{anchor}@markPrice@1s") if self._cross_market_state is not None else ()),
        )

        profile = self._stream_profile()
        if profile == "current":
            specs = [
                self._combined_spec("public-current", PUBLIC_WS_BASE, (*primary_public, *public_context)),
                self._combined_spec("market-current", MARKET_WS_BASE, (*primary_market, *market_context)),
            ]
        elif profile == "hot-combined":
            specs = [
                self._combined_spec("public-hot", PUBLIC_WS_BASE, primary_public),
                self._combined_spec("market-hot", MARKET_WS_BASE, primary_market),
                self._combined_spec("public-context", PUBLIC_WS_BASE, public_context),
                self._combined_spec("market-context", MARKET_WS_BASE, market_context),
            ]
        elif profile == "hot-split":
            specs = [
                self._combined_spec("bookticker", PUBLIC_WS_BASE, (book_ticker_stream,) if book_ticker_stream else ()),
                self._combined_spec("depth", PUBLIC_WS_BASE, (depth_stream,) if depth_stream else ()),
                self._combined_spec("aggtrade", MARKET_WS_BASE, primary_market),
                self._combined_spec("public-context", PUBLIC_WS_BASE, public_context),
                self._combined_spec("market-context", MARKET_WS_BASE, market_context),
            ]
        else:
            supported = "current, hot-combined, hot-split"
            raise ValueError(f"Unsupported BINANCE_STREAM_PROFILE {self.settings.binance_stream_profile!r}. Supported: {supported}.")

        return [spec for spec in specs if spec.streams]

    def _stream_profile(self) -> str:
        return self.settings.binance_stream_profile.strip().lower().replace("_", "-")

    def _combined_spec(self, name: str, base_url: str, streams: tuple[str, ...]) -> StreamConnectionSpec:
        return StreamConnectionSpec(name=name, url=base_url + "/".join(streams), streams=streams)

    async def _consume(self, spec: StreamConnectionSpec) -> None:
        url = spec.url
        assert self._stop is not None
        while not self._stop.is_set():
            try:
                async with websockets.connect(url, ping_interval=150, ping_timeout=30) as ws:
                    self.state.set_connected(True)
                    connected_at = datetime.now(timezone.utc)
                    self._stream_connected_at[url] = connected_at
                    self.audit.write(
                        "stream_connected",
                        {
                            "name": spec.name,
                            "url": url,
                            "streams": list(spec.streams),
                            "profile": self._stream_profile(),
                        },
                    )
                    async for raw in ws:
                        if self._stop.is_set():
                            break
                        received_at = datetime.now(timezone.utc)
                        envelope = json.loads(raw)
                        payload = envelope.get("data", envelope)
                        if self.settings.record_raw_ws:
                            self._record_raw(envelope, received_at=received_at)
                        if self._should_ingest_payload(envelope, payload, received_at):
                            self._ingest_payload(envelope, payload, received_at=received_at)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.state.set_connected(False)
                disconnected_at = datetime.now(timezone.utc)
                connected_at = self._stream_connected_at.pop(url, None)
                session_seconds = (
                    (disconnected_at - connected_at).total_seconds()
                    if connected_at is not None
                    else None
                )
                logger.warning("Binance stream error: %s", exc)
                self.audit.write(
                    "stream_error",
                    {
                        "name": spec.name,
                        "url": url,
                        "streams": list(spec.streams),
                        "profile": self._stream_profile(),
                        "error": str(exc),
                        "session_seconds": session_seconds,
                        "reconnect_sleep_seconds": 2,
                    },
                )
                await asyncio.sleep(2)

    def _record_raw(self, envelope: dict[str, Any], *, received_at: datetime | None = None) -> None:
        payload = envelope.get("data", envelope)
        event = self._raw_event_name(envelope, payload)
        symbol = self._payload_symbol(envelope, payload) or self.settings.symbol.upper()
        now = received_at or datetime.now(timezone.utc)
        if not self._should_record_event(symbol, event, now):
            return

        bucket = self._bucket(now)
        self._compress_previous_bucket(symbol, event, bucket)
        path = self.raw_dir / f"{symbol}_{event}_{bucket}.jsonl"
        row = {
            "received_at": now.isoformat(),
            "message": envelope,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, separators=(",", ":"), default=str))
            handle.write("\n")

    def _should_record_event(self, symbol: str, event: str, now: datetime) -> bool:
        if event in {"depthUpdate", "partialDepth"} and not self.settings.record_depth_stream:
            return False
        if event != "bookTicker":
            return True
        interval = max(0, self.settings.record_book_ticker_min_interval_ms)
        if interval <= 0:
            return True
        key = f"{symbol}:{event}"
        last = self._last_recorded_at.get(key)
        if last is not None and (now - last).total_seconds() * 1000 < interval:
            return False
        self._last_recorded_at[key] = now
        return True

    def _bucket(self, now: datetime) -> str:
        rotation = max(1, self.settings.raw_rotation_minutes)
        minute = (now.minute // rotation) * rotation if rotation < 60 else 0
        bucket = now.replace(minute=minute, second=0, microsecond=0)
        return bucket.strftime("%Y-%m-%dT%H%MZ")

    def _compress_previous_bucket(self, symbol: str, event: str, current_bucket: str) -> None:
        key = f"{symbol}:{event}"
        previous_bucket = self._current_buckets.get(key)
        if previous_bucket is None:
            self._current_buckets[key] = current_bucket
            return
        if previous_bucket == current_bucket:
            return
        self._current_buckets[key] = current_bucket
        if not self.settings.compress_rotated_raw:
            return
        raw_path = self.raw_dir / f"{symbol}_{event}_{previous_bucket}.jsonl"
        gz_path = raw_path.with_suffix(raw_path.suffix + ".gz")
        if not raw_path.exists() or gz_path.exists():
            return
        with raw_path.open("rb") as source, gzip.open(gz_path, "wb") as target:
            target.writelines(source)
        raw_path.unlink()

    def _raw_event_name(self, envelope: dict[str, Any], payload: dict[str, Any]) -> str:
        event = payload.get("e")
        if event is not None:
            return str(event)
        stream = str(envelope.get("stream") or "")
        if "@depth" in stream:
            return "partialDepth"
        return "unknown"

    def _should_ingest_payload(self, envelope: dict[str, Any], payload: dict[str, Any], received_at: datetime) -> bool:
        event = self._raw_event_name(envelope, payload)
        if event != "bookTicker":
            return True
        interval = max(0, self.settings.consume_book_ticker_min_interval_ms)
        if interval <= 0:
            return True
        symbol = self._payload_symbol(envelope, payload) or self.settings.symbol.upper()
        key = f"{symbol}:{event}"
        last = self._last_ingested_at.get(key)
        if last is not None and (received_at - last).total_seconds() * 1000 < interval:
            return False
        self._last_ingested_at[key] = received_at
        return True

    def _ingest_payload(self, envelope: dict[str, Any], payload: dict[str, Any], *, received_at: datetime | None = None) -> None:
        received_at = received_at or datetime.now(timezone.utc)
        symbol = self._payload_symbol(envelope, payload)
        if self._cross_market_state is not None and symbol == self._anchor_symbol:
            accepted = self._cross_market_state.ingest(payload, received_at=received_at)
            if not accepted:
                self._audit_stale_event(symbol or self._anchor_symbol, payload)
                return
            self._refresh_cross_market_context()
            return
        accepted = self.state.ingest(payload, received_at=received_at)
        if not accepted:
            self._audit_stale_event(symbol or self.settings.symbol.upper(), payload)
            return
        self._refresh_cross_market_context()

    def _audit_stale_event(self, symbol: str, payload: dict[str, Any]) -> None:
        event_ms = payload.get("E") or payload.get("T")
        if event_ms is None:
            return
        now = datetime.now(timezone.utc)
        key = f"{symbol}:{self._raw_event_name({}, payload)}"
        last = self._last_stale_event_audit_at.get(key)
        if last is not None and (now - last).total_seconds() < 60:
            return
        lag_ms = max(0.0, (now - datetime.fromtimestamp(int(event_ms) / 1000, tz=timezone.utc)).total_seconds() * 1000)
        self._last_stale_event_audit_at[key] = now
        self.audit.write(
            "stream_stale_event_dropped",
            {
                "symbol": symbol,
                "event": self._raw_event_name({}, payload),
                "lag_ms": lag_ms,
                "max_exchange_event_lag_ms": self.settings.max_exchange_event_lag_ms,
            },
        )

    def _refresh_cross_market_context(self) -> None:
        if self._cross_market_state is None:
            return
        now = datetime.now(timezone.utc)
        primary = self.state.snapshot(current=now)
        anchor = self._cross_market_state.snapshot(current=now)
        context = build_cross_market_context(self.settings, primary, anchor, updated_at=now)
        self.state.set_cross_market_context(context, updated_at=now)

    def _payload_symbol(self, envelope: dict[str, Any], payload: dict[str, Any]) -> str | None:
        symbol = payload.get("s")
        if symbol:
            return str(symbol).upper()
        stream = str(envelope.get("stream") or "")
        if "@" in stream:
            return stream.split("@", 1)[0].upper()
        return None

    def _build_cross_market_state(self) -> MarketStateBook | None:
        if not self.settings.cross_market_enabled:
            return None
        if self._anchor_symbol == self.settings.symbol.upper():
            return None
        return MarketStateBook(self.settings.model_copy(update={"symbol": self._anchor_symbol}))

    def _supported_depth_levels(self, requested: int) -> int:
        for level in (5, 10, 20):
            if requested <= level:
                return level
        return 20
