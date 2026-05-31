import asyncio
import gzip
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import websockets

from futures_lab.binance_streams import MARKET_WS_BASE, PUBLIC_WS_BASE
from futures_lab.config import Settings


@dataclass(frozen=True)
class ProbeStreamSpec:
    name: str
    url: str
    streams: tuple[str, ...]


@dataclass
class StreamLatencyStats:
    messages: int = 0
    missing_event_time: int = 0
    event_lag_ms: list[float] = field(default_factory=list)
    trade_lag_ms: list[float] = field(default_factory=list)
    receive_gap_ms: list[float] = field(default_factory=list)
    payload_bytes: list[int] = field(default_factory=list)

    def add(
        self,
        *,
        event_lag_ms: float | None,
        trade_lag_ms: float | None,
        receive_gap_ms: float | None,
        payload_bytes: int,
    ) -> None:
        self.messages += 1
        if event_lag_ms is None:
            self.missing_event_time += 1
        else:
            self.event_lag_ms.append(event_lag_ms)
        if trade_lag_ms is not None:
            self.trade_lag_ms.append(trade_lag_ms)
        if receive_gap_ms is not None:
            self.receive_gap_ms.append(receive_gap_ms)
        self.payload_bytes.append(payload_bytes)

    def summary(self) -> dict[str, Any]:
        return {
            "messages": self.messages,
            "missing_event_time": self.missing_event_time,
            "event_lag_ms": summarize_values(self.event_lag_ms),
            "trade_lag_ms": summarize_values(self.trade_lag_ms),
            "receive_gap_ms": summarize_values(self.receive_gap_ms),
            "payload_bytes": summarize_values(self.payload_bytes),
        }


@dataclass
class ConnectionStats:
    name: str
    url: str
    streams: tuple[str, ...]
    connects: int = 0
    errors: int = 0
    messages: int = 0
    error_messages: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "streams": list(self.streams),
            "connects": self.connects,
            "errors": self.errors,
            "messages": self.messages,
            "error_messages": self.error_messages[-10:],
        }


def summarize_values(values: list[float] | list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0}
    ordered = sorted(float(value) for value in values)
    count = len(ordered)

    def percentile(pct: float) -> float:
        index = min(count - 1, max(0, int((pct / 100) * count + 0.999999) - 1))
        return ordered[index]

    return {
        "count": count,
        "min": ordered[0],
        "avg": sum(ordered) / count,
        "p50": percentile(50),
        "p75": percentile(75),
        "p90": percentile(90),
        "p95": percentile(95),
        "p99": percentile(99),
        "max": ordered[-1],
    }


def build_probe_streams(
    *,
    profile: str,
    symbol: str,
    anchor_symbol: str = "BTCUSDT",
    depth_levels: int = 5,
    include_liquidations: bool = True,
) -> list[ProbeStreamSpec]:
    symbol_lower = symbol.strip().lower()
    anchor_lower = anchor_symbol.strip().lower()
    depth = _supported_depth_levels(depth_levels)
    profile_key = profile.strip().lower().replace("_", "-")

    public_hot = (f"{symbol_lower}@bookTicker", f"{symbol_lower}@depth{depth}@100ms")
    market_hot = (f"{symbol_lower}@aggTrade",)
    public_context = (f"{anchor_lower}@bookTicker",) if anchor_lower != symbol_lower else ()
    market_context = (
        f"{symbol_lower}@markPrice@1s",
        f"{symbol_lower}@kline_1m",
        *((f"{symbol_lower}@forceOrder",) if include_liquidations else ()),
        *((f"{anchor_lower}@aggTrade", f"{anchor_lower}@markPrice@1s") if anchor_lower != symbol_lower else ()),
    )

    if profile_key == "current":
        specs = [
            _combined_spec("public-current", PUBLIC_WS_BASE, (*public_hot, *public_context)),
            _combined_spec("market-current", MARKET_WS_BASE, (*market_hot, *market_context)),
        ]
    elif profile_key == "hot-combined":
        specs = [
            _combined_spec("public-hot", PUBLIC_WS_BASE, public_hot),
            _combined_spec("market-hot", MARKET_WS_BASE, market_hot),
        ]
    elif profile_key == "hot-split":
        specs = [
            _combined_spec("bookticker", PUBLIC_WS_BASE, (f"{symbol_lower}@bookTicker",)),
            _combined_spec("depth", PUBLIC_WS_BASE, (f"{symbol_lower}@depth{depth}@100ms",)),
            _combined_spec("aggtrade", MARKET_WS_BASE, (f"{symbol_lower}@aggTrade",)),
        ]
    elif profile_key == "aggtrade":
        specs = [_combined_spec("aggtrade", MARKET_WS_BASE, (f"{symbol_lower}@aggTrade",))]
    elif profile_key == "bookticker":
        specs = [_combined_spec("bookticker", PUBLIC_WS_BASE, (f"{symbol_lower}@bookTicker",))]
    elif profile_key == "depth":
        specs = [_combined_spec("depth", PUBLIC_WS_BASE, (f"{symbol_lower}@depth{depth}@100ms",))]
    else:
        supported = "current, hot-combined, hot-split, aggtrade, bookticker, depth"
        raise ValueError(f"Unsupported latency probe profile {profile!r}. Supported profiles: {supported}.")

    return [spec for spec in specs if spec.streams]


async def run_latency_probe(
    settings: Settings,
    *,
    seconds: int,
    profile: str,
    symbol: str | None = None,
    anchor_symbol: str | None = None,
    depth_levels: int | None = None,
    write_log: bool = True,
    output_path: Path | None = None,
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    deadline = started_at + timedelta(seconds=max(0, seconds))
    probe_symbol = (symbol or settings.symbol).upper()
    probe_anchor = (anchor_symbol or settings.cross_market_anchor_symbol).upper()
    specs = build_probe_streams(
        profile=profile,
        symbol=probe_symbol,
        anchor_symbol=probe_anchor,
        depth_levels=depth_levels if depth_levels is not None else settings.depth_levels,
        include_liquidations=settings.consume_liquidation_stream,
    )
    stats: dict[str, StreamLatencyStats] = {}
    connections = {spec.name: ConnectionStats(spec.name, spec.url, spec.streams) for spec in specs}
    log_handle = None
    resolved_output_path = None
    if write_log:
        resolved_output_path = output_path or _default_output_path(settings, probe_symbol, profile, started_at)
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
        if resolved_output_path.suffix == ".gz":
            log_handle = gzip.open(resolved_output_path, "at", encoding="utf-8")
        else:
            log_handle = resolved_output_path.open("a", encoding="utf-8")

    try:
        await asyncio.gather(
            *[
                _consume_probe_connection(
                    spec,
                    deadline=deadline,
                    stats=stats,
                    connection=connections[spec.name],
                    log_handle=log_handle,
                )
                for spec in specs
            ]
        )
    finally:
        if log_handle is not None:
            log_handle.close()

    finished_at = datetime.now(timezone.utc)
    total_messages = sum(stream.messages for stream in stats.values())
    return {
        "profile": profile,
        "symbol": probe_symbol,
        "anchor_symbol": probe_anchor,
        "seconds_requested": seconds,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_seconds": (finished_at - started_at).total_seconds(),
        "total_messages": total_messages,
        "output_path": str(resolved_output_path) if resolved_output_path is not None else None,
        "connections": {name: conn.summary() for name, conn in connections.items()},
        "streams": {stream: stream_stats.summary() for stream, stream_stats in sorted(stats.items())},
    }


async def _consume_probe_connection(
    spec: ProbeStreamSpec,
    *,
    deadline: datetime,
    stats: dict[str, StreamLatencyStats],
    connection: ConnectionStats,
    log_handle: Any,
) -> None:
    last_received_at_by_stream: dict[str, datetime] = {}
    while datetime.now(timezone.utc) < deadline:
        try:
            connect_remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
            if connect_remaining <= 0:
                return
            async with websockets.connect(
                spec.url,
                open_timeout=max(1.0, min(10.0, connect_remaining)),
                close_timeout=1,
                ping_interval=150,
                ping_timeout=30,
            ) as ws:
                connection.connects += 1
                if datetime.now(timezone.utc) >= deadline:
                    return
                while datetime.now(timezone.utc) < deadline:
                    remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
                    if remaining <= 0:
                        break
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                    except TimeoutError:
                        break
                    received_at = datetime.now(timezone.utc)
                    if received_at >= deadline:
                        break
                    envelope = json.loads(raw)
                    payload = envelope.get("data", envelope)
                    stream_name = str(envelope.get("stream") or _infer_stream_name(spec, payload))
                    event_ms = payload.get("E") or payload.get("T")
                    trade_ms = payload.get("T")
                    event_lag_ms = _lag_ms(received_at, event_ms)
                    trade_lag_ms = _lag_ms(received_at, trade_ms) if trade_ms != event_ms else None
                    last_received_at = last_received_at_by_stream.get(stream_name)
                    receive_gap_ms = (
                        (received_at - last_received_at).total_seconds() * 1000
                        if last_received_at is not None
                        else None
                    )
                    last_received_at_by_stream[stream_name] = received_at
                    stats.setdefault(stream_name, StreamLatencyStats()).add(
                        event_lag_ms=event_lag_ms,
                        trade_lag_ms=trade_lag_ms,
                        receive_gap_ms=receive_gap_ms,
                        payload_bytes=len(raw),
                    )
                    connection.messages += 1
                    if log_handle is not None:
                        row = {
                            "received_at": received_at.isoformat(),
                            "connection": spec.name,
                            "stream": stream_name,
                            "event": payload.get("e"),
                            "symbol": payload.get("s"),
                            "event_time": _event_time_iso(event_ms),
                            "trade_time": _event_time_iso(trade_ms),
                            "event_lag_ms": event_lag_ms,
                            "trade_lag_ms": trade_lag_ms,
                            "receive_gap_ms": receive_gap_ms,
                            "payload_bytes": len(raw),
                        }
                        log_handle.write(json.dumps(row, separators=(",", ":"), default=str))
                        log_handle.write("\n")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            connection.errors += 1
            connection.error_messages.append(str(exc))
            remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
            if remaining <= 0:
                return
            await asyncio.sleep(min(2.0, remaining))


def _combined_spec(name: str, base_url: str, streams: tuple[str, ...]) -> ProbeStreamSpec:
    return ProbeStreamSpec(name=name, url=base_url + "/".join(streams), streams=streams)


def _lag_ms(received_at: datetime, event_ms: Any) -> float | None:
    if event_ms is None:
        return None
    try:
        event_time = datetime.fromtimestamp(int(event_ms) / 1000, tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return None
    return max(0.0, (received_at - event_time).total_seconds() * 1000)


def _event_time_iso(event_ms: Any) -> str | None:
    if event_ms is None:
        return None
    try:
        return datetime.fromtimestamp(int(event_ms) / 1000, tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return None


def _infer_stream_name(spec: ProbeStreamSpec, payload: dict[str, Any]) -> str:
    if len(spec.streams) == 1:
        return spec.streams[0]
    event = payload.get("e")
    symbol = str(payload.get("s") or "").lower()
    if event == "aggTrade" and symbol:
        return f"{symbol}@aggTrade"
    if event == "markPriceUpdate" and symbol:
        return f"{symbol}@markPrice@1s"
    if event == "kline" and symbol:
        return f"{symbol}@kline_1m"
    if event == "forceOrder" and symbol:
        return f"{symbol}@forceOrder"
    return spec.name


def _default_output_path(settings: Settings, symbol: str, profile: str, started_at: datetime) -> Path:
    safe_profile = profile.strip().lower().replace("_", "-")
    stamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    return Path(settings.data_dir) / "latency_probes" / f"{symbol}_{safe_profile}_{stamp}.jsonl"


def _supported_depth_levels(requested: int) -> int:
    for level in (5, 10, 20):
        if requested <= level:
            return level
    return 20
