import asyncio
import json
import logging
import math
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlencode

from futures_lab.audit import AuditLog
from futures_lab.config import Settings
from futures_lab.market_state import MarketStateBook, now_utc

logger = logging.getLogger(__name__)


OPEN_INTEREST_URL = "https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}"
KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _signed_clamp(value: float) -> float:
    return max(-1.0, min(1.0, value))


@dataclass
class BinanceContextPoller:
    settings: Settings
    state: MarketStateBook
    audit: AuditLog

    def __post_init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._stop: asyncio.Event | None = None

    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        if self.settings.open_interest_poll_seconds <= 0 or self.is_running():
            return
        self._stop = asyncio.Event()
        self._task = asyncio.create_task(self._run(), name="binance-context-poller")

    async def stop(self) -> None:
        if self._stop is not None:
            self._stop.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run(self) -> None:
        assert self._stop is not None
        interval = max(5, min(self.settings.open_interest_poll_seconds, self.settings.higher_timeframe_poll_seconds))
        last_higher_timeframe_poll: datetime | None = None
        while not self._stop.is_set():
            current = now_utc()
            if self.settings.open_interest_poll_seconds > 0:
                try:
                    value = await asyncio.to_thread(self._fetch_open_interest)
                    self.state.set_open_interest(value, updated_at=current)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning("Binance context poll error: %s", exc)
                    self.audit.write("context_poll_error", {"source": "open_interest", "error": str(exc)})
            if self._should_poll_higher_timeframes(last_higher_timeframe_poll, current):
                try:
                    context = await asyncio.to_thread(self._fetch_higher_timeframe_context)
                    self.state.set_higher_timeframe_context(context, updated_at=current)
                    last_higher_timeframe_poll = current
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning("Binance higher timeframe context error: %s", exc)
                    self.audit.write("context_poll_error", {"source": "higher_timeframe", "error": str(exc)})
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
            except TimeoutError:
                pass

    def _fetch_open_interest(self) -> float:
        url = OPEN_INTEREST_URL.format(symbol=self.settings.symbol.upper())
        request = urllib.request.Request(url, headers={"User-Agent": "futures-lab/0.1"})
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"open interest request failed: {exc}") from exc
        return float(payload["openInterest"])

    def _should_poll_higher_timeframes(self, previous: datetime | None, current: datetime) -> bool:
        if not self.settings.higher_timeframe_enabled:
            return False
        if previous is None:
            return True
        return (current - previous).total_seconds() >= max(60, self.settings.higher_timeframe_poll_seconds)

    def _fetch_higher_timeframe_context(self) -> dict[str, Any]:
        intervals = self._higher_timeframe_intervals()
        frames: dict[str, Any] = {}
        for interval in intervals:
            candles = self._fetch_closed_klines(interval)
            if len(candles) < max(8, self.settings.higher_timeframe_min_closed_candles):
                continue
            frames[interval] = self._summarize_candles(interval, candles)
        return self._aggregate_higher_timeframes(frames)

    def _higher_timeframe_intervals(self) -> list[str]:
        values = [item.strip() for item in self.settings.higher_timeframe_intervals.split(",")]
        return [item for item in values if item]

    def _fetch_closed_klines(self, interval: str) -> list[dict[str, float | int]]:
        query = urlencode(
            {
                "symbol": self.settings.symbol.upper(),
                "interval": interval,
                "limit": max(30, min(1500, self.settings.higher_timeframe_limit)),
            }
        )
        request = urllib.request.Request(f"{KLINES_URL}?{query}", headers={"User-Agent": "futures-lab/0.1"})
        try:
            with urllib.request.urlopen(request, timeout=8) as response:
                rows = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"kline request failed for {interval}: {exc}") from exc
        now_ms = int(now_utc().timestamp() * 1000)
        candles = []
        for row in rows:
            close_time = int(row[6])
            if close_time > now_ms:
                continue
            candles.append(
                {
                    "open_time": int(row[0]),
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                    "close_time": close_time,
                    "quote_volume": float(row[7]),
                    "trades": int(row[8]),
                    "taker_buy_base": float(row[9]),
                    "taker_buy_quote": float(row[10]),
                }
            )
        return candles

    def _summarize_candles(self, interval: str, candles: list[dict[str, float | int]]) -> dict[str, Any]:
        lookback = min(30, len(candles))
        window = candles[-lookback:]
        closes = [float(candle["close"]) for candle in candles]
        recent_closes = [float(candle["close"]) for candle in window]
        highs = [float(candle["high"]) for candle in window]
        lows = [float(candle["low"]) for candle in window]
        volumes = [float(candle["volume"]) for candle in window]
        last = window[-1]
        close = float(last["close"])
        start = recent_closes[0]
        high = max(highs)
        low = min(lows)
        return_pct = (close - start) / start if start > 0 else 0.0
        range_pct = (high - low) / close if close > 0 else 0.0
        range_position = (close - low) / (high - low) if high > low else 0.5
        fast_ema = self._ema(closes, 8)
        slow_ema = self._ema(closes, 21)
        ema_spread = (fast_ema - slow_ema) / close if close > 0 and slow_ema > 0 else 0.0
        volatility = self._avg_true_range_pct(window)
        trend_score = self._trend_score(return_pct, ema_spread, volatility, range_position)
        taker_buy_ratio = self._taker_buy_ratio(window)
        volume_z = self._volume_z_score(volumes)
        support_distance_pct = (close - low) / close if close > 0 else None
        resistance_distance_pct = (high - close) / close if close > 0 else None
        structure = self._structure_label(trend_score, range_position, return_pct)
        return {
            "interval": interval,
            "closed_candles": lookback,
            "last_close_time": datetime.fromtimestamp(int(last["close_time"]) / 1000, tz=timezone.utc).isoformat(),
            "close": close,
            "return_pct": round(return_pct, 6),
            "range_pct": round(range_pct, 6),
            "range_position": round(range_position, 4),
            "ema_fast": round(fast_ema, 4),
            "ema_slow": round(slow_ema, 4),
            "ema_spread_pct": round(ema_spread, 6),
            "volatility_pct": round(volatility, 6),
            "trend_score": round(trend_score, 4),
            "structure": structure,
            "support": round(low, 4),
            "resistance": round(high, 4),
            "support_distance_pct": round(support_distance_pct, 6) if support_distance_pct is not None else None,
            "resistance_distance_pct": round(resistance_distance_pct, 6) if resistance_distance_pct is not None else None,
            "taker_buy_ratio": round(taker_buy_ratio, 4) if taker_buy_ratio is not None else None,
            "volume_z_score": round(volume_z, 4),
        }

    def _aggregate_higher_timeframes(self, frames: dict[str, Any]) -> dict[str, Any]:
        weights = {"5m": 0.12, "15m": 0.14, "1h": 0.24, "4h": 0.26, "1d": 0.24, "1w": 0.14}
        weighted = 0.0
        total_weight = 0.0
        reasons = []
        for interval, frame in frames.items():
            weight = weights.get(interval, 0.12)
            score = float(frame.get("trend_score") or 0.0)
            weighted += weight * score
            total_weight += weight
            if interval in {"1h", "4h", "1d", "1w"}:
                reasons.append(f"{interval}:{frame.get('structure')}({score:+.2f})")
        aggregate = weighted / total_weight if total_weight > 0 else 0.0
        if aggregate >= 0.18:
            side = "long"
        elif aggregate <= -0.18:
            side = "short"
        else:
            side = "neutral"
        return {
            "updated_at": now_utc().isoformat(),
            "symbol": self.settings.symbol.upper(),
            "bias": {
                "side": side,
                "strength": round(min(1.0, abs(aggregate)), 4),
                "score": round(aggregate, 4),
                "reason": ", ".join(reasons),
            },
            "timeframes": frames,
        }

    def _ema(self, values: list[float], period: int) -> float:
        if not values:
            return 0.0
        alpha = 2 / (period + 1)
        ema = values[0]
        for value in values[1:]:
            ema = alpha * value + (1 - alpha) * ema
        return ema

    def _avg_true_range_pct(self, candles: list[dict[str, float | int]]) -> float:
        if len(candles) < 2:
            return 0.0
        trs = []
        previous_close = float(candles[0]["close"])
        for candle in candles[1:]:
            high = float(candle["high"])
            low = float(candle["low"])
            tr = max(high - low, abs(high - previous_close), abs(low - previous_close))
            close = float(candle["close"])
            if close > 0:
                trs.append(tr / close)
            previous_close = close
        return sum(trs) / len(trs) if trs else 0.0

    def _trend_score(self, return_pct: float, ema_spread: float, volatility: float, range_position: float) -> float:
        vol_floor = max(0.002, volatility * 2.5)
        momentum = _signed_clamp(return_pct / vol_floor)
        ema_component = _signed_clamp(ema_spread / max(0.001, volatility))
        location = _signed_clamp((range_position - 0.5) * 2.0)
        return _signed_clamp(0.50 * momentum + 0.35 * ema_component + 0.15 * location)

    def _taker_buy_ratio(self, candles: list[dict[str, float | int]]) -> float | None:
        taker_buy = sum(float(candle["taker_buy_base"]) for candle in candles)
        volume = sum(float(candle["volume"]) for candle in candles)
        if volume <= 0:
            return None
        return _clamp(taker_buy / volume)

    def _volume_z_score(self, volumes: list[float]) -> float:
        if len(volumes) < 3:
            return 0.0
        mean = sum(volumes[:-1]) / (len(volumes) - 1)
        variance = sum((volume - mean) ** 2 for volume in volumes[:-1]) / (len(volumes) - 2)
        std = math.sqrt(variance)
        if std <= 0:
            return 0.0
        return _signed_clamp((volumes[-1] - mean) / (3 * std))

    def _structure_label(self, trend_score: float, range_position: float, return_pct: float) -> str:
        if trend_score >= 0.45:
            return "uptrend_breakout" if range_position >= 0.75 else "uptrend_pullback"
        if trend_score <= -0.45:
            return "downtrend_breakdown" if range_position <= 0.25 else "downtrend_bounce"
        if range_position >= 0.78 and return_pct > 0:
            return "range_resistance_test"
        if range_position <= 0.22 and return_pct < 0:
            return "range_support_test"
        return "balanced"
