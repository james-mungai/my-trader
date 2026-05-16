import asyncio
import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass

from futures_lab.audit import AuditLog
from futures_lab.config import Settings
from futures_lab.market_state import MarketStateBook, now_utc

logger = logging.getLogger(__name__)


OPEN_INTEREST_URL = "https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}"


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
        interval = max(5, self.settings.open_interest_poll_seconds)
        while not self._stop.is_set():
            try:
                value = await asyncio.to_thread(self._fetch_open_interest)
                self.state.set_open_interest(value, updated_at=now_utc())
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Binance context poll error: %s", exc)
                self.audit.write("context_poll_error", {"source": "open_interest", "error": str(exc)})
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
