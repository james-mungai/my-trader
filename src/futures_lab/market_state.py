import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

from futures_lab.config import Settings
from futures_lab.models import MarketState, Regime


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class PricePoint:
    ts: datetime
    price: float


@dataclass(frozen=True)
class FlowPoint:
    ts: datetime
    taker_buy_qty: float
    taker_sell_qty: float


@dataclass
class MarketStateBook:
    settings: Settings
    connected: bool = False
    started_at: datetime | None = None
    last_event_at: datetime | None = None
    last_received_at: datetime | None = None
    best_bid: float | None = None
    best_ask: float | None = None
    best_bid_qty: float | None = None
    best_ask_qty: float | None = None
    last_trade_price: float | None = None
    mark_price: float | None = None
    funding_rate: float | None = None
    prices: deque[PricePoint] = field(default_factory=deque)
    flows: deque[FlowPoint] = field(default_factory=deque)

    def set_connected(self, connected: bool) -> None:
        self.connected = connected
        if connected and self.started_at is None:
            self.started_at = now_utc()

    def ingest(self, payload: dict) -> None:
        received_at = now_utc()
        self.last_received_at = received_at
        event_ms = payload.get("E") or payload.get("T")
        if event_ms is not None:
            self.last_event_at = datetime.fromtimestamp(int(event_ms) / 1000, tz=timezone.utc)

        event_type = payload.get("e")
        if event_type == "bookTicker":
            self.best_bid = float(payload["b"])
            self.best_ask = float(payload["a"])
            self.best_bid_qty = float(payload["B"])
            self.best_ask_qty = float(payload["A"])
            mid = self.mid_price
            if mid is not None:
                self.prices.append(PricePoint(ts=received_at, price=mid))
        elif event_type == "aggTrade":
            price = float(payload["p"])
            qty = float(payload["q"])
            self.last_trade_price = price
            if self.mid_price is None:
                self.prices.append(PricePoint(ts=received_at, price=price))
            buyer_is_maker = bool(payload.get("m"))
            self.flows.append(
                FlowPoint(
                    ts=received_at,
                    taker_buy_qty=0.0 if buyer_is_maker else qty,
                    taker_sell_qty=qty if buyer_is_maker else 0.0,
                )
            )
        elif event_type == "markPriceUpdate":
            self.mark_price = float(payload["p"])
            self.funding_rate = float(payload["r"])
            if self.mid_price is None:
                self.prices.append(PricePoint(ts=received_at, price=self.mark_price))

        self._trim(received_at)

    @property
    def mid_price(self) -> float | None:
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2
        return self.last_trade_price or self.mark_price

    def snapshot(self) -> MarketState:
        current = now_utc()
        mid = self.mid_price
        spread_bps = None
        if mid and self.best_bid is not None and self.best_ask is not None:
            spread_bps = ((self.best_ask - self.best_bid) / mid) * 10_000

        observed = 0.0
        if self.prices:
            observed = (current - self.prices[0].ts).total_seconds()

        data_age = None
        if self.last_received_at is not None:
            data_age = (current - self.last_received_at).total_seconds()

        range_high, range_low, range_pct, range_position = self._range(current, 180)
        state = MarketState(
            symbol=self.settings.symbol.upper(),
            connected=self.connected,
            last_event_at=self.last_event_at,
            last_received_at=self.last_received_at,
            data_age_seconds=data_age,
            observed_seconds=observed,
            best_bid=self.best_bid,
            best_ask=self.best_ask,
            best_bid_qty=self.best_bid_qty,
            best_ask_qty=self.best_ask_qty,
            mid_price=mid,
            spread_bps=spread_bps,
            last_trade_price=self.last_trade_price,
            mark_price=self.mark_price,
            funding_rate=self.funding_rate,
            return_15s_pct=self._return_pct(current, 15),
            return_60s_pct=self._return_pct(current, 60),
            return_180s_pct=self._return_pct(current, 180),
            realized_vol_60s_pct=self._realized_vol(current, 60),
            realized_vol_180s_pct=self._realized_vol(current, 180),
            range_180s_pct=range_pct,
            range_high_180s=range_high,
            range_low_180s=range_low,
            range_position_180s=range_position,
            taker_buy_ratio_10s=self._taker_buy_ratio(current, 10),
            taker_buy_ratio_30s=self._taker_buy_ratio(current, 30),
            book_imbalance_top=self._book_imbalance_top(),
        )
        state.regime = self._classify_regime(state)
        return state

    def _trim(self, current: datetime) -> None:
        cutoff = current.timestamp() - self.settings.state_window_seconds
        while self.prices and self.prices[0].ts.timestamp() < cutoff:
            self.prices.popleft()
        while self.flows and self.flows[0].ts.timestamp() < cutoff:
            self.flows.popleft()

    def _prices_since(self, current: datetime, seconds: int) -> list[PricePoint]:
        cutoff = current.timestamp() - seconds
        return [point for point in self.prices if point.ts.timestamp() >= cutoff]

    def _return_pct(self, current: datetime, seconds: int) -> float | None:
        points = self._prices_since(current, seconds)
        if len(points) < 2 or points[0].price <= 0:
            return None
        return (points[-1].price - points[0].price) / points[0].price

    def _realized_vol(self, current: datetime, seconds: int) -> float | None:
        points = self._prices_since(current, seconds)
        if len(points) < 3:
            return None
        returns = []
        for prev, cur in zip(points, points[1:]):
            if prev.price > 0:
                returns.append((cur.price - prev.price) / prev.price)
        if len(returns) < 2:
            return None
        mean = sum(returns) / len(returns)
        variance = sum((ret - mean) ** 2 for ret in returns) / (len(returns) - 1)
        return math.sqrt(variance)

    def _range(self, current: datetime, seconds: int) -> tuple[float | None, float | None, float | None, float | None]:
        points = self._prices_since(current, seconds)
        if len(points) < 2:
            return None, None, None, None
        high = max(point.price for point in points)
        low = min(point.price for point in points)
        mid = self.mid_price
        if mid is None or mid <= 0 or high <= low:
            return high, low, None, None
        return high, low, (high - low) / mid, (mid - low) / (high - low)

    def _taker_buy_ratio(self, current: datetime, seconds: int) -> float | None:
        cutoff = current.timestamp() - seconds
        buy = 0.0
        sell = 0.0
        for point in self.flows:
            if point.ts.timestamp() < cutoff:
                continue
            buy += point.taker_buy_qty
            sell += point.taker_sell_qty
        total = buy + sell
        if total <= 0:
            return None
        return buy / total

    def _book_imbalance_top(self) -> float | None:
        if self.best_bid_qty is None or self.best_ask_qty is None:
            return None
        total = self.best_bid_qty + self.best_ask_qty
        if total <= 0:
            return None
        return (self.best_bid_qty - self.best_ask_qty) / total

    def _classify_regime(self, state: MarketState) -> Regime:
        if state.data_age_seconds is None or state.data_age_seconds > self.settings.stale_after_seconds:
            return Regime.stale
        if state.observed_seconds < self.settings.min_warmup_seconds:
            return Regime.warming_up
        if state.range_180s_pct is None or state.realized_vol_180s_pct is None:
            return Regime.unknown
        abs_return = abs(state.return_180s_pct or 0.0)
        if state.range_180s_pct > 0.018:
            return Regime.volatile
        if abs_return > 0.004 and state.range_180s_pct > 0.006:
            return Regime.directional
        return Regime.sideways

