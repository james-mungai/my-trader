import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

from futures_lab.config import Settings
from futures_lab.liquidation_phase import classify_liquidation_phase
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


@dataclass(frozen=True)
class OrderFlowImbalancePoint:
    ts: datetime
    imbalance: float
    magnitude: float


@dataclass(frozen=True)
class SpreadPoint:
    ts: datetime
    spread_bps: float


@dataclass(frozen=True)
class DepthPoint:
    ts: datetime
    bid_qty: float
    ask_qty: float


@dataclass(frozen=True)
class LiquidationPoint:
    ts: datetime
    side: str
    notional: float


@dataclass(frozen=True)
class LatencyPoint:
    ts: datetime
    lag_ms: float


@dataclass(frozen=True)
class OpenInterestPoint:
    ts: datetime
    value: float


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
    open_interest: float | None = None
    open_interest_updated_at: datetime | None = None
    prices: deque[PricePoint] = field(default_factory=deque)
    flows: deque[FlowPoint] = field(default_factory=deque)
    order_flow_imbalances: deque[OrderFlowImbalancePoint] = field(default_factory=deque)
    spreads: deque[SpreadPoint] = field(default_factory=deque)
    depth_points: deque[DepthPoint] = field(default_factory=deque)
    liquidations: deque[LiquidationPoint] = field(default_factory=deque)
    latencies: deque[LatencyPoint] = field(default_factory=deque)
    open_interest_points: deque[OpenInterestPoint] = field(default_factory=deque)
    higher_timeframe_context: dict = field(default_factory=dict)
    higher_timeframe_updated_at: datetime | None = None
    cross_market_context: dict = field(default_factory=dict)
    cross_market_updated_at: datetime | None = None
    depth_bids: dict[float, float] = field(default_factory=dict)
    depth_asks: dict[float, float] = field(default_factory=dict)
    last_stream_event_type: str | None = None
    last_depth_top_book_sample_at: datetime | None = None

    def set_connected(self, connected: bool) -> None:
        self.connected = connected
        if connected and self.started_at is None:
            self.started_at = now_utc()

    def ingest(self, payload: dict, received_at: datetime | None = None) -> bool:
        received_at = received_at or now_utc()
        event_ms = payload.get("E") or payload.get("T")
        if event_ms is not None:
            event_at = datetime.fromtimestamp(int(event_ms) / 1000, tz=timezone.utc)
            lag_ms = max(0.0, (received_at - event_at).total_seconds() * 1000)
            self.latencies.append(LatencyPoint(ts=received_at, lag_ms=lag_ms))
            if lag_ms > self.settings.max_exchange_event_lag_ms:
                self._trim(received_at)
                return False
            self.last_event_at = event_at
        self.last_received_at = received_at

        event_type = self._event_type(payload)
        self.last_stream_event_type = event_type
        if event_type == "bookTicker":
            bid = float(payload["b"])
            ask = float(payload["a"])
            bid_qty = float(payload["B"])
            ask_qty = float(payload["A"])
            self._append_order_flow_imbalance(received_at, bid, ask, bid_qty, ask_qty)
            self.best_bid = bid
            self.best_ask = ask
            self.best_bid_qty = bid_qty
            self.best_ask_qty = ask_qty
            self._append_spread(received_at)
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
        elif event_type in {"depthUpdate", "partialDepth"}:
            sampled = self._ingest_depth(payload, received_at)
            if sampled:
                self._append_depth_point(received_at)
        elif event_type == "forceOrder":
            order = payload.get("o", {})
            price = float(order.get("p") or order.get("ap") or 0.0)
            qty = float(order.get("q") or 0.0)
            notional = price * qty
            if notional > 0:
                self.liquidations.append(
                    LiquidationPoint(
                        ts=received_at,
                        side=str(order.get("S", "")).upper(),
                        notional=notional,
                    )
                )

        self._trim(received_at)
        return True

    def set_open_interest(self, value: float, updated_at: datetime | None = None) -> None:
        updated_at = updated_at or now_utc()
        self.open_interest = value
        self.open_interest_updated_at = updated_at
        self.open_interest_points.append(OpenInterestPoint(ts=updated_at, value=value))
        self._trim(updated_at)

    def set_higher_timeframe_context(self, context: dict, updated_at: datetime | None = None) -> None:
        self.higher_timeframe_context = context
        self.higher_timeframe_updated_at = updated_at or now_utc()

    def set_cross_market_context(self, context: dict, updated_at: datetime | None = None) -> None:
        self.cross_market_context = context
        self.cross_market_updated_at = updated_at or now_utc()

    @property
    def mid_price(self) -> float | None:
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2
        return self.last_trade_price or self.mark_price

    def snapshot(self, current: datetime | None = None) -> MarketState:
        current = current or now_utc()
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
            depth_bid_qty_top5=self._depth_qty(self.depth_bids, reverse=True),
            depth_ask_qty_top5=self._depth_qty(self.depth_asks, reverse=False),
            depth_imbalance_top5=self._depth_imbalance(),
            depth_bid_wall_ratio_top5=self._depth_wall_ratio(self.depth_bids, reverse=True),
            depth_ask_wall_ratio_top5=self._depth_wall_ratio(self.depth_asks, reverse=False),
            mid_price=mid,
            spread_bps=spread_bps,
            spread_bps_avg_5s=self._spread_avg(current, 5),
            spread_bps_std_5s=self._spread_std(current, 5),
            spread_bps_max_5s=self._spread_max(current, 5),
            last_trade_price=self.last_trade_price,
            mark_price=self.mark_price,
            mark_last_basis_bps=self._mark_last_basis_bps(),
            funding_rate=self.funding_rate,
            open_interest=self.open_interest,
            open_interest_age_seconds=(
                (current - self.open_interest_updated_at).total_seconds()
                if self.open_interest_updated_at is not None
                else None
            ),
            open_interest_change_5m_pct=self._open_interest_change_pct(current, 300),
            return_15s_pct=self._return_pct(current, 15),
            return_60s_pct=self._return_pct(current, 60),
            return_180s_pct=self._return_pct(current, 180),
            realized_vol_60s_pct=self._realized_vol(current, 60),
            realized_vol_180s_pct=self._realized_vol(current, 180),
            range_180s_pct=range_pct,
            range_high_180s=range_high,
            range_low_180s=range_low,
            range_position_180s=range_position,
            order_flow_imbalance_250ms=self._order_flow_imbalance(current, 0.250),
            order_flow_imbalance_1s=self._order_flow_imbalance(current, 1),
            order_flow_imbalance_5s=self._order_flow_imbalance(current, 5),
            taker_aggression_imbalance_1s=self._taker_aggression_imbalance(current, 1),
            taker_aggression_imbalance_5s=self._taker_aggression_imbalance(current, 5),
            taker_aggression_imbalance_15s=self._taker_aggression_imbalance(current, 15),
            taker_buy_ratio_10s=self._taker_buy_ratio(current, 10),
            taker_buy_ratio_30s=self._taker_buy_ratio(current, 30),
            book_imbalance_top=self._book_imbalance_top(),
            microprice=self._microprice(),
            microprice_mid_bps=self._price_mid_bps(self._microprice(), mid),
            vamp_price_top=self._vamp_price(),
            vamp_mid_bps=self._price_mid_bps(self._vamp_price(), mid),
            weighted_depth_price_top=self._weighted_depth_price(),
            weighted_depth_mid_bps=self._price_mid_bps(self._weighted_depth_price(), mid),
            bid_depth_refill_rate_5s=self._depth_change_rate(current, 5, side="bid", positive=True),
            ask_depth_refill_rate_5s=self._depth_change_rate(current, 5, side="ask", positive=True),
            bid_depth_evaporation_rate_5s=self._depth_change_rate(current, 5, side="bid", positive=False),
            ask_depth_evaporation_rate_5s=self._depth_change_rate(current, 5, side="ask", positive=False),
            liquidation_notional_30s=self._liquidation_notional(current, 30),
            long_liquidation_notional_30s=self._liquidation_notional(current, 30, side="SELL"),
            short_liquidation_notional_30s=self._liquidation_notional(current, 30, side="BUY"),
            liquidation_buy_ratio_30s=self._liquidation_buy_ratio(current, 30),
            last_stream_event_type=self.last_stream_event_type,
            exchange_event_lag_ms=self.latencies[-1].lag_ms if self.latencies else None,
            avg_event_lag_30s_ms=self._latency_avg(current, 30),
            max_event_lag_30s_ms=self._latency_max(current, 30),
            higher_timeframe_context=self.higher_timeframe_context,
            higher_timeframe_context_age_seconds=(
                (current - self.higher_timeframe_updated_at).total_seconds()
                if self.higher_timeframe_updated_at is not None
                else None
            ),
            higher_timeframe_bias_side=self._higher_timeframe_bias_side(),
            higher_timeframe_bias_strength=self._higher_timeframe_bias_strength(),
            higher_timeframe_bias_reason=self._higher_timeframe_bias_reason(),
            cross_market_context=self.cross_market_context,
            cross_market_context_age_seconds=(
                (current - self.cross_market_updated_at).total_seconds()
                if self.cross_market_updated_at is not None
                else None
            ),
            btc_order_flow_imbalance_1s=self._cross_anchor_value("order_flow_imbalance_1s"),
            btc_taker_aggression_imbalance_1s=self._cross_anchor_value("taker_aggression_imbalance_1s"),
            btc_microprice_mid_bps=self._cross_anchor_value("microprice_mid_bps"),
            btc_return_15s_pct=self._cross_anchor_value("return_15s_pct"),
            btc_return_60s_pct=self._cross_anchor_value("return_60s_pct"),
            eth_btc_relative_return_15s_pct=self._cross_relative_value("return_15s_pct"),
            eth_btc_relative_return_60s_pct=self._cross_relative_value("return_60s_pct"),
        )
        liquidation_phase = classify_liquidation_phase(state, self.settings)
        state.liquidation_phase = liquidation_phase.phase
        state.liquidation_phase_side = liquidation_phase.side
        state.liquidation_phase_confidence = liquidation_phase.confidence
        state.liquidation_phase_context = liquidation_phase.model_dump()
        state.regime = self._classify_regime(state)
        return state

    def _trim(self, current: datetime) -> None:
        cutoff = current.timestamp() - self.settings.state_window_seconds
        while self.prices and self.prices[0].ts.timestamp() < cutoff:
            self.prices.popleft()
        while self.flows and self.flows[0].ts.timestamp() < cutoff:
            self.flows.popleft()
        while self.order_flow_imbalances and self.order_flow_imbalances[0].ts.timestamp() < cutoff:
            self.order_flow_imbalances.popleft()
        while self.spreads and self.spreads[0].ts.timestamp() < cutoff:
            self.spreads.popleft()
        while self.depth_points and self.depth_points[0].ts.timestamp() < cutoff:
            self.depth_points.popleft()
        while self.liquidations and self.liquidations[0].ts.timestamp() < cutoff:
            self.liquidations.popleft()
        while self.latencies and self.latencies[0].ts.timestamp() < cutoff:
            self.latencies.popleft()
        while self.open_interest_points and self.open_interest_points[0].ts.timestamp() < cutoff:
            self.open_interest_points.popleft()

    def _event_type(self, payload: dict) -> str | None:
        event_type = payload.get("e")
        if event_type is None and ("bids" in payload or "asks" in payload):
            return "partialDepth"
        return event_type

    def _ingest_depth(self, payload: dict, received_at: datetime) -> bool:
        bids = payload.get("b") or payload.get("bids") or []
        asks = payload.get("a") or payload.get("asks") or []
        if payload.get("e") == "depthUpdate":
            self._apply_depth_delta(self.depth_bids, bids)
            self._apply_depth_delta(self.depth_asks, asks)
            self._trim_depth_books()
            return self._sync_top_of_book_from_depth(received_at)
        self.depth_bids = {float(price): float(qty) for price, qty in bids if float(qty) > 0}
        self.depth_asks = {float(price): float(qty) for price, qty in asks if float(qty) > 0}
        self._trim_depth_books()
        return self._sync_top_of_book_from_depth(received_at)

    def _apply_depth_delta(self, book: dict[float, float], levels: list) -> None:
        for price_raw, qty_raw in levels:
            price = float(price_raw)
            qty = float(qty_raw)
            if qty <= 0:
                book.pop(price, None)
            else:
                book[price] = qty

    def _trim_depth_books(self) -> None:
        levels = max(1, self.settings.depth_levels)
        self.depth_bids = dict(sorted(self.depth_bids.items(), reverse=True)[:levels])
        self.depth_asks = dict(sorted(self.depth_asks.items())[:levels])

    def _sync_top_of_book_from_depth(self, ts: datetime) -> bool:
        if not self.depth_bids or not self.depth_asks:
            return False
        bid, bid_qty = max(self.depth_bids.items())
        ask, ask_qty = min(self.depth_asks.items())
        should_sample = self._should_sample_depth_top_book(ts)
        if should_sample:
            self._append_order_flow_imbalance(ts, bid, ask, bid_qty, ask_qty)
        self.best_bid = bid
        self.best_ask = ask
        self.best_bid_qty = bid_qty
        self.best_ask_qty = ask_qty
        if should_sample:
            self._append_spread(ts)
            mid = self.mid_price
            if mid is not None:
                self.prices.append(PricePoint(ts=ts, price=mid))
        return should_sample

    def _should_sample_depth_top_book(self, ts: datetime) -> bool:
        interval_ms = max(0, self.settings.consume_depth_top_book_min_interval_ms)
        if interval_ms <= 0:
            self.last_depth_top_book_sample_at = ts
            return True
        last = self.last_depth_top_book_sample_at
        if last is not None and (ts - last).total_seconds() * 1000 < interval_ms:
            return False
        self.last_depth_top_book_sample_at = ts
        return True

    def _append_order_flow_imbalance(
        self,
        ts: datetime,
        bid: float,
        ask: float,
        bid_qty: float,
        ask_qty: float,
    ) -> None:
        if (
            self.best_bid is None
            or self.best_ask is None
            or self.best_bid_qty is None
            or self.best_ask_qty is None
        ):
            return
        imbalance = 0.0
        magnitude = 0.0
        if bid > self.best_bid:
            imbalance += bid_qty
            magnitude += bid_qty
        elif bid == self.best_bid:
            delta = bid_qty - self.best_bid_qty
            imbalance += delta
            magnitude += abs(delta)
        else:
            imbalance -= self.best_bid_qty
            magnitude += self.best_bid_qty

        if ask < self.best_ask:
            imbalance -= ask_qty
            magnitude += ask_qty
        elif ask == self.best_ask:
            delta = self.best_ask_qty - ask_qty
            imbalance += delta
            magnitude += abs(delta)
        else:
            imbalance += self.best_ask_qty
            magnitude += self.best_ask_qty

        if magnitude > 0:
            self.order_flow_imbalances.append(
                OrderFlowImbalancePoint(ts=ts, imbalance=imbalance, magnitude=magnitude)
            )

    def _append_spread(self, ts: datetime) -> None:
        mid = self.mid_price
        if mid is None or mid <= 0 or self.best_bid is None or self.best_ask is None:
            return
        self.spreads.append(SpreadPoint(ts=ts, spread_bps=((self.best_ask - self.best_bid) / mid) * 10_000))

    def _append_depth_point(self, ts: datetime) -> None:
        bid_qty = self._depth_qty(self.depth_bids, reverse=True)
        ask_qty = self._depth_qty(self.depth_asks, reverse=False)
        if bid_qty is None or ask_qty is None:
            return
        self.depth_points.append(DepthPoint(ts=ts, bid_qty=bid_qty, ask_qty=ask_qty))

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

    def _taker_aggression_imbalance(self, current: datetime, seconds: int) -> float | None:
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
        return (buy - sell) / total

    def _order_flow_imbalance(self, current: datetime, seconds: float) -> float | None:
        cutoff = current.timestamp() - seconds
        imbalance = 0.0
        magnitude = 0.0
        for point in self.order_flow_imbalances:
            if point.ts.timestamp() < cutoff:
                continue
            imbalance += point.imbalance
            magnitude += point.magnitude
        if magnitude <= 0:
            return None
        return max(-1.0, min(1.0, imbalance / magnitude))

    def _book_imbalance_top(self) -> float | None:
        if self.best_bid_qty is None or self.best_ask_qty is None:
            return None
        total = self.best_bid_qty + self.best_ask_qty
        if total <= 0:
            return None
        return (self.best_bid_qty - self.best_ask_qty) / total

    def _depth_qty(self, book: dict[float, float], reverse: bool) -> float | None:
        if not book:
            return None
        levels = sorted(book.items(), reverse=reverse)[: max(1, self.settings.depth_levels)]
        return sum(qty for _, qty in levels)

    def _depth_imbalance(self) -> float | None:
        bid_qty = self._depth_qty(self.depth_bids, reverse=True)
        ask_qty = self._depth_qty(self.depth_asks, reverse=False)
        if bid_qty is None or ask_qty is None:
            return None
        total = bid_qty + ask_qty
        if total <= 0:
            return None
        return (bid_qty - ask_qty) / total

    def _depth_wall_ratio(self, book: dict[float, float], reverse: bool) -> float | None:
        if not book:
            return None
        levels = [qty for _, qty in sorted(book.items(), reverse=reverse)[: max(1, self.settings.depth_levels)]]
        total = sum(levels)
        if total <= 0:
            return None
        return max(levels) / total

    def _spread_values_since(self, current: datetime, seconds: int) -> list[float]:
        cutoff = current.timestamp() - seconds
        return [point.spread_bps for point in self.spreads if point.ts.timestamp() >= cutoff]

    def _spread_avg(self, current: datetime, seconds: int) -> float | None:
        values = self._spread_values_since(current, seconds)
        if not values:
            return None
        return sum(values) / len(values)

    def _spread_std(self, current: datetime, seconds: int) -> float | None:
        values = self._spread_values_since(current, seconds)
        if len(values) < 2:
            return None
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        return math.sqrt(variance)

    def _spread_max(self, current: datetime, seconds: int) -> float | None:
        values = self._spread_values_since(current, seconds)
        if not values:
            return None
        return max(values)

    def _microprice(self) -> float | None:
        if (
            self.best_bid is None
            or self.best_ask is None
            or self.best_bid_qty is None
            or self.best_ask_qty is None
        ):
            return None
        total = self.best_bid_qty + self.best_ask_qty
        if total <= 0:
            return None
        return ((self.best_bid * self.best_ask_qty) + (self.best_ask * self.best_bid_qty)) / total

    def _vamp_price(self) -> float | None:
        bids = sorted(self.depth_bids.items(), reverse=True)[: max(1, self.settings.depth_levels)]
        asks = sorted(self.depth_asks.items())[: max(1, self.settings.depth_levels)]
        if not bids or not asks:
            return self._microprice()
        numerator = 0.0
        denominator = 0.0
        for (bid_price, bid_qty), (ask_price, ask_qty) in zip(bids, asks):
            numerator += bid_price * ask_qty + ask_price * bid_qty
            denominator += bid_qty + ask_qty
        if denominator <= 0:
            return None
        return numerator / denominator

    def _weighted_depth_price(self) -> float | None:
        levels = list(sorted(self.depth_bids.items(), reverse=True)[: max(1, self.settings.depth_levels)])
        levels += list(sorted(self.depth_asks.items())[: max(1, self.settings.depth_levels)])
        if not levels:
            return self.mid_price
        denominator = sum(qty for _, qty in levels)
        if denominator <= 0:
            return None
        return sum(price * qty for price, qty in levels) / denominator

    def _price_mid_bps(self, price: float | None, mid: float | None) -> float | None:
        if price is None or mid is None or mid <= 0:
            return None
        return ((price - mid) / mid) * 10_000

    def _depth_change_rate(self, current: datetime, seconds: int, side: str, positive: bool) -> float | None:
        cutoff = current.timestamp() - seconds
        points = [point for point in self.depth_points if point.ts.timestamp() >= cutoff]
        if len(points) < 2:
            return None
        first = points[0].bid_qty if side == "bid" else points[0].ask_qty
        last = points[-1].bid_qty if side == "bid" else points[-1].ask_qty
        elapsed = max(0.001, (points[-1].ts - points[0].ts).total_seconds())
        if first <= 0:
            return None
        change = (last - first) / first
        directional_change = max(0.0, change) if positive else max(0.0, -change)
        return directional_change / elapsed

    def _mark_last_basis_bps(self) -> float | None:
        if self.mark_price is None or self.last_trade_price is None or self.last_trade_price <= 0:
            return None
        return ((self.mark_price - self.last_trade_price) / self.last_trade_price) * 10_000

    def _liquidation_notional(self, current: datetime, seconds: int, side: str | None = None) -> float | None:
        cutoff = current.timestamp() - seconds
        total = 0.0
        seen = False
        for point in self.liquidations:
            if point.ts.timestamp() < cutoff:
                continue
            if side is not None and point.side != side:
                continue
            seen = True
            total += point.notional
        return total if seen else None

    def _liquidation_buy_ratio(self, current: datetime, seconds: int) -> float | None:
        buy = self._liquidation_notional(current, seconds, side="BUY") or 0.0
        sell = self._liquidation_notional(current, seconds, side="SELL") or 0.0
        total = buy + sell
        if total <= 0:
            return None
        return buy / total

    def _latencies_since(self, current: datetime, seconds: int) -> list[LatencyPoint]:
        cutoff = current.timestamp() - seconds
        return [point for point in self.latencies if point.ts.timestamp() >= cutoff]

    def _latency_avg(self, current: datetime, seconds: int) -> float | None:
        points = self._latencies_since(current, seconds)
        if not points:
            return None
        return sum(point.lag_ms for point in points) / len(points)

    def _latency_max(self, current: datetime, seconds: int) -> float | None:
        points = self._latencies_since(current, seconds)
        if not points:
            return None
        return max(point.lag_ms for point in points)

    def _open_interest_change_pct(self, current: datetime, seconds: int) -> float | None:
        cutoff = current.timestamp() - seconds
        points = [point for point in self.open_interest_points if point.ts.timestamp() >= cutoff]
        if len(points) < 2 or points[0].value <= 0:
            return None
        return (points[-1].value - points[0].value) / points[0].value

    def _classify_regime(self, state: MarketState) -> Regime:
        if state.data_age_seconds is None or state.data_age_seconds > self.settings.stale_after_seconds:
            return Regime.stale
        lag_ms = state.avg_event_lag_30s_ms if state.avg_event_lag_30s_ms is not None else state.exchange_event_lag_ms
        if lag_ms is not None and lag_ms > self.settings.max_exchange_event_lag_ms:
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

    def _higher_timeframe_bias_side(self) -> str:
        bias = self.higher_timeframe_context.get("bias") if self.higher_timeframe_context else None
        side = str((bias or {}).get("side") or "neutral").lower()
        return side if side in {"long", "short", "neutral"} else "neutral"

    def _higher_timeframe_bias_strength(self) -> float:
        bias = self.higher_timeframe_context.get("bias") if self.higher_timeframe_context else None
        strength = (bias or {}).get("strength", 0.0)
        try:
            return max(0.0, min(1.0, float(strength)))
        except (TypeError, ValueError):
            return 0.0

    def _higher_timeframe_bias_reason(self) -> str:
        bias = self.higher_timeframe_context.get("bias") if self.higher_timeframe_context else None
        return str((bias or {}).get("reason") or "")

    def _cross_anchor_value(self, key: str) -> float | None:
        anchor = self.cross_market_context.get("anchor") if self.cross_market_context else None
        value = (anchor or {}).get(key)
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _cross_relative_value(self, key: str) -> float | None:
        relative = self.cross_market_context.get("relative") if self.cross_market_context else None
        value = (relative or {}).get(key)
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None
