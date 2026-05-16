from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum

from futures_lab.models import MarketState, Regime, TradeMode


class MarketSequenceState(str, Enum):
    neutral = "neutral"
    data_blocked = "data_blocked"
    bullish_pressure = "bullish_pressure"
    bearish_pressure = "bearish_pressure"
    pullback = "pullback"
    bounce = "bounce"
    reclaim = "reclaim"
    rejection = "rejection"
    long_continuation_confirmed = "long_continuation_confirmed"
    short_continuation_confirmed = "short_continuation_confirmed"
    exhaustion = "exhaustion"


@dataclass(frozen=True)
class MarketRegimeSnapshot:
    state: MarketSequenceState
    previous_state: MarketSequenceState | None
    confidence: float
    allowed_modes: tuple[TradeMode, ...] = ()
    blockers: tuple[str, ...] = ()
    path: tuple[MarketSequenceState, ...] = ()
    transition_count: int = 0
    transition_probability: float | None = None

    @property
    def allows_long(self) -> bool:
        return self.state == MarketSequenceState.long_continuation_confirmed

    @property
    def allows_short(self) -> bool:
        return self.state == MarketSequenceState.short_continuation_confirmed

    def model_dump(self) -> dict:
        return {
            "state": self.state.value,
            "previous_state": self.previous_state.value if self.previous_state else None,
            "confidence": self.confidence,
            "allowed_modes": [mode.value for mode in self.allowed_modes],
            "blockers": list(self.blockers),
            "path": [state.value for state in self.path],
            "transition_count": self.transition_count,
            "transition_probability": self.transition_probability,
            "allows_long": self.allows_long,
            "allows_short": self.allows_short,
        }


@dataclass
class MarkovTransitionLogger:
    transition_counts: Counter[tuple[MarketSequenceState, MarketSequenceState]] = field(default_factory=Counter)
    from_counts: Counter[MarketSequenceState] = field(default_factory=Counter)
    state_counts: Counter[MarketSequenceState] = field(default_factory=Counter)

    def record(self, previous: MarketSequenceState | None, current: MarketSequenceState) -> None:
        self.state_counts[current] += 1
        if previous is None:
            return
        self.transition_counts[(previous, current)] += 1
        self.from_counts[previous] += 1

    def transition_count(self, previous: MarketSequenceState | None, current: MarketSequenceState) -> int:
        if previous is None:
            return 0
        return self.transition_counts[(previous, current)]

    def transition_probability(self, previous: MarketSequenceState | None, current: MarketSequenceState) -> float | None:
        if previous is None:
            return None
        total = self.from_counts[previous]
        if total <= 0:
            return None
        return self.transition_counts[(previous, current)] / total

    def model_dump(self) -> dict:
        transitions: dict[str, dict[str, float | int]] = {}
        for (previous, current), count in sorted(
            self.transition_counts.items(),
            key=lambda item: (item[0][0].value, item[0][1].value),
        ):
            key = f"{previous.value}->{current.value}"
            total = self.from_counts[previous]
            transitions[key] = {
                "count": count,
                "probability": count / total if total else 0.0,
            }
        return {
            "states": {state.value: count for state, count in sorted(self.state_counts.items(), key=lambda item: item[0].value)},
            "transitions": transitions,
        }


@dataclass
class MarketStateMachine:
    history_size: int = 8
    logger: MarkovTransitionLogger = field(default_factory=MarkovTransitionLogger)

    def __post_init__(self) -> None:
        self._state: MarketSequenceState | None = None
        self._path: deque[MarketSequenceState] = deque(maxlen=max(2, self.history_size))
        self.latest: MarketRegimeSnapshot | None = None

    def update(self, market: MarketState) -> MarketRegimeSnapshot:
        previous = self._state
        current, confidence, blockers = self._classify(market)
        self._state = current
        self._path.append(current)
        self.logger.record(previous, current)
        snapshot = MarketRegimeSnapshot(
            state=current,
            previous_state=previous,
            confidence=confidence,
            allowed_modes=self._allowed_modes(current),
            blockers=tuple(blockers),
            path=tuple(self._path),
            transition_count=self.logger.transition_count(previous, current),
            transition_probability=self.logger.transition_probability(previous, current),
        )
        self.latest = snapshot
        return snapshot

    def model_dump(self) -> dict:
        return self.logger.model_dump()

    def _classify(self, market: MarketState) -> tuple[MarketSequenceState, float, list[str]]:
        blockers = self._data_blockers(market)
        if blockers:
            return MarketSequenceState.data_blocked, 0.0, blockers

        buy_flow = market.taker_buy_ratio_10s or 0.5
        sell_flow = 1.0 - buy_flow
        r15 = market.return_15s_pct or 0.0
        r60 = market.return_60s_pct or 0.0
        r180 = market.return_180s_pct or 0.0
        book = market.book_imbalance_top or 0.0
        depth = market.depth_imbalance_top5 if market.depth_imbalance_top5 is not None else book
        pressure = (book + depth) / 2.0

        if self._is_exhaustion(market):
            return MarketSequenceState.exhaustion, 0.65, ["liquidation/exhaustion pulse active"]

        if self._recent(MarketSequenceState.reclaim) and self._long_reaccelerating(r15, buy_flow, pressure):
            return MarketSequenceState.long_continuation_confirmed, self._bounded(0.60 + buy_flow * 0.30 + max(0.0, pressure) * 0.10), []
        if self._recent(MarketSequenceState.pullback) and self._long_reaccelerating(r15, buy_flow, pressure):
            return MarketSequenceState.reclaim, self._bounded(0.55 + buy_flow * 0.35 + max(0.0, pressure) * 0.10), []

        if self._recent(MarketSequenceState.rejection) and self._short_reaccelerating(r15, sell_flow, pressure):
            return MarketSequenceState.short_continuation_confirmed, self._bounded(0.60 + sell_flow * 0.30 + max(0.0, -pressure) * 0.10), []
        if self._recent(MarketSequenceState.bounce) and self._short_reaccelerating(r15, sell_flow, pressure):
            return MarketSequenceState.rejection, self._bounded(0.55 + sell_flow * 0.35 + max(0.0, -pressure) * 0.10), []

        if self._recent(MarketSequenceState.bullish_pressure) and self._controlled_pullback(r15, r180):
            return MarketSequenceState.pullback, self._bounded(0.55 + abs(r15) * 80 + max(0.0, r180) * 90), []
        if self._recent(MarketSequenceState.bearish_pressure) and self._controlled_bounce(r15, r180):
            return MarketSequenceState.bounce, self._bounded(0.55 + abs(r15) * 80 + abs(min(0.0, r180)) * 90), []

        if self._bullish_pressure(r60, r180, buy_flow, pressure):
            confidence = self._bounded(0.45 + buy_flow * 0.25 + max(0.0, r180) * 80 + max(0.0, pressure) * 0.15)
            return MarketSequenceState.bullish_pressure, confidence, []
        if self._bearish_pressure(r60, r180, sell_flow, pressure):
            confidence = self._bounded(0.45 + sell_flow * 0.25 + abs(min(0.0, r180)) * 80 + max(0.0, -pressure) * 0.15)
            return MarketSequenceState.bearish_pressure, confidence, []

        return MarketSequenceState.neutral, 0.35, ["no confirmed sequence state"]

    def _data_blockers(self, market: MarketState) -> list[str]:
        blockers = []
        if not market.connected:
            blockers.append("stream disconnected")
        if market.regime in {Regime.stale, Regime.warming_up, Regime.unknown}:
            blockers.append(f"regime={market.regime.value}")
        if market.mid_price is None:
            blockers.append("missing mid price")
        if market.range_position_180s is None:
            blockers.append("missing range position")
        if market.taker_buy_ratio_10s is None:
            blockers.append("missing taker flow")
        return blockers

    def _allowed_modes(self, state: MarketSequenceState) -> tuple[TradeMode, ...]:
        if state in {MarketSequenceState.long_continuation_confirmed, MarketSequenceState.short_continuation_confirmed}:
            return (TradeMode.fast,)
        return ()

    def _recent(self, state: MarketSequenceState, lookback: int = 4) -> bool:
        recent = list(self._path)[-lookback:]
        return state in recent

    def _is_exhaustion(self, market: MarketState) -> bool:
        liquidation = market.liquidation_notional_30s or 0.0
        range_pct = market.range_180s_pct or 0.0
        return liquidation > 250_000 and range_pct > 0.002

    def _bullish_pressure(self, r60: float, r180: float, buy_flow: float, pressure: float) -> bool:
        return r180 > 0.0007 and r60 > -0.0003 and buy_flow >= 0.62 and pressure >= 0.05

    def _bearish_pressure(self, r60: float, r180: float, sell_flow: float, pressure: float) -> bool:
        return r180 < -0.0007 and r60 < 0.0003 and sell_flow >= 0.62 and pressure <= -0.05

    def _controlled_pullback(self, r15: float, r180: float) -> bool:
        return r180 > 0.0004 and -0.0012 <= r15 <= 0.00015

    def _controlled_bounce(self, r15: float, r180: float) -> bool:
        return r180 < -0.0004 and -0.00015 <= r15 <= 0.0012

    def _long_reaccelerating(self, r15: float, buy_flow: float, pressure: float) -> bool:
        return r15 > 0.00012 and buy_flow >= 0.58 and pressure >= 0.0

    def _short_reaccelerating(self, r15: float, sell_flow: float, pressure: float) -> bool:
        return r15 < -0.00012 and sell_flow >= 0.58 and pressure <= 0.0

    def _bounded(self, value: float) -> float:
        return round(max(0.0, min(1.0, value)), 4)
