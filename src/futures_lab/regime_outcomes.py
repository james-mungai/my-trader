from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from futures_lab.config import Settings
from futures_lab.models import Decision, MarketState, utc_now


@dataclass
class RegimeOutcomeEpisode:
    episode_id: str
    symbol: str
    side: str
    state: str
    path: list[str]
    entry_price: float
    opened_at: datetime
    confidence: float
    strategy_score: float
    quality_score: float
    quality_components: dict[str, float]
    target_moves_pct: list[float]
    stop_moves_pct: list[float]
    horizons_seconds: list[int]
    source: str = "fsm_continuation"
    adaptive_allowed: bool = False
    target_feasible: bool = False
    blockers: list[str] = field(default_factory=list)
    max_favorable_move_pct: float = 0.0
    max_adverse_move_pct: float = 0.0
    last_price: float | None = None
    last_seen_at: datetime | None = None
    target_hits: dict[str, dict] = field(default_factory=dict)
    stop_hits: dict[str, dict] = field(default_factory=dict)
    horizons: dict[str, dict] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for target in self.target_moves_pct:
            self.target_hits.setdefault(_key(target), {"hit": False, "first_hit_seconds": None})
        for stop in self.stop_moves_pct:
            self.stop_hits.setdefault(_key(stop), {"hit": False, "first_hit_seconds": None})
        for horizon in self.horizons_seconds:
            self.horizons.setdefault(str(horizon), {})


@dataclass
class RegimeOutcomeTracker:
    settings: Settings
    episodes: list[RegimeOutcomeEpisode] = field(default_factory=list)
    opened_count: int = 0
    closed_count: int = 0
    last_opened_at_by_key: dict[tuple[str, str], datetime] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.horizons_seconds = sorted(set(_parse_ints(self.settings.regime_outcome_horizons_seconds)))
        self.early_horizons_seconds = sorted(set(_parse_ints(self.settings.regime_outcome_early_horizons_seconds)))
        self.target_moves_pct = sorted(set(_parse_floats(self.settings.regime_outcome_target_moves_pct)))
        self.stop_moves_pct = sorted(set(_parse_floats(self.settings.regime_outcome_stop_moves_pct)))
        if not self.horizons_seconds:
            self.horizons_seconds = [15, 30, 60, 180, 300, 900]
        if not self.early_horizons_seconds:
            self.early_horizons_seconds = [15, 30, 60]
        if not self.target_moves_pct:
            self.target_moves_pct = [0.001, 0.002, 0.0035, 0.005]
        if not self.stop_moves_pct:
            self.stop_moves_pct = [0.001, 0.002]

    def open_from_decision(
        self,
        decision: Decision,
        market: MarketState,
        opened_at: datetime | None = None,
    ) -> dict | None:
        if not self.settings.write_regime_outcome_log:
            return None
        if market.mid_price is None:
            return None
        stateful = decision.evidence.get("stateful_momentum_filter") or {}
        if not stateful.get("confirmed"):
            return None
        side = stateful.get("side")
        state = stateful.get("state")
        if side not in {"long", "short"} or state not in {"long_continuation_confirmed", "short_continuation_confirmed"}:
            return None
        current = opened_at or market.last_received_at or decision.timestamp
        key = (side, state)
        if self._has_active(key):
            return None
        last_opened = self.last_opened_at_by_key.get(key)
        if last_opened is not None and (current - last_opened).total_seconds() < self.settings.regime_outcome_cooldown_seconds:
            return None

        self.opened_count += 1
        quality_components = self._quality_components(side, stateful, market)
        episode = RegimeOutcomeEpisode(
            episode_id=f"regime-{self.opened_count:06d}",
            symbol=decision.symbol,
            side=side,
            state=state,
            path=list(stateful.get("path") or []),
            entry_price=market.mid_price,
            opened_at=current,
            confidence=float(stateful.get("sequence_confidence") or decision.confidence or 0.0),
            strategy_score=float(stateful.get("score") or decision.confidence or 0.0),
            quality_score=self._quality_score(quality_components),
            quality_components=quality_components,
            target_moves_pct=self.target_moves_pct,
            stop_moves_pct=self.stop_moves_pct,
            horizons_seconds=self.horizons_seconds,
            adaptive_allowed=bool(stateful.get("adaptive_entry_allowed")),
            target_feasible=bool(stateful.get("target_feasible")),
            blockers=list(stateful.get("blockers") or []),
        )
        self._update_episode(episode, market, current)
        self.episodes.append(episode)
        self.last_opened_at_by_key[key] = current
        return self._event("open", episode)

    def mark(self, market: MarketState, timestamp: datetime | None = None) -> list[dict]:
        if not self.episodes or market.mid_price is None:
            return []
        current = timestamp or market.last_received_at or utc_now()
        closed = []
        active = []
        for episode in self.episodes:
            self._update_episode(episode, market, current)
            if self._is_complete(episode, current):
                closed.append(self._close_event(episode, current, "max_horizon_elapsed"))
            else:
                active.append(episode)
        self.episodes = active
        return closed

    def close_all(self, market: MarketState | None, reason: str = "session_end", timestamp: datetime | None = None) -> list[dict]:
        if not self.episodes:
            return []
        current = timestamp or (market.last_received_at if market else None) or utc_now()
        if market is not None and market.mid_price is not None:
            for episode in self.episodes:
                self._update_episode(episode, market, current)
        closed = [self._close_event(episode, current, reason) for episode in self.episodes]
        self.episodes = []
        return closed

    def _has_active(self, key: tuple[str, str]) -> bool:
        side, state = key
        return any(episode.side == side and episode.state == state for episode in self.episodes)

    def _update_episode(self, episode: RegimeOutcomeEpisode, market: MarketState, current: datetime) -> None:
        if market.mid_price is None:
            return
        move = _underlying_move_pct(episode.side, episode.entry_price, market.mid_price)
        elapsed = max(0.0, (current - episode.opened_at).total_seconds())
        episode.last_price = market.mid_price
        episode.last_seen_at = current
        episode.max_favorable_move_pct = max(episode.max_favorable_move_pct, move)
        episode.max_adverse_move_pct = min(episode.max_adverse_move_pct, move)
        for target in episode.target_moves_pct:
            hit = episode.target_hits[_key(target)]
            if not hit["hit"] and move >= target:
                hit["hit"] = True
                hit["first_hit_seconds"] = elapsed
        for stop in episode.stop_moves_pct:
            hit = episode.stop_hits[_key(stop)]
            if not hit["hit"] and move <= -abs(stop):
                hit["hit"] = True
                hit["first_hit_seconds"] = elapsed
        for horizon in episode.horizons_seconds:
            row = episode.horizons[str(horizon)]
            if row or elapsed < horizon:
                continue
            row.update(
                {
                    "elapsed_seconds": elapsed,
                    "move_pct": move,
                    "direction_correct": move > 0,
                    "max_favorable_move_pct": episode.max_favorable_move_pct,
                    "max_adverse_move_pct": episode.max_adverse_move_pct,
                    "targets_hit": [key for key, value in episode.target_hits.items() if value["hit"]],
                    "stops_hit": [key for key, value in episode.stop_hits.items() if value["hit"]],
                }
            )

    def _is_complete(self, episode: RegimeOutcomeEpisode, current: datetime) -> bool:
        elapsed = (current - episode.opened_at).total_seconds()
        return elapsed >= max(episode.horizons_seconds)

    def _close_event(self, episode: RegimeOutcomeEpisode, current: datetime, reason: str) -> dict:
        self.closed_count += 1
        row = self._event("close", episode)
        row.update(
            {
                "closed_at": current.isoformat(),
                "close_reason": reason,
                "time_in_episode_seconds": (current - episode.opened_at).total_seconds(),
                "last_price": episode.last_price,
                "max_favorable_move_pct": episode.max_favorable_move_pct,
                "max_adverse_move_pct": episode.max_adverse_move_pct,
                "target_hits": episode.target_hits,
                "stop_hits": episode.stop_hits,
                "horizons": episode.horizons,
                "target_before_stop": self._target_before_stop(episode),
                "early_follow_through": self._early_follow_through(episode),
            }
        )
        return row

    def _event(self, event: str, episode: RegimeOutcomeEpisode) -> dict:
        return {
            "event": event,
            "episode_id": episode.episode_id,
            "symbol": episode.symbol,
            "side": episode.side,
            "state": episode.state,
            "source": episode.source,
            "path": episode.path,
            "entry_price": episode.entry_price,
            "opened_at": episode.opened_at.isoformat(),
            "confidence": episode.confidence,
            "strategy_score": episode.strategy_score,
            "quality_score": episode.quality_score,
            "quality_components": episode.quality_components,
            "adaptive_allowed": episode.adaptive_allowed,
            "target_feasible": episode.target_feasible,
            "blockers": episode.blockers,
            "target_moves_pct": episode.target_moves_pct,
            "stop_moves_pct": episode.stop_moves_pct,
            "horizons_seconds": episode.horizons_seconds,
        }

    def _target_before_stop(self, episode: RegimeOutcomeEpisode) -> dict[str, dict[str, bool | None]]:
        matrix: dict[str, dict[str, bool | None]] = {}
        for target_key, target in episode.target_hits.items():
            target_time = target["first_hit_seconds"]
            matrix[target_key] = {}
            for stop_key, stop in episode.stop_hits.items():
                stop_time = stop["first_hit_seconds"]
                if target_time is None:
                    matrix[target_key][stop_key] = False
                elif stop_time is None:
                    matrix[target_key][stop_key] = True
                else:
                    matrix[target_key][stop_key] = target_time <= stop_time
        return matrix

    def _early_follow_through(self, episode: RegimeOutcomeEpisode) -> dict:
        horizon_rows = [episode.horizons.get(str(horizon)) or {} for horizon in self.early_horizons_seconds]
        direction_correct = sum(1 for row in horizon_rows if row.get("direction_correct") is True)
        best_target_key = _key(min(self.target_moves_pct)) if self.target_moves_pct else None
        first_stop_key = _key(min(self.stop_moves_pct)) if self.stop_moves_pct else None
        target_hit = bool(best_target_key and episode.target_hits.get(best_target_key, {}).get("hit"))
        target_time = episode.target_hits.get(best_target_key, {}).get("first_hit_seconds") if best_target_key else None
        stop_time = episode.stop_hits.get(first_stop_key, {}).get("first_hit_seconds") if first_stop_key else None
        target_before_stop = target_hit and (stop_time is None or (target_time is not None and target_time <= stop_time))
        return {
            "horizons_seconds": self.early_horizons_seconds,
            "direction_correct_count": direction_correct,
            "required_correct_count": self.settings.regime_outcome_min_early_correct,
            "target_key": best_target_key,
            "stop_key": first_stop_key,
            "target_before_stop": target_before_stop,
            "qualified": (
                direction_correct >= self.settings.regime_outcome_min_early_correct
                and target_before_stop
            ),
        }

    def _quality_components(self, side: str, stateful: dict, market: MarketState) -> dict[str, float]:
        score = float(stateful.get("score") or 0.0)
        sequence = float(stateful.get("sequence_confidence") or 0.0)
        range_pct = float(market.range_180s_pct or 0.0)
        range_component = min(1.0, range_pct / max(0.000001, self.settings.fast_target_move_pct))
        flow = market.taker_buy_ratio_30s if market.taker_buy_ratio_30s is not None else market.taker_buy_ratio_10s
        flow = flow if flow is not None else 0.5
        flow_component = flow if side == "long" else 1.0 - flow
        book = market.book_imbalance_top or 0.0
        depth = market.depth_imbalance_top5 if market.depth_imbalance_top5 is not None else book
        pressure = (book + depth) / 2.0
        pressure_component = (pressure + 1.0) / 2.0 if side == "long" else (-pressure + 1.0) / 2.0
        trend = market.return_180s_pct or 0.0
        trend_component = _clamp((trend if side == "long" else -trend) / max(0.000001, self.settings.stateful_adaptive_target_move_pct))
        impulse = market.return_60s_pct or 0.0
        impulse_component = _clamp((impulse if side == "long" else -impulse) / max(0.000001, self.settings.stateful_adaptive_target_move_pct * 0.60))
        oi = market.open_interest_change_5m_pct or 0.0
        oi_component = min(1.0, max(0.0, (oi + 0.001) / 0.003))
        stale_penalty = 1.0 if (market.data_age_seconds or 0.0) <= self.settings.stale_after_seconds else 0.0
        return {
            "strategy_score": _clamp(score),
            "sequence_confidence": _clamp(sequence),
            "range_expansion": _clamp(range_component),
            "flow_alignment": _clamp(flow_component),
            "pressure_alignment": _clamp(pressure_component),
            "trend_alignment": trend_component,
            "impulse_alignment": impulse_component,
            "open_interest_context": _clamp(oi_component),
            "fresh_data": stale_penalty,
        }

    def _quality_score(self, components: dict[str, float]) -> float:
        score = (
            0.18 * components["strategy_score"]
            + 0.18 * components["sequence_confidence"]
            + 0.12 * components["range_expansion"]
            + 0.14 * components["flow_alignment"]
            + 0.10 * components["pressure_alignment"]
            + 0.10 * components["trend_alignment"]
            + 0.05 * components["impulse_alignment"]
            + 0.05 * components["open_interest_context"]
            + 0.08 * components["fresh_data"]
        )
        return round(_clamp(score), 4)


def _parse_floats(raw: str) -> list[float]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    return values


def _parse_ints(raw: str) -> list[int]:
    return [int(value) for value in _parse_floats(raw)]


def _underlying_move_pct(side: str, entry: float, price: float) -> float:
    direction = 1 if side == "long" else -1
    return ((price - entry) / entry) * direction


def _key(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
