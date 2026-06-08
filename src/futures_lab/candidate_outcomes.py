from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from futures_lab.config import Settings
from futures_lab.models import Decision, MarketState, utc_now


@dataclass
class CandidateOutcomeEpisode:
    candidate_id: str
    symbol: str
    strategy: str
    family: str
    side: str
    selected: bool
    viable: bool
    entry_price: float
    opened_at: datetime
    score: float
    expected_ev_bps: float
    expected_cost_bps: float
    target_bps: float
    stop_bps: float
    max_hold_ms: int
    blockers: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    max_favorable_move_pct: float = 0.0
    max_adverse_move_pct: float = 0.0
    last_price: float | None = None
    last_seen_at: datetime | None = None
    first_target_seconds: float | None = None
    first_cost_adjusted_target_seconds: float | None = None
    first_stop_seconds: float | None = None
    first_soft_invalidation_seconds: float | None = None
    horizons: dict[str, dict] = field(default_factory=dict)


@dataclass
class CandidateOutcomeTracker:
    settings: Settings
    episodes: list[CandidateOutcomeEpisode] = field(default_factory=list)
    opened_count: int = 0
    closed_count: int = 0
    last_opened_at_by_key: dict[tuple[str, str, str], datetime] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.horizons_seconds = sorted(set(_parse_ints(self.settings.candidate_outcome_horizons_seconds)))
        if not self.horizons_seconds:
            self.horizons_seconds = [1, 3, 5, 10, 30, 60]

    def open_from_decision(
        self,
        decision: Decision,
        market: MarketState,
        opened_at: datetime | None = None,
    ) -> list[dict]:
        if not self.settings.write_candidate_outcome_log:
            return []
        edge_router = decision.evidence.get("edge_router") or {}
        candidates = edge_router.get("candidates") or []
        if not candidates or market.mid_price is None:
            return []
        lag_ms = market.book_freshness_lag_ms if market.book_freshness_lag_ms is not None else market.hot_freshness_lag_ms
        if lag_ms is not None and lag_ms > self.settings.max_exchange_event_lag_ms:
            return []
        current = opened_at or market.last_received_at or decision.timestamp
        selected_strategy = (edge_router.get("selected") or {}).get("strategy")
        opened = []
        for candidate in candidates:
            strategy = str(candidate.get("strategy") or "")
            side = str(candidate.get("side") or "")
            if side not in {"long", "short"} or not strategy:
                continue
            selected = strategy == selected_strategy
            viable = bool(candidate.get("viable"))
            accepted = selected and viable
            key = (strategy, side, "accepted" if accepted else "rejected")
            last_opened = self.last_opened_at_by_key.get(key)
            if (
                last_opened is not None
                and (current - last_opened).total_seconds() < self.settings.candidate_outcome_cooldown_seconds
            ):
                continue
            self.opened_count += 1
            episode = CandidateOutcomeEpisode(
                candidate_id=f"candidate-{self.opened_count:06d}",
                symbol=decision.symbol,
                strategy=strategy,
                family=str(candidate.get("family") or ""),
                side=side,
                selected=selected,
                viable=viable,
                entry_price=market.mid_price,
                opened_at=current,
                score=float(candidate.get("score") or 0.0),
                expected_ev_bps=float(candidate.get("expected_ev_bps") or 0.0),
                expected_cost_bps=float(candidate.get("expected_cost_bps") or 0.0),
                target_bps=float(candidate.get("target_bps") or 0.0),
                stop_bps=float(candidate.get("stop_bps") or 0.0),
                max_hold_ms=int(candidate.get("max_hold_ms") or 0),
                blockers=list(candidate.get("blockers") or []),
                reasons=list(candidate.get("reasons") or []),
                horizons={str(horizon): {} for horizon in self.horizons_seconds},
            )
            self._update_episode(episode, market, current)
            self.episodes.append(episode)
            self.last_opened_at_by_key[key] = current
            opened.append(self._event("open", episode))
        return opened

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

    def _update_episode(self, episode: CandidateOutcomeEpisode, market: MarketState, current: datetime) -> None:
        if market.mid_price is None:
            return
        move = _underlying_move_pct(episode.side, episode.entry_price, market.mid_price)
        elapsed = max(0.0, (current - episode.opened_at).total_seconds())
        episode.last_price = market.mid_price
        episode.last_seen_at = current
        episode.max_favorable_move_pct = max(episode.max_favorable_move_pct, move)
        episode.max_adverse_move_pct = min(episode.max_adverse_move_pct, move)
        if episode.first_target_seconds is None and move >= _bps_to_pct(episode.target_bps):
            episode.first_target_seconds = elapsed
        if episode.first_cost_adjusted_target_seconds is None and move >= _bps_to_pct(episode.target_bps + episode.expected_cost_bps):
            episode.first_cost_adjusted_target_seconds = elapsed
        if episode.first_stop_seconds is None and move <= -abs(_bps_to_pct(episode.stop_bps)):
            episode.first_stop_seconds = elapsed
        if episode.first_soft_invalidation_seconds is None and _soft_invalidated(episode.side, market):
            episode.first_soft_invalidation_seconds = elapsed
        for horizon in self.horizons_seconds:
            row = episode.horizons[str(horizon)]
            if row or elapsed < horizon:
                continue
            row.update(
                {
                    "elapsed_seconds": elapsed,
                    "move_pct": move,
                    "mfe_pct": episode.max_favorable_move_pct,
                    "mae_pct": episode.max_adverse_move_pct,
                    "direction_correct": move > 0,
                }
            )

    def _is_complete(self, episode: CandidateOutcomeEpisode, current: datetime) -> bool:
        elapsed = (current - episode.opened_at).total_seconds()
        return elapsed >= max(self.horizons_seconds)

    def _close_event(self, episode: CandidateOutcomeEpisode, current: datetime, reason: str) -> dict:
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
                "mfe_mae_horizons": episode.horizons,
                "target_hits": {
                    "gross": {"hit": episode.first_target_seconds is not None, "first_hit_seconds": episode.first_target_seconds},
                    "cost_adjusted": {
                        "hit": episode.first_cost_adjusted_target_seconds is not None,
                        "first_hit_seconds": episode.first_cost_adjusted_target_seconds,
                    },
                },
                "stop_hits": {
                    "gross": {"hit": episode.first_stop_seconds is not None, "first_hit_seconds": episode.first_stop_seconds}
                },
                "target_before_stop": self._target_before_stop(episode),
                "outcome_label": self._outcome_label(episode),
            }
        )
        return row

    def _event(self, event: str, episode: CandidateOutcomeEpisode) -> dict:
        return {
            "event": event,
            "candidate_id": episode.candidate_id,
            "symbol": episode.symbol,
            "strategy": episode.strategy,
            "family": episode.family,
            "side": episode.side,
            "selected": episode.selected,
            "viable": episode.viable,
            "accepted": episode.selected and episode.viable,
            "entry_price": episode.entry_price,
            "opened_at": episode.opened_at.isoformat(),
            "score": episode.score,
            "expected_ev_bps": episode.expected_ev_bps,
            "expected_cost_bps": episode.expected_cost_bps,
            "target_bps": episode.target_bps,
            "cost_adjusted_target_bps": episode.target_bps + episode.expected_cost_bps,
            "stop_bps": episode.stop_bps,
            "max_hold_ms": episode.max_hold_ms,
            "blockers": episode.blockers,
            "reasons": episode.reasons,
            "horizons_seconds": self.horizons_seconds,
        }

    def _target_before_stop(self, episode: CandidateOutcomeEpisode) -> dict[str, dict[str, bool]]:
        stop_time = episode.first_stop_seconds
        return {
            "gross": {"gross": _before(episode.first_target_seconds, stop_time)},
            "cost_adjusted": {"gross": _before(episode.first_cost_adjusted_target_seconds, stop_time)},
        }

    def _outcome_label(self, episode: CandidateOutcomeEpisode) -> str:
        events = []
        if episode.first_cost_adjusted_target_seconds is not None:
            events.append(("target_first", episode.first_cost_adjusted_target_seconds))
        elif episode.first_target_seconds is not None:
            events.append(("target_first", episode.first_target_seconds))
        if episode.first_stop_seconds is not None:
            events.append(("stop_first", episode.first_stop_seconds))
        if episode.first_soft_invalidation_seconds is not None:
            events.append(("soft_invalidation_first", episode.first_soft_invalidation_seconds))
        if not events:
            return "timeout"
        return min(events, key=lambda item: item[1])[0]


def _soft_invalidated(side: str, market: MarketState) -> bool:
    ofi = market.order_flow_imbalance_1s
    micro = market.microprice_mid_bps
    if side == "long":
        return (ofi is not None and ofi < -0.15) or (micro is not None and micro < 0)
    return (ofi is not None and ofi > 0.15) or (micro is not None and micro > 0)


def _before(target_time: float | None, stop_time: float | None) -> bool:
    if target_time is None:
        return False
    if stop_time is None:
        return True
    return target_time <= stop_time


def _underlying_move_pct(side: str, entry: float, price: float) -> float:
    direction = 1 if side == "long" else -1
    return ((price - entry) / entry) * direction


def _bps_to_pct(bps: float) -> float:
    return bps / 10_000


def _parse_ints(raw: str) -> list[int]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(float(part)))
    return values
