import json
from datetime import datetime, timedelta, timezone

from futures_lab.config import Settings
from futures_lab.models import MarketState
from futures_lab.shadow_arena import ShadowArena, ShadowArenaConfig, summarize_shadow_arena


class NoopRecorder:
    def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


class FixedState:
    def __init__(self, market: MarketState) -> None:
        self.market = market

    def snapshot(self, current=None) -> MarketState:
        return self.market.model_copy(update={"last_received_at": current or self.market.last_received_at})


def test_symmetric_long_and_short_barriers_include_costs(tmp_path) -> None:
    settings = Settings(DATA_DIR=tmp_path, SYMBOL="ETHUSDT")
    market = _market(mark=100.0)
    arena = ShadowArena(
        settings,
        ShadowArenaConfig(target_bps=60, stop_bps=60, cost_bps=10, warmup_seconds=0),
        state=FixedState(market),
        recorder=NoopRecorder(),
    )
    timestamp = market.last_received_at
    assert timestamp is not None

    long_arm = arena.arms["raw_micro_momentum"]
    assert arena._try_signal_arm(long_arm, 1.0, 100.0, timestamp, trigger="test")
    arena._mark(long_arm, 100.6, timestamp + timedelta(seconds=1))
    assert long_arm.targets == 1
    assert round(long_arm.cumulative_net_bps, 8) == 50.0

    short_arm = arena.arms["clock_coin_control"]
    assert arena._try_signal_arm(short_arm, -1.0, 100.0, timestamp, trigger="test")
    arena._mark(short_arm, 100.6, timestamp + timedelta(seconds=1))
    assert short_arm.stops == 1
    assert round(short_arm.cumulative_net_bps, 8) == -70.0
    assert arena.break_even_win_rate == 70 / 120


def test_arms_are_independent_and_coin_is_reproducible(tmp_path) -> None:
    settings = Settings(DATA_DIR=tmp_path, SYMBOL="ETHUSDT")
    market = _market(mark=100.0)
    arena = ShadowArena(
        settings,
        ShadowArenaConfig(warmup_seconds=0, cooldown_seconds=60),
        state=FixedState(market),
        recorder=NoopRecorder(),
    )
    timestamp = market.last_received_at
    assert timestamp is not None
    assert arena._coin_side(timestamp, "clock") == arena._coin_side(timestamp, "clock")

    assert arena._try_signal_arm(arena.arms["squeeze_breakout"], 1.0, 100.0, timestamp, trigger="one")
    assert arena._try_signal_arm(arena.arms["raw_micro_momentum"], -1.0, 100.0, timestamp, trigger="two")
    assert arena.arms["squeeze_breakout"].position is not None
    assert arena.arms["raw_micro_momentum"].position is not None
    assert arena.arms["squeeze_breakout"].position.side == 1
    assert arena.arms["raw_micro_momentum"].position.side == -1


def test_timeout_and_summary_reconstruct_events(tmp_path) -> None:
    settings = Settings(DATA_DIR=tmp_path, SYMBOL="ETHUSDT")
    market = _market(mark=100.0)
    arena = ShadowArena(
        settings,
        ShadowArenaConfig(horizon_seconds=10, warmup_seconds=0, cooldown_seconds=0),
        state=FixedState(market),
        recorder=NoopRecorder(),
    )
    timestamp = market.last_received_at
    assert timestamp is not None
    arm = arena.arms["raw_micro_momentum"]
    arena._try_signal_arm(arm, 1.0, 100.0, timestamp, trigger="test")
    arena._mark(arm, 100.1, timestamp + timedelta(seconds=10))

    summary = summarize_shadow_arena(settings)
    assert arm.timeouts == 1
    assert summary["arms"]["raw_micro_momentum"]["timeouts"] == 1
    assert summary["arms"]["raw_micro_momentum"]["closed"] == 1
    rows = [json.loads(line) for line in arena.events.path.read_text(encoding="utf-8").splitlines()]
    assert [row["event"] for row in rows] == ["arena_open", "arena_close"]


def _market(*, mark: float) -> MarketState:
    timestamp = datetime(2026, 7, 11, tzinfo=timezone.utc)
    return MarketState(
        symbol="ETHUSDT",
        connected=True,
        last_received_at=timestamp,
        data_age_seconds=0.0,
        observed_seconds=300.0,
        mark_price=mark,
        mid_price=mark,
    )
