import gzip
import json
from datetime import datetime, timedelta, timezone

from futures_lab.config import Settings
from futures_lab.models import Decision, DecisionAction, RiskVerdict, TradeMode
import futures_lab.replay as replay_module
from futures_lab.replay import discover_raw_files, replay_files


def _write_row(handle, received_at: datetime, payload: dict) -> None:
    handle.write(
        json.dumps(
            {
                "received_at": received_at.isoformat(),
                "message": {"stream": "btcusdt@test", "data": payload},
            }
        )
    )
    handle.write("\n")


def test_replay_loads_recorded_jsonl_and_runs_decisions(tmp_path):
    raw_dir = tmp_path / "raw_ws"
    raw_dir.mkdir(parents=True)
    path = raw_dir / "BTCUSDT_fixture_2026-05-03.jsonl"
    start = datetime(2026, 5, 3, 8, 0, tzinfo=timezone.utc)
    with path.open("w", encoding="utf-8") as handle:
        for idx in range(20):
            ts = start + timedelta(seconds=idx)
            price = 100.0 + (idx * 0.01)
            _write_row(
                handle,
                ts,
                {
                    "e": "bookTicker",
                    "E": int(ts.timestamp() * 1000),
                    "b": str(price - 0.005),
                    "a": str(price + 0.005),
                    "B": "12",
                    "A": "8",
                },
            )
            _write_row(
                handle,
                ts + timedelta(milliseconds=10),
                {
                    "e": "aggTrade",
                    "E": int(ts.timestamp() * 1000),
                    "p": str(price),
                    "q": "1.0",
                    "m": False,
                },
            )

    settings = Settings(
        DATA_DIR=str(tmp_path),
        MIN_WARMUP_SECONDS=1,
        STALE_AFTER_SECONDS=999,
        STATE_WINDOW_SECONDS=60,
    )
    files = discover_raw_files(settings, pattern="BTCUSDT_fixture_*.jsonl")
    summary = replay_files(settings, files, decision_interval_ms=1000)

    assert files == [path]
    assert summary.messages == 40
    assert summary.decisions > 0
    assert summary.model_dump()["messages"] == 40
    assert "states" in summary.model_dump()["markov"]


def test_replay_discovers_and_loads_compressed_jsonl(tmp_path):
    raw_dir = tmp_path / "raw_ws"
    raw_dir.mkdir(parents=True)
    path = raw_dir / "BTCUSDT_fixture_2026-05-03.jsonl.gz"
    start = datetime(2026, 5, 3, 8, 0, tzinfo=timezone.utc)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        _write_row(
            handle,
            start,
            {
                "e": "bookTicker",
                "E": int(start.timestamp() * 1000),
                "b": "99.995",
                "a": "100.005",
                "B": "12",
                "A": "8",
            },
        )

    settings = Settings(DATA_DIR=str(tmp_path), SYMBOL="BTCUSDT")
    files = discover_raw_files(settings)
    summary = replay_files(settings, files)

    assert files == [path]
    assert summary.messages == 1


def test_replay_reports_open_position_and_can_flatten_at_end(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw_ws"
    raw_dir.mkdir(parents=True)
    path = raw_dir / "BTCUSDT_fixture_2026-05-03.jsonl"
    start = datetime(2026, 5, 3, 8, 0, tzinfo=timezone.utc)
    with path.open("w", encoding="utf-8") as handle:
        for idx, price in enumerate([100.0, 101.0]):
            ts = start + timedelta(seconds=idx)
            _write_row(
                handle,
                ts,
                {
                    "e": "bookTicker",
                    "E": int(ts.timestamp() * 1000),
                    "b": str(price - 0.005),
                    "a": str(price + 0.005),
                    "B": "12",
                    "A": "8",
                },
            )

    class FakeStrategy:
        def __init__(self, settings):
            self._proposed = False

        def decide(self, market):
            if self._proposed:
                return Decision(symbol="BTCUSDT", action=DecisionAction.wait, confidence=0.0, reason="test")
            self._proposed = True
            return Decision(
                symbol="BTCUSDT",
                action=DecisionAction.propose_long,
                mode=TradeMode.fast,
                confidence=0.9,
                reason="test",
                entry_price=100.0,
                take_profit_price=200.0,
                stop_loss_price=50.0,
                leverage=200,
            )

    class FakeRisk:
        def __init__(self, settings):
            pass

        def evaluate(self, decision, market, paper_state):
            return RiskVerdict(allowed=decision.action == DecisionAction.propose_long, reason="test")

    monkeypatch.setattr(replay_module, "HitAndRunStrategy", FakeStrategy)
    monkeypatch.setattr(replay_module, "RiskEngine", FakeRisk)
    settings = Settings(DATA_DIR=str(tmp_path), TAKER_FEE_BPS=4.0)

    open_summary = replay_files(settings, [path], decision_interval_ms=100)
    assert open_summary.paper_opens == 1
    assert open_summary.paper_closes == 0
    assert open_summary.open_position is not None
    assert open_summary.open_position.max_favorable_move_pct > 0
    assert open_summary.net_unrealized_pnl_usd > 0

    flat_summary = replay_files(settings, [path], decision_interval_ms=100, flatten_at_end=True)
    assert flat_summary.paper_opens == 1
    assert flat_summary.paper_closes == 1
    assert flat_summary.open_position is None
    assert flat_summary.trades[0].exit_reason == "session_end"
    assert flat_summary.position_stats[0].status == "flattened"


def test_hostile_replay_pessimizes_taker_entry_and_exit(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw_ws"
    raw_dir.mkdir(parents=True)
    path = raw_dir / "BTCUSDT_fixture_2026-05-03.jsonl"
    start = datetime(2026, 5, 3, 8, 0, tzinfo=timezone.utc)
    with path.open("w", encoding="utf-8") as handle:
        for idx, price in enumerate([100.0, 101.0]):
            ts = start + timedelta(seconds=idx)
            _write_row(
                handle,
                ts,
                {
                    "e": "bookTicker",
                    "E": int(ts.timestamp() * 1000),
                    "b": str(price - 0.01),
                    "a": str(price + 0.01),
                    "B": "12",
                    "A": "8",
                },
            )

    class FakeStrategy:
        def __init__(self, settings):
            self._proposed = False

        def decide(self, market):
            if self._proposed:
                return Decision(symbol="BTCUSDT", action=DecisionAction.wait, confidence=0.0, reason="test")
            self._proposed = True
            return Decision(
                symbol="BTCUSDT",
                action=DecisionAction.propose_long,
                mode=TradeMode.fast,
                confidence=0.9,
                reason="test",
                entry_price=100.0,
                take_profit_price=100.2,
                stop_loss_price=99.5,
                target_move_pct=0.002,
                stop_move_pct=0.005,
                leverage=200,
            )

    class FakeRisk:
        def __init__(self, settings):
            pass

        def evaluate(self, decision, market, paper_state):
            return RiskVerdict(allowed=decision.action == DecisionAction.propose_long, reason="test")

    monkeypatch.setattr(replay_module, "HitAndRunStrategy", FakeStrategy)
    monkeypatch.setattr(replay_module, "RiskEngine", FakeRisk)
    settings = Settings(
        DATA_DIR=str(tmp_path),
        TAKER_FEE_BPS=0.0,
        HOSTILE_REPLAY_ENTRY_SLIPPAGE_BPS=10.0,
        HOSTILE_REPLAY_EXIT_SLIPPAGE_BPS=10.0,
        HOSTILE_REPLAY_LATENCY_PENALTY_BPS=0.0,
        HOSTILE_REPLAY_LATENCY_BPS_PER_SECOND=0.0,
    )

    normal = replay_files(settings, [path], decision_interval_ms=100, flatten_at_end=True)
    hostile = replay_files(settings, [path], decision_interval_ms=100, flatten_at_end=True, hostile=True)

    assert normal.replay_mode == "normal"
    assert hostile.replay_mode == "hostile"
    assert hostile.hostile_replay is not None
    assert hostile.hostile_replay.entry_fills == 1
    assert hostile.hostile_replay.exit_fills == 1
    assert hostile.trades[0].entry_price > normal.trades[0].entry_price
    assert hostile.trades[0].exit_price < normal.trades[0].exit_price
    assert hostile.trades[0].net_pnl_usd < normal.trades[0].net_pnl_usd


def test_hostile_replay_rejects_stale_book_entries(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw_ws"
    raw_dir.mkdir(parents=True)
    path = raw_dir / "BTCUSDT_fixture_2026-05-03.jsonl"
    received_at = datetime(2026, 5, 3, 8, 0, tzinfo=timezone.utc)
    stale_event_at = received_at - timedelta(seconds=5)
    with path.open("w", encoding="utf-8") as handle:
        _write_row(
            handle,
            received_at,
            {
                "e": "bookTicker",
                "E": int(stale_event_at.timestamp() * 1000),
                "b": "99.99",
                "a": "100.01",
                "B": "12",
                "A": "8",
            },
        )

    class FakeStrategy:
        def __init__(self, settings):
            pass

        def decide(self, market):
            return Decision(
                symbol="BTCUSDT",
                action=DecisionAction.propose_long,
                mode=TradeMode.fast,
                confidence=0.9,
                reason="test",
                entry_price=100.0,
                take_profit_price=100.2,
                stop_loss_price=99.5,
                target_move_pct=0.002,
                stop_move_pct=0.005,
                leverage=200,
            )

    class FakeRisk:
        def __init__(self, settings):
            pass

        def evaluate(self, decision, market, paper_state):
            return RiskVerdict(allowed=True, reason="test")

    monkeypatch.setattr(replay_module, "HitAndRunStrategy", FakeStrategy)
    monkeypatch.setattr(replay_module, "RiskEngine", FakeRisk)
    settings = Settings(DATA_DIR=str(tmp_path), HOSTILE_REPLAY_MAX_BOOK_AGE_MS=500)

    summary = replay_files(settings, [path], decision_interval_ms=100, hostile=True)

    assert summary.risk_allowed == 1
    assert summary.paper_opens == 0
    assert summary.hostile_replay is not None
    assert summary.hostile_replay.stale_book_rejections == 1


def test_hostile_replay_penalizes_stop_execution(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw_ws"
    raw_dir.mkdir(parents=True)
    path = raw_dir / "BTCUSDT_fixture_2026-05-03.jsonl"
    start = datetime(2026, 5, 3, 8, 0, tzinfo=timezone.utc)
    with path.open("w", encoding="utf-8") as handle:
        for idx, price in enumerate([100.0, 99.4]):
            ts = start + timedelta(seconds=idx)
            _write_row(
                handle,
                ts,
                {
                    "e": "bookTicker",
                    "E": int(ts.timestamp() * 1000),
                    "b": str(price - 0.01),
                    "a": str(price + 0.01),
                    "B": "12",
                    "A": "8",
                },
            )

    class FakeStrategy:
        def __init__(self, settings):
            self._proposed = False

        def decide(self, market):
            if self._proposed:
                return Decision(symbol="BTCUSDT", action=DecisionAction.wait, confidence=0.0, reason="test")
            self._proposed = True
            return Decision(
                symbol="BTCUSDT",
                action=DecisionAction.propose_long,
                mode=TradeMode.fast,
                confidence=0.9,
                reason="test",
                entry_price=100.0,
                take_profit_price=101.0,
                stop_loss_price=99.5,
                target_move_pct=0.01,
                stop_move_pct=0.005,
                leverage=200,
            )

    class FakeRisk:
        def __init__(self, settings):
            pass

        def evaluate(self, decision, market, paper_state):
            return RiskVerdict(allowed=decision.action == DecisionAction.propose_long, reason="test")

    monkeypatch.setattr(replay_module, "HitAndRunStrategy", FakeStrategy)
    monkeypatch.setattr(replay_module, "RiskEngine", FakeRisk)
    settings = Settings(
        DATA_DIR=str(tmp_path),
        TAKER_FEE_BPS=0.0,
        HOSTILE_REPLAY_ENTRY_SLIPPAGE_BPS=0.0,
        HOSTILE_REPLAY_EXIT_SLIPPAGE_BPS=0.0,
        HOSTILE_REPLAY_STOP_PENALTY_BPS=10.0,
        HOSTILE_REPLAY_LATENCY_PENALTY_BPS=0.0,
        HOSTILE_REPLAY_LATENCY_BPS_PER_SECOND=0.0,
    )

    normal = replay_files(settings, [path], decision_interval_ms=100)
    hostile = replay_files(settings, [path], decision_interval_ms=100, hostile=True)

    assert normal.trades[0].exit_reason == "stop_loss"
    assert hostile.trades[0].exit_reason == "stop_loss"
    assert hostile.trades[0].exit_price < normal.trades[0].exit_price
    assert hostile.hostile_replay is not None
    assert hostile.hostile_replay.stop_penalties_usd > 0


def test_hostile_replay_approximates_maker_queue_rejection(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw_ws"
    raw_dir.mkdir(parents=True)
    path = raw_dir / "BTCUSDT_fixture_2026-05-03.jsonl"
    start = datetime(2026, 5, 3, 8, 0, tzinfo=timezone.utc)
    with path.open("w", encoding="utf-8") as handle:
        _write_row(
            handle,
            start,
            {
                "e": "bookTicker",
                "E": int(start.timestamp() * 1000),
                "b": "99.99",
                "a": "100.01",
                "B": "12",
                "A": "8",
            },
        )

    class FakeStrategy:
        def __init__(self, settings):
            pass

        def decide(self, market):
            return Decision(
                symbol="BTCUSDT",
                action=DecisionAction.propose_long,
                mode=TradeMode.fast,
                confidence=0.5,
                reason="test",
                entry_price=100.0,
                take_profit_price=100.2,
                stop_loss_price=99.5,
                target_move_pct=0.002,
                stop_move_pct=0.005,
                leverage=200,
            )

    class FakeRisk:
        def __init__(self, settings):
            pass

        def evaluate(self, decision, market, paper_state):
            return RiskVerdict(allowed=True, reason="test")

    monkeypatch.setattr(replay_module, "HitAndRunStrategy", FakeStrategy)
    monkeypatch.setattr(replay_module, "RiskEngine", FakeRisk)
    settings = Settings(
        DATA_DIR=str(tmp_path),
        DEFAULT_ENTRY_ORDER_TYPE="maker",
        HOSTILE_REPLAY_MAKER_QUEUE_FILL_PROBABILITY=0.1,
    )

    summary = replay_files(settings, [path], decision_interval_ms=100, hostile=True)

    assert summary.paper_opens == 0
    assert summary.hostile_replay is not None
    assert summary.hostile_replay.maker_queue_rejections == 1

