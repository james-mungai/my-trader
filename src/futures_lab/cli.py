import argparse
import asyncio
import json
import signal
from datetime import datetime, timedelta, timezone
from pathlib import Path

from futures_lab.audit import AuditLog
from futures_lab.binance_private import BinancePrivateClient, BinancePrivateError
from futures_lab.config import Settings
from futures_lab.data_ops import (
    compress_raw,
    prune_raw,
    summarize_candidate_outcomes,
    summarize_data,
    summarize_exit_shadow,
    summarize_range_exit_counterfactuals,
    summarize_regime_outcomes,
)
from futures_lab.exit_laboratory import run_exit_laboratory
from futures_lab.feature_value import run_feature_value_analysis
from futures_lab.first_touch import run_first_touch_study
from futures_lab.latency_probe import build_probe_streams, run_latency_probe
from futures_lab.live_canary import run_live_canary
from futures_lab.market_atlas import run_market_atlas
from futures_lab.replay import discover_raw_files, replay_files
from futures_lab.readiness import evaluate_readiness
from futures_lab.regime_atlas import run_regime_atlas
from futures_lab.runtime import TradingRuntime
from futures_lab.shadow_arena import ShadowArena, ShadowArenaConfig, summarize_shadow_arena
from futures_lab.strategy_tournament import run_strategy_tournament


def _record_deadline(seconds: int, *, current: datetime | None = None) -> datetime:
    now = current or datetime.now(timezone.utc)
    return now + timedelta(seconds=max(0, seconds))


def _deadline_remaining_seconds(deadline: datetime, *, current: datetime | None = None) -> float:
    now = current or datetime.now(timezone.utc)
    return (deadline - now).total_seconds()


async def watch(seconds: int, quiet: bool = False) -> None:
    settings = Settings()
    runtime = TradingRuntime.create(settings)
    stop_requested = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_requested.set)
        except (NotImplementedError, RuntimeError):
            signal.signal(sig, lambda *_: loop.call_soon_threadsafe(stop_requested.set))
    runtime.start()
    try:
        deadline = _record_deadline(seconds)
        while (remaining := _deadline_remaining_seconds(deadline)) > 0 and not stop_requested.is_set():
            try:
                await asyncio.wait_for(stop_requested.wait(), timeout=min(1, remaining))
                break
            except TimeoutError:
                pass
            market, decision, risk = runtime.decide_once()
            if not quiet:
                print(
                    {
                        "symbol": market.symbol,
                        "mid": market.mid_price,
                        "spread_bps": market.spread_bps,
                        "regime": market.regime.value,
                        "decision": decision.action.value,
                        "confidence": decision.confidence,
                        "risk": risk.allowed,
                        "paper_pnl": runtime.paper_state().realized_pnl_usd,
                    }
                )
    finally:
        try:
            await asyncio.wait_for(asyncio.shield(runtime.stop()), timeout=settings.shutdown_timeout_seconds)
        except TimeoutError:
            print("Timed out while stopping runtime; event loop shutdown will cancel remaining tasks.")


async def shadow_arena(
    seconds: int,
    quiet: bool,
    target_bps: float,
    stop_bps: float,
    cost_bps: float,
    horizon_seconds: int,
    cooldown_seconds: int,
    warmup_seconds: int,
    max_trades_per_arm: int,
) -> None:
    settings = Settings()
    config = ShadowArenaConfig(
        target_bps=target_bps,
        stop_bps=stop_bps,
        cost_bps=cost_bps,
        horizon_seconds=horizon_seconds,
        cooldown_seconds=cooldown_seconds,
        warmup_seconds=warmup_seconds,
        max_trades_per_arm=max_trades_per_arm,
    )
    arena = ShadowArena(settings, config)
    stop_requested = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_requested.set)
        except (NotImplementedError, RuntimeError):
            signal.signal(sig, lambda *_: loop.call_soon_threadsafe(stop_requested.set))
    arena.start()
    try:
        deadline = _record_deadline(seconds)
        while (remaining := _deadline_remaining_seconds(deadline)) > 0 and not stop_requested.is_set():
            try:
                await asyncio.wait_for(stop_requested.wait(), timeout=min(1, remaining))
                break
            except TimeoutError:
                pass
            market = arena.tick()
            if not quiet:
                print(
                    {
                        "symbol": market.symbol,
                        "mark": market.mark_price,
                        "data_age_seconds": market.data_age_seconds,
                        "arms": arena.summary()["arms"],
                    }
                )
            if arena.all_arms_complete():
                break
    finally:
        try:
            await asyncio.wait_for(asyncio.shield(arena.stop()), timeout=settings.shutdown_timeout_seconds)
        except TimeoutError:
            print("Timed out while stopping shadow arena; event loop shutdown will cancel remaining tasks.")
    print(json.dumps(arena.summary(), indent=2, default=str))


def replay(
    pattern: str | None,
    decision_interval_ms: int | None,
    include_depth: bool,
    book_ticker_min_interval_ms: int,
    flatten_at_end: bool,
    hostile: bool,
) -> None:
    settings = Settings()
    files = discover_raw_files(settings, pattern=pattern)
    if not files:
        raise SystemExit("No raw WebSocket JSONL files found. Run `futures-lab record` first.")
    summary = replay_files(
        settings=settings,
        paths=files,
        decision_interval_ms=decision_interval_ms,
        include_depth=include_depth,
        book_ticker_min_interval_ms=book_ticker_min_interval_ms,
        flatten_at_end=flatten_at_end,
        audit=AuditLog(settings),
        hostile=hostile,
    )
    print(json.dumps(summary.model_dump(), indent=2, default=str))


def replay_compare(
    pattern: str | None,
    variants: list[str],
    decision_interval_ms: int | None,
    include_depth: bool,
    book_ticker_min_interval_ms: int,
    flatten_at_end: bool,
    hostile: bool,
) -> None:
    rows = []
    for variant in variants:
        settings = Settings(STRATEGY_VARIANT=variant)
        files = discover_raw_files(settings, pattern=pattern)
        if not files:
            raise SystemExit("No raw WebSocket JSONL files found. Run `futures-lab record` first.")
        summary = replay_files(
            settings=settings,
            paths=files,
            decision_interval_ms=decision_interval_ms,
            include_depth=include_depth,
            book_ticker_min_interval_ms=book_ticker_min_interval_ms,
            flatten_at_end=flatten_at_end,
            hostile=hostile,
        )
        rows.append(
            {
                "variant": variant,
                "replay_mode": summary.replay_mode,
                "files": len(files),
                "messages": summary.messages,
                "decisions": summary.decisions,
                "proposals": summary.proposals,
                "risk_allowed": summary.risk_allowed,
                "paper_opens": summary.paper_opens,
                "paper_closes": summary.paper_closes,
                "net_pnl_usd": summary.net_pnl_usd,
                "wins": summary.wins,
                "losses": summary.losses,
                "win_rate": summary.win_rate,
                "trades": [trade.model_dump() for trade in summary.trades],
                "position_stats": [stats.model_dump() for stats in summary.position_stats],
                "open_position": summary.open_position.model_dump() if summary.open_position else None,
                "net_unrealized_pnl_usd": summary.net_unrealized_pnl_usd,
                "markov": summary.markov,
                "hostile_replay": summary.hostile_replay.model_dump() if summary.hostile_replay else None,
            }
        )
    print(json.dumps(rows, indent=2, default=str))


def readiness_report(
    pattern: str | None,
    decision_interval_ms: int | None,
    include_depth: bool,
    book_ticker_min_interval_ms: int,
    flatten_at_end: bool,
) -> None:
    settings = Settings()
    try:
        report = evaluate_readiness(
            settings=settings,
            pattern=pattern,
            decision_interval_ms=decision_interval_ms,
            include_depth=include_depth,
            book_ticker_min_interval_ms=book_ticker_min_interval_ms,
            flatten_at_end=flatten_at_end,
        )
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(report.model_dump(), indent=2, default=str))


def binance_account_probe() -> None:
    settings = Settings()
    try:
        print(json.dumps(BinancePrivateClient(settings).account_probe(), indent=2, default=str))
    except BinancePrivateError as exc:
        raise SystemExit(str(exc)) from exc


def binance_live_preflight(symbol: str | None) -> None:
    settings = Settings()
    try:
        print(json.dumps(BinancePrivateClient(settings).live_preflight(symbol=symbol), indent=2, default=str))
    except BinancePrivateError as exc:
        raise SystemExit(str(exc)) from exc


def binance_order_test(side: str, symbol: str | None, notional_usd: float | None) -> None:
    settings = Settings()
    try:
        result = BinancePrivateClient(settings).market_order_test_probe(
            side,
            symbol=symbol,
            target_notional_usd=notional_usd,
        )
    except BinancePrivateError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, indent=2, default=str))


def binance_dust_round_trip(side: str, symbol: str | None, confirm_live_order: bool) -> None:
    if not confirm_live_order:
        raise SystemExit("Refusing to place a live order without --confirm-live-order.")
    settings = Settings()
    try:
        result = BinancePrivateClient(settings).live_dust_round_trip(side, symbol=symbol)
    except BinancePrivateError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, indent=2, default=str))


def binance_dust_open(side: str, symbol: str | None, confirm_live_order: bool) -> None:
    if not confirm_live_order:
        raise SystemExit("Refusing to place a live order without --confirm-live-order.")
    settings = Settings()
    try:
        result = BinancePrivateClient(settings).live_dust_open(side, symbol=symbol)
    except BinancePrivateError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, indent=2, default=str))


def binance_flatten_position(symbol: str | None, confirm_live_order: bool) -> None:
    if not confirm_live_order:
        raise SystemExit("Refusing to place a live order without --confirm-live-order.")
    settings = Settings()
    try:
        result = BinancePrivateClient(settings).flatten_position(symbol=symbol)
    except BinancePrivateError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, indent=2, default=str))


def binance_treasury_rebalance(symbol: str | None, confirm_live_transfer: bool) -> None:
    if not confirm_live_transfer:
        raise SystemExit("Refusing to transfer funds without --confirm-live-transfer.")
    settings = Settings(SYMBOL=symbol) if symbol else Settings()
    try:
        result = BinancePrivateClient(settings).live_treasury_rebalance(symbol=symbol, reason="cli")
    except BinancePrivateError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, indent=2, default=str))


async def binance_live_canary(seconds: int, max_trades: int, symbol: str | None, confirm_live_order: bool) -> None:
    if not confirm_live_order:
        raise SystemExit("Refusing to run live canary without --confirm-live-order.")
    settings = Settings(SYMBOL=symbol) if symbol else Settings()
    stop_requested = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_requested.set)
        except (NotImplementedError, RuntimeError):
            signal.signal(sig, lambda *_: loop.call_soon_threadsafe(stop_requested.set))
    try:
        result = await run_live_canary(
            settings,
            seconds=seconds,
            max_trades=max_trades,
            stop_requested=stop_requested,
        )
    except BinancePrivateError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, indent=2, default=str))


async def latency_probe(
    seconds: int,
    profile: str,
    symbol: str | None,
    anchor_symbol: str | None,
    depth_levels: int | None,
    no_write_log: bool,
) -> None:
    settings = Settings()
    summary = await run_latency_probe(
        settings,
        seconds=seconds,
        profile=profile,
        symbol=symbol,
        anchor_symbol=anchor_symbol,
        depth_levels=depth_levels,
        write_log=not no_write_log,
    )
    print(json.dumps(summary, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description="Futures Lab CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    watch_parser = sub.add_parser("watch", help="Watch live public Binance data and decisions.")
    watch_parser.add_argument("--seconds", type=int, default=30)
    watch_parser.add_argument("--quiet", action="store_true", help="Record without printing every second.")

    record_parser = sub.add_parser("record", help="Record public Binance data and run paper decisions.")
    record_parser.add_argument("--seconds", type=int, default=1800)
    record_parser.add_argument("--quiet", action="store_true")

    arena_parser = sub.add_parser(
        "shadow-arena",
        help="Run independent squeeze, micro-momentum, and fair-coin paper ledgers on one public feed.",
    )
    arena_parser.add_argument("--seconds", type=int, default=172_800)
    arena_parser.add_argument("--quiet", action="store_true")
    arena_parser.add_argument("--target-bps", type=float, default=60.0)
    arena_parser.add_argument("--stop-bps", type=float, default=60.0)
    arena_parser.add_argument("--cost-bps", type=float, default=10.0)
    arena_parser.add_argument("--horizon-seconds", type=int, default=21_600)
    arena_parser.add_argument("--cooldown-seconds", type=int, default=60)
    arena_parser.add_argument("--warmup-seconds", type=int, default=180)
    arena_parser.add_argument("--max-trades-per-arm", type=int, default=300)

    sub.add_parser("shadow-arena-summary", help="Summarize forward shadow-arena outcomes and checkpoints.")

    replay_parser = sub.add_parser("replay", help="Replay recorded raw WebSocket JSONL.")
    replay_parser.add_argument("--pattern", default=None, help="Glob under data/raw_ws, e.g. BTCUSDT_*_2026-05-03.jsonl")
    replay_parser.add_argument("--decision-interval-ms", type=int, default=None)
    replay_parser.add_argument("--include-depth", action="store_true", help="Include depthUpdate messages. Off by default because current strategy does not consume them.")
    replay_parser.add_argument("--book-ticker-min-interval-ms", type=int, default=100)
    replay_parser.add_argument("--flatten-at-end", action="store_true", help="Close any open paper position at the final replay price for session accounting.")
    replay_parser.add_argument("--hostile", action="store_true", help="Use pessimistic replay fills, latency penalties, and stale-book rejection.")

    compare_parser = sub.add_parser("replay-compare", help="Replay the same raw data across strategy variants.")
    compare_parser.add_argument("--pattern", default=None, help="Glob under data/raw_ws.")
    compare_parser.add_argument(
        "--variants",
        nargs="+",
        default=["baseline", "liquidity_sweep_reversal", "momentum_pullback", "stateful_momentum"],
    )
    compare_parser.add_argument("--decision-interval-ms", type=int, default=None)
    compare_parser.add_argument("--include-depth", action="store_true")
    compare_parser.add_argument("--book-ticker-min-interval-ms", type=int, default=100)
    compare_parser.add_argument("--flatten-at-end", action="store_true")
    compare_parser.add_argument("--hostile", action="store_true", help="Use hostile replay assumptions for every variant.")

    readiness_parser = sub.add_parser("readiness-report", help="Evaluate live-readiness gates using normal and hostile replay.")
    readiness_parser.add_argument("--pattern", default=None, help="Glob under data/raw_ws.")
    readiness_parser.add_argument("--decision-interval-ms", type=int, default=None)
    readiness_parser.add_argument("--include-depth", action="store_true")
    readiness_parser.add_argument("--book-ticker-min-interval-ms", type=int, default=100)
    readiness_parser.add_argument("--flatten-at-end", action="store_true")

    sub.add_parser(
        "binance-account-probe",
        help="Read-only signed Binance USD-M Futures account/auth probe. Does not place or modify orders.",
    )
    preflight_parser = sub.add_parser(
        "binance-live-preflight",
        help="Read-only Binance USD-M Futures preflight for live dust testing. Does not place or modify orders.",
    )
    preflight_parser.add_argument("--symbol", default=None)

    order_test_parser = sub.add_parser(
        "binance-order-test",
        help="Validate a Binance USD-M Futures MARKET order using /fapi/v1/order/test. Places nothing.",
    )
    order_test_parser.add_argument("--side", choices=["BUY", "SELL"], required=True)
    order_test_parser.add_argument("--symbol", default=None)
    order_test_parser.add_argument(
        "--notional-usd",
        type=float,
        default=None,
        help="Target notional before filter rounding. Must not exceed LIVE_MAX_NOTIONAL_USD.",
    )

    dust_parser = sub.add_parser(
        "binance-dust-round-trip",
        help="Place one tiny live Binance USD-M MARKET order, immediately close reduce-only, then verify flat.",
    )
    dust_parser.add_argument("--side", choices=["BUY", "SELL"], required=True)
    dust_parser.add_argument("--symbol", default=None)
    dust_parser.add_argument("--confirm-live-order", action="store_true", help="Required. This command places real orders.")

    dust_open_parser = sub.add_parser(
        "binance-dust-open",
        help="Place one tiny live Binance USD-M MARKET order and leave it open for a flatten drill.",
    )
    dust_open_parser.add_argument("--side", choices=["BUY", "SELL"], required=True)
    dust_open_parser.add_argument("--symbol", default=None)
    dust_open_parser.add_argument("--confirm-live-order", action="store_true", help="Required. This command places a real order.")

    flatten_parser = sub.add_parser(
        "binance-flatten-position",
        help="Flatten the current Binance USD-M position with one reduce-only MARKET order.",
    )
    flatten_parser.add_argument("--symbol", default=None)
    flatten_parser.add_argument("--confirm-live-order", action="store_true", help="Required. This command can place a real reduce-only order.")

    treasury_parser = sub.add_parser(
        "binance-treasury-rebalance",
        help="Flat-only USDT rebalance between Binance USD-M Futures and Funding wallet.",
    )
    treasury_parser.add_argument("--symbol", default=None)
    treasury_parser.add_argument(
        "--confirm-live-transfer",
        action="store_true",
        help="Required. This command can transfer USDT between Binance wallets.",
    )

    live_canary_parser = sub.add_parser(
        "binance-live-canary",
        help="Run a tightly capped live canary that mirrors paper opens/closes with tiny Binance orders.",
    )
    live_canary_parser.add_argument("--seconds", type=int, default=3600)
    live_canary_parser.add_argument("--max-trades", type=int, default=1)
    live_canary_parser.add_argument("--symbol", default=None)
    live_canary_parser.add_argument("--confirm-live-order", action="store_true", help="Required. This command can place real orders.")

    latency_parser = sub.add_parser("latency-probe", help="Measure Binance WebSocket event lag without running strategy logic.")
    latency_parser.add_argument("--seconds", type=int, default=300)
    latency_parser.add_argument(
        "--profile",
        default="current",
        choices=["current", "hot-combined", "hot-split", "aggtrade", "bookticker", "depth"],
        help="Stream profile to probe. `current` mirrors the recorder's public/market combined sockets.",
    )
    latency_parser.add_argument("--symbol", default=None, help="Override SYMBOL for this probe, e.g. ETHUSDT.")
    latency_parser.add_argument("--anchor-symbol", default=None, help="Override BTC cross-market anchor for current profile.")
    latency_parser.add_argument("--depth-levels", type=int, default=None)
    latency_parser.add_argument("--no-write-log", action="store_true", help="Only print summary JSON; do not write sample JSONL.")
    latency_parser.add_argument("--list-profiles", action="store_true", help="Show stream URLs for every built-in profile and exit.")

    data_summary_parser = sub.add_parser("data-summary", help="Summarize recon data storage.")
    data_summary_parser.add_argument("--largest", type=int, default=20)

    sub.add_parser("regime-outcome-summary", help="Summarize deduped FSM regime outcome episodes.")

    sub.add_parser("candidate-outcome-summary", help="Summarize edge-router candidate triple-barrier outcomes.")

    sub.add_parser("exit-shadow-summary", help="Summarize fee-aware shadow exit policy outcomes.")

    sub.add_parser("range-exit-counterfactual-summary", help="Summarize range time-decay hindsight outcomes.")

    first_touch_parser = sub.add_parser(
        "first-touch-study",
        help="Run a purged chronological +/- barrier study over stored feature and mark-price logs.",
    )
    first_touch_parser.add_argument("--runs-root", type=Path, default=Path("/app/data/runs"))
    first_touch_parser.add_argument("--symbol", default="ETHUSDT")
    first_touch_parser.add_argument("--target-bps", type=float, default=60.0)
    first_touch_parser.add_argument(
        "--stop-bps",
        type=float,
        nargs="+",
        default=[40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 120.0, 150.0, 450.0],
    )
    first_touch_parser.add_argument("--cost-bps", type=float, default=10.0)
    first_touch_parser.add_argument("--horizon-seconds", type=int, default=21_600)
    first_touch_parser.add_argument("--sample-seconds", type=int, default=60)
    first_touch_parser.add_argument("--max-gap-seconds", type=int, default=5)
    first_touch_parser.add_argument("--train-fraction", type=float, default=0.70)
    first_touch_parser.add_argument("--account-exposure", type=float, default=20.0)
    first_touch_parser.add_argument("--max-selected-stop-bps", type=float, default=150.0)

    atlas_parser = sub.add_parser(
        "market-atlas",
        help="Map ETH first-passage behavior and strategy economics across barriers, horizons, and regimes.",
    )
    atlas_parser.add_argument("--runs-root", type=Path, default=Path("/app/data/runs"))
    atlas_parser.add_argument("--symbol", default="ETHUSDT")
    atlas_parser.add_argument("--cost-bps", type=float, default=10.0)
    atlas_parser.add_argument("--sample-seconds", type=int, default=60)
    atlas_parser.add_argument("--max-gap-seconds", type=int, default=5)
    atlas_parser.add_argument("--discovery-fraction", type=float, default=0.70)
    atlas_parser.add_argument("--account-exposure", type=float, default=4.95)
    atlas_parser.add_argument("--minimum-robust-trades", type=int, default=20)
    atlas_parser.add_argument("--output", type=Path, default=None)

    tournament_parser = sub.add_parser(
        "strategy-tournament",
        help="Select one locked configuration per deterministic strategy family and judge it on holdout.",
    )
    tournament_parser.add_argument("--runs-root", type=Path, default=Path("/app/data/runs"))
    tournament_parser.add_argument("--symbol", default="ETHUSDT")
    tournament_parser.add_argument("--cost-bps", type=float, default=10.0)
    tournament_parser.add_argument("--sample-seconds", type=int, default=60)
    tournament_parser.add_argument("--max-gap-seconds", type=int, default=5)
    tournament_parser.add_argument("--account-exposure", type=float, default=4.95)
    tournament_parser.add_argument("--matched-coin-seeds", type=int, default=32)
    tournament_parser.add_argument("--output", type=Path, default=None)

    exit_lab_parser = sub.add_parser(
        "exit-laboratory",
        help="Compare deterministic exits on one locked squeeze-entry cohort.",
    )
    exit_lab_parser.add_argument("--runs-root", type=Path, default=Path("/app/data/runs"))
    exit_lab_parser.add_argument("--symbol", default="ETHUSDT")
    exit_lab_parser.add_argument("--cost-bps", type=float, default=10.0)
    exit_lab_parser.add_argument("--sample-seconds", type=int, default=60)
    exit_lab_parser.add_argument("--max-gap-seconds", type=int, default=5)
    exit_lab_parser.add_argument("--account-exposure", type=float, default=4.95)
    exit_lab_parser.add_argument("--matched-coin-seeds", type=int, default=32)
    exit_lab_parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    exit_lab_parser.add_argument("--output", type=Path, default=None)

    regime_atlas_parser = sub.add_parser(
        "regime-atlas",
        help="Test side-neutral permission regimes on the locked squeeze-entry cohort.",
    )
    regime_atlas_parser.add_argument("--runs-root", type=Path, default=Path("/app/data/runs"))
    regime_atlas_parser.add_argument("--symbol", default="ETHUSDT")
    regime_atlas_parser.add_argument("--cost-bps", type=float, default=10.0)
    regime_atlas_parser.add_argument("--sample-seconds", type=int, default=60)
    regime_atlas_parser.add_argument("--max-gap-seconds", type=int, default=5)
    regime_atlas_parser.add_argument("--account-exposure", type=float, default=4.95)
    regime_atlas_parser.add_argument("--matched-coin-seeds", type=int, default=32)
    regime_atlas_parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    regime_atlas_parser.add_argument("--output", type=Path, default=None)

    feature_value_parser = sub.add_parser(
        "feature-value",
        help="Test side-neutral feature value with nested discovery, calibration, and holdout periods.",
    )
    feature_value_parser.add_argument("--runs-root", type=Path, default=Path("/app/data/runs"))
    feature_value_parser.add_argument("--symbol", default="ETHUSDT")
    feature_value_parser.add_argument("--cost-bps", type=float, default=10.0)
    feature_value_parser.add_argument("--sample-seconds", type=int, default=60)
    feature_value_parser.add_argument("--max-gap-seconds", type=int, default=5)
    feature_value_parser.add_argument("--account-exposure", type=float, default=4.95)
    feature_value_parser.add_argument("--matched-coin-seeds", type=int, default=32)
    feature_value_parser.add_argument("--permutation-samples", type=int, default=2_000)
    feature_value_parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    feature_value_parser.add_argument("--output", type=Path, default=None)

    compress_parser = sub.add_parser("compress-raw", help="Gzip raw JSONL files under data/raw_ws.")
    compress_parser.add_argument("--older-than-minutes", type=int, default=5)
    compress_parser.add_argument("--all", action="store_true", help="Compress even recently modified files. Use after a run has stopped.")

    prune_parser = sub.add_parser("prune-raw", help="Delete raw JSONL/GZIP files older than a threshold.")
    prune_parser.add_argument("--older-than-hours", type=float, required=True)
    prune_parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.command == "watch":
        asyncio.run(watch(args.seconds, quiet=args.quiet))
    elif args.command == "record":
        asyncio.run(watch(args.seconds, quiet=args.quiet))
    elif args.command == "shadow-arena":
        asyncio.run(
            shadow_arena(
                args.seconds,
                args.quiet,
                args.target_bps,
                args.stop_bps,
                args.cost_bps,
                args.horizon_seconds,
                args.cooldown_seconds,
                args.warmup_seconds,
                args.max_trades_per_arm,
            )
        )
    elif args.command == "shadow-arena-summary":
        print(json.dumps(summarize_shadow_arena(Settings()), indent=2, default=str))
    elif args.command == "replay":
        replay(
            pattern=args.pattern,
            decision_interval_ms=args.decision_interval_ms,
            include_depth=args.include_depth,
            book_ticker_min_interval_ms=args.book_ticker_min_interval_ms,
            flatten_at_end=args.flatten_at_end,
            hostile=args.hostile,
        )
    elif args.command == "replay-compare":
        replay_compare(
            pattern=args.pattern,
            variants=args.variants,
            decision_interval_ms=args.decision_interval_ms,
            include_depth=args.include_depth,
            book_ticker_min_interval_ms=args.book_ticker_min_interval_ms,
            flatten_at_end=args.flatten_at_end,
            hostile=args.hostile,
        )
    elif args.command == "readiness-report":
        readiness_report(
            pattern=args.pattern,
            decision_interval_ms=args.decision_interval_ms,
            include_depth=args.include_depth,
            book_ticker_min_interval_ms=args.book_ticker_min_interval_ms,
            flatten_at_end=args.flatten_at_end,
        )
    elif args.command == "binance-account-probe":
        binance_account_probe()
    elif args.command == "binance-live-preflight":
        binance_live_preflight(args.symbol)
    elif args.command == "binance-order-test":
        binance_order_test(args.side, args.symbol, args.notional_usd)
    elif args.command == "binance-dust-round-trip":
        binance_dust_round_trip(args.side, args.symbol, args.confirm_live_order)
    elif args.command == "binance-dust-open":
        binance_dust_open(args.side, args.symbol, args.confirm_live_order)
    elif args.command == "binance-flatten-position":
        binance_flatten_position(args.symbol, args.confirm_live_order)
    elif args.command == "binance-treasury-rebalance":
        binance_treasury_rebalance(args.symbol, args.confirm_live_transfer)
    elif args.command == "binance-live-canary":
        asyncio.run(binance_live_canary(args.seconds, args.max_trades, args.symbol, args.confirm_live_order))
    elif args.command == "latency-probe":
        if args.list_profiles:
            settings = Settings()
            symbol = args.symbol or settings.symbol
            anchor = args.anchor_symbol or settings.cross_market_anchor_symbol
            depth_levels = args.depth_levels if args.depth_levels is not None else settings.depth_levels
            profiles = {}
            for profile in ["current", "hot-combined", "hot-split", "aggtrade", "bookticker", "depth"]:
                profiles[profile] = [
                    {"name": spec.name, "url": spec.url, "streams": list(spec.streams)}
                    for spec in build_probe_streams(
                        profile=profile,
                        symbol=symbol,
                        anchor_symbol=anchor,
                        depth_levels=depth_levels,
                        include_liquidations=settings.consume_liquidation_stream,
                    )
                ]
            print(json.dumps(profiles, indent=2))
            return
        asyncio.run(
            latency_probe(
                args.seconds,
                args.profile,
                args.symbol,
                args.anchor_symbol,
                args.depth_levels,
                args.no_write_log,
            )
        )
    elif args.command == "data-summary":
        print(json.dumps(summarize_data(Settings(), largest=args.largest).model_dump(), indent=2))
    elif args.command == "regime-outcome-summary":
        print(json.dumps(summarize_regime_outcomes(Settings()).model_dump(), indent=2))
    elif args.command == "candidate-outcome-summary":
        print(json.dumps(summarize_candidate_outcomes(Settings()).model_dump(), indent=2))
    elif args.command == "exit-shadow-summary":
        print(json.dumps(summarize_exit_shadow(Settings()).model_dump(), indent=2))
    elif args.command == "range-exit-counterfactual-summary":
        print(json.dumps(summarize_range_exit_counterfactuals(Settings()).model_dump(), indent=2))
    elif args.command == "first-touch-study":
        print(
            json.dumps(
                run_first_touch_study(
                    args.runs_root,
                    symbol=args.symbol,
                    target_bps=args.target_bps,
                    stop_bps_values=args.stop_bps,
                    cost_bps=args.cost_bps,
                    horizon_seconds=args.horizon_seconds,
                    sample_seconds=args.sample_seconds,
                    max_gap_seconds=args.max_gap_seconds,
                    train_fraction=args.train_fraction,
                    account_exposure=args.account_exposure,
                    max_selected_stop_bps=args.max_selected_stop_bps,
                ),
                indent=2,
            )
        )
    elif args.command == "market-atlas":
        report = run_market_atlas(
            args.runs_root,
            symbol=args.symbol,
            cost_bps=args.cost_bps,
            sample_seconds=args.sample_seconds,
            max_gap_seconds=args.max_gap_seconds,
            discovery_fraction=args.discovery_fraction,
            account_exposure=args.account_exposure,
            minimum_robust_trades=args.minimum_robust_trades,
        )
        encoded = json.dumps(report, indent=2)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded + "\n", encoding="utf-8")
            print(json.dumps({"output": str(args.output), "bytes": args.output.stat().st_size}, indent=2))
        else:
            print(encoded)
    elif args.command == "strategy-tournament":
        report = run_strategy_tournament(
            args.runs_root,
            symbol=args.symbol,
            cost_bps=args.cost_bps,
            sample_seconds=args.sample_seconds,
            max_gap_seconds=args.max_gap_seconds,
            account_exposure=args.account_exposure,
            matched_coin_seeds=args.matched_coin_seeds,
        )
        encoded = json.dumps(report, indent=2)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded + "\n", encoding="utf-8")
            print(json.dumps({"output": str(args.output), "bytes": args.output.stat().st_size}, indent=2))
        else:
            print(encoded)
    elif args.command == "exit-laboratory":
        report = run_exit_laboratory(
            args.runs_root,
            symbol=args.symbol,
            cost_bps=args.cost_bps,
            sample_seconds=args.sample_seconds,
            max_gap_seconds=args.max_gap_seconds,
            account_exposure=args.account_exposure,
            matched_coin_seeds=args.matched_coin_seeds,
            bootstrap_samples=args.bootstrap_samples,
        )
        encoded = json.dumps(report, indent=2)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded + "\n", encoding="utf-8")
            print(json.dumps({"output": str(args.output), "bytes": args.output.stat().st_size}, indent=2))
        else:
            print(encoded)
    elif args.command == "regime-atlas":
        report = run_regime_atlas(
            args.runs_root,
            symbol=args.symbol,
            cost_bps=args.cost_bps,
            sample_seconds=args.sample_seconds,
            max_gap_seconds=args.max_gap_seconds,
            account_exposure=args.account_exposure,
            matched_coin_seeds=args.matched_coin_seeds,
            bootstrap_samples=args.bootstrap_samples,
        )
        encoded = json.dumps(report, indent=2)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded + "\n", encoding="utf-8")
            print(json.dumps({"output": str(args.output), "bytes": args.output.stat().st_size}, indent=2))
        else:
            print(encoded)
    elif args.command == "feature-value":
        report = run_feature_value_analysis(
            args.runs_root,
            symbol=args.symbol,
            cost_bps=args.cost_bps,
            sample_seconds=args.sample_seconds,
            max_gap_seconds=args.max_gap_seconds,
            account_exposure=args.account_exposure,
            matched_coin_seeds=args.matched_coin_seeds,
            permutation_samples=args.permutation_samples,
            bootstrap_samples=args.bootstrap_samples,
        )
        encoded = json.dumps(report, indent=2)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded + "\n", encoding="utf-8")
            print(json.dumps({"output": str(args.output), "bytes": args.output.stat().st_size}, indent=2))
        else:
            print(encoded)
    elif args.command == "compress-raw":
        print(
            json.dumps(
                compress_raw(Settings(), older_than_minutes=args.older_than_minutes, include_current=args.all).model_dump(),
                indent=2,
            )
        )
    elif args.command == "prune-raw":
        print(
            json.dumps(
                prune_raw(Settings(), older_than_hours=args.older_than_hours, dry_run=args.dry_run).model_dump(),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
