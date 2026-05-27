import argparse
import asyncio
import json
from datetime import datetime, timedelta, timezone

from futures_lab.audit import AuditLog
from futures_lab.config import Settings
from futures_lab.data_ops import (
    compress_raw,
    prune_raw,
    summarize_candidate_outcomes,
    summarize_data,
    summarize_exit_shadow,
    summarize_regime_outcomes,
)
from futures_lab.replay import discover_raw_files, replay_files
from futures_lab.runtime import TradingRuntime


def _record_deadline(seconds: int, *, current: datetime | None = None) -> datetime:
    now = current or datetime.now(timezone.utc)
    return now + timedelta(seconds=max(0, seconds))


def _deadline_remaining_seconds(deadline: datetime, *, current: datetime | None = None) -> float:
    now = current or datetime.now(timezone.utc)
    return (deadline - now).total_seconds()


async def watch(seconds: int, quiet: bool = False) -> None:
    settings = Settings()
    runtime = TradingRuntime.create(settings)
    runtime.start()
    try:
        deadline = _record_deadline(seconds)
        while (remaining := _deadline_remaining_seconds(deadline)) > 0:
            await asyncio.sleep(min(1, remaining))
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
            await asyncio.wait_for(runtime.stop(), timeout=10)
        except TimeoutError:
            print("Timed out while stopping runtime; event loop shutdown will cancel remaining tasks.")


def replay(
    pattern: str | None,
    decision_interval_ms: int | None,
    include_depth: bool,
    book_ticker_min_interval_ms: int,
    flatten_at_end: bool,
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
    )
    print(json.dumps(summary.model_dump(), indent=2, default=str))


def replay_compare(
    pattern: str | None,
    variants: list[str],
    decision_interval_ms: int | None,
    include_depth: bool,
    book_ticker_min_interval_ms: int,
    flatten_at_end: bool,
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
        )
        rows.append(
            {
                "variant": variant,
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
            }
        )
    print(json.dumps(rows, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description="Futures Lab CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    watch_parser = sub.add_parser("watch", help="Watch live public Binance data and decisions.")
    watch_parser.add_argument("--seconds", type=int, default=30)
    watch_parser.add_argument("--quiet", action="store_true", help="Record without printing every second.")

    record_parser = sub.add_parser("record", help="Record public Binance data and run paper decisions.")
    record_parser.add_argument("--seconds", type=int, default=1800)
    record_parser.add_argument("--quiet", action="store_true")

    replay_parser = sub.add_parser("replay", help="Replay recorded raw WebSocket JSONL.")
    replay_parser.add_argument("--pattern", default=None, help="Glob under data/raw_ws, e.g. BTCUSDT_*_2026-05-03.jsonl")
    replay_parser.add_argument("--decision-interval-ms", type=int, default=None)
    replay_parser.add_argument("--include-depth", action="store_true", help="Include depthUpdate messages. Off by default because current strategy does not consume them.")
    replay_parser.add_argument("--book-ticker-min-interval-ms", type=int, default=100)
    replay_parser.add_argument("--flatten-at-end", action="store_true", help="Close any open paper position at the final replay price for session accounting.")

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

    data_summary_parser = sub.add_parser("data-summary", help="Summarize recon data storage.")
    data_summary_parser.add_argument("--largest", type=int, default=20)

    sub.add_parser("regime-outcome-summary", help="Summarize deduped FSM regime outcome episodes.")

    sub.add_parser("candidate-outcome-summary", help="Summarize edge-router candidate triple-barrier outcomes.")

    sub.add_parser("exit-shadow-summary", help="Summarize fee-aware shadow exit policy outcomes.")

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
    elif args.command == "replay":
        replay(
            pattern=args.pattern,
            decision_interval_ms=args.decision_interval_ms,
            include_depth=args.include_depth,
            book_ticker_min_interval_ms=args.book_ticker_min_interval_ms,
            flatten_at_end=args.flatten_at_end,
        )
    elif args.command == "replay-compare":
        replay_compare(
            pattern=args.pattern,
            variants=args.variants,
            decision_interval_ms=args.decision_interval_ms,
            include_depth=args.include_depth,
            book_ticker_min_interval_ms=args.book_ticker_min_interval_ms,
            flatten_at_end=args.flatten_at_end,
        )
    elif args.command == "data-summary":
        print(json.dumps(summarize_data(Settings(), largest=args.largest).model_dump(), indent=2))
    elif args.command == "regime-outcome-summary":
        print(json.dumps(summarize_regime_outcomes(Settings()).model_dump(), indent=2))
    elif args.command == "candidate-outcome-summary":
        print(json.dumps(summarize_candidate_outcomes(Settings()).model_dump(), indent=2))
    elif args.command == "exit-shadow-summary":
        print(json.dumps(summarize_exit_shadow(Settings()).model_dump(), indent=2))
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
