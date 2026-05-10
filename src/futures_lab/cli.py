import argparse
import asyncio
import json
import time

from futures_lab.audit import AuditLog
from futures_lab.config import Settings
from futures_lab.replay import discover_raw_files, replay_files
from futures_lab.runtime import TradingRuntime


async def watch(seconds: int, quiet: bool = False) -> None:
    settings = Settings()
    runtime = TradingRuntime.create(settings)
    runtime.start()
    try:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            await asyncio.sleep(1)
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
        audit=AuditLog(settings),
    )
    print(json.dumps(summary.model_dump(), indent=2, default=str))


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
        )


if __name__ == "__main__":
    main()
