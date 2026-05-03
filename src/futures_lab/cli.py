import argparse
import asyncio

from futures_lab.config import Settings
from futures_lab.runtime import TradingRuntime


async def watch(seconds: int) -> None:
    settings = Settings()
    runtime = TradingRuntime.create(settings)
    runtime.start()
    try:
        for _ in range(seconds):
            await asyncio.sleep(1)
            market, decision, risk = runtime.decide_once()
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
        await runtime.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Futures Lab CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    watch_parser = sub.add_parser("watch", help="Watch live public Binance data and decisions.")
    watch_parser.add_argument("--seconds", type=int, default=30)
    args = parser.parse_args()

    if args.command == "watch":
        asyncio.run(watch(args.seconds))


if __name__ == "__main__":
    main()

