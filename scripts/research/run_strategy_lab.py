import argparse
import json
import os
from pathlib import Path

from futures_lab.strategy_lab import run_symmetric_strategy_lab


DEFAULT_DATA_ROOT = Path(
    os.environ.get(
        "FUTURES_LAB_RESEARCH_DATA",
        Path.home() / "projects" / "my-trader-research-data" / "20260711",
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the purged symmetric first-touch strategy laboratory.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_DATA_ROOT / "dataset" / "runs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DATA_ROOT / "strategy-lab-local.json",
    )
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--barrier-bps", type=float, default=60.0)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--horizon-seconds", type=int, default=21_600)
    parser.add_argument("--sample-seconds", type=int, default=60)
    parser.add_argument("--account-exposure", type=float, default=4.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_symmetric_strategy_lab(
        args.runs_root,
        symbol=args.symbol,
        barrier_bps=args.barrier_bps,
        cost_bps=args.cost_bps,
        horizon_seconds=args.horizon_seconds,
        sample_seconds=args.sample_seconds,
        account_exposure=args.account_exposure,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
