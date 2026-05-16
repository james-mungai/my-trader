# Codex Project Instructions

This repository is a fresh project. Do not treat the old sibling `scalper/` screenshot app as the
architecture source of truth.

## Project Goal

Build an API-first crypto futures research, paper-trading, and eventually demo/live-ready system for
high-conviction hit-and-run trades.

The current target model is:

- Account reference: 1000 USDT
- Stake fraction: 15%, or 150 USDT
- Fast mode: 200x leverage, 0.5% underlying target
- Slow mode: lower leverage, wider/longer window when market structure supports it
- Stop after daily target or max loss

## Key Design Rules

- Exchange API data is source of truth.
- Do not use screenshots as a live trading signal.
- Do not build discretionary LLM-driven live execution.
- Keep strategy logic deterministic and testable.
- Keep risk gating separate from strategy creativity.
- Keep live real-money execution out of scope until paper/replay/demo results justify it.
- Use NautilusTrader where it helps with backtesting, venue integration, order lifecycle, and live/demo plumbing.
- Keep the custom decision layer independent so strategy research is not constrained by Nautilus abstractions.

## Current Architecture

```text
Binance public WebSockets
  -> BinanceStreamRecorder
  -> MarketStateBook
  -> HitAndRunStrategy
  -> RiskEngine
  -> PaperBroker
  -> FastAPI / dashboard / CLI / compact recon logs / audit log
```

## Recon Storage Policy

The repo is patched for all-day recon:

- `CONSUME_DEPTH_STREAM=true` by default keeps top-5 depth features live.
- `RECORD_DEPTH_STREAM=false` by default because raw depth storage is noisy and only needed for targeted order-book replay research.
- `CONSUME_LIQUIDATION_STREAM=true` by default keeps forced-order/liquidation context live.
- `OPEN_INTEREST_POLL_SECONDS=30` refreshes open-interest context outside the hot decision path.
- `RECORD_BOOK_TICKER_MIN_INTERVAL_MS=250` down-samples raw bookTicker storage.
- `RAW_ROTATION_MINUTES=60` writes hourly raw files.
- `COMPRESS_ROTATED_RAW=true` compresses completed raw files.
- `WRITE_FEATURE_LOG=true` writes compact feature snapshots.
- `WRITE_DECISION_LOG=true` writes strategy/risk decisions.
- `WRITE_PAPER_TRADE_LOG=true` writes closed paper trades.
- `ENABLE_FAST_FAILURE_EXIT=false` by default because recent replay showed early fast-failure exits can cut valid winners before TP.
- `ENABLE_MARKOV_STATE_MACHINE=true` logs deterministic finite-state/Markov sequence state in decisions and replay summaries.
- `STRATEGY_VARIANT=stateful_momentum` enables the hybrid FSM gate: impulse -> pullback/bounce -> reclaim/rejection -> continuation confirmation, with a fast-target feasibility check. Confirmed-but-blocked near-misses are logged in decision evidence as `stateful_momentum_filter`.

Important output locations:

```text
data/raw_ws/        replayable raw WebSocket JSONL, ignored by Git
data/features/      compact market feature JSONL, ignored by Git
data/decisions/     decision/risk JSONL, ignored by Git
data/paper_trades/  closed paper-trade JSONL, ignored by Git
```

## Development Commands

Prefer Docker for OS-agnostic local work:

```bash
cp .env.example .env
docker compose up --build
docker compose run --rm api pytest -q
docker compose run --rm api futures-lab watch --seconds 30
docker compose run --rm api futures-lab record --seconds 28800 --quiet
docker compose run --rm api futures-lab replay --flatten-at-end
docker compose run --rm api futures-lab data-summary
docker compose run --rm api futures-lab compress-raw --all
```

Local Python is still supported when needed. Use Python 3.11+ from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
python -m pip install -e ".[dev]"
pytest -q
uvicorn futures_lab.api:app --reload --host 127.0.0.1 --port 8090
futures-lab watch --seconds 30
```

Serious recon run:

```bash
docker compose run --rm api futures-lab record --seconds 28800 --quiet
```

Replay latest captured raw data:

```bash
docker compose run --rm api futures-lab replay --flatten-at-end
```

Local dashboard:

```text
http://127.0.0.1:8090/
```

## Cloud Deployment

AWS deployment scaffolding is available:

```text
Dockerfile.aws
docker-compose.aws.yml
deployments/aws/
scripts/aws/
docs/cloud_deployment.md
```

Use Lightsail Containers in Tokyo (`ap-northeast-1`) as the first cloud API/paper-monitoring target.
Use a Lightsail/EC2 VM with `docker-compose.aws.yml` when durable recon data and easier file retrieval
matter more than managed container convenience.

## Current Local Checkpoint

Latest important commit as of 2026-05-11:

```text
680b28c Add all-day recon storage and replay tools
```

Current tests:

```text
9 passed
```

## Safety Notes

This repo intentionally does not place live orders. Future live execution must require explicit user
permission, deterministic rules, risk checks, and careful handling of API keys.
