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

- `RECORD_DEPTH_STREAM=false` by default because current strategy does not consume depth updates.
- `RECORD_BOOK_TICKER_MIN_INTERVAL_MS=250` down-samples raw bookTicker storage.
- `RAW_ROTATION_MINUTES=60` writes hourly raw files.
- `COMPRESS_ROTATED_RAW=true` compresses completed raw files.
- `WRITE_FEATURE_LOG=true` writes compact feature snapshots.
- `WRITE_DECISION_LOG=true` writes strategy/risk decisions.
- `WRITE_PAPER_TRADE_LOG=true` writes closed paper trades.

Important output locations:

```text
data/raw_ws/        replayable raw WebSocket JSONL, ignored by Git
data/features/      compact market feature JSONL, ignored by Git
data/decisions/     decision/risk JSONL, ignored by Git
data/paper_trades/  closed paper-trade JSONL, ignored by Git
```

## Development Commands

Use Python 3.11+.

```bash
cd /Users/jamesmungai/myprojects/futures-lab
source .venv/bin/activate
pytest -q
uvicorn futures_lab.api:app --reload --host 127.0.0.1 --port 8090
futures-lab watch --seconds 30
```

Serious recon run:

```bash
futures-lab record --seconds 28800 --quiet
```

Replay latest captured raw data:

```bash
futures-lab replay
```

Local dashboard:

```text
http://127.0.0.1:8090/
```

If `.venv` is absent:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

On this machine, Python 3.13 was available at `/opt/homebrew/bin/python3.13`.

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
