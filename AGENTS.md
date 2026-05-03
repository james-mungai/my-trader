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
  -> FastAPI / CLI / audit log
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

If `.venv` is absent:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

On this machine, Python 3.13 was available at `/opt/homebrew/bin/python3.13`.

## Safety Notes

This repo intentionally does not place live orders. Future live execution must require explicit user
permission, deterministic rules, risk checks, and careful handling of API keys.

