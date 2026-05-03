# Futures Lab

Fresh API-first crypto futures research and paper-trading system.

This project intentionally does not inherit the old screenshot scalper architecture. Screenshots can be
used later for review, but not as the live signal source.

## Core Idea

The system watches Binance USD-M Futures market data directly, builds microstructure state, makes
deterministic hit-and-run proposals, runs those through a hard risk engine, then paper-trades the
result.

The target model from the project brief is represented in config:

```text
1000 USDT account
15% stake = 150 USDT
fast mode = 200x leverage
fast target = 0.5% underlying move
gross target = about 150 USDT before fees/slippage
```

The slow mode exists for the tradeoff we discussed:

```text
reduce leverage
allow wider/slower move window
stay open longer only when the regime supports it
```

## Data Sources

Public Binance USD-M Futures WebSockets:

- `btcusdt@bookTicker`
- `btcusdt@depth@100ms`
- `btcusdt@aggTrade`
- `btcusdt@markPrice@1s`
- `btcusdt@kline_1m`

The code uses Binance's upgraded split WebSocket routes:

- `wss://fstream.binance.com/public/stream?...`
- `wss://fstream.binance.com/market/stream?...`

Private account/order updates are deliberately not connected yet. The next safe phase is Binance demo
or testnet through NautilusTrader, then gated live execution only after paper data proves behavior.

## Decision Loop

No cron for trade decisions. The runtime is event-driven:

```text
Binance WebSockets
  -> MarketStateBook
  -> feature snapshot
  -> HitAndRunStrategy
  -> RiskEngine
  -> PaperBroker
  -> audit log
```

The strategy evaluates every `DECISION_INTERVAL_MS` milliseconds. Default is 500ms.

## What Drives Decisions

The first strategy uses:

- spread in bps
- 180s range position
- 15s/60s/180s returns
- 60s/180s realized volatility
- taker buy/sell ratio
- top-of-book imbalance
- regime classification: warming up, stale, sideways, directional, volatile

Fast mode is preferred in sideways conditions. Slow mode is available for less sideways regimes.

This is only the first strategy shell. The design expects many strategies to be added and replay-tested.

## NautilusTrader Role

NautilusTrader should be used for:

- production-grade event-driven backtesting
- instrument modeling
- Binance Futures demo/live venue integration
- order and position lifecycle handling
- moving from paper simulation to exchange-compatible execution

This project keeps our creative strategy/feature layer separate so Nautilus does not constrain the
decision logic. The bridge is in:

```text
src/futures_lab/nautilus_bridge.py
```

Install it later with:

```bash
pip install -e '.[nautilus]'
```

## Setup

```bash
cd /Users/jamesmungai/myprojects/futures-lab
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
```

Run tests:

```bash
pytest -q
```

Run API:

```bash
uvicorn futures_lab.api:app --reload --host 127.0.0.1 --port 8090
```

Start streaming:

```bash
curl -X POST http://127.0.0.1:8090/runtime/start
```

Inspect state:

```bash
curl http://127.0.0.1:8090/market
curl http://127.0.0.1:8090/decision
curl http://127.0.0.1:8090/paper
```

## Safety Boundary

This system does not place live trades. It produces decisions, runs paper trades, records raw market
data, and prepares a clean path to demo/testnet/live integration.

Live real-money futures execution must remain behind deterministic strategy rules, the risk engine,
and explicit user-controlled permissions.

