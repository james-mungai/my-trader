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

The default recon risk model now allows one accepted paper trade per day/session. That matches the
current research goal: prefer one high-quality setup, then stop and analyze rather than churn.
Fast-failure exits are disabled by default after replay showed they can cut valid slow-developing
winners before TP. They remain available as an explicit experiment switch.

The slow mode exists for the tradeoff we discussed:

```text
reduce leverage
allow wider/slower move window
stay open longer only when the regime supports it
```

## Data Sources

Public Binance USD-M Futures WebSockets:

- `btcusdt@bookTicker`
- `btcusdt@depth5@100ms`
- `btcusdt@aggTrade`
- `btcusdt@markPrice@1s`
- `btcusdt@kline_1m`
- `btcusdt@forceOrder`

The code uses Binance's upgraded split WebSocket routes:

- `wss://fstream.binance.com/public/stream?...`
- `wss://fstream.binance.com/market/stream?...`

To separate provider/network latency from strategy behavior, use the standalone latency probe:

```bash
futures-lab latency-probe --seconds 300 --profile current --symbol ETHUSDT
futures-lab latency-probe --seconds 300 --profile hot-combined --symbol ETHUSDT
futures-lab latency-probe --seconds 300 --profile hot-split --symbol ETHUSDT
futures-lab latency-probe --list-profiles --symbol ETHUSDT
```

Profiles:

- `current`: mirrors the recorder's public and market combined sockets.
- `hot-combined`: probes only ETH book/depth plus ETH aggregate trades.
- `hot-split`: probes ETH bookTicker, depth, and aggregate trades on separate connections.
- `aggtrade`, `bookticker`, `depth`: isolate one stream family.

The probe writes compact samples under `data/latency_probes/` and prints per-stream event-lag,
receive-gap, connection, and error statistics. This is the quickest way to compare laptop, AWS, GCP,
and trading VPS routing before running the full strategy.

The recorder can also use the same hot-stream routing by setting `BINANCE_STREAM_PROFILE`:

- `current`: existing two-socket public/context and market/context layout.
- `hot-combined`: primary book/depth and primary trades on separate hot sockets, context on separate sockets.
- `hot-split`: primary bookTicker, depth, and trades each on their own socket, context on separate sockets.

Use `hot-split` for latency-sensitive AWS recon runs when the probe confirms it is the fastest route.
`CONSUME_BOOK_TICKER_MIN_INTERVAL_MS` throttles only state ingestion for high-volume bookTicker updates;
depth/trade streams remain unthrottled, and raw bookTicker storage keeps its separate
`RECORD_BOOK_TICKER_MIN_INTERVAL_MS` setting.

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
Deeper market context is updated continuously before the decision pass, so the live decision path only
reads already-computed rolling features.

Select the deterministic strategy with `STRATEGY_VARIANT`:

- `baseline`: range-location hit-and-run with flow/book/depth confirmation.
- `liquidity_sweep_reversal`: waits for a low/high sweep plus reclaim/rejection before entering.
- `momentum_pullback`: joins short-horizon trend continuation after a controlled pullback or bounce.
- `stateful_momentum`: hybrid finite-state/Markov variant that only enters after impulse -> controlled pullback/bounce -> re-acceleration confirmation, and only when the recent range can plausibly support the fast target.

`ENABLE_MARKOV_STATE_MACHINE=true` logs the rolling sequence state and transition counts in decision
evidence and replay summaries. This is deterministic bookkeeping, not ML.
When `stateful_momentum` confirms a sequence, decision evidence also includes
`stateful_momentum_filter` so near-misses show the confirmed side, score, target feasibility,
required 180s range, and blocker that prevented entry.

## What Drives Decisions

The first strategy uses:

- spread in bps
- 180s range position
- 15s/60s/180s returns
- 60s/180s realized volatility
- taker buy/sell ratio
- top-of-book imbalance
- top-5 depth imbalance and wall ratios
- 30s liquidation pulse
- open-interest drift, refreshed outside the hot path
- exchange event lag / local decision latency
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

## Docker Setup

Docker is the recommended default so the project runs the same way on Windows, macOS, and Linux.

```bash
cp .env.example .env
docker compose up --build
```

Open the local dashboard:

```text
http://127.0.0.1:8090/
```

Run tests in the same container image:

```bash
docker compose run --rm api pytest -q
```

Record a recon session from the CLI:

```bash
docker compose run --rm api futures-lab record --seconds 1800
```

Replay recorded raw WebSocket JSONL:

```bash
docker compose run --rm api futures-lab replay
```

Replay with session-end flattening for accounting:

```bash
docker compose run --rm api futures-lab replay --flatten-at-end
```

Compare strategy variants on the same raw tape:

```bash
docker compose run --rm api futures-lab replay-compare --flatten-at-end
```

Summarize and maintain recon storage:

```bash
docker compose run --rm api futures-lab data-summary
docker compose run --rm api futures-lab compress-raw --all
docker compose run --rm api futures-lab prune-raw --older-than-hours 24 --dry-run
```

The Compose service stores runtime data in the `futures_lab_data` Docker volume mounted at
`/app/data` inside the container.

## AWS Deployment

AWS deployment scaffolding lives under `deployments/aws/` and `scripts/aws/`.

For the first cloud test, use Lightsail Containers in Tokyo:

```text
Region: ap-northeast-1
Service size: medium
Scale: 1
```

Build/push/deploy helpers:

```powershell
.\scripts\aws\build-lightsail-image.ps1
.\scripts\aws\push-lightsail-image.ps1 -ServiceName futures-lab -Region ap-northeast-1
.\scripts\aws\deploy-lightsail-container.ps1 -Image ":futures-lab.futures-lab.1" -ApiToken "<long-token>"
```

See `docs/cloud_deployment.md` for the full workflow and the tradeoff between Lightsail Containers
and a Lightsail VM. Containers are convenient for API/live paper monitoring; a VM is better for
durable recon data until S3 export is added.

## Local Python Setup

Use this only when you specifically want a host-machine development environment.

Python 3.11+ is required. From the repository root:

```bash
python -m venv .venv
```

Activate the virtualenv for your shell:

```bash
# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows cmd.exe
.\.venv\Scripts\activate.bat
```

Install dependencies:

```bash
python -m pip install -e ".[dev]"
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

Open the local dashboard:

```text
http://127.0.0.1:8090/
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

Record a recon session from the CLI:

```bash
futures-lab record --seconds 1800
```

Replay recorded raw WebSocket JSONL:

```bash
futures-lab replay
```

Replay one date or subset:

```bash
futures-lab replay --pattern 'BTCUSDT_*_2026-05-03.jsonl'
```

## All-Day Recon Storage

The recorder is configured for all-day public-data collection without keeping every noisy tick forever:

- `RECORD_DEPTH_STREAM=false` by default because depth is useful live context but too noisy to store raw unless researching order-book behavior.
- `CONSUME_DEPTH_STREAM=true` keeps top-5 depth features in memory even when raw depth storage is off.
- `CONSUME_LIQUIDATION_STREAM=true` records forced-order/liquidation pulses for context.
- `OPEN_INTEREST_POLL_SECONDS=30` refreshes open interest outside the hot decision path.
- `RECORD_BOOK_TICKER_MIN_INTERVAL_MS=250` stores bookTicker at most four times per second.
- `RAW_ROTATION_MINUTES=60` writes hourly raw files.
- `COMPRESS_ROTATED_RAW=true` gzips completed hourly raw files.
- `WRITE_FEATURE_LOG=true` writes compact feature snapshots.
- `WRITE_DECISION_LOG=true` writes strategy/risk decisions.
- `WRITE_PAPER_TRADE_LOG=true` writes closed paper trades.

Useful output locations:

```text
data/raw_ws/        raw replayable WebSocket data
data/features/      compact market features
data/decisions/     decision/risk tape
data/paper_trades/  closed paper trades
```

Tomorrow's recon command:

```bash
docker compose run --rm api futures-lab record --seconds 28800 --quiet
```

That is an 8-hour run.

## Safety Boundary

This system does not place live trades. It produces decisions, runs paper trades, records raw market
data, and prepares a clean path to demo/testnet/live integration.

Live real-money futures execution must remain behind deterministic strategy rules, the risk engine,
and explicit user-controlled permissions.
