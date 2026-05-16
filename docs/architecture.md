# Architecture

## Why Not Screenshots

Screenshots are too lossy and too slow for high-leverage futures scalping. They hide bid/ask,
spread, depth, mark price, funding, position state, and fills. This system treats exchange API state
as source of truth.

## Why Not Pure NautilusTrader First

NautilusTrader is strong infrastructure: venue adapters, backtesting, execution clients, and market
models. But the strategy research loop needs room to evolve quickly. The project uses a clean custom
core first, then adds Nautilus where the engine value is highest.

## Components

```text
BinanceStreamRecorder
  -> MarketStateBook
  -> BinanceContextPoller
  -> HitAndRunStrategy
  -> RiskEngine
  -> PaperBroker
  -> API / dashboard / audit log
```

## Novel Strategy Surface

The strategy does not need to imitate common TradingView indicators. It can mix:

- microstructure pressure
- range position
- realized volatility compression/expansion
- short horizon returns
- top-5 depth imbalance and wall concentration
- 30s forced-liquidation pulse
- mark-price/funding context
- open-interest drift refreshed outside the hot path
- exchange-event lag and decision latency
- adaptive leverage and target windows
- kill-after-profit daily behavior

The current first pass is deliberately interpretable. Once enough raw WebSocket data is recorded, the
next step is replay and parameter search.

`STRATEGY_VARIANT` selects the deterministic decision surface:

- `baseline`: range-location hit-and-run.
- `liquidity_sweep_reversal`: wick/sweep reclaim or rejection.
- `momentum_pullback`: continuation after a controlled pullback or bounce.
- `stateful_momentum`: finite-state sequence gate plus Markov transition logger and a fast-target feasibility check.

Variants stay long/short symmetric so they do not overfit a temporary bearish or bullish news cycle.

The finite-state layer labels the market path as `bullish_pressure`, `pullback`, `reclaim`,
`long_continuation_confirmed` or the short-side mirror: `bearish_pressure`, `bounce`, `rejection`,
`short_continuation_confirmed`. Replay summaries include observed transition counts/probabilities so
we can later see which paths actually lead to TP versus stop. Confirmed continuation is not enough on
its own: the recent 180s range must also be large enough to make the configured fast target plausible.
Near-misses are logged in decision evidence as `stateful_momentum_filter`, including the confirmed
side, score, required range, target feasibility, and blockers.

## Next Phases

1. Record public market data for several sessions.
2. Replay recorded JSONL through the same market-state, strategy, risk, and paper stack.
3. Add multiple strategy candidates.
4. Score strategies by net PnL after fees, max adverse excursion, time-in-trade, and daily target hit rate.
5. Add NautilusTrader backtest adapter.
6. Add Binance demo trading through Nautilus.
7. Add gated live execution only after demo results are acceptable.

## Recon Tools

- Local dashboard: `http://127.0.0.1:8090/`
- CLI recording: `docker compose run --rm api futures-lab record --seconds 1800`
- CLI replay: `docker compose run --rm api futures-lab replay`
- Container data location: `/app/data/raw_ws/`
- Default Docker volume: `futures_lab_data`

## Storage Policy

Raw WebSocket data is for short-term forensic replay. It is not the primary long-term research store.

All-day recon should rely on:

- compressed hourly raw files for replay
- compact feature snapshots for analysis
- decision/risk JSONL for strategy debugging
- paper-trade JSONL for outcome review

Depth updates are consumed by default for rolling top-5 order-book features, but raw depth storage
stays disabled by default. Re-enable `RECORD_DEPTH_STREAM=true` only when actively researching
order-book-depth replay data because depth files grow quickly.
