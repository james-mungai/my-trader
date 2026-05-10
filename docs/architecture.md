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
- mark-price/funding context
- adaptive leverage and target windows
- kill-after-profit daily behavior

The current first pass is deliberately interpretable. Once enough raw WebSocket data is recorded, the
next step is replay and parameter search.

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
- CLI recording: `futures-lab record --seconds 1800`
- CLI replay: `futures-lab replay`
- Raw data location: `data/raw_ws/`

## Storage Policy

Raw WebSocket data is for short-term forensic replay. It is not the primary long-term research store.

All-day recon should rely on:

- compressed hourly raw files for replay
- compact feature snapshots for analysis
- decision/risk JSONL for strategy debugging
- paper-trade JSONL for outcome review

Depth updates are disabled by default because the current strategy does not use them. Re-enable
`RECORD_DEPTH_STREAM=true` only when actively researching order-book-depth features.
