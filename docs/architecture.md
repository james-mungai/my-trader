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
  -> API / audit log
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
2. Add replay tests against recorded JSONL.
3. Add multiple strategy candidates.
4. Score strategies by net PnL after fees, max adverse excursion, time-in-trade, and daily target hit rate.
5. Add NautilusTrader backtest adapter.
6. Add Binance demo trading through Nautilus.
7. Add gated live execution only after demo results are acceptable.

