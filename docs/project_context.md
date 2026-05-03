# Portable Project Context

This file exists so the project can move to another machine or another Codex session without relying
on local chat history.

## Origin Story

The project started from a rejected screenshot-based trading idea. The old prototype sent TradingView
screenshots to an LLM and asked for a decision/confidence level. We concluded that cannot be the live
trading core because screenshots are too slow and lossy for high-leverage futures scalping.

This repository is the clean rebuild.

## User Intent

The user wants to automate a crypto futures scalping workflow as much as possible while staying inside
safe boundaries. The desired trading style is "hit and run":

- Wait for a good market pattern.
- Open only when the setup is strong.
- Close quickly once the allocated stake has achieved the target gain.
- Prefer one successful high-quality trade per day over many weak trades.

The reference example:

```text
1000 USDT futures account
15% stake = 150 USDT
Remaining balance acts as margin protection
200x leverage -> 30,000 USDT notional
0.5% favorable underlying move -> about 150 USDT gross PnL
```

The tradeoff under active discussion:

- High leverage + very short open/close window
- Lower leverage + longer decision window, for example 3m/5m-style market structure

The implementation should allow both modes.

## Assistant Boundary

The assistant can:

- Design architecture
- Write code
- Build data pipelines
- Build strategy/risk/paper/backtest modules
- Analyze logs and strategy results
- Suggest parameter changes
- Integrate NautilusTrader for demo/backtest/live-ready infrastructure

The assistant should not:

- Act as the autonomous discretionary live trader for real funds
- Place live leveraged trades from natural-language judgment
- Override risk controls
- Use screenshots as the live signal source

## Data Plan

Primary data source is Binance USD-M Futures API/WebSocket data:

- `bookTicker`
- `depth@100ms`
- `aggTrade`
- `markPrice@1s`
- `kline_1m`

Future private/demo/live integration should use:

- User order updates
- Account updates
- Position updates
- NautilusTrader Binance integration where useful

## Strategy Research Plan

The first strategy is an interpretable baseline, not the final edge:

- classify regime
- prefer fast mode in sideways conditions
- use range position
- use spread
- use short-horizon returns
- use realized volatility
- use taker buy/sell pressure
- use top-of-book imbalance
- open paper position only when risk allows
- close at take-profit or stop-loss

Next research direction:

1. Record raw public WebSocket data for several market sessions.
2. Build replay from recorded JSONL.
3. Score hit-and-run variants by:
   - net PnL after fees
   - win rate
   - daily target hit rate
   - max adverse excursion
   - time in trade
   - slippage sensitivity
4. Add NautilusTrader adapter/backtest path.
5. Add Binance demo/testnet trading.
6. Consider real execution only after demo behavior is boring and robust.

## Repository State At Context Capture

Initial clean project commit:

```text
48499c8 Initial futures trading lab
```

Remote:

```text
origin https://github.com/james-mungai/my-trader.git
```

Tests passed:

```text
6 passed
```

Live public Binance WebSocket smoke test succeeded and produced BTCUSDT market state.

