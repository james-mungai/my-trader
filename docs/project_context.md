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

Current all-day recon defaults:

- `depth@100ms` is available but not recorded by default: `RECORD_DEPTH_STREAM=false`.
- raw `bookTicker` storage is downsampled with `RECORD_BOOK_TICKER_MIN_INTERVAL_MS=250`.
- raw files rotate hourly with `RAW_ROTATION_MINUTES=60`.
- completed raw files are compressed with `COMPRESS_ROTATED_RAW=true`.
- compact feature, decision, and paper-trade logs are written by default.

Important output locations:

```text
data/raw_ws/        replayable raw WebSocket JSONL
data/features/      compact market feature JSONL
data/decisions/     strategy/risk decision JSONL
data/paper_trades/  closed paper-trade JSONL
```

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
2. Replay recorded JSONL through `MarketStateBook -> HitAndRunStrategy -> RiskEngine -> PaperBroker`.
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

## Recon Commands

Short check:

```bash
futures-lab record --seconds 1800 --quiet
futures-lab replay
```

Serious all-day-style run:

```bash
futures-lab record --seconds 28800 --quiet
```

Dashboard:

```bash
uvicorn futures_lab.api:app --reload --host 127.0.0.1 --port 8090
```

Then open:

```text
http://127.0.0.1:8090/
```

## Latest Recon Result

A 30-minute-plus BTCUSDT public-data run on 2026-05-10 captured about 164 MB of raw data before the
all-day storage patch. Replay after timestamp/performance fixes produced:

```text
messages:      41,345
decisions:     3,009
proposals:     133
risk_allowed:  1
paper_opens:   1
paper_closes:  0
realized PnL:  0.0
```

The accepted paper trade:

```text
side:       long
mode:       fast
entry:      81431.15
TP:         81838.31
stop:       81268.29
leverage:   200x
notional:   30000
confidence: 0.7619
```

It did not close within the captured window. This indicates the next strategy work should add better
exit handling: time stop, flatten-at-session-end for replay, partial profit, or adaptive TP/SL based
on volatility.

## Repository State At Context Capture

Initial clean project commit:

```text
48499c8 Initial futures trading lab
```

Portable context commit:

```text
00ae587 Add portable project context
```

All-day recon/storage commit:

```text
680b28c Add all-day recon storage and replay tools
```

Remote:

```text
origin https://github.com/james-mungai/my-trader.git
```

Tests passed:

```text
9 passed
```

Live public Binance WebSocket smoke test succeeded and produced BTCUSDT market state.
