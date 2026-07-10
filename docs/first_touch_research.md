# First-Touch Strategy Research

The range-risk track is now evaluated as a first-touch problem:

> Given the current observable market state, which side should be traded so that the favorable
> barrier is reached before the adverse barrier?

This framing keeps prediction, execution cost, exit shape, and position sizing separate.

## Economics

For a 60 bps target, a 50 bps stop, and 10 bps round-trip taker cost:

```text
net win  = +50 bps
net loss = -60 bps
break-even win rate = 60 / (50 + 60) = 54.55%
```

A wider stop raises the observed hit rate but also raises the break-even hit rate. At a 450 bps
liquidation proxy, the required win rate is 90.20%. A high win rate is not sufficient by itself.

## Study Command

```bash
futures-lab first-touch-study \
  --runs-root /app/data/runs \
  --symbol ETHUSDT \
  --target-bps 60 \
  --stop-bps 40 50 60 70 80 90 100 120 150 450 \
  --cost-bps 10 \
  --horizon-seconds 21600 \
  --sample-seconds 60 \
  --account-exposure 20
```

The study:

- deduplicates overlapping run archives;
- reconstructs first barrier touches from one-second mark prices;
- splits data chronologically and purges one full label horizon from training;
- simulates sequential, non-overlapping trades;
- compares long/short, range-reversion, micro-momentum, HTF-momentum, and regime-router rules;
- fits a regularized linear probability model as a research benchmark;
- reports fees, timeouts, confidence intervals, losing streaks, and account-level drawdown;
- selects stop widths on training data only, then reports the frozen holdout result.

## Current Findings

The prior range-reversion rule did not clear the 60/50 bps economics on the chronological holdout.
Its router also treated a hand-built score as `p_hit_tp_before_sl`, which was not statistically
calibrated and must not be interpreted as a probability.

A side-neutral micro-momentum rule was the strongest recent candidate at a 60 bps target and 70
bps stop, but it is not yet proven across regimes. It should be validated with controlled exposure,
real fill costs, and rolling out-of-sample monitoring before notional is increased.

The matching paper strategy is `STRATEGY_VARIANT=first_touch_micro_momentum`. Its score is explicitly
logged as signal strength, not as a probability. A first-touch candidate remains blocked from the
paper-live edge gate until an out-of-sample probability calibration is promoted separately.

For a 100 USDT research account, `ACCOUNT_EQUITY_USD=100`, `STAKE_FRACTION=0.033`, and
`FIRST_TOUCH_LEVERAGE=150` produce approximately 495 USDT notional, or 4.95x effective account
exposure. Exchange leverage controls required initial margin; notional divided by account equity
controls economic exposure and drawdown.

## Promotion Rules

A strategy is not promoted because it has the highest backtest return. It must satisfy all of the
following:

1. Positive average net bps after the account's actual fee tier and hostile cost assumptions.
2. Holdout barrier win rate above the break-even rate, with enough samples to narrow uncertainty.
3. Positive or acceptably flat behavior for both long and short predictions.
4. Stability across chronological blocks and market regimes.
5. Drawdown survivable at the proposed effective account exposure.
6. Live shadow fills close enough to modeled fills to preserve the expected edge.
