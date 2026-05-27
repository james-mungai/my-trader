# Microstructure Edge Router Roadmap

This roadmap is the working ledger for the revamp from a regime-gated strategy into a microstructure-first, HTF-vetoed edge router. Each completed substage should be checked off in this file and committed with the implementation that made it true.

## Operating Principle

The bot should trade only when the expected short-horizon move is large enough to beat fees, spread, slippage, latency, and adverse selection.

```text
HTF / regime layer     -> permission, side bias, thresholds
Microstructure layer   -> exact entry edge
Execution cost layer   -> fee/spread/slippage/latency viability
Risk layer             -> liquidation buffer, stale data, daily controls
Exit layer             -> fixed baseline plus state-conditioned exits
```

## Stage 0 - Project Ledger

- [x] Create this roadmap and start the microstructure revamp branch.
- [x] Keep each implementation substage as its own commit.
- [x] Update this roadmap whenever a substage is completed or deliberately deferred.

## Stage 1 - Microstructure Feature Engine

Goal: make the next-second to next-minute state visible before changing entry logic.

- [x] Add event-time order-flow imbalance windows: `250ms`, `1s`, `5s`.
- [x] Add taker aggression imbalance windows: `1s`, `5s`, `15s`.
- [x] Add microprice and `microprice_mid_bps`.
- [x] Add VAMP / weighted-depth price for available top depth levels.
- [x] Add `vamp_mid_bps` and depth-weighted fair-value deviation.
- [x] Add spread stability metrics over short windows.
- [x] Add bid/ask depth refill and evaporation rates.
- [x] Add mark-last dislocation in bps.
- [x] Add feature fields to `MarketState`, feature logs, decision evidence, and tests.
- [x] Keep the first pass shadow/log-only. No entry loosening in this stage.

Acceptance:

- [x] Unit tests cover OFI, microprice, VAMP, spread stability, refill, and mark-last basis.
- [x] A short Docker watch/recon run writes the new fields without breaking existing logs.

## Stage 2 - Effective Cost Model

Goal: stop judging targets by fees alone.

- [x] Add venue/order-type cost settings: maker fee bps, taker fee bps, expected slippage bps, latency penalty bps.
- [x] Add expected spread crossing cost from current book.
- [x] Compute `expected_round_trip_cost_bps` by candidate entry/exit type.
- [x] Replace thin-target checks with `expected_target_bps >= multiple * expected_total_cost_bps`.
- [x] Log cost decomposition in decisions and shadow trades.

Acceptance:

- [x] Tests prove Binance-style taker/taker 0.10% scalps are blocked when cost multiple fails.
- [x] Tests prove lower-cost or larger-target candidates can pass when all other gates agree.

## Stage 3 - Edge Candidate Router

Goal: split one blended strategy into explicit deterministic candidates.

- [x] Add candidate model with side, family, target bps, stop bps, max hold, expected cost, score, EV proxy, and blockers.
- [x] Implement router that evaluates candidates and chooses highest viable EV candidate.
- [x] Keep current `stateful_momentum` path as a fallback/baseline candidate.
- [x] Log all rejected candidates so rejected-vs-accepted quality can be measured.

Initial candidates:

- [x] `taker_impulse_long`
- [x] `taker_impulse_short`
- [x] `liquidation_continuation_long`
- [x] `liquidation_continuation_short`
- [x] `liquidation_exhaustion_bounce_long`
- [x] `liquidation_exhaustion_bounce_short`

Deferred candidates:

- [ ] `maker_reversion_long`
- [ ] `maker_reversion_short`

Acceptance:

- [x] Decisions include candidate table/evidence.
- [x] Only one accepted candidate can open a paper position at a time.
- [x] Rejected candidates are persisted for later scoring.

## Stage 4 - Taker Impulse Continuation

Goal: make ETHUSDT taker impulse continuation the first primary microstructure strategy.

Entry concept:

- [x] HTF is aligned or non-hostile.
- [x] Book is fresh and synchronized enough for local state.
- [x] Spread is below profile threshold.
- [x] OFI and taker aggression agree over `1s` and `5s`.
- [x] Microprice and VAMP are on the trade side of mid.
- [x] Opposite-side depth is thinning or failing to refill.
- [x] Expected target clears total cost multiple.

Exit concept:

- [x] Fixed TP/stop remains baseline.
- [x] MFE trailing activates only after gross move clears estimated round-trip cost.
- [x] Soft exit on OFI flip, microprice reclaim/loss, spread expansion, or depth disappearance.
- [x] Timeout if impulse does not pay quickly.

Acceptance:

- [x] Shadow-only mode can score impulse candidates for at least one recon run.
- [x] Paper mode can be enabled by env flag after shadow metrics look sane.

## Stage 5 - Liquidation Phase State Machine

Goal: stop treating liquidation prints as complete flow and instead detect forced-flow phase.

- [x] Treat Binance `forceOrder` as largest-pulse event flags, not full volume truth.
- [x] Add liquidation states: normal, pressure_building, liquidation_impulse, cascade_continuation, exhaustion_candidate, reclaim_or_failed_reclaim.
- [x] Combine liquidation pulse, mark-last basis, OFI, taker aggression, depth evaporation/refill, spread expansion/compression, and OI change.
- [x] Prefer cascade continuation while forced flow accelerates.
- [x] Allow countertrend bounce only after exhaustion/reclaim conditions prove out.

Acceptance:

- [x] Tests cover sell-cascade continuation and delayed long-bounce eligibility.
- [x] Recon logs show phase transitions and rejected early bounces.

## Stage 6 - Triple-Barrier Outcome Labels

Goal: score regimes and candidates by what happens first after costs.

- [x] Record MFE/MAE at `1s`, `3s`, `5s`, `10s`, `30s`, `60s`.
- [x] Add candidate outcome labels: target first, stop first, timeout, soft invalidation first.
- [x] Add cost-adjusted target-before-stop matrix.
- [x] Compare accepted candidates against rejected candidates.

Acceptance:

- [x] Summary command reports accepted-vs-rejected quality.
- [x] Higher-score candidates outperform lower-score candidates over replay windows.

## Stage 7 - Hostile Replay

Goal: make paper/replay fills harsher and closer to live trading.

- [x] Add taker fill at ask/bid plus configurable slippage.
- [x] Add latency injection for feed, decision, and order path.
- [x] Add stale-book rejection.
- [x] Add stop execution penalty under fast adverse movement.
- [x] Add optional maker queue approximation before any maker strategy is trusted.

Acceptance:

- [x] Replay can run in normal and hostile modes.
- [ ] Strategy must remain positive under hostile assumptions before live/demo consideration.

## Stage 8 - Cross-Market Confirmation

Goal: use BTC and relative ETH strength as veto/confirmation context.

- [ ] Add optional BTCUSDT public stream context.
- [ ] Add ETH/BTC relative strength proxy.
- [ ] Add BTC contradiction veto for moderate ETH signals.
- [ ] Allow ETH counter-move only when ETH strength is extreme and BTC pressure decelerates.

Acceptance:

- [ ] ETH decisions log BTC confirmation/veto state.
- [ ] Tests cover BTC veto and ETH relative-strength override.

## Stage 9 - Maker Strategy Research

Goal: only consider passive execution in calm, low-toxicity regimes.

- [ ] Add maker candidate as shadow-only.
- [ ] Require stable spread, low OFI, low aggression, low volatility, no liquidation pulse, and acceptable queue estimate.
- [ ] Add adverse-selection penalty to maker fills in hostile replay.

Acceptance:

- [ ] Maker candidate remains disabled by default.
- [ ] Maker results are reported separately from taker impulse results.

## Live-Readiness Gates

No real-money execution until all of these are true:

- [ ] Positive EV after real fee tier.
- [ ] Positive EV after hostile slippage and latency.
- [ ] Positive EV across at least three distinct market regimes.
- [ ] No single event/day explains most profit.
- [ ] Rejected trades perform worse than accepted trades.
- [ ] Accepted high-score trades outperform accepted low-score trades.
- [ ] Paper/demo fills are close enough to simulated fills.
- [ ] Max drawdown remains acceptable under 2x worse slippage.

## Current Running Experiment

- [ ] Analyze `futures-lab-recon_16h_eth_mfe_exit_20260527_091309` after completion.
- [ ] Compare profile-gated MFE paper exits against fixed TP/stop shadow outcomes.
- [ ] Use that result to decide whether MFE trailing remains profile-specific, becomes stricter, or returns to shadow-only.
