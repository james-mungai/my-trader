from dataclasses import dataclass, field

from futures_lab.config import Settings
from futures_lab.markov import MarketRegimeSnapshot, MarketStateMachine
from futures_lab.models import Decision, DecisionAction, MarketState, Regime, TradeMode


@dataclass
class TradeProfile:
    name: str
    mode: TradeMode
    target_move_pct: float
    stop_move_pct: float
    leverage: int


@dataclass
class HitAndRunStrategy:
    settings: Settings
    sequence: MarketStateMachine = field(init=False)

    def __post_init__(self) -> None:
        self.sequence = MarketStateMachine(history_size=self.settings.markov_state_history)

    def decide(self, market: MarketState) -> Decision:
        variant = self.settings.strategy_variant.strip().lower()
        blockers = self._blockers(market)
        evidence = self._evidence(market)
        sequence_snapshot = self._sequence_snapshot(market)
        if sequence_snapshot is not None:
            evidence["market_sequence"] = sequence_snapshot.model_dump()
            evidence["markov_summary"] = self.sequence.model_dump()
        if blockers:
            return Decision(
                symbol=market.symbol,
                action=DecisionAction.wait,
                confidence=0.0,
                reason="; ".join(blockers),
                evidence=evidence,
            )

        assert market.mid_price is not None
        assert market.range_position_180s is not None
        assert market.taker_buy_ratio_10s is not None

        mode = self._choose_mode(market)
        buy_flow = market.taker_buy_ratio_10s
        sell_flow = 1.0 - buy_flow
        book = market.book_imbalance_top or 0.0
        depth = market.depth_imbalance_top5 if market.depth_imbalance_top5 is not None else book
        liquidation_bias = self._liquidation_bias(market)

        long_score, short_score, reasons = self._score_variant(
            variant=variant,
            market=market,
            sequence_snapshot=sequence_snapshot,
            buy_flow=buy_flow,
            sell_flow=sell_flow,
            book=book,
            depth=depth,
            liquidation_bias=liquidation_bias,
        )
        long_score, short_score = self._apply_session_bias(long_score, short_score, evidence)
        stateful_filter = self._stateful_momentum_filter(variant, sequence_snapshot, market, long_score, short_score)
        if stateful_filter is not None:
            evidence["stateful_momentum_filter"] = stateful_filter

        if long_score >= self._required_score("long", short_score, stateful_filter):
            sequence_blocker = self._sequence_blocker(variant, sequence_snapshot, DecisionAction.propose_long, stateful_filter)
            if sequence_blocker:
                return Decision(
                    symbol=market.symbol,
                    action=DecisionAction.wait,
                    confidence=long_score,
                    reason=sequence_blocker,
                    evidence=evidence | {"strategy_variant": variant, "long_score": long_score, "short_score": short_score},
                )
            profile = self._trade_profile(mode, stateful_filter, "long")
            return self._build_trade_decision(
                market=market,
                action=DecisionAction.propose_long,
                profile=profile,
                confidence=long_score,
                reason=reasons["long"],
                evidence=evidence | {"strategy_variant": variant, "long_score": long_score, "short_score": short_score},
            )

        short_blockers = self._short_blockers(market)
        if short_score >= self._required_score("short", long_score, stateful_filter):
            if short_blockers:
                return Decision(
                    symbol=market.symbol,
                    action=DecisionAction.wait,
                    confidence=short_score,
                    reason="; ".join(short_blockers),
                    evidence=evidence | {"strategy_variant": variant, "long_score": long_score, "short_score": short_score},
                )
            sequence_blocker = self._sequence_blocker(variant, sequence_snapshot, DecisionAction.propose_short, stateful_filter)
            if sequence_blocker:
                return Decision(
                    symbol=market.symbol,
                    action=DecisionAction.wait,
                    confidence=short_score,
                    reason=sequence_blocker,
                    evidence=evidence | {"strategy_variant": variant, "long_score": long_score, "short_score": short_score},
                )
            profile = self._trade_profile(mode, stateful_filter, "short")
            return self._build_trade_decision(
                market=market,
                action=DecisionAction.propose_short,
                profile=profile,
                confidence=short_score,
                reason=reasons["short"],
                evidence=evidence | {"strategy_variant": variant, "long_score": long_score, "short_score": short_score},
            )

        return Decision(
            symbol=market.symbol,
            action=DecisionAction.wait,
            confidence=max(long_score, short_score),
            reason=self._wait_reason(stateful_filter),
            evidence=evidence | {"strategy_variant": variant, "long_score": long_score, "short_score": short_score},
        )

    def sequence_summary(self) -> dict:
        return self.sequence.model_dump()

    def _sequence_snapshot(self, market: MarketState) -> MarketRegimeSnapshot | None:
        if not self.settings.enable_markov_state_machine:
            return None
        return self.sequence.update(market)

    def _sequence_blocker(
        self,
        variant: str,
        snapshot: MarketRegimeSnapshot | None,
        action: DecisionAction,
        stateful_filter: dict | None = None,
    ) -> str | None:
        if variant not in {"stateful_momentum", "markov_momentum", "fsm_momentum"}:
            return None
        if stateful_filter is not None and stateful_filter.get("blocked"):
            return self._wait_reason(stateful_filter)
        if snapshot is None:
            return "stateful momentum blocked: sequence monitor disabled"
        if action == DecisionAction.propose_long and snapshot.allows_long:
            return None
        if action == DecisionAction.propose_short and snapshot.allows_short:
            return None
        path = " > ".join(state.value for state in snapshot.path)
        return f"stateful momentum waiting for confirmed sequence; state={snapshot.state.value}; path={path}"

    def _stateful_momentum_filter(
        self,
        variant: str,
        snapshot: MarketRegimeSnapshot | None,
        market: MarketState,
        long_score: float,
        short_score: float,
    ) -> dict | None:
        if variant not in {"stateful_momentum", "markov_momentum", "fsm_momentum"}:
            return None
        if snapshot is None or not (snapshot.allows_long or snapshot.allows_short):
            return None

        side = "long" if snapshot.allows_long else "short"
        score = long_score if snapshot.allows_long else short_score
        opposing_score = short_score if snapshot.allows_long else long_score
        adaptive_gate = self._adaptive_low_range_gate(side, snapshot, market, score, opposing_score)
        adaptive_entry_allowed = adaptive_gate["allowed"]
        min_score = self.settings.stateful_adaptive_min_score if adaptive_entry_allowed else self.settings.min_confidence
        required_score = max(min_score, opposing_score + 0.04)
        target_feasible = self._target_move_feasible(market)
        blockers = []
        if not target_feasible and not adaptive_entry_allowed:
            blockers.append("target_feasibility")
        if score < min_score:
            blockers.append("min_confidence")
        if score < opposing_score + 0.04:
            blockers.append("side_separation")

        row = {
            "confirmed": True,
            "side": side,
            "state": snapshot.state.value,
            "score": score,
            "opposing_score": opposing_score,
            "min_confidence": min_score,
            "base_min_confidence": self.settings.min_confidence,
            "required_score": round(required_score, 4),
            "blocked": bool(blockers),
            "blockers": blockers,
            "target_feasible": target_feasible,
            "adaptive_entry_allowed": adaptive_entry_allowed,
            "adaptive_gate": adaptive_gate,
            "range_180s_pct": market.range_180s_pct,
            "min_required_range_180s_pct": self._min_target_feasible_range_pct(),
            "min_adaptive_range_180s_pct": self._min_adaptive_range_pct(),
            "target_move_pct": self.settings.fast_target_move_pct,
            "sequence_confidence": snapshot.confidence,
            "path": [state.value for state in snapshot.path],
        }
        if row["blocked"]:
            shadow = self._shadow_trade_signal(side, market, score, blockers)
            if shadow is not None:
                row["shadow_trade"] = shadow
        return row

    def _wait_reason(self, stateful_filter: dict | None) -> str:
        if stateful_filter is None or not stateful_filter.get("blocked"):
            return "No asymmetric hit-and-run edge yet."
        if "target_feasibility" in stateful_filter["blockers"]:
            return (
                "stateful momentum confirmed but adaptive target feasibility blocked: "
                f"range_180s_pct={stateful_filter['range_180s_pct']} "
                f"< required={stateful_filter['min_required_range_180s_pct']}"
            )
        return (
            "stateful momentum confirmed but score blocked: "
            f"score={stateful_filter['score']} < required={stateful_filter['required_score']}"
        )

    def _blockers(self, market: MarketState) -> list[str]:
        blockers = []
        if not market.connected:
            blockers.append("stream disconnected")
        if market.regime in {Regime.stale, Regime.warming_up, Regime.unknown}:
            blockers.append(f"regime={market.regime.value}")
        if market.mid_price is None:
            blockers.append("missing mid price")
        if market.spread_bps is None or market.spread_bps > self.settings.max_spread_bps:
            blockers.append(f"spread not tradable: {market.spread_bps}")
        if market.range_position_180s is None:
            blockers.append("missing range position")
        if market.taker_buy_ratio_10s is None:
            blockers.append("missing taker flow")
        if market.regime == Regime.volatile:
            blockers.append("volatility regime too unstable")
        return blockers

    def _apply_session_bias(self, long_score: float, short_score: float, evidence: dict) -> tuple[float, float]:
        side = self.settings.session_bias_side.strip().lower()
        strength = self._clamp(self.settings.session_bias_strength)
        if side not in {"long", "short", "neutral"}:
            side = "neutral"
        evidence["session_context"] = {
            "bias_side": side,
            "bias_strength": strength,
            "bias_reason": self.settings.session_bias_reason,
        }
        if side == "long" and strength > 0:
            return round(min(1.0, long_score + 0.02 * strength), 4), round(max(0.0, short_score - 0.01 * strength), 4)
        if side == "short" and strength > 0:
            return round(max(0.0, long_score - 0.01 * strength), 4), round(min(1.0, short_score + 0.02 * strength), 4)
        return long_score, short_score

    def _short_blockers(self, market: MarketState) -> list[str]:
        blockers = []
        if (
            market.taker_buy_ratio_30s is not None
            and market.taker_buy_ratio_30s > self.settings.max_short_taker_buy_ratio_30s
        ):
            blockers.append(
                "short blocked: 30s taker buy ratio "
                f"{market.taker_buy_ratio_30s:.3f} > {self.settings.max_short_taker_buy_ratio_30s:.3f}"
            )
        return blockers

    def _choose_mode(self, market: MarketState) -> TradeMode:
        # Fast mode is the user's 200x / 0.5% target design. Slow mode lowers leverage
        # and widens the target when market motion is directional or slower.
        if market.regime == Regime.sideways:
            return TradeMode.fast
        return TradeMode.slow

    def _score_variant(
        self,
        variant: str,
        market: MarketState,
        sequence_snapshot: MarketRegimeSnapshot | None,
        buy_flow: float,
        sell_flow: float,
        book: float,
        depth: float,
        liquidation_bias: float,
    ) -> tuple[float, float, dict[str, str]]:
        if variant in {"liquidity_sweep_reversal", "sweep_reversal", "sweep"}:
            return (
                self._score_sweep_long(market, buy_flow, book, depth, liquidation_bias),
                self._score_sweep_short(market, sell_flow, book, depth, liquidation_bias),
                {
                    "long": "Liquidity-sweep long: range-low reclaim with buy-flow recovery and bid-side depth support.",
                    "short": "Liquidity-sweep short: range-high rejection with sell-flow recovery and ask-side depth support.",
                },
            )
        if variant in {"momentum_pullback", "momentum_pullback_continuation", "pullback"}:
            return (
                self._score_momentum_long(market, buy_flow, book, depth, liquidation_bias),
                self._score_momentum_short(market, sell_flow, book, depth, liquidation_bias),
                {
                    "long": "Momentum-pullback long: bullish short-horizon structure after a controlled pullback.",
                    "short": "Momentum-pullback short: bearish short-horizon structure after a controlled bounce.",
                },
            )
        if variant in {"stateful_momentum", "markov_momentum", "fsm_momentum"}:
            return (
                self._score_stateful_momentum_long(market, sequence_snapshot, buy_flow, book, depth, liquidation_bias),
                self._score_stateful_momentum_short(market, sequence_snapshot, sell_flow, book, depth, liquidation_bias),
                {
                    "long": "Stateful momentum long: confirmed continuation after impulse, pullback, and reclaim.",
                    "short": "Stateful momentum short: confirmed continuation after impulse, bounce, and rejection.",
                },
            )
        return (
            self._score_long(market, buy_flow, book, depth, liquidation_bias),
            self._score_short(market, sell_flow, book, depth, liquidation_bias),
            {
                "long": "Hit-and-run long: range-low location with positive taker flow and supportive top-book pressure.",
                "short": "Hit-and-run short: range-high location with positive taker-sell flow and supportive top-book pressure.",
            },
        )

    def _score_long(
        self,
        market: MarketState,
        buy_flow: float,
        book_imbalance: float,
        depth_imbalance: float,
        liquidation_bias: float,
    ) -> float:
        assert market.range_position_180s is not None
        range_edge = max(0.0, 1.0 - market.range_position_180s / 0.28)
        flow = max(0.0, min(1.0, (buy_flow - 0.52) / 0.28))
        book = max(0.0, min(1.0, (book_imbalance + 0.20) / 0.40))
        depth = max(0.0, min(1.0, (depth_imbalance + 0.20) / 0.40))
        liquidation = max(0.0, min(1.0, (liquidation_bias + 1.0) / 2.0))
        reversion_ok = 1.0 if (market.return_15s_pct or 0.0) > -0.0015 else 0.5
        regime = 1.0 if market.regime == Regime.sideways else 0.7
        return round(
            0.32 * range_edge
            + 0.30 * flow
            + 0.12 * book
            + 0.08 * depth
            + 0.05 * liquidation
            + 0.08 * reversion_ok
            + 0.05 * regime,
            4,
        )

    def _score_short(
        self,
        market: MarketState,
        sell_flow: float,
        book_imbalance: float,
        depth_imbalance: float,
        liquidation_bias: float,
    ) -> float:
        assert market.range_position_180s is not None
        range_edge = max(0.0, 1.0 - (1.0 - market.range_position_180s) / 0.28)
        flow = max(0.0, min(1.0, (sell_flow - 0.52) / 0.28))
        book = max(0.0, min(1.0, (-book_imbalance + 0.20) / 0.40))
        depth = max(0.0, min(1.0, (-depth_imbalance + 0.20) / 0.40))
        liquidation = max(0.0, min(1.0, (-liquidation_bias + 1.0) / 2.0))
        reversion_ok = 1.0 if (market.return_15s_pct or 0.0) < 0.0015 else 0.5
        regime = 1.0 if market.regime == Regime.sideways else 0.7
        return round(
            0.32 * range_edge
            + 0.30 * flow
            + 0.12 * book
            + 0.08 * depth
            + 0.05 * liquidation
            + 0.08 * reversion_ok
            + 0.05 * regime,
            4,
        )

    def _liquidation_bias(self, market: MarketState) -> float:
        buy = market.short_liquidation_notional_30s or 0.0
        sell = market.long_liquidation_notional_30s or 0.0
        total = buy + sell
        if total <= 0:
            return 0.0
        return (buy - sell) / total

    def _score_sweep_long(
        self,
        market: MarketState,
        buy_flow: float,
        book_imbalance: float,
        depth_imbalance: float,
        liquidation_bias: float,
    ) -> float:
        assert market.range_position_180s is not None
        sweep_location = max(0.0, 1.0 - market.range_position_180s / 0.35)
        reclaim = max(0.0, min(1.0, ((market.return_15s_pct or 0.0) + 0.0002) / 0.0012))
        prior_pressure = max(0.0, min(1.0, abs(min(0.0, market.return_60s_pct or 0.0)) / 0.0015))
        flow_flip = max(0.0, min(1.0, (buy_flow - 0.56) / 0.24))
        book = max(0.0, min(1.0, (book_imbalance + 0.15) / 0.35))
        depth = max(0.0, min(1.0, (depth_imbalance + 0.10) / 0.45))
        liquidation = max(0.0, min(1.0, (liquidation_bias + 1.0) / 2.0))
        return round(
            0.24 * sweep_location
            + 0.18 * reclaim
            + 0.12 * prior_pressure
            + 0.22 * flow_flip
            + 0.10 * book
            + 0.09 * depth
            + 0.05 * liquidation,
            4,
        )

    def _score_sweep_short(
        self,
        market: MarketState,
        sell_flow: float,
        book_imbalance: float,
        depth_imbalance: float,
        liquidation_bias: float,
    ) -> float:
        assert market.range_position_180s is not None
        sweep_location = max(0.0, 1.0 - (1.0 - market.range_position_180s) / 0.35)
        rejection = max(0.0, min(1.0, (0.0002 - (market.return_15s_pct or 0.0)) / 0.0012))
        prior_pressure = max(0.0, min(1.0, max(0.0, market.return_60s_pct or 0.0) / 0.0015))
        flow_flip = max(0.0, min(1.0, (sell_flow - 0.56) / 0.24))
        book = max(0.0, min(1.0, (-book_imbalance + 0.15) / 0.35))
        depth = max(0.0, min(1.0, (-depth_imbalance + 0.10) / 0.45))
        liquidation = max(0.0, min(1.0, (-liquidation_bias + 1.0) / 2.0))
        return round(
            0.24 * sweep_location
            + 0.18 * rejection
            + 0.12 * prior_pressure
            + 0.22 * flow_flip
            + 0.10 * book
            + 0.09 * depth
            + 0.05 * liquidation,
            4,
        )

    def _score_momentum_long(
        self,
        market: MarketState,
        buy_flow: float,
        book_imbalance: float,
        depth_imbalance: float,
        liquidation_bias: float,
    ) -> float:
        trend = max(0.0, min(1.0, (market.return_180s_pct or 0.0) / 0.003))
        pullback_control = max(0.0, min(1.0, (0.0008 + (market.return_15s_pct or 0.0)) / 0.0016))
        flow = max(0.0, min(1.0, (buy_flow - 0.55) / 0.25))
        structure = 1.0 if 0.30 <= (market.range_position_180s or 0.0) <= 0.78 else 0.35
        book = max(0.0, min(1.0, (book_imbalance + 0.10) / 0.35))
        depth = max(0.0, min(1.0, (depth_imbalance + 0.10) / 0.45))
        oi = max(0.0, min(1.0, ((market.open_interest_change_5m_pct or 0.0) + 0.001) / 0.003))
        liquidation = max(0.0, min(1.0, (liquidation_bias + 1.0) / 2.0))
        return round(
            0.24 * trend
            + 0.14 * pullback_control
            + 0.20 * flow
            + 0.12 * structure
            + 0.10 * book
            + 0.08 * depth
            + 0.07 * oi
            + 0.05 * liquidation,
            4,
        )

    def _score_momentum_short(
        self,
        market: MarketState,
        sell_flow: float,
        book_imbalance: float,
        depth_imbalance: float,
        liquidation_bias: float,
    ) -> float:
        trend = max(0.0, min(1.0, abs(min(0.0, market.return_180s_pct or 0.0)) / 0.003))
        bounce_control = max(0.0, min(1.0, (0.0008 - (market.return_15s_pct or 0.0)) / 0.0016))
        flow = max(0.0, min(1.0, (sell_flow - 0.55) / 0.25))
        structure = 1.0 if 0.22 <= (market.range_position_180s or 0.0) <= 0.70 else 0.35
        book = max(0.0, min(1.0, (-book_imbalance + 0.10) / 0.35))
        depth = max(0.0, min(1.0, (-depth_imbalance + 0.10) / 0.45))
        oi = max(0.0, min(1.0, ((market.open_interest_change_5m_pct or 0.0) + 0.001) / 0.003))
        liquidation = max(0.0, min(1.0, (-liquidation_bias + 1.0) / 2.0))
        return round(
            0.24 * trend
            + 0.14 * bounce_control
            + 0.20 * flow
            + 0.12 * structure
            + 0.10 * book
            + 0.08 * depth
            + 0.07 * oi
            + 0.05 * liquidation,
            4,
        )

    def _score_stateful_momentum_long(
        self,
        market: MarketState,
        sequence_snapshot: MarketRegimeSnapshot | None,
        buy_flow: float,
        book_imbalance: float,
        depth_imbalance: float,
        liquidation_bias: float,
    ) -> float:
        base = self._score_momentum_long(market, buy_flow, book_imbalance, depth_imbalance, liquidation_bias)
        if sequence_snapshot is None or not sequence_snapshot.allows_long:
            return base

        trend = self._clamp((market.return_180s_pct or 0.0) / 0.003)
        reacceleration = self._clamp((market.return_15s_pct or 0.0) / 0.00035)
        flow = self._clamp((buy_flow - 0.58) / 0.35)
        pressure = self._clamp(((book_imbalance + depth_imbalance) / 2.0 + 0.10) / 0.50)
        range_breakout = self._long_breakout_location(market.range_position_180s or 0.0)
        oi = self._clamp(((market.open_interest_change_5m_pct or 0.0) + 0.001) / 0.003)
        liquidation = self._clamp((liquidation_bias + 1.0) / 2.0)

        continuation = (
            0.22 * sequence_snapshot.confidence
            + 0.20 * trend
            + 0.16 * reacceleration
            + 0.16 * flow
            + 0.12 * pressure
            + 0.09 * range_breakout
            + 0.03 * oi
            + 0.02 * liquidation
        )
        return round(max(base, continuation), 4)

    def _score_stateful_momentum_short(
        self,
        market: MarketState,
        sequence_snapshot: MarketRegimeSnapshot | None,
        sell_flow: float,
        book_imbalance: float,
        depth_imbalance: float,
        liquidation_bias: float,
    ) -> float:
        base = self._score_momentum_short(market, sell_flow, book_imbalance, depth_imbalance, liquidation_bias)
        if sequence_snapshot is None or not sequence_snapshot.allows_short:
            return base

        trend = self._clamp(abs(min(0.0, market.return_180s_pct or 0.0)) / 0.003)
        reacceleration = self._clamp(abs(min(0.0, market.return_15s_pct or 0.0)) / 0.00035)
        flow = self._clamp((sell_flow - 0.58) / 0.35)
        pressure = self._clamp((-(book_imbalance + depth_imbalance) / 2.0 + 0.10) / 0.50)
        range_breakout = self._short_breakout_location(market.range_position_180s or 0.0)
        oi = self._clamp(((market.open_interest_change_5m_pct or 0.0) + 0.001) / 0.003)
        liquidation = self._clamp((-liquidation_bias + 1.0) / 2.0)

        continuation = (
            0.22 * sequence_snapshot.confidence
            + 0.20 * trend
            + 0.16 * reacceleration
            + 0.16 * flow
            + 0.12 * pressure
            + 0.09 * range_breakout
            + 0.03 * oi
            + 0.02 * liquidation
        )
        return round(max(base, continuation), 4)

    def _long_breakout_location(self, range_position: float) -> float:
        if range_position >= 0.78:
            return 1.0
        if range_position >= 0.55:
            return 0.75
        if range_position >= 0.30:
            return 0.55
        return 0.20

    def _short_breakout_location(self, range_position: float) -> float:
        if range_position <= 0.22:
            return 1.0
        if range_position <= 0.45:
            return 0.75
        if range_position <= 0.70:
            return 0.55
        return 0.20

    def _target_move_feasible(self, market: MarketState) -> bool:
        return (market.range_180s_pct or 0.0) >= self._min_target_feasible_range_pct()

    def _min_target_feasible_range_pct(self) -> float:
        return self.settings.fast_target_move_pct * self.settings.stateful_target_feasibility_fraction

    def _min_adaptive_range_pct(self) -> float:
        return self.settings.fast_target_move_pct * self.settings.stateful_adaptive_target_fraction

    def _adaptive_low_range_gate(
        self,
        side: str,
        snapshot: MarketRegimeSnapshot,
        market: MarketState,
        score: float,
        opposing_score: float,
    ) -> dict:
        blockers = []
        range_pct = market.range_180s_pct or 0.0
        target_feasible = range_pct >= self._min_target_feasible_range_pct()
        adaptive_range = range_pct >= self._min_adaptive_range_pct()
        trend = market.return_180s_pct or 0.0
        trend_agrees = trend > 0 if side == "long" else trend < 0
        strong_structure = (
            market.regime == Regime.directional
            or (adaptive_range and trend_agrees and abs(trend) >= self.settings.stateful_adaptive_target_move_pct * 0.50)
        )
        flow_agrees = self._flow_agrees(side, market)

        if target_feasible:
            blockers.append("target_already_feasible")
        if not adaptive_range:
            blockers.append("insufficient_adaptive_range")
        if not strong_structure:
            blockers.append("weak_structure")
        if not flow_agrees:
            blockers.append("flow_not_confirmed")
        if snapshot.confidence < self.settings.stateful_adaptive_min_sequence_confidence:
            blockers.append("sequence_confidence")
        if score < self.settings.stateful_adaptive_min_score:
            blockers.append("adaptive_min_score")
        if score < opposing_score + 0.04:
            blockers.append("side_separation")

        return {
            "allowed": not blockers,
            "blockers": blockers,
            "range_180s_pct": market.range_180s_pct,
            "min_adaptive_range_180s_pct": self._min_adaptive_range_pct(),
            "target_feasible": target_feasible,
            "strong_structure": strong_structure,
            "flow_agrees": flow_agrees,
            "sequence_confidence": snapshot.confidence,
            "min_sequence_confidence": self.settings.stateful_adaptive_min_sequence_confidence,
            "adaptive_min_score": self.settings.stateful_adaptive_min_score,
        }

    def _flow_agrees(self, side: str, market: MarketState) -> bool:
        buy_10s = market.taker_buy_ratio_10s
        buy_30s = market.taker_buy_ratio_30s
        book = market.book_imbalance_top or 0.0
        depth = market.depth_imbalance_top5 if market.depth_imbalance_top5 is not None else book
        pressure = (book + depth) / 2.0
        if buy_10s is None or buy_30s is None:
            return False
        if side == "long":
            return buy_10s >= 0.66 and buy_30s >= 0.54 and pressure >= 0.05
        return buy_10s <= 0.34 and buy_30s <= self.settings.max_short_taker_buy_ratio_30s and pressure <= -0.05

    def _required_score(self, side: str, opposing_score: float, stateful_filter: dict | None) -> float:
        min_score = self.settings.min_confidence
        if (
            stateful_filter is not None
            and stateful_filter.get("side") == side
            and stateful_filter.get("adaptive_entry_allowed")
        ):
            min_score = min(min_score, self.settings.stateful_adaptive_min_score)
        return max(min_score, opposing_score + 0.04)

    def _trade_profile(self, mode: TradeMode, stateful_filter: dict | None, side: str) -> TradeProfile:
        if (
            stateful_filter is not None
            and stateful_filter.get("side") == side
            and stateful_filter.get("adaptive_entry_allowed")
            and not stateful_filter.get("target_feasible")
        ):
            return TradeProfile(
                name="adaptive_low_range",
                mode=TradeMode.slow,
                target_move_pct=self.settings.stateful_adaptive_target_move_pct,
                stop_move_pct=self.settings.stateful_adaptive_stop_move_pct,
                leverage=self.settings.slow_leverage,
            )
        if mode == TradeMode.fast:
            return TradeProfile(
                name="fast",
                mode=TradeMode.fast,
                target_move_pct=self.settings.fast_target_move_pct,
                stop_move_pct=self.settings.fast_stop_move_pct,
                leverage=self.settings.fast_leverage,
            )
        return TradeProfile(
            name="slow",
            mode=TradeMode.slow,
            target_move_pct=self.settings.slow_target_move_pct,
            stop_move_pct=self.settings.slow_stop_move_pct,
            leverage=self.settings.slow_leverage,
        )

    def _shadow_trade_signal(self, side: str, market: MarketState, confidence: float, blockers: list[str]) -> dict | None:
        if market.mid_price is None:
            return None
        direction = 1 if side == "long" else -1
        target = self.settings.fast_target_move_pct
        stop = self.settings.fast_stop_move_pct
        return {
            "source": "blocked_stateful_continuation",
            "side": side,
            "confidence": confidence,
            "blocked_by": blockers,
            "entry_price": market.mid_price,
            "take_profit_price": market.mid_price * (1 + direction * target),
            "stop_loss_price": market.mid_price * (1 - direction * stop),
            "target_move_pct": target,
            "stop_move_pct": stop,
            "leverage": self.settings.fast_leverage,
            "stake_usd": self.settings.stake_usd,
            "notional_usd": self.settings.stake_usd * self.settings.fast_leverage,
        }

    def _clamp(self, value: float) -> float:
        return max(0.0, min(1.0, value))

    def _build_trade_decision(
        self,
        market: MarketState,
        action: DecisionAction,
        profile: TradeProfile,
        confidence: float,
        reason: str,
        evidence: dict,
    ) -> Decision:
        assert market.mid_price is not None
        direction = 1 if action == DecisionAction.propose_long else -1
        target = profile.target_move_pct
        stop = profile.stop_move_pct
        leverage = profile.leverage
        stake = self.settings.stake_usd
        notional = stake * leverage
        return Decision(
            symbol=market.symbol,
            action=action,
            mode=profile.mode,
            confidence=confidence,
            reason=reason,
            entry_price=market.mid_price,
            take_profit_price=market.mid_price * (1 + direction * target),
            stop_loss_price=market.mid_price * (1 - direction * stop),
            target_move_pct=target,
            stop_move_pct=stop,
            leverage=leverage,
            stake_usd=stake,
            notional_usd=notional,
            evidence=evidence | {"trade_profile": profile.name},
        )

    def _evidence(self, market: MarketState) -> dict:
        return {
            "regime": market.regime.value,
            "mid_price": market.mid_price,
            "spread_bps": market.spread_bps,
            "range_position_180s": market.range_position_180s,
            "range_180s_pct": market.range_180s_pct,
            "return_15s_pct": market.return_15s_pct,
            "return_60s_pct": market.return_60s_pct,
            "return_180s_pct": market.return_180s_pct,
            "realized_vol_60s_pct": market.realized_vol_60s_pct,
            "taker_buy_ratio_10s": market.taker_buy_ratio_10s,
            "taker_buy_ratio_30s": market.taker_buy_ratio_30s,
            "book_imbalance_top": market.book_imbalance_top,
            "depth_imbalance_top5": market.depth_imbalance_top5,
            "depth_bid_qty_top5": market.depth_bid_qty_top5,
            "depth_ask_qty_top5": market.depth_ask_qty_top5,
            "liquidation_notional_30s": market.liquidation_notional_30s,
            "long_liquidation_notional_30s": market.long_liquidation_notional_30s,
            "short_liquidation_notional_30s": market.short_liquidation_notional_30s,
            "liquidation_buy_ratio_30s": market.liquidation_buy_ratio_30s,
            "open_interest": market.open_interest,
            "open_interest_change_5m_pct": market.open_interest_change_5m_pct,
            "exchange_event_lag_ms": market.exchange_event_lag_ms,
            "avg_event_lag_30s_ms": market.avg_event_lag_30s_ms,
            "funding_rate": market.funding_rate,
        }

