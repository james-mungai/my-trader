from dataclasses import dataclass, field

from futures_lab.config import Settings
from futures_lab.costs import estimate_effective_cost
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
    recent_stateful_signals: dict[str, dict] = field(init=False)

    def __post_init__(self) -> None:
        self.sequence = MarketStateMachine(history_size=self.settings.markov_state_history)
        self.recent_stateful_signals = {}

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
        long_score, short_score = self._apply_higher_timeframe_context(long_score, short_score, market, evidence)
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
        target_feasible = self._target_move_feasible(market)
        higher_timeframe_gate = self._higher_timeframe_execution_gate(
            side=side,
            snapshot=snapshot,
            market=market,
            score=score,
            adaptive_gate=adaptive_gate,
            target_feasible=target_feasible,
        )
        if higher_timeframe_gate.get("profile") == "eth_counter_htf_bounce":
            min_score = min(min_score, self.settings.counter_htf_bounce_min_score)
        required_score = max(min_score, opposing_score + 0.04)
        local_execution_gate = self._local_execution_gate(side, market, higher_timeframe_gate)
        fee_edge_gate = self._fee_edge_quality_gate(
            snapshot=snapshot,
            adaptive_gate=adaptive_gate,
            market=market,
            target_move_pct=float(higher_timeframe_gate["target_move_pct"]),
            trade_profile=str(higher_timeframe_gate.get("profile") or "fast"),
        )
        duplicate_gate = self._duplicate_signal_gate(
            side=side,
            snapshot=snapshot,
            market=market,
            score=score,
            quality_score=float(adaptive_gate.get("quality_score") or 0.0),
            profile=str(higher_timeframe_gate.get("profile") or "fast"),
        )
        entry_follow_through_gate = self._entry_follow_through_gate(
            side=side,
            market=market,
            target_move_pct=float(higher_timeframe_gate["target_move_pct"]),
            trade_profile=str(higher_timeframe_gate.get("profile") or "fast"),
        )
        weak_neutral_short_gate = self._weak_neutral_short_gate(
            side=side,
            snapshot=snapshot,
            adaptive_gate=adaptive_gate,
            higher_timeframe_gate=higher_timeframe_gate,
            local_execution_gate=local_execution_gate,
            entry_follow_through_gate=entry_follow_through_gate,
        )
        weak_neutral_long_gate = self._weak_neutral_long_gate(
            side=side,
            market=market,
            snapshot=snapshot,
            adaptive_gate=adaptive_gate,
            higher_timeframe_gate=higher_timeframe_gate,
            entry_follow_through_gate=entry_follow_through_gate,
        )
        blockers = []
        if not target_feasible and not adaptive_entry_allowed:
            blockers.append("target_feasibility")
        if score < min_score:
            blockers.append("min_confidence")
        if score < opposing_score + 0.04:
            blockers.append("side_separation")
        if not higher_timeframe_gate["allowed"]:
            blockers.append(higher_timeframe_gate["blocker"])
        if not local_execution_gate["allowed"]:
            blockers.append(local_execution_gate["blocker"])
        if not fee_edge_gate["allowed"]:
            blockers.append(fee_edge_gate["blocker"])
        if not duplicate_gate["allowed"]:
            blockers.append(duplicate_gate["blocker"])
        if not entry_follow_through_gate["allowed"]:
            blockers.append(entry_follow_through_gate["blocker"])
        if not weak_neutral_short_gate["allowed"]:
            blockers.append(weak_neutral_short_gate["blocker"])
        if not weak_neutral_long_gate["allowed"]:
            blockers.append(weak_neutral_long_gate["blocker"])

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
            "higher_timeframe_gate": higher_timeframe_gate,
            "local_execution_gate": local_execution_gate,
            "fee_edge_gate": fee_edge_gate,
            "duplicate_signal_gate": duplicate_gate,
            "entry_follow_through_gate": entry_follow_through_gate,
            "weak_neutral_short_gate": weak_neutral_short_gate,
            "weak_neutral_long_gate": weak_neutral_long_gate,
            "range_180s_pct": market.range_180s_pct,
            "min_required_range_180s_pct": self._min_target_feasible_range_pct(),
            "min_adaptive_range_180s_pct": self._min_adaptive_range_pct(),
            "target_move_pct": higher_timeframe_gate["target_move_pct"],
            "sequence_confidence": snapshot.confidence,
            "path": [state.value for state in snapshot.path],
        }
        if not row["blocked"]:
            self._remember_stateful_signal(
                side=side,
                snapshot=snapshot,
                market=market,
                score=score,
                quality_score=float(adaptive_gate.get("quality_score") or 0.0),
                profile=str(higher_timeframe_gate.get("profile") or "fast"),
            )
        if row["blocked"] and "duplicate_signal" not in blockers:
            shadow = self._shadow_trade_signal(side, market, score, blockers, fee_edge_gate)
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
        if "higher_timeframe_countertrend" in stateful_filter["blockers"]:
            gate = stateful_filter.get("higher_timeframe_gate") or {}
            return (
                "stateful momentum confirmed but higher-timeframe gate blocked countertrend trade: "
                f"bias={gate.get('bias_side')} strength={gate.get('bias_strength')}"
            )
        if "local_execution_countertrend" in stateful_filter["blockers"]:
            gate = stateful_filter.get("local_execution_gate") or {}
            return (
                "stateful momentum confirmed but local execution gate blocked early trigger: "
                f"5m={gate.get('structure_5m')} trend={gate.get('trend_score_5m')}"
            )
        if "fee_edge_quality" in stateful_filter["blockers"]:
            gate = stateful_filter.get("fee_edge_gate") or {}
            return (
                "stateful momentum confirmed but fee-edge quality gate blocked thin target: "
                f"quality={gate.get('quality_score')} required={gate.get('min_quality')}"
            )
        if "duplicate_signal" in stateful_filter["blockers"]:
            gate = stateful_filter.get("duplicate_signal_gate") or {}
            return (
                "stateful momentum confirmed but duplicate signal throttle blocked repeat entry: "
                f"elapsed={gate.get('elapsed_seconds')}s"
            )
        if "entry_follow_through" in stateful_filter["blockers"]:
            gate = stateful_filter.get("entry_follow_through_gate") or {}
            return (
                "stateful momentum confirmed but entry follow-through gate blocked early entry: "
                f"confirmations={gate.get('confirmations')} score={gate.get('score')}"
            )
        if "weak_neutral_long_quality" in stateful_filter["blockers"]:
            gate = stateful_filter.get("weak_neutral_long_gate") or {}
            return (
                "stateful momentum confirmed but weak/neutral long gate blocked trade: "
                f"quality={gate.get('quality_score')} trend_1h={gate.get('trend_score_1h')}"
            )
        if "weak_neutral_short_quality" in stateful_filter["blockers"]:
            gate = stateful_filter.get("weak_neutral_short_gate") or {}
            return (
                "stateful momentum confirmed but weak/neutral short gate blocked trade: "
                f"quality={gate.get('quality_score')} follow={gate.get('follow_score')}"
            )
        return (
            "stateful momentum confirmed but score blocked: "
            f"score={stateful_filter['score']} < required={stateful_filter['required_score']}"
        )

    def _fee_edge_quality_gate(
        self,
        snapshot: MarketRegimeSnapshot,
        adaptive_gate: dict,
        market: MarketState,
        target_move_pct: float,
        trade_profile: str,
    ) -> dict:
        effective_cost = estimate_effective_cost(self.settings, market)
        required_target_pct = effective_cost.required_target_pct(self.settings.min_gross_target_fee_multiple)
        quality_score = float(adaptive_gate.get("quality_score") or 0.0)
        fee_buffer = max(1.0, self.settings.fee_edge_target_fee_buffer)
        fee_thin_target = target_move_pct <= required_target_pct * fee_buffer
        enabled = self.settings.fee_edge_quality_gate_enabled
        min_quality = self.settings.fee_edge_fast_min_quality
        min_sequence_confidence = self.settings.fee_edge_fast_min_sequence_confidence
        if trade_profile == "eth_counter_htf_bounce":
            min_quality = self.settings.fee_edge_counter_htf_bounce_min_quality
            min_sequence_confidence = self.settings.fee_edge_counter_htf_bounce_min_sequence_confidence
        allowed = True
        blocker = None
        if enabled and fee_thin_target:
            allowed = (
                quality_score >= min_quality
                and snapshot.confidence >= min_sequence_confidence
            )
            if not allowed:
                blocker = "fee_edge_quality"
        return {
            "enabled": enabled,
            "allowed": allowed,
            "blocker": blocker,
            "target_move_pct": target_move_pct,
            "round_trip_fee_pct": effective_cost.total_cost_pct,
            "required_target_pct": required_target_pct,
            "fee_buffer": fee_buffer,
            "fee_thin_target": fee_thin_target,
            "trade_profile": trade_profile,
            "effective_cost": effective_cost.model_dump(),
            "quality_score": quality_score,
            "min_quality": min_quality,
            "sequence_confidence": snapshot.confidence,
            "min_sequence_confidence": min_sequence_confidence,
        }

    def _duplicate_signal_gate(
        self,
        side: str,
        snapshot: MarketRegimeSnapshot,
        market: MarketState,
        score: float,
        quality_score: float,
        profile: str,
    ) -> dict:
        key = self._stateful_signal_key(market.symbol, side, snapshot, profile)
        previous = self.recent_stateful_signals.get(key)
        previous_meta = None
        if previous is not None:
            previous_meta = {
                "seen_at": previous["seen_at"].isoformat(),
                "score": previous["score"],
                "quality_score": previous["quality_score"],
                "state": previous["state"],
                "path": previous["path"],
            }
        base = {
            "enabled": self.settings.duplicate_signal_suppression_seconds > 0,
            "allowed": True,
            "blocker": None,
            "key": key,
            "profile": profile,
            "previous": previous_meta,
            "score": score,
            "quality_score": quality_score,
            "min_quality_improvement": self.settings.duplicate_signal_min_quality_improvement,
            "min_score_improvement": self.settings.duplicate_signal_min_score_improvement,
        }
        if not base["enabled"] or previous is None or market.last_received_at is None:
            return base
        elapsed = (market.last_received_at - previous["seen_at"]).total_seconds()
        materially_better = (
            quality_score >= previous["quality_score"] + self.settings.duplicate_signal_min_quality_improvement
            or score >= previous["score"] + self.settings.duplicate_signal_min_score_improvement
        )
        if elapsed < self.settings.duplicate_signal_suppression_seconds and not materially_better:
            return base | {
                "allowed": False,
                "blocker": "duplicate_signal",
                "elapsed_seconds": elapsed,
                "suppression_seconds": self.settings.duplicate_signal_suppression_seconds,
                "materially_better": materially_better,
            }
        return base | {
            "elapsed_seconds": elapsed,
            "suppression_seconds": self.settings.duplicate_signal_suppression_seconds,
            "materially_better": materially_better,
        }

    def _remember_stateful_signal(
        self,
        side: str,
        snapshot: MarketRegimeSnapshot,
        market: MarketState,
        score: float,
        quality_score: float,
        profile: str,
    ) -> None:
        if market.last_received_at is None:
            return
        key = self._stateful_signal_key(market.symbol, side, snapshot, profile)
        self.recent_stateful_signals[key] = {
            "seen_at": market.last_received_at,
            "score": score,
            "quality_score": quality_score,
            "state": snapshot.state.value,
            "path": [state.value for state in snapshot.path],
        }

    def _entry_follow_through_gate(
        self, side: str, market: MarketState, target_move_pct: float, trade_profile: str
    ) -> dict:
        enabled = self.settings.entry_follow_through_gate_enabled
        applies = target_move_pct <= self.settings.fast_target_move_pct
        pressure = self._market_pressure(market)
        r15 = market.return_15s_pct or 0.0
        r60 = market.return_60s_pct or 0.0
        buy_10s = market.taker_buy_ratio_10s
        buy_30s = market.taker_buy_ratio_30s

        if side == "long":
            return_15s = r15 >= self.settings.entry_follow_through_return_15s_pct
            return_60s = r60 >= self.settings.entry_follow_through_return_60s_pct
            flow_10s = buy_10s is not None and buy_10s >= 0.66
            flow_30s = buy_30s is not None and buy_30s >= 0.55
            pressure_aligned = pressure >= self.settings.entry_follow_through_pressure
            return_15s_score = self._clamp(r15 / self.settings.entry_follow_through_return_15s_pct)
            return_60s_score = self._clamp(r60 / self.settings.entry_follow_through_return_60s_pct)
            flow_10s_score = self._clamp(((buy_10s or 0.0) - 0.50) / 0.20)
            flow_30s_score = self._clamp(((buy_30s or 0.0) - 0.50) / 0.16)
            pressure_score = self._clamp((pressure + 0.05) / (self.settings.entry_follow_through_pressure + 0.05))
        else:
            return_15s = r15 <= -self.settings.entry_follow_through_return_15s_pct
            return_60s = r60 <= -self.settings.entry_follow_through_return_60s_pct
            flow_10s = buy_10s is not None and buy_10s <= 0.34
            flow_30s = buy_30s is not None and buy_30s <= 0.45
            pressure_aligned = pressure <= -self.settings.entry_follow_through_pressure
            return_15s_score = self._clamp((-r15) / self.settings.entry_follow_through_return_15s_pct)
            return_60s_score = self._clamp((-r60) / self.settings.entry_follow_through_return_60s_pct)
            flow_10s_score = self._clamp((0.50 - (buy_10s or 1.0)) / 0.20)
            flow_30s_score = self._clamp((0.50 - (buy_30s or 1.0)) / 0.16)
            pressure_score = self._clamp((-pressure + 0.05) / (self.settings.entry_follow_through_pressure + 0.05))

        checks = {
            "return_15s": return_15s,
            "return_60s": return_60s,
            "flow_10s": flow_10s,
            "flow_30s": flow_30s,
            "pressure": pressure_aligned,
        }
        confirmations = sum(1 for value in checks.values() if value)
        score = round(
            0.28 * return_15s_score
            + 0.22 * return_60s_score
            + 0.20 * flow_10s_score
            + 0.12 * flow_30s_score
            + 0.18 * pressure_score,
            4,
        )
        required_confirmations = max(1, min(5, self.settings.entry_follow_through_min_confirmations))
        strict_mandatory = checks["return_15s"] and checks["flow_10s"]
        htf_aligned_override = (
            trade_profile == "htf_aligned_fast"
            and checks["return_60s"]
            and checks["flow_10s"]
            and checks["pressure"]
            and confirmations >= min(required_confirmations, 3)
            and score >= self.settings.entry_follow_through_htf_aligned_override_score
        )
        counter_htf_bounce_override = (
            trade_profile == "eth_counter_htf_bounce"
            and checks["return_60s"]
            and checks["flow_10s"]
            and checks["pressure"]
            and confirmations >= min(required_confirmations, 4)
            and score >= self.settings.entry_follow_through_counter_htf_bounce_override_score
        )
        mandatory = strict_mandatory or htf_aligned_override or counter_htf_bounce_override
        allowed = (
            not enabled
            or not applies
            or (
                confirmations >= required_confirmations
                and score >= self.settings.entry_follow_through_min_score
                and mandatory
            )
        )
        return {
            "enabled": enabled,
            "applies": applies,
            "allowed": allowed,
            "blocker": None if allowed else "entry_follow_through",
            "side": side,
            "trade_profile": trade_profile,
            "target_move_pct": target_move_pct,
            "score": score,
            "min_score": self.settings.entry_follow_through_min_score,
            "htf_aligned_override_score": self.settings.entry_follow_through_htf_aligned_override_score,
            "counter_htf_bounce_override_score": self.settings.entry_follow_through_counter_htf_bounce_override_score,
            "confirmations": confirmations,
            "min_confirmations": required_confirmations,
            "mandatory_confirmed": mandatory,
            "strict_mandatory_confirmed": strict_mandatory,
            "htf_aligned_override_confirmed": htf_aligned_override,
            "counter_htf_bounce_override_confirmed": counter_htf_bounce_override,
            "checks": checks,
            "return_15s_pct": r15,
            "return_60s_pct": r60,
            "taker_buy_ratio_10s": buy_10s,
            "taker_buy_ratio_30s": buy_30s,
            "pressure": round(pressure, 4),
        }

    def _weak_neutral_short_gate(
        self,
        side: str,
        snapshot: MarketRegimeSnapshot,
        adaptive_gate: dict,
        higher_timeframe_gate: dict,
        local_execution_gate: dict,
        entry_follow_through_gate: dict,
    ) -> dict:
        profile = str(higher_timeframe_gate.get("profile") or "")
        applies = side == "short" and profile == "weak_or_neutral_htf"
        enabled = self.settings.weak_neutral_short_gate_enabled
        quality_score = float(adaptive_gate.get("quality_score") or 0.0)
        follow_score = float(entry_follow_through_gate.get("score") or 0.0)
        blockers: list[str] = []
        if enabled and applies:
            if quality_score < self.settings.weak_neutral_short_min_quality:
                blockers.append("quality")
            if snapshot.confidence < self.settings.weak_neutral_short_min_sequence_confidence:
                blockers.append("sequence_confidence")
            if follow_score < self.settings.weak_neutral_short_min_follow_score:
                blockers.append("follow_score")
            if not entry_follow_through_gate.get("strict_mandatory_confirmed"):
                blockers.append("strict_follow_through")
            if local_execution_gate.get("local_reversal_confirmed") is not True:
                blockers.append("local_reversal")
        return {
            "enabled": enabled,
            "applies": applies,
            "allowed": not blockers,
            "blocker": None if not blockers else "weak_neutral_short_quality",
            "blockers": blockers,
            "profile": profile,
            "quality_score": quality_score,
            "min_quality": self.settings.weak_neutral_short_min_quality,
            "sequence_confidence": snapshot.confidence,
            "min_sequence_confidence": self.settings.weak_neutral_short_min_sequence_confidence,
            "follow_score": follow_score,
            "min_follow_score": self.settings.weak_neutral_short_min_follow_score,
            "strict_mandatory_confirmed": bool(entry_follow_through_gate.get("strict_mandatory_confirmed")),
            "local_reversal_confirmed": local_execution_gate.get("local_reversal_confirmed"),
        }

    def _weak_neutral_long_gate(
        self,
        side: str,
        market: MarketState,
        snapshot: MarketRegimeSnapshot,
        adaptive_gate: dict,
        higher_timeframe_gate: dict,
        entry_follow_through_gate: dict,
    ) -> dict:
        profile = str(higher_timeframe_gate.get("profile") or "")
        applies = side == "long" and profile == "weak_or_neutral_htf"
        enabled = self.settings.weak_neutral_long_gate_enabled
        quality_score = float(adaptive_gate.get("quality_score") or 0.0)
        follow_score = float(entry_follow_through_gate.get("score") or 0.0)
        timeframes = market.higher_timeframe_context.get("timeframes", {}) if market.higher_timeframe_context else {}
        frame_1h = timeframes.get("1h") or {}
        structure_1h = str(frame_1h.get("structure") or "")
        trend_1h = float(frame_1h.get("trend_score") or 0.0)
        return_1h = float(frame_1h.get("return_pct") or 0.0)
        range_position_1h = frame_1h.get("range_position")
        one_hour_supportive = (
            structure_1h in {"uptrend_breakout", "uptrend_pullback"}
            or trend_1h >= self.settings.weak_neutral_long_min_1h_trend_score
            or return_1h >= self.settings.weak_neutral_long_min_1h_return_pct
        )
        blockers: list[str] = []
        if enabled and applies:
            if quality_score < self.settings.weak_neutral_long_min_quality:
                blockers.append("quality")
            if snapshot.confidence < self.settings.weak_neutral_long_min_sequence_confidence:
                blockers.append("sequence_confidence")
            if follow_score < self.settings.weak_neutral_long_min_follow_score:
                blockers.append("follow_score")
            if not entry_follow_through_gate.get("strict_mandatory_confirmed"):
                blockers.append("strict_follow_through")
            if not one_hour_supportive:
                blockers.append("one_hour_support")
        return {
            "enabled": enabled,
            "applies": applies,
            "allowed": not blockers,
            "blocker": None if not blockers else "weak_neutral_long_quality",
            "blockers": blockers,
            "profile": profile,
            "quality_score": quality_score,
            "min_quality": self.settings.weak_neutral_long_min_quality,
            "sequence_confidence": snapshot.confidence,
            "min_sequence_confidence": self.settings.weak_neutral_long_min_sequence_confidence,
            "follow_score": follow_score,
            "min_follow_score": self.settings.weak_neutral_long_min_follow_score,
            "strict_mandatory_confirmed": bool(entry_follow_through_gate.get("strict_mandatory_confirmed")),
            "one_hour_supportive": one_hour_supportive,
            "structure_1h": structure_1h,
            "trend_score_1h": round(trend_1h, 4),
            "min_trend_score_1h": self.settings.weak_neutral_long_min_1h_trend_score,
            "return_1h_pct": return_1h,
            "min_return_1h_pct": self.settings.weak_neutral_long_min_1h_return_pct,
            "range_position_1h": range_position_1h,
        }

    def _market_pressure(self, market: MarketState) -> float:
        book = market.book_imbalance_top or 0.0
        depth = market.depth_imbalance_top5 if market.depth_imbalance_top5 is not None else book
        return (book + depth) / 2.0

    def _stateful_signal_key(
        self,
        symbol: str,
        side: str,
        snapshot: MarketRegimeSnapshot,
        profile: str,
    ) -> str:
        return f"{symbol.upper()}:{side}:{snapshot.state.value}:{profile}"

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

    def _apply_higher_timeframe_context(
        self,
        long_score: float,
        short_score: float,
        market: MarketState,
        evidence: dict,
    ) -> tuple[float, float]:
        context_stale = self._higher_timeframe_context_stale(market)
        side = market.higher_timeframe_bias_side
        strength = self._clamp(market.higher_timeframe_bias_strength)
        evidence["higher_timeframe_context"] = {
            "bias_side": side,
            "bias_strength": strength,
            "bias_reason": market.higher_timeframe_bias_reason,
            "age_seconds": market.higher_timeframe_context_age_seconds,
            "stale": context_stale,
            "timeframes": market.higher_timeframe_context.get("timeframes", {}),
        }
        if context_stale or side == "neutral" or strength <= 0:
            return long_score, short_score
        boost = self.settings.higher_timeframe_score_boost * strength
        penalty = self.settings.higher_timeframe_score_penalty * strength
        if side == "long":
            return round(min(1.0, long_score + boost), 4), round(max(0.0, short_score - penalty), 4)
        if side == "short":
            return round(max(0.0, long_score - penalty), 4), round(min(1.0, short_score + boost), 4)
        return long_score, short_score

    def _higher_timeframe_context_stale(self, market: MarketState) -> bool:
        if not self.settings.higher_timeframe_enabled:
            return True
        if not market.higher_timeframe_context:
            return True
        age = market.higher_timeframe_context_age_seconds
        if age is None:
            return True
        return age > max(900, self.settings.higher_timeframe_poll_seconds * 3)

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
        quality_components = self._adaptive_quality_components(side, snapshot, market, score)
        quality_score = self._weighted_quality_score(quality_components)
        min_quality = (
            self.settings.stateful_adaptive_min_quality_long
            if side == "long"
            else self.settings.stateful_adaptive_min_quality_short
        )
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
        if quality_score < min_quality:
            blockers.append("regime_quality")

        return {
            "allowed": not blockers,
            "blockers": blockers,
            "quality_score": quality_score,
            "min_quality": min_quality,
            "quality_components": quality_components,
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
            return buy_10s >= 0.70 and buy_30s >= 0.60 and pressure >= 0.12 and (market.return_180s_pct or 0.0) > 0
        return buy_10s <= 0.34 and buy_30s <= 0.52 and pressure <= -0.05 and (market.return_180s_pct or 0.0) < 0

    def _adaptive_quality_components(
        self,
        side: str,
        snapshot: MarketRegimeSnapshot,
        market: MarketState,
        score: float,
    ) -> dict[str, float]:
        range_pct = market.range_180s_pct or 0.0
        range_component = self._clamp(range_pct / max(0.000001, self.settings.fast_target_move_pct))
        buy_10s = market.taker_buy_ratio_10s if market.taker_buy_ratio_10s is not None else 0.5
        buy_30s = market.taker_buy_ratio_30s if market.taker_buy_ratio_30s is not None else buy_10s
        flow_10s = buy_10s if side == "long" else 1.0 - buy_10s
        flow_30s = buy_30s if side == "long" else 1.0 - buy_30s
        book = market.book_imbalance_top or 0.0
        depth = market.depth_imbalance_top5 if market.depth_imbalance_top5 is not None else book
        pressure = (book + depth) / 2.0
        pressure_component = (pressure + 1.0) / 2.0 if side == "long" else (-pressure + 1.0) / 2.0
        trend = market.return_180s_pct or 0.0
        trend_component = self._clamp((trend if side == "long" else -trend) / self.settings.stateful_adaptive_target_move_pct)
        r60 = market.return_60s_pct or 0.0
        impulse_component = self._clamp((r60 if side == "long" else -r60) / (self.settings.stateful_adaptive_target_move_pct * 0.60))
        freshness = 1.0 if (market.data_age_seconds or 0.0) <= self.settings.stale_after_seconds else 0.0
        htf_alignment = self._higher_timeframe_alignment(side, market)
        return {
            "strategy_score": self._clamp(score),
            "sequence_confidence": self._clamp(snapshot.confidence),
            "range_expansion": range_component,
            "flow_10s_alignment": self._clamp(flow_10s),
            "flow_30s_alignment": self._clamp(flow_30s),
            "pressure_alignment": self._clamp(pressure_component),
            "trend_alignment": trend_component,
            "impulse_alignment": impulse_component,
            "higher_timeframe_alignment": htf_alignment,
            "fresh_data": freshness,
        }

    def _higher_timeframe_alignment(self, side: str, market: MarketState) -> float:
        if self._higher_timeframe_context_stale(market):
            return 0.5
        bias_side = market.higher_timeframe_bias_side
        strength = self._clamp(market.higher_timeframe_bias_strength)
        if bias_side == "neutral":
            return 0.5
        if bias_side == side:
            return 0.5 + 0.5 * strength
        return 0.5 - 0.5 * strength

    def _weighted_quality_score(self, components: dict[str, float]) -> float:
        score = (
            0.16 * components["strategy_score"]
            + 0.16 * components["sequence_confidence"]
            + 0.11 * components["range_expansion"]
            + 0.13 * components["flow_10s_alignment"]
            + 0.09 * components["flow_30s_alignment"]
            + 0.09 * components["pressure_alignment"]
            + 0.09 * components["trend_alignment"]
            + 0.04 * components["impulse_alignment"]
            + 0.10 * components["higher_timeframe_alignment"]
            + 0.03 * components["fresh_data"]
        )
        return round(self._clamp(score), 4)

    def _higher_timeframe_execution_gate(
        self,
        side: str,
        snapshot: MarketRegimeSnapshot,
        market: MarketState,
        score: float,
        adaptive_gate: dict,
        target_feasible: bool,
    ) -> dict:
        quality_score = float(adaptive_gate.get("quality_score") or 0.0)
        base = {
            "enabled": self.settings.higher_timeframe_execution_gate_enabled,
            "allowed": True,
            "blocker": None,
            "bias_side": market.higher_timeframe_bias_side,
            "bias_strength": self._clamp(market.higher_timeframe_bias_strength),
            "bias_reason": market.higher_timeframe_bias_reason,
            "context_stale": self._higher_timeframe_context_stale(market),
            "quality_score": quality_score,
            "target_move_pct": self.settings.fast_target_move_pct,
            "stop_move_pct": self.settings.fast_stop_move_pct,
            "profile": "fast",
            "exception_allowed": False,
        }
        if not self.settings.higher_timeframe_execution_gate_enabled:
            return base
        if base["context_stale"]:
            return base | {"profile": "no_htf_context"}
        bias_side = market.higher_timeframe_bias_side
        strength = self._clamp(market.higher_timeframe_bias_strength)
        if bias_side not in {"long", "short"} or strength < self.settings.higher_timeframe_gate_min_strength:
            return base | {"profile": "weak_or_neutral_htf"}
        if bias_side == side:
            return base | {
                "profile": "htf_aligned_fast",
                "target_move_pct": self.settings.fast_target_move_pct,
                "stop_move_pct": self.settings.fast_stop_move_pct,
            }

        counter_bounce_gate = self._counter_higher_timeframe_bounce_gate(
            side=side,
            snapshot=snapshot,
            market=market,
            score=score,
            adaptive_gate=adaptive_gate,
            target_feasible=target_feasible,
        )
        if counter_bounce_gate["allowed"]:
            return base | {
                "profile": "eth_counter_htf_bounce",
                "exception_allowed": True,
                "target_move_pct": self.settings.fast_target_move_pct,
                "stop_move_pct": self.settings.fast_stop_move_pct,
                "counter_bounce_gate": counter_bounce_gate,
            }

        exception_allowed = (
            target_feasible
            and quality_score >= self.settings.higher_timeframe_countertrend_min_quality
            and score >= self.settings.higher_timeframe_countertrend_min_score
            and snapshot.confidence >= self.settings.higher_timeframe_countertrend_min_sequence_confidence
            and adaptive_gate.get("strong_structure")
            and adaptive_gate.get("flow_agrees")
        )
        if not exception_allowed:
            return base | {
                "allowed": False,
                "blocker": "higher_timeframe_countertrend",
                "profile": "countertrend_shadow_only",
                "required_quality": self.settings.higher_timeframe_countertrend_min_quality,
                "required_score": self.settings.higher_timeframe_countertrend_min_score,
                "required_sequence_confidence": self.settings.higher_timeframe_countertrend_min_sequence_confidence,
                "counter_bounce_gate": counter_bounce_gate,
            }

        target = self.settings.higher_timeframe_exception_target_move_pct
        if quality_score >= 0.92 and score >= 0.90:
            target = self.settings.higher_timeframe_strong_exception_target_move_pct
        return base | {
            "profile": "countertrend_exception",
            "exception_allowed": True,
            "target_move_pct": target,
            "stop_move_pct": self.settings.slow_stop_move_pct,
            "counter_bounce_gate": counter_bounce_gate,
        }

    def _counter_higher_timeframe_bounce_gate(
        self,
        side: str,
        snapshot: MarketRegimeSnapshot,
        market: MarketState,
        score: float,
        adaptive_gate: dict,
        target_feasible: bool,
    ) -> dict:
        quality_score = float(adaptive_gate.get("quality_score") or 0.0)
        range_pct = market.range_180s_pct or 0.0
        blockers: list[str] = []
        local_context = self._counter_higher_timeframe_local_context(market)
        allowed_symbols = {
            symbol.strip().upper()
            for symbol in self.settings.counter_htf_bounce_symbols.split(",")
            if symbol.strip()
        }
        if not self.settings.counter_htf_bounce_enabled:
            blockers.append("disabled")
        if market.symbol.upper() not in allowed_symbols:
            blockers.append("symbol_not_enabled")
        if side != "long":
            blockers.append("side_not_long")
        if market.higher_timeframe_bias_side != "short":
            blockers.append("htf_not_short")
        if not target_feasible or range_pct < self.settings.counter_htf_bounce_min_range_pct:
            blockers.append("target_range")
        if quality_score < self.settings.counter_htf_bounce_min_quality:
            blockers.append("quality")
        if score < self.settings.counter_htf_bounce_min_score:
            blockers.append("score")
        if snapshot.confidence < self.settings.counter_htf_bounce_min_sequence_confidence:
            blockers.append("sequence_confidence")
        if not adaptive_gate.get("flow_agrees"):
            blockers.append("flow_not_confirmed")
        if not local_context["allowed"]:
            blockers.append("local_context")

        return {
            "enabled": self.settings.counter_htf_bounce_enabled,
            "allowed": not blockers,
            "blockers": blockers,
            "symbol": market.symbol.upper(),
            "required_symbols": sorted(allowed_symbols),
            "quality_score": quality_score,
            "min_quality": self.settings.counter_htf_bounce_min_quality,
            "score": score,
            "min_score": self.settings.counter_htf_bounce_min_score,
            "sequence_confidence": snapshot.confidence,
            "min_sequence_confidence": self.settings.counter_htf_bounce_min_sequence_confidence,
            "range_180s_pct": range_pct,
            "min_range_180s_pct": self.settings.counter_htf_bounce_min_range_pct,
            "target_feasible": target_feasible,
            "flow_agrees": bool(adaptive_gate.get("flow_agrees")),
            "local_context": local_context,
        }

    def _counter_higher_timeframe_local_context(self, market: MarketState) -> dict:
        timeframes = market.higher_timeframe_context.get("timeframes", {}) if market.higher_timeframe_context else {}
        frame_5m = timeframes.get("5m") or {}
        frame_1h = timeframes.get("1h") or {}
        structure_5m = str(frame_5m.get("structure") or "")
        structure_1h = str(frame_1h.get("structure") or "")
        trend_5m = float(frame_5m.get("trend_score") or 0.0)
        trend_1h = float(frame_1h.get("trend_score") or 0.0)
        return_1h = float(frame_1h.get("return_pct") or 0.0)
        range_position_5m = frame_5m.get("range_position")
        range_position_1h = frame_1h.get("range_position")
        pressure = self._market_pressure(market)
        buy_10s = market.taker_buy_ratio_10s
        buy_30s = market.taker_buy_ratio_30s

        five_minute_reclaim = (
            structure_5m in {"uptrend_breakout", "uptrend_pullback", "range_support_test"}
            or trend_5m >= self.settings.local_5m_countertrend_trend_threshold
            or (range_position_5m is not None and range_position_5m >= 0.60 and trend_5m > 0.15)
        )
        one_hour_improving = (
            structure_1h in {"balanced", "range_resistance_test", "uptrend_breakout", "uptrend_pullback"}
            or trend_1h >= -0.15
            or return_1h >= 0.0
            or (range_position_1h is not None and range_position_1h >= 0.50 and trend_1h > -0.30)
        )
        immediate_flow_agrees = (
            buy_10s is not None
            and buy_30s is not None
            and buy_10s >= 0.70
            and buy_30s >= 0.60
            and pressure >= self.settings.local_reversal_pressure
            and (market.return_60s_pct or 0.0) >= self.settings.local_reversal_return_60s_pct
        )
        one_hour_led_relief = (
            one_hour_improving
            and immediate_flow_agrees
            and (
                return_1h >= self.settings.counter_htf_bounce_1h_relief_return_pct
                or structure_1h in {"balanced", "uptrend_breakout", "uptrend_pullback"}
                or trend_1h >= 0.0
            )
        )
        return {
            "allowed": bool(frame_5m and frame_1h and one_hour_improving and (five_minute_reclaim or one_hour_led_relief)),
            "structure_5m": structure_5m,
            "trend_score_5m": round(trend_5m, 4),
            "range_position_5m": range_position_5m,
            "structure_1h": structure_1h,
            "trend_score_1h": round(trend_1h, 4),
            "return_1h_pct": return_1h,
            "range_position_1h": range_position_1h,
            "pressure": round(pressure, 4),
            "immediate_flow_agrees": immediate_flow_agrees,
            "five_minute_reclaim": five_minute_reclaim,
            "one_hour_improving": one_hour_improving,
            "one_hour_led_relief": one_hour_led_relief,
        }

    def _local_execution_gate(self, side: str, market: MarketState, higher_timeframe_gate: dict) -> dict:
        timeframes = market.higher_timeframe_context.get("timeframes", {}) if market.higher_timeframe_context else {}
        frame_5m = timeframes.get("5m") or {}
        structure = str(frame_5m.get("structure") or "")
        trend_score = float(frame_5m.get("trend_score") or 0.0)
        taker_buy_ratio_5m = frame_5m.get("taker_buy_ratio")
        range_position_5m = frame_5m.get("range_position")
        book = market.book_imbalance_top or 0.0
        depth = market.depth_imbalance_top5 if market.depth_imbalance_top5 is not None else book
        pressure = (book + depth) / 2.0
        local_reversal_confirmed = self._local_reversal_confirmed(side, market, pressure)
        profile = higher_timeframe_gate.get("profile")
        enabled = self.settings.local_execution_gate_enabled
        base = {
            "enabled": enabled,
            "allowed": True,
            "blocker": None,
            "profile": profile,
            "structure_5m": structure,
            "trend_score_5m": round(trend_score, 4),
            "range_position_5m": range_position_5m,
            "taker_buy_ratio_5m": taker_buy_ratio_5m,
            "pressure": round(pressure, 4),
            "local_reversal_confirmed": local_reversal_confirmed,
        }
        if not enabled or not frame_5m:
            return base
        if profile not in {"htf_aligned_fast", "countertrend_exception", "weak_or_neutral_htf", "fast"}:
            return base

        threshold = self.settings.local_5m_countertrend_trend_threshold
        if side == "short":
            local_uptrend = structure in {"uptrend_breakout", "uptrend_pullback"} or trend_score >= threshold
            local_buy_pressure = (
                (taker_buy_ratio_5m is not None and taker_buy_ratio_5m >= 0.515)
                or (range_position_5m is not None and range_position_5m >= 0.60)
            )
            if local_uptrend and local_buy_pressure and not local_reversal_confirmed:
                return base | {"allowed": False, "blocker": "local_execution_countertrend"}
        if side == "long":
            local_downtrend = structure in {"downtrend_breakdown", "downtrend_bounce"} or trend_score <= -threshold
            local_sell_pressure = (
                (taker_buy_ratio_5m is not None and taker_buy_ratio_5m <= 0.485)
                or (range_position_5m is not None and range_position_5m <= 0.40)
            )
            if local_downtrend and local_sell_pressure and not local_reversal_confirmed:
                return base | {"allowed": False, "blocker": "local_execution_countertrend"}
        return base

    def _local_reversal_confirmed(self, side: str, market: MarketState, pressure: float) -> bool:
        r15 = market.return_15s_pct or 0.0
        r60 = market.return_60s_pct or 0.0
        buy_10s = market.taker_buy_ratio_10s
        buy_30s = market.taker_buy_ratio_30s
        if buy_10s is None or buy_30s is None:
            return False
        if side == "short":
            return (
                r15 <= -self.settings.local_reversal_return_15s_pct
                and r60 <= -self.settings.local_reversal_return_60s_pct
                and buy_10s <= 0.34
                and buy_30s <= 0.52
                and pressure <= -self.settings.local_reversal_pressure
            )
        return (
            r15 >= self.settings.local_reversal_return_15s_pct
            and r60 >= self.settings.local_reversal_return_60s_pct
            and buy_10s >= 0.66
            and buy_30s >= 0.55
            and pressure >= self.settings.local_reversal_pressure
        )

    def _required_score(self, side: str, opposing_score: float, stateful_filter: dict | None) -> float:
        min_score = self.settings.min_confidence
        if (
            stateful_filter is not None
            and stateful_filter.get("side") == side
            and stateful_filter.get("adaptive_entry_allowed")
        ):
            min_score = min(min_score, self.settings.stateful_adaptive_min_score)
        higher_timeframe_gate = (stateful_filter or {}).get("higher_timeframe_gate") or {}
        if (
            stateful_filter is not None
            and stateful_filter.get("side") == side
            and higher_timeframe_gate.get("profile") == "eth_counter_htf_bounce"
        ):
            min_score = min(min_score, self.settings.counter_htf_bounce_min_score)
        return max(min_score, opposing_score + 0.04)

    def _trade_profile(self, mode: TradeMode, stateful_filter: dict | None, side: str) -> TradeProfile:
        higher_timeframe_gate = (stateful_filter or {}).get("higher_timeframe_gate") or {}
        if (
            stateful_filter is not None
            and stateful_filter.get("side") == side
            and higher_timeframe_gate.get("profile") == "eth_counter_htf_bounce"
        ):
            return TradeProfile(
                name="eth_counter_htf_bounce",
                mode=TradeMode.fast,
                target_move_pct=float(higher_timeframe_gate["target_move_pct"]),
                stop_move_pct=float(higher_timeframe_gate["stop_move_pct"]),
                leverage=self.settings.fast_leverage,
            )
        if (
            stateful_filter is not None
            and stateful_filter.get("side") == side
            and higher_timeframe_gate.get("profile") == "countertrend_exception"
        ):
            return TradeProfile(
                name="countertrend_exception",
                mode=TradeMode.slow,
                target_move_pct=float(higher_timeframe_gate["target_move_pct"]),
                stop_move_pct=float(higher_timeframe_gate["stop_move_pct"]),
                leverage=self.settings.slow_leverage,
            )
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
        if (
            stateful_filter is not None
            and stateful_filter.get("side") == side
            and higher_timeframe_gate.get("profile") == "htf_aligned_fast"
        ):
            return TradeProfile(
                name="htf_aligned_fast",
                mode=TradeMode.fast,
                target_move_pct=float(higher_timeframe_gate["target_move_pct"]),
                stop_move_pct=float(higher_timeframe_gate["stop_move_pct"]),
                leverage=self.settings.fast_leverage,
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

    def _shadow_trade_signal(
        self,
        side: str,
        market: MarketState,
        confidence: float,
        blockers: list[str],
        fee_edge_gate: dict | None = None,
    ) -> dict | None:
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
            "effective_cost": (fee_edge_gate or {}).get("effective_cost"),
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
            "spread_bps_avg_5s": market.spread_bps_avg_5s,
            "spread_bps_std_5s": market.spread_bps_std_5s,
            "spread_bps_max_5s": market.spread_bps_max_5s,
            "range_position_180s": market.range_position_180s,
            "range_180s_pct": market.range_180s_pct,
            "return_15s_pct": market.return_15s_pct,
            "return_60s_pct": market.return_60s_pct,
            "return_180s_pct": market.return_180s_pct,
            "realized_vol_60s_pct": market.realized_vol_60s_pct,
            "taker_buy_ratio_10s": market.taker_buy_ratio_10s,
            "taker_buy_ratio_30s": market.taker_buy_ratio_30s,
            "taker_aggression_imbalance_1s": market.taker_aggression_imbalance_1s,
            "taker_aggression_imbalance_5s": market.taker_aggression_imbalance_5s,
            "taker_aggression_imbalance_15s": market.taker_aggression_imbalance_15s,
            "book_imbalance_top": market.book_imbalance_top,
            "order_flow_imbalance_250ms": market.order_flow_imbalance_250ms,
            "order_flow_imbalance_1s": market.order_flow_imbalance_1s,
            "order_flow_imbalance_5s": market.order_flow_imbalance_5s,
            "microprice": market.microprice,
            "microprice_mid_bps": market.microprice_mid_bps,
            "vamp_price_top": market.vamp_price_top,
            "vamp_mid_bps": market.vamp_mid_bps,
            "weighted_depth_price_top": market.weighted_depth_price_top,
            "weighted_depth_mid_bps": market.weighted_depth_mid_bps,
            "depth_imbalance_top5": market.depth_imbalance_top5,
            "depth_bid_qty_top5": market.depth_bid_qty_top5,
            "depth_ask_qty_top5": market.depth_ask_qty_top5,
            "bid_depth_refill_rate_5s": market.bid_depth_refill_rate_5s,
            "ask_depth_refill_rate_5s": market.ask_depth_refill_rate_5s,
            "bid_depth_evaporation_rate_5s": market.bid_depth_evaporation_rate_5s,
            "ask_depth_evaporation_rate_5s": market.ask_depth_evaporation_rate_5s,
            "liquidation_notional_30s": market.liquidation_notional_30s,
            "long_liquidation_notional_30s": market.long_liquidation_notional_30s,
            "short_liquidation_notional_30s": market.short_liquidation_notional_30s,
            "liquidation_buy_ratio_30s": market.liquidation_buy_ratio_30s,
            "open_interest": market.open_interest,
            "open_interest_change_5m_pct": market.open_interest_change_5m_pct,
            "mark_last_basis_bps": market.mark_last_basis_bps,
            "exchange_event_lag_ms": market.exchange_event_lag_ms,
            "avg_event_lag_30s_ms": market.avg_event_lag_30s_ms,
            "funding_rate": market.funding_rate,
            "effective_cost": estimate_effective_cost(self.settings, market).model_dump(),
        }

