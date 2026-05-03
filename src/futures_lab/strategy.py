from dataclasses import dataclass

from futures_lab.config import Settings
from futures_lab.models import Decision, DecisionAction, MarketState, Regime, TradeMode


@dataclass
class HitAndRunStrategy:
    settings: Settings

    def decide(self, market: MarketState) -> Decision:
        blockers = self._blockers(market)
        evidence = self._evidence(market)
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

        long_score = self._score_long(market, buy_flow, book)
        short_score = self._score_short(market, sell_flow, book)

        if long_score >= max(self.settings.min_confidence, short_score + 0.04):
            return self._build_trade_decision(
                market=market,
                action=DecisionAction.propose_long,
                mode=mode,
                confidence=long_score,
                reason="Hit-and-run long: range-low location with positive taker flow and supportive top-book pressure.",
                evidence=evidence | {"long_score": long_score, "short_score": short_score},
            )

        if short_score >= max(self.settings.min_confidence, long_score + 0.04):
            return self._build_trade_decision(
                market=market,
                action=DecisionAction.propose_short,
                mode=mode,
                confidence=short_score,
                reason="Hit-and-run short: range-high location with positive taker-sell flow and supportive top-book pressure.",
                evidence=evidence | {"long_score": long_score, "short_score": short_score},
            )

        return Decision(
            symbol=market.symbol,
            action=DecisionAction.wait,
            confidence=max(long_score, short_score),
            reason="No asymmetric hit-and-run edge yet.",
            evidence=evidence | {"long_score": long_score, "short_score": short_score},
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

    def _choose_mode(self, market: MarketState) -> TradeMode:
        # Fast mode is the user's 200x / 0.5% target design. Slow mode lowers leverage
        # and widens the target when market motion is directional or slower.
        if market.regime == Regime.sideways:
            return TradeMode.fast
        return TradeMode.slow

    def _score_long(self, market: MarketState, buy_flow: float, book_imbalance: float) -> float:
        assert market.range_position_180s is not None
        range_edge = max(0.0, 1.0 - market.range_position_180s / 0.28)
        flow = max(0.0, min(1.0, (buy_flow - 0.52) / 0.28))
        book = max(0.0, min(1.0, (book_imbalance + 0.20) / 0.40))
        reversion_ok = 1.0 if (market.return_15s_pct or 0.0) > -0.0015 else 0.5
        regime = 1.0 if market.regime == Regime.sideways else 0.7
        return round(0.34 * range_edge + 0.33 * flow + 0.18 * book + 0.10 * reversion_ok + 0.05 * regime, 4)

    def _score_short(self, market: MarketState, sell_flow: float, book_imbalance: float) -> float:
        assert market.range_position_180s is not None
        range_edge = max(0.0, 1.0 - (1.0 - market.range_position_180s) / 0.28)
        flow = max(0.0, min(1.0, (sell_flow - 0.52) / 0.28))
        book = max(0.0, min(1.0, (-book_imbalance + 0.20) / 0.40))
        reversion_ok = 1.0 if (market.return_15s_pct or 0.0) < 0.0015 else 0.5
        regime = 1.0 if market.regime == Regime.sideways else 0.7
        return round(0.34 * range_edge + 0.33 * flow + 0.18 * book + 0.10 * reversion_ok + 0.05 * regime, 4)

    def _build_trade_decision(
        self,
        market: MarketState,
        action: DecisionAction,
        mode: TradeMode,
        confidence: float,
        reason: str,
        evidence: dict,
    ) -> Decision:
        assert market.mid_price is not None
        direction = 1 if action == DecisionAction.propose_long else -1
        target = self.settings.fast_target_move_pct if mode == TradeMode.fast else self.settings.slow_target_move_pct
        stop = self.settings.fast_stop_move_pct if mode == TradeMode.fast else self.settings.slow_stop_move_pct
        leverage = self.settings.fast_leverage if mode == TradeMode.fast else self.settings.slow_leverage
        stake = self.settings.stake_usd
        notional = stake * leverage
        return Decision(
            symbol=market.symbol,
            action=action,
            mode=mode,
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
            evidence=evidence,
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
            "funding_rate": market.funding_rate,
        }

