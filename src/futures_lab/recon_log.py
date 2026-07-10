import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from futures_lab.config import Settings
from futures_lab.models import Decision, MarketState, PaperTrade, RiskVerdict


@dataclass
class ReconLogger:
    settings: Settings

    def __post_init__(self) -> None:
        self._feature_rows = 0
        self._decision_rows = 0
        self.features_dir = Path(self.settings.data_dir) / "features"
        self.decisions_dir = Path(self.settings.data_dir) / "decisions"
        self.paper_dir = Path(self.settings.data_dir) / "paper_trades"
        self.shadow_dir = Path(self.settings.data_dir) / "shadow_trades"
        self.regime_outcomes_dir = Path(self.settings.data_dir) / "regime_outcomes"
        self.candidate_outcomes_dir = Path(self.settings.data_dir) / "candidate_outcomes"
        for path in (
            self.features_dir,
            self.decisions_dir,
            self.paper_dir,
            self.shadow_dir,
            self.regime_outcomes_dir,
            self.candidate_outcomes_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def write_feature(self, market: MarketState) -> None:
        if not self.settings.write_feature_log:
            return
        self._feature_rows += 1
        sample_interval = max(1, self.settings.feature_log_sample_interval)
        if sample_interval > 1 and self._feature_rows % sample_interval != 0:
            return
        self._write(self.features_dir, "features", self._market_row(market))

    def write_decision(
        self,
        market: MarketState,
        decision: Decision,
        risk: RiskVerdict,
        decision_latency_ms: float | None = None,
    ) -> None:
        if not self.settings.write_decision_log:
            return
        self._decision_rows += 1
        sample_interval = max(1, self.settings.decision_log_wait_sample_interval)
        if (
            sample_interval > 1
            and self._decision_rows % sample_interval != 0
            and not risk.allowed
        ):
            return
        self._write(
            self.decisions_dir,
            "decisions",
            {
                "ts": decision.timestamp.isoformat(),
                "symbol": decision.symbol,
                "action": decision.action.value,
                "mode": decision.mode.value if decision.mode else None,
                "confidence": decision.confidence,
                "risk_allowed": risk.allowed,
                "risk_reason": risk.reason,
                "risk_blockers": risk.blockers,
                "entry_price": decision.entry_price,
                "take_profit_price": decision.take_profit_price,
                "stop_loss_price": decision.stop_loss_price,
                "leverage": decision.leverage,
                "notional_usd": decision.notional_usd,
                "decision_latency_ms": decision_latency_ms,
                "reason": decision.reason,
                "market": self._market_row(market),
                "evidence": decision.evidence,
            },
        )

    def write_paper_trade(self, trade: PaperTrade) -> None:
        if not self.settings.write_paper_trade_log:
            return
        row = trade.model_dump()
        row["side"] = trade.side.value
        row["mode"] = trade.mode.value
        row["opened_at"] = trade.opened_at.isoformat()
        row["closed_at"] = trade.closed_at.isoformat()
        self._write(self.paper_dir, "paper_trades", row)

    def write_shadow_trade(self, row: dict[str, Any]) -> None:
        if not self.settings.write_shadow_trade_log:
            return
        self._write(self.shadow_dir, "shadow_trades", row)

    def write_regime_outcome(self, row: dict[str, Any]) -> None:
        if not self.settings.write_regime_outcome_log:
            return
        self._write(self.regime_outcomes_dir, "regime_outcomes", row)

    def write_candidate_outcome(self, row: dict[str, Any]) -> None:
        if not self.settings.write_candidate_outcome_log:
            return
        self._write(self.candidate_outcomes_dir, "candidate_outcomes", row)

    def _write(self, directory: Path, prefix: str, row: dict[str, Any]) -> None:
        path = directory / f"{self.settings.symbol.upper()}_{prefix}_{datetime.now(timezone.utc).date().isoformat()}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, separators=(",", ":"), default=str))
            handle.write("\n")

    def _market_row(self, market: MarketState) -> dict[str, Any]:
        return {
            "ts": (market.last_received_at.isoformat() if market.last_received_at else datetime.now(timezone.utc).isoformat()),
            "symbol": market.symbol,
            "regime": market.regime.value,
            "connected": market.connected,
            "data_age_seconds": market.data_age_seconds,
            "observed_seconds": market.observed_seconds,
            "best_bid_qty": market.best_bid_qty,
            "best_ask_qty": market.best_ask_qty,
            "depth_bid_qty_top5": market.depth_bid_qty_top5,
            "depth_ask_qty_top5": market.depth_ask_qty_top5,
            "depth_imbalance_top5": market.depth_imbalance_top5,
            "depth_bid_wall_ratio_top5": market.depth_bid_wall_ratio_top5,
            "depth_ask_wall_ratio_top5": market.depth_ask_wall_ratio_top5,
            "mid_price": market.mid_price,
            "spread_bps": market.spread_bps,
            "spread_bps_avg_5s": market.spread_bps_avg_5s,
            "spread_bps_std_5s": market.spread_bps_std_5s,
            "spread_bps_max_5s": market.spread_bps_max_5s,
            "microprice": market.microprice,
            "microprice_mid_bps": market.microprice_mid_bps,
            "vamp_price_top": market.vamp_price_top,
            "vamp_mid_bps": market.vamp_mid_bps,
            "weighted_depth_price_top": market.weighted_depth_price_top,
            "weighted_depth_mid_bps": market.weighted_depth_mid_bps,
            "order_flow_imbalance_250ms": market.order_flow_imbalance_250ms,
            "order_flow_imbalance_1s": market.order_flow_imbalance_1s,
            "order_flow_imbalance_5s": market.order_flow_imbalance_5s,
            "taker_aggression_imbalance_1s": market.taker_aggression_imbalance_1s,
            "taker_aggression_imbalance_5s": market.taker_aggression_imbalance_5s,
            "taker_aggression_imbalance_15s": market.taker_aggression_imbalance_15s,
            "bid_depth_refill_rate_5s": market.bid_depth_refill_rate_5s,
            "ask_depth_refill_rate_5s": market.ask_depth_refill_rate_5s,
            "bid_depth_evaporation_rate_5s": market.bid_depth_evaporation_rate_5s,
            "ask_depth_evaporation_rate_5s": market.ask_depth_evaporation_rate_5s,
            "mark_price": market.mark_price,
            "mark_last_basis_bps": market.mark_last_basis_bps,
            "funding_rate": market.funding_rate,
            "open_interest": market.open_interest,
            "open_interest_age_seconds": market.open_interest_age_seconds,
            "open_interest_change_5m_pct": market.open_interest_change_5m_pct,
            "return_15s_pct": market.return_15s_pct,
            "return_60s_pct": market.return_60s_pct,
            "return_180s_pct": market.return_180s_pct,
            "realized_vol_60s_pct": market.realized_vol_60s_pct,
            "realized_vol_180s_pct": market.realized_vol_180s_pct,
            "range_180s_pct": market.range_180s_pct,
            "range_position_180s": market.range_position_180s,
            "taker_buy_ratio_10s": market.taker_buy_ratio_10s,
            "taker_buy_ratio_30s": market.taker_buy_ratio_30s,
            "book_imbalance_top": market.book_imbalance_top,
            "liquidation_notional_30s": market.liquidation_notional_30s,
            "long_liquidation_notional_30s": market.long_liquidation_notional_30s,
            "short_liquidation_notional_30s": market.short_liquidation_notional_30s,
            "liquidation_buy_ratio_30s": market.liquidation_buy_ratio_30s,
            "liquidation_phase": market.liquidation_phase,
            "liquidation_phase_side": market.liquidation_phase_side,
            "liquidation_phase_confidence": market.liquidation_phase_confidence,
            "liquidation_phase_context": market.liquidation_phase_context,
            "last_stream_event_type": market.last_stream_event_type,
            "exchange_event_lag_ms": market.exchange_event_lag_ms,
            "avg_event_lag_30s_ms": market.avg_event_lag_30s_ms,
            "max_event_lag_30s_ms": market.max_event_lag_30s_ms,
            "hot_event_lag_ms": market.hot_event_lag_ms,
            "avg_hot_event_lag_30s_ms": market.avg_hot_event_lag_30s_ms,
            "max_hot_event_lag_30s_ms": market.max_hot_event_lag_30s_ms,
            "book_event_lag_ms": market.book_event_lag_ms,
            "avg_book_event_lag_30s_ms": market.avg_book_event_lag_30s_ms,
            "max_book_event_lag_30s_ms": market.max_book_event_lag_30s_ms,
            "trade_event_lag_ms": market.trade_event_lag_ms,
            "avg_trade_event_lag_30s_ms": market.avg_trade_event_lag_30s_ms,
            "max_trade_event_lag_30s_ms": market.max_trade_event_lag_30s_ms,
            "context_event_lag_ms": market.context_event_lag_ms,
            "avg_context_event_lag_30s_ms": market.avg_context_event_lag_30s_ms,
            "max_context_event_lag_30s_ms": market.max_context_event_lag_30s_ms,
            "higher_timeframe_context_age_seconds": market.higher_timeframe_context_age_seconds,
            "higher_timeframe_bias_side": market.higher_timeframe_bias_side,
            "higher_timeframe_bias_strength": market.higher_timeframe_bias_strength,
            "higher_timeframe_bias_reason": market.higher_timeframe_bias_reason,
            "higher_timeframe_context": market.higher_timeframe_context,
            "cross_market_context_age_seconds": market.cross_market_context_age_seconds,
            "btc_order_flow_imbalance_1s": market.btc_order_flow_imbalance_1s,
            "btc_taker_aggression_imbalance_1s": market.btc_taker_aggression_imbalance_1s,
            "btc_microprice_mid_bps": market.btc_microprice_mid_bps,
            "btc_return_15s_pct": market.btc_return_15s_pct,
            "btc_return_60s_pct": market.btc_return_60s_pct,
            "eth_btc_relative_return_15s_pct": market.eth_btc_relative_return_15s_pct,
            "eth_btc_relative_return_60s_pct": market.eth_btc_relative_return_60s_pct,
            "cross_market_context": market.cross_market_context,
        }
