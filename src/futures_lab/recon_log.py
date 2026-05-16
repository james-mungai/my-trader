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
        self.features_dir = Path(self.settings.data_dir) / "features"
        self.decisions_dir = Path(self.settings.data_dir) / "decisions"
        self.paper_dir = Path(self.settings.data_dir) / "paper_trades"
        self.shadow_dir = Path(self.settings.data_dir) / "shadow_trades"
        for path in (self.features_dir, self.decisions_dir, self.paper_dir, self.shadow_dir):
            path.mkdir(parents=True, exist_ok=True)

    def write_feature(self, market: MarketState) -> None:
        if not self.settings.write_feature_log:
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
            "mark_price": market.mark_price,
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
            "last_stream_event_type": market.last_stream_event_type,
            "exchange_event_lag_ms": market.exchange_event_lag_ms,
            "avg_event_lag_30s_ms": market.avg_event_lag_30s_ms,
            "max_event_lag_30s_ms": market.max_event_lag_30s_ms,
        }
