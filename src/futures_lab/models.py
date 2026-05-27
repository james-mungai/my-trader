from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Side(str, Enum):
    long = "long"
    short = "short"


class DecisionAction(str, Enum):
    wait = "wait"
    propose_long = "propose_long"
    propose_short = "propose_short"
    close = "close"


class TradeMode(str, Enum):
    fast = "fast"
    slow = "slow"


class Regime(str, Enum):
    warming_up = "warming_up"
    stale = "stale"
    sideways = "sideways"
    directional = "directional"
    volatile = "volatile"
    unknown = "unknown"


class MarketState(BaseModel):
    symbol: str
    connected: bool = False
    last_event_at: datetime | None = None
    last_received_at: datetime | None = None
    data_age_seconds: float | None = None
    observed_seconds: float = 0.0
    best_bid: float | None = None
    best_ask: float | None = None
    best_bid_qty: float | None = None
    best_ask_qty: float | None = None
    depth_bid_qty_top5: float | None = None
    depth_ask_qty_top5: float | None = None
    depth_imbalance_top5: float | None = None
    depth_bid_wall_ratio_top5: float | None = None
    depth_ask_wall_ratio_top5: float | None = None
    mid_price: float | None = None
    spread_bps: float | None = None
    spread_bps_avg_5s: float | None = None
    spread_bps_std_5s: float | None = None
    spread_bps_max_5s: float | None = None
    last_trade_price: float | None = None
    mark_price: float | None = None
    mark_last_basis_bps: float | None = None
    funding_rate: float | None = None
    open_interest: float | None = None
    open_interest_age_seconds: float | None = None
    open_interest_change_5m_pct: float | None = None
    return_15s_pct: float | None = None
    return_60s_pct: float | None = None
    return_180s_pct: float | None = None
    realized_vol_60s_pct: float | None = None
    realized_vol_180s_pct: float | None = None
    range_180s_pct: float | None = None
    range_high_180s: float | None = None
    range_low_180s: float | None = None
    range_position_180s: float | None = None
    order_flow_imbalance_250ms: float | None = None
    order_flow_imbalance_1s: float | None = None
    order_flow_imbalance_5s: float | None = None
    taker_aggression_imbalance_1s: float | None = None
    taker_aggression_imbalance_5s: float | None = None
    taker_aggression_imbalance_15s: float | None = None
    taker_buy_ratio_10s: float | None = None
    taker_buy_ratio_30s: float | None = None
    book_imbalance_top: float | None = None
    microprice: float | None = None
    microprice_mid_bps: float | None = None
    vamp_price_top: float | None = None
    vamp_mid_bps: float | None = None
    weighted_depth_price_top: float | None = None
    weighted_depth_mid_bps: float | None = None
    bid_depth_refill_rate_5s: float | None = None
    ask_depth_refill_rate_5s: float | None = None
    bid_depth_evaporation_rate_5s: float | None = None
    ask_depth_evaporation_rate_5s: float | None = None
    liquidation_notional_30s: float | None = None
    long_liquidation_notional_30s: float | None = None
    short_liquidation_notional_30s: float | None = None
    liquidation_buy_ratio_30s: float | None = None
    last_stream_event_type: str | None = None
    exchange_event_lag_ms: float | None = None
    avg_event_lag_30s_ms: float | None = None
    max_event_lag_30s_ms: float | None = None
    higher_timeframe_context: dict[str, Any] = Field(default_factory=dict)
    higher_timeframe_context_age_seconds: float | None = None
    higher_timeframe_bias_side: str = "neutral"
    higher_timeframe_bias_strength: float = 0.0
    higher_timeframe_bias_reason: str = ""
    regime: Regime = Regime.unknown


class Decision(BaseModel):
    timestamp: datetime = Field(default_factory=utc_now)
    symbol: str
    action: DecisionAction
    mode: TradeMode | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    entry_price: float | None = None
    take_profit_price: float | None = None
    stop_loss_price: float | None = None
    target_move_pct: float | None = None
    stop_move_pct: float | None = None
    leverage: int | None = None
    stake_usd: float | None = None
    notional_usd: float | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)


class RiskVerdict(BaseModel):
    allowed: bool
    reason: str
    blockers: list[str] = Field(default_factory=list)


class PaperPosition(BaseModel):
    symbol: str
    side: Side
    mode: TradeMode
    trade_profile: str = "unknown"
    exit_policy: str = "fixed_tp_stop"
    entry_price: float
    quantity: float
    stake_usd: float
    notional_usd: float
    leverage: int
    take_profit_price: float
    stop_loss_price: float
    opened_at: datetime
    confidence: float
    max_favorable_move_pct: float = 0.0
    max_adverse_move_pct: float = 0.0


class PaperTrade(BaseModel):
    symbol: str
    side: Side
    mode: TradeMode
    trade_profile: str = "unknown"
    exit_policy: str = "fixed_tp_stop"
    entry_price: float
    exit_price: float
    quantity: float
    stake_usd: float
    notional_usd: float
    leverage: int
    gross_pnl_usd: float
    fees_usd: float
    net_pnl_usd: float
    exit_reason: str
    opened_at: datetime
    closed_at: datetime
    exit_shadow: dict[str, Any] = Field(default_factory=dict)


class PaperState(BaseModel):
    day: str
    account_equity_usd: float
    realized_pnl_usd: float
    trades_today: int
    daily_target_hit: bool
    daily_max_loss_hit: bool
    open_position: PaperPosition | None = None
    last_trade: PaperTrade | None = None

