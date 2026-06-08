from __future__ import annotations

from datetime import datetime

from futures_lab.config import Settings
from futures_lab.models import MarketState, Side


def build_cross_market_context(
    settings: Settings,
    primary: MarketState,
    anchor: MarketState,
    updated_at: datetime | None = None,
) -> dict:
    relative_15s = _relative(primary.return_15s_pct, anchor.return_15s_pct)
    relative_60s = _relative(primary.return_60s_pct, anchor.return_60s_pct)
    contradiction = {
        "long": _btc_contradicts(Side.long, anchor),
        "short": _btc_contradicts(Side.short, anchor),
    }
    eth_strength = {
        "long": _eth_strength(Side.long, primary, anchor, relative_15s, relative_60s),
        "short": _eth_strength(Side.short, primary, anchor, relative_15s, relative_60s),
    }
    return {
        "enabled": settings.cross_market_enabled,
        "anchor_symbol": settings.cross_market_anchor_symbol.upper(),
        "updated_at": updated_at.isoformat() if updated_at is not None else None,
        "anchor": {
            "symbol": anchor.symbol,
            "mid_price": anchor.mid_price,
            "return_15s_pct": anchor.return_15s_pct,
            "return_60s_pct": anchor.return_60s_pct,
            "order_flow_imbalance_1s": anchor.order_flow_imbalance_1s,
            "order_flow_imbalance_5s": anchor.order_flow_imbalance_5s,
            "taker_aggression_imbalance_1s": anchor.taker_aggression_imbalance_1s,
            "taker_aggression_imbalance_5s": anchor.taker_aggression_imbalance_5s,
            "microprice_mid_bps": anchor.microprice_mid_bps,
            "vamp_mid_bps": anchor.vamp_mid_bps,
            "spread_bps": anchor.spread_bps,
            "exchange_event_lag_ms": anchor.exchange_event_lag_ms,
            "avg_event_lag_30s_ms": anchor.avg_event_lag_30s_ms,
            "hot_event_lag_ms": anchor.hot_event_lag_ms,
            "avg_hot_event_lag_30s_ms": anchor.avg_hot_event_lag_30s_ms,
            "book_event_lag_ms": anchor.book_event_lag_ms,
            "avg_book_event_lag_30s_ms": anchor.avg_book_event_lag_30s_ms,
            "trade_event_lag_ms": anchor.trade_event_lag_ms,
            "avg_trade_event_lag_30s_ms": anchor.avg_trade_event_lag_30s_ms,
            "context_event_lag_ms": anchor.context_event_lag_ms,
            "avg_context_event_lag_30s_ms": anchor.avg_context_event_lag_30s_ms,
        },
        "relative": {
            "primary_symbol": primary.symbol,
            "anchor_symbol": anchor.symbol,
            "return_15s_pct": relative_15s,
            "return_60s_pct": relative_60s,
        },
        "contradiction": contradiction,
        "eth_strength": eth_strength,
    }


def cross_market_gate(settings: Settings, market: MarketState, side: Side, score: float) -> dict:
    context = market.cross_market_context or {}
    enabled = settings.cross_market_enabled
    stale = _context_stale(settings, market)
    side_key = side.value
    contradiction = bool((context.get("contradiction") or {}).get(side_key))
    strength = float((context.get("eth_strength") or {}).get(side_key) or 0.0)
    moderate_signal = score < settings.cross_market_btc_veto_score_ceiling
    extreme_override = score >= settings.cross_market_relative_strength_override_score and strength >= settings.cross_market_min_relative_strength
    allowed = True
    blocker = None
    if enabled and not stale and contradiction and moderate_signal and not extreme_override:
        allowed = False
        blocker = "btc_microstructure_contradiction"
    return {
        "enabled": enabled,
        "allowed": allowed,
        "blocker": blocker,
        "stale": stale,
        "side": side_key,
        "score": score,
        "moderate_signal": moderate_signal,
        "contradiction": contradiction,
        "eth_strength": round(strength, 4),
        "min_relative_strength": settings.cross_market_min_relative_strength,
        "override_score": settings.cross_market_relative_strength_override_score,
        "extreme_override": extreme_override,
        "context": context,
    }


def _relative(primary: float | None, anchor: float | None) -> float | None:
    if primary is None or anchor is None:
        return None
    return primary - anchor


def _btc_contradicts(side: Side, anchor: MarketState) -> bool:
    ofi = anchor.order_flow_imbalance_1s or 0.0
    aggression = anchor.taker_aggression_imbalance_1s or 0.0
    micro = anchor.microprice_mid_bps or 0.0
    ret = anchor.return_15s_pct or 0.0
    pressure = 0.35 * ofi + 0.35 * aggression + 0.20 * _scaled_bps(micro) + 0.10 * _scaled_return(ret)
    if side == Side.long:
        return pressure <= -0.35
    return pressure >= 0.35


def _eth_strength(
    side: Side,
    primary: MarketState,
    anchor: MarketState,
    relative_15s: float | None,
    relative_60s: float | None,
) -> float:
    sign = 1.0 if side == Side.long else -1.0
    primary_pressure = (
        0.35 * sign * (primary.order_flow_imbalance_1s or 0.0)
        + 0.35 * sign * (primary.taker_aggression_imbalance_1s or 0.0)
        + 0.15 * sign * _scaled_bps(primary.microprice_mid_bps or 0.0)
        + 0.15 * sign * _scaled_return(primary.return_15s_pct or 0.0)
    )
    relative_pressure = (
        0.55 * sign * _scaled_return(relative_15s or 0.0)
        + 0.45 * sign * _scaled_return(relative_60s or 0.0)
    )
    anchor_decelerates = _anchor_pressure_decelerates(side, anchor)
    score = 0.65 * _unit(primary_pressure) + 0.25 * _unit(relative_pressure)
    if anchor_decelerates:
        score += 0.10
    return max(0.0, min(1.0, score))


def _anchor_pressure_decelerates(side: Side, anchor: MarketState) -> bool:
    one = anchor.taker_aggression_imbalance_1s
    five = anchor.taker_aggression_imbalance_5s
    if one is None or five is None:
        return False
    if side == Side.long:
        return five < -0.25 and one > five
    return five > 0.25 and one < five


def _context_stale(settings: Settings, market: MarketState) -> bool:
    if not market.cross_market_context:
        return True
    age = market.cross_market_context_age_seconds
    if age is None:
        return True
    lag_ms = (market.cross_market_context.get("anchor") or {}).get("avg_book_event_lag_30s_ms")
    if lag_ms is None:
        lag_ms = (market.cross_market_context.get("anchor") or {}).get("book_event_lag_ms")
    if lag_ms is None:
        lag_ms = (market.cross_market_context.get("anchor") or {}).get("avg_hot_event_lag_30s_ms")
    if lag_ms is None:
        lag_ms = (market.cross_market_context.get("anchor") or {}).get("hot_event_lag_ms")
    if lag_ms is None:
        lag_ms = (market.cross_market_context.get("anchor") or {}).get("exchange_event_lag_ms")
    if lag_ms is not None and float(lag_ms) > settings.max_exchange_event_lag_ms:
        return True
    return age > max(1.0, settings.cross_market_max_context_age_seconds)


def _scaled_bps(value: float) -> float:
    return max(-1.0, min(1.0, value / 2.0))


def _scaled_return(value: float) -> float:
    return max(-1.0, min(1.0, value / 0.0015))


def _unit(value: float) -> float:
    return max(0.0, min(1.0, (value + 1.0) / 2.0))
