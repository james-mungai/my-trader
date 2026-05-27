from datetime import datetime, timezone

from futures_lab.config import Settings
from futures_lab.cross_market import build_cross_market_context, cross_market_gate
from futures_lab.edge_router import EdgeRouter
from futures_lab.models import MarketState, Regime, Side
from futures_lab.strategy import HitAndRunStrategy


def _market(symbol: str = "ETHUSDT", price: float = 100.0, **overrides) -> MarketState:
    base = {
        "symbol": symbol,
        "connected": True,
        "data_age_seconds": 0.1,
        "observed_seconds": 300,
        "best_bid": price - 0.01,
        "best_ask": price + 0.01,
        "best_bid_qty": 12.0,
        "best_ask_qty": 8.0,
        "mid_price": price,
        "spread_bps": 0.5,
        "last_trade_price": price,
        "mark_price": price,
        "funding_rate": 0.0,
        "return_15s_pct": 0.0004,
        "return_60s_pct": 0.0008,
        "return_180s_pct": 0.0002,
        "realized_vol_60s_pct": 0.00004,
        "realized_vol_180s_pct": 0.00006,
        "range_180s_pct": 0.007,
        "range_position_180s": 0.05,
        "taker_buy_ratio_10s": 0.78,
        "taker_buy_ratio_30s": 0.66,
        "book_imbalance_top": 0.2,
        "order_flow_imbalance_1s": 0.70,
        "order_flow_imbalance_5s": 0.50,
        "taker_aggression_imbalance_1s": 0.72,
        "taker_aggression_imbalance_5s": 0.45,
        "microprice_mid_bps": 0.4,
        "vamp_mid_bps": 0.35,
        "depth_imbalance_top5": 0.4,
        "bid_depth_refill_rate_5s": 0.2,
        "ask_depth_evaporation_rate_5s": 0.2,
        "regime": Regime.sideways,
    }
    base.update(overrides)
    return MarketState(**base)


def test_cross_market_context_builds_btc_anchor_and_relative_strength():
    settings = Settings(CROSS_MARKET_ENABLED=True, CROSS_MARKET_ANCHOR_SYMBOL="BTCUSDT")
    eth = _market(return_15s_pct=0.0010, return_60s_pct=0.0015)
    btc = _market(
        symbol="BTCUSDT",
        price=100_000.0,
        return_15s_pct=-0.0005,
        return_60s_pct=-0.0002,
        order_flow_imbalance_1s=-0.70,
        taker_aggression_imbalance_1s=-0.70,
        microprice_mid_bps=-0.5,
    )

    context = build_cross_market_context(
        settings,
        eth,
        btc,
        updated_at=datetime(2026, 5, 27, tzinfo=timezone.utc),
    )

    assert context["anchor"]["symbol"] == "BTCUSDT"
    assert context["relative"]["return_15s_pct"] == 0.0015
    assert context["contradiction"]["long"] is True
    assert context["eth_strength"]["long"] > 0.5


def test_cross_market_gate_blocks_moderate_eth_long_when_btc_sells_hard():
    settings = Settings(CROSS_MARKET_ENABLED=True)
    market = _market(
        cross_market_context={
            "contradiction": {"long": True, "short": False},
            "eth_strength": {"long": 0.55, "short": 0.2},
        },
        cross_market_context_age_seconds=0.2,
    )

    gate = cross_market_gate(settings, market, Side.long, score=0.80)

    assert gate["allowed"] is False
    assert gate["blocker"] == "btc_microstructure_contradiction"


def test_cross_market_gate_allows_extreme_eth_relative_strength_override():
    settings = Settings(CROSS_MARKET_ENABLED=True)
    market = _market(
        cross_market_context={
            "contradiction": {"long": True, "short": False},
            "eth_strength": {"long": 0.90, "short": 0.2},
        },
        cross_market_context_age_seconds=0.2,
    )

    gate = cross_market_gate(settings, market, Side.long, score=0.95)

    assert gate["allowed"] is True
    assert gate["extreme_override"] is True


def test_edge_router_adds_btc_veto_blocker_to_moderate_candidate():
    settings = Settings(CROSS_MARKET_ENABLED=True, EDGE_ROUTER_MIN_SCORE=0.5, EDGE_ROUTER_MIN_EV_BPS=-100)
    market = _market(
        cross_market_context={
            "contradiction": {"long": True, "short": False},
            "eth_strength": {"long": 0.4, "short": 0.2},
        },
        cross_market_context_age_seconds=0.1,
    )

    evidence = EdgeRouter(settings).evaluate(market)
    long_candidate = next(candidate for candidate in evidence["candidates"] if candidate["strategy"] == "taker_impulse_long")

    assert "btc_microstructure_contradiction" in long_candidate["blockers"]


def test_strategy_logs_cross_market_context_in_decisions():
    settings = Settings(CROSS_MARKET_ENABLED=True, MIN_CONFIDENCE=0.70)
    strategy = HitAndRunStrategy(settings)
    market = _market(
        cross_market_context={
            "anchor": {"symbol": "BTCUSDT", "order_flow_imbalance_1s": -0.5},
            "relative": {"return_15s_pct": 0.001, "return_60s_pct": 0.0012},
            "contradiction": {"long": False, "short": True},
            "eth_strength": {"long": 0.75, "short": 0.2},
        },
        cross_market_context_age_seconds=0.2,
        btc_order_flow_imbalance_1s=-0.5,
        eth_btc_relative_return_15s_pct=0.001,
    )

    decision = strategy.decide(market)

    logged = decision.evidence["cross_market_context"]
    assert logged["enabled"] is True
    assert logged["anchor_symbol"] == "BTCUSDT"
    assert logged["btc_order_flow_imbalance_1s"] == -0.5
    assert logged["eth_btc_relative_return_15s_pct"] == 0.001
