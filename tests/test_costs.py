import pytest

from futures_lab.config import Settings
from futures_lab.costs import estimate_effective_cost
from futures_lab.models import MarketState


def _market(spread_bps: float = 1.0) -> MarketState:
    return MarketState(symbol="ETHUSDT", spread_bps=spread_bps)


def test_effective_cost_models_taker_round_trip_costs():
    settings = Settings(
        MAKER_FEE_BPS=2.0,
        TAKER_FEE_BPS=4.0,
        DEFAULT_ENTRY_ORDER_TYPE="taker",
        DEFAULT_EXIT_ORDER_TYPE="taker",
        EXPECTED_SLIPPAGE_BPS=0.5,
        LATENCY_ADVERSE_SELECTION_BPS=0.5,
    )

    cost = estimate_effective_cost(settings, _market(spread_bps=1.0))

    assert cost.entry_fee_bps == 4.0
    assert cost.exit_fee_bps == 4.0
    assert cost.fee_bps == 8.0
    assert cost.spread_cross_bps == 1.0
    assert cost.slippage_bps == 1.0
    assert cost.latency_penalty_bps == 0.5
    assert cost.total_cost_bps == pytest.approx(10.5)
    assert cost.required_target_pct(2.0) == pytest.approx(0.0021)


def test_effective_cost_models_maker_taker_hybrid_costs():
    settings = Settings(
        MAKER_FEE_BPS=2.0,
        TAKER_FEE_BPS=4.0,
        DEFAULT_ENTRY_ORDER_TYPE="maker",
        DEFAULT_EXIT_ORDER_TYPE="taker",
        EXPECTED_SLIPPAGE_BPS=0.5,
        LATENCY_ADVERSE_SELECTION_BPS=0.5,
    )

    cost = estimate_effective_cost(settings, _market(spread_bps=1.0))

    assert cost.entry_fee_bps == 2.0
    assert cost.exit_fee_bps == 4.0
    assert cost.spread_cross_bps == 0.5
    assert cost.slippage_bps == 0.5
    assert cost.total_cost_bps == pytest.approx(7.5)
