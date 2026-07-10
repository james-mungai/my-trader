import json

from futures_lab.config import Settings
from futures_lab.models import Decision, DecisionAction, MarketState, RiskVerdict, TradeMode
from futures_lab.recon_log import ReconLogger


def test_blocked_trade_proposals_are_sampled_but_allowed_entries_are_not(tmp_path) -> None:
    settings = Settings(
        DATA_DIR=tmp_path,
        DECISION_LOG_WAIT_SAMPLE_INTERVAL=3,
    )
    logger = ReconLogger(settings)
    market = MarketState(symbol="ETHUSDT", connected=True, mid_price=100.0)
    decision = Decision(
        symbol="ETHUSDT",
        action=DecisionAction.propose_long,
        mode=TradeMode.fast,
        confidence=0.6,
        reason="test proposal",
    )
    blocked = RiskVerdict(allowed=False, reason="blocked", blockers=["paper position already open"])

    logger.write_decision(market, decision, blocked)
    logger.write_decision(market, decision, blocked)
    assert list(logger.decisions_dir.glob("*.jsonl")) == []

    logger.write_decision(market, decision, blocked)
    path = next(logger.decisions_dir.glob("*.jsonl"))
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1

    allowed = RiskVerdict(allowed=True, reason="allowed")
    logger.write_decision(market, decision, allowed)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[-1]["risk_allowed"] is True
