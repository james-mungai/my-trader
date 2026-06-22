from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from decimal import Decimal
from time import monotonic
from typing import Any

from futures_lab.binance_private import BinancePrivateClient, BinancePrivateError
from futures_lab.config import Settings
from futures_lab.models import Decision, DecisionAction
from futures_lab.runtime import TradingRuntime


@dataclass
class LiveCanarySummary:
    ok: bool
    symbol: str
    seconds_requested: int
    max_trades: int
    live_opens: int = 0
    live_closes: int = 0
    paper_opens: int = 0
    paper_closes: int = 0
    decisions: int = 0
    risk_allowed: int = 0
    started_wallet_balance: str | None = None
    ended_wallet_balance: str | None = None
    wallet_delta_usd: str | None = None
    final_position: dict[str, Any] | None = None
    open_orders_count: int = 0
    stop_reason: str = "completed"
    events: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "symbol": self.symbol,
            "seconds_requested": self.seconds_requested,
            "max_trades": self.max_trades,
            "live_opens": self.live_opens,
            "live_closes": self.live_closes,
            "paper_opens": self.paper_opens,
            "paper_closes": self.paper_closes,
            "decisions": self.decisions,
            "risk_allowed": self.risk_allowed,
            "started_wallet_balance": self.started_wallet_balance,
            "ended_wallet_balance": self.ended_wallet_balance,
            "wallet_delta_usd": self.wallet_delta_usd,
            "final_position": self.final_position,
            "open_orders_count": self.open_orders_count,
            "stop_reason": self.stop_reason,
            "events": self.events,
        }


@dataclass
class LiveProfitProtection:
    previous_close_wallet: Decimal
    peak_profit_usd: Decimal = Decimal("0")
    consecutive_losses: int = 0


@dataclass
class LiveProfitProtectionVerdict:
    stop: bool
    reason: str | None
    state: LiveProfitProtection
    payload: dict[str, Any]


def validate_live_canary_settings(settings: Settings, max_trades: int) -> None:
    if not settings.live_trading_enabled:
        raise BinancePrivateError("LIVE_TRADING_ENABLED must be true before running live canary.")
    if settings.live_dry_run:
        raise BinancePrivateError("LIVE_DRY_RUN must be false before running live canary.")
    if max_trades <= 0:
        raise BinancePrivateError("max live canary trades must be positive.")
    if settings.live_max_trades_per_day > 0 and max_trades > settings.live_max_trades_per_day:
        raise BinancePrivateError(
            f"Requested canary max trades {max_trades} exceeds LIVE_MAX_TRADES_PER_DAY "
            f"{settings.live_max_trades_per_day}."
        )
    if settings.live_max_open_positions != 1:
        raise BinancePrivateError("LIVE_MAX_OPEN_POSITIONS must be 1 for live canary.")
    if settings.live_canary_max_notional_usd <= 0:
        raise BinancePrivateError("LIVE_CANARY_MAX_NOTIONAL_USD must be positive.")
    if settings.live_canary_max_loss_usd <= 0:
        raise BinancePrivateError("LIVE_CANARY_MAX_LOSS_USD must be positive.")
    if settings.live_max_notional_usd > settings.live_canary_max_notional_usd:
        raise BinancePrivateError(
            f"LIVE_MAX_NOTIONAL_USD {settings.live_max_notional_usd} exceeds "
            f"LIVE_CANARY_MAX_NOTIONAL_USD {settings.live_canary_max_notional_usd}."
        )
    if settings.live_daily_max_loss_usd > settings.live_canary_max_loss_usd:
        raise BinancePrivateError(
            f"LIVE_DAILY_MAX_LOSS_USD {settings.live_daily_max_loss_usd} exceeds "
            f"LIVE_CANARY_MAX_LOSS_USD {settings.live_canary_max_loss_usd}."
        )
    if settings.live_canary_position_check_seconds <= 0:
        raise BinancePrivateError("LIVE_CANARY_POSITION_CHECK_SECONDS must be positive.")
    if settings.live_canary_max_consecutive_losses < 0:
        raise BinancePrivateError("LIVE_CANARY_MAX_CONSECUTIVE_LOSSES must be non-negative.")
    if settings.live_canary_profit_lock_min_profit_usd < 0:
        raise BinancePrivateError("LIVE_CANARY_PROFIT_LOCK_MIN_PROFIT_USD must be non-negative.")
    if not 0 <= settings.live_canary_max_profit_giveback_fraction <= 1:
        raise BinancePrivateError("LIVE_CANARY_MAX_PROFIT_GIVEBACK_FRACTION must be between 0 and 1.")


def live_order_side_from_decision(decision: Decision) -> str | None:
    if decision.action == DecisionAction.propose_long:
        return "BUY"
    if decision.action == DecisionAction.propose_short:
        return "SELL"
    return None


async def run_live_canary(
    settings: Settings,
    *,
    seconds: int,
    max_trades: int,
    stop_requested: asyncio.Event | None = None,
) -> dict[str, Any]:
    validate_live_canary_settings(settings, max_trades)
    symbol = settings.symbol.upper()
    client = BinancePrivateClient(settings)
    preflight = client.live_preflight(symbol=symbol)
    if not preflight.get("ok"):
        raise BinancePrivateError(f"Live preflight failed: {preflight.get('hard_failures')}")

    start_wallet = str(preflight["account"].get("total_wallet_balance") or "0")
    summary = LiveCanarySummary(
        ok=False,
        symbol=symbol,
        seconds_requested=seconds,
        max_trades=max_trades,
        started_wallet_balance=start_wallet,
    )
    runtime = TradingRuntime.create(settings)
    live_position_open = False
    profit_protection = LiveProfitProtection(previous_close_wallet=Decimal(start_wallet))
    deadline = monotonic() + max(0, seconds)
    interval = max(0.1, settings.decision_interval_ms / 1000)
    next_position_check = monotonic()

    runtime.recorder.start()
    runtime.context.start()
    runtime.audit.write(
        "live_canary_start",
        {
            "symbol": symbol,
            "seconds": seconds,
            "max_trades": max_trades,
            "live_max_notional_usd": settings.live_max_notional_usd,
            "live_daily_max_loss_usd": settings.live_daily_max_loss_usd,
            "live_profit_protection_enabled": settings.live_canary_profit_protection_enabled,
            "live_max_consecutive_losses": settings.live_canary_max_consecutive_losses,
            "live_profit_lock_min_profit_usd": settings.live_canary_profit_lock_min_profit_usd,
            "live_max_profit_giveback_fraction": settings.live_canary_max_profit_giveback_fraction,
        },
    )
    try:
        while monotonic() < deadline and not (stop_requested is not None and stop_requested.is_set()):
            loop_started = monotonic()
            market, decision, risk = runtime.decide_once()
            summary.decisions += 1
            if risk.allowed:
                summary.risk_allowed += 1
            runtime.recon_log.write_feature(market)
            runtime.recon_log.write_decision(
                market,
                decision,
                risk,
                decision_latency_ms=(monotonic() - loop_started) * 1000,
            )

            if live_position_open and monotonic() >= next_position_check:
                position_check = _live_position_loss_check(client, symbol, settings.live_canary_intratrade_max_loss_usd)
                next_position_check = monotonic() + settings.live_canary_position_check_seconds
                if position_check is not None:
                    runtime.audit.write("live_canary_position_check", position_check)
                    summary.events.append({"event": "live_position_check", "payload": position_check})
                    if position_check["loss_limit_hit"]:
                        flatten = client.flatten_position(symbol=symbol)
                        live_position_open = False
                        summary.live_closes += int(bool(flatten.get("live_order_placed")))
                        summary.events.append({"event": "live_flatten_on_intratrade_loss", "payload": flatten})
                        runtime.audit.write("live_canary_flatten", flatten)
                        summary.stop_reason = "live_intratrade_max_loss_hit"
                        break

            closed = runtime.paper.mark(market)
            if closed is not None:
                summary.paper_closes += 1
                runtime.audit.write("paper_close", closed.model_dump())
                runtime.recon_log.write_paper_trade(closed)
                if live_position_open:
                    flatten = client.flatten_position(symbol=symbol)
                    live_position_open = False
                    summary.live_closes += int(bool(flatten.get("live_order_placed")))
                    summary.events.append({"event": "live_flatten_on_paper_close", "payload": flatten})
                    runtime.audit.write("live_canary_flatten", flatten)
                    if _wallet_loss_exceeded(start_wallet, flatten.get("wallet_balance_after"), settings.live_daily_max_loss_usd):
                        summary.stop_reason = "live_daily_max_loss_hit"
                        break
                    protection_verdict = _live_profit_protection_check(
                        settings,
                        start_wallet=start_wallet,
                        current_wallet=flatten.get("wallet_balance_after"),
                        state=profit_protection,
                    )
                    profit_protection = protection_verdict.state
                    runtime.audit.write("live_canary_profit_protection", protection_verdict.payload)
                    summary.events.append({"event": "live_profit_protection", "payload": protection_verdict.payload})
                    if protection_verdict.stop:
                        summary.stop_reason = protection_verdict.reason or "live_profit_protection_hit"
                        break
                if summary.live_closes >= max_trades:
                    summary.stop_reason = "max_live_trades_hit"
                    break

            for event, payload in runtime.paper.drain_audit_events():
                runtime.audit.write(event, payload)
            for event in runtime.shadow.mark(market):
                runtime.audit.write("shadow_trade", event)
                runtime.recon_log.write_shadow_trade(event)
            for event in runtime.regime_outcomes.mark(market):
                runtime.audit.write("regime_outcome", event)
                runtime.recon_log.write_regime_outcome(event)
            for event in runtime.candidate_outcomes.mark(market):
                runtime.audit.write("candidate_outcome", event)
                runtime.recon_log.write_candidate_outcome(event)

            order_side = live_order_side_from_decision(decision)
            if (
                risk.allowed
                and order_side is not None
                and not live_position_open
                and summary.live_opens < max_trades
            ):
                live_open = client.live_dust_open(order_side, symbol=symbol)
                live_position_open = True
                summary.live_opens += 1
                summary.events.append({"event": "live_open", "payload": live_open})
                runtime.audit.write("live_canary_open", live_open)
                opened = runtime.paper.open_from_decision(
                    decision,
                    opened_at=market.last_received_at,
                    notional_usd=_filled_notional_usd(live_open),
                )
                if opened is None:
                    flatten = client.flatten_position(symbol=symbol)
                    live_position_open = False
                    summary.live_closes += int(bool(flatten.get("live_order_placed")))
                    runtime.audit.write("live_canary_flatten_after_paper_open_failure", flatten)
                    raise BinancePrivateError("Live opened but paper broker did not open; flattened immediately.")
                summary.paper_opens += 1
                runtime.audit.write(
                    "paper_open",
                    {
                        "symbol": opened.symbol,
                        "side": opened.side.value,
                        "mode": opened.mode.value,
                        "trade_profile": opened.trade_profile,
                        "exit_policy": opened.exit_policy,
                        "entry_price": opened.entry_price,
                        "take_profit_price": opened.take_profit_price,
                        "stop_loss_price": opened.stop_loss_price,
                        "structural_invalidation_price": opened.structural_invalidation_price,
                        "structural_adverse_move_pct": opened.structural_adverse_move_pct,
                        "confidence": opened.confidence,
                        "leverage": opened.leverage,
                        "notional_usd": opened.notional_usd,
                    },
                )

            shadow_opened = runtime.shadow.open_from_decision(decision, opened_at=market.last_received_at)
            if shadow_opened is not None:
                runtime.audit.write("shadow_trade", shadow_opened)
                runtime.recon_log.write_shadow_trade(shadow_opened)
            regime_opened = runtime.regime_outcomes.open_from_decision(decision, market, opened_at=market.last_received_at)
            if regime_opened is not None:
                runtime.audit.write("regime_outcome", regime_opened)
                runtime.recon_log.write_regime_outcome(regime_opened)
            for event in runtime.candidate_outcomes.open_from_decision(decision, market, opened_at=market.last_received_at):
                runtime.audit.write("candidate_outcome", event)
                runtime.recon_log.write_candidate_outcome(event)

            sleep_for = interval - (monotonic() - loop_started)
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
        else:
            summary.stop_reason = "time_elapsed"
        if stop_requested is not None and stop_requested.is_set() and summary.stop_reason == "completed":
            summary.stop_reason = "signal_requested"
    finally:
        if live_position_open:
            flatten = client.flatten_position(symbol=symbol)
            live_position_open = False
            summary.live_closes += int(bool(flatten.get("live_order_placed")))
            summary.events.append({"event": "live_flatten_on_shutdown", "payload": flatten})
            runtime.audit.write("live_canary_flatten", flatten)
        if runtime.paper.open_position is not None:
            closed = runtime.paper.close_open_position(runtime.latest_market, reason="live_canary_session_end")
            if closed is not None:
                summary.paper_closes += 1
                runtime.audit.write("paper_close", closed.model_dump())
                runtime.recon_log.write_paper_trade(closed)
        await runtime.recorder.stop()
        await runtime.context.stop()
        for event, payload in runtime.paper.close_range_exit_counterfactuals(runtime.latest_market, reason="live_canary_session_end"):
            runtime.audit.write(event, payload)
        for event in runtime.shadow.close_all(runtime.latest_market, reason="live_canary_session_end"):
            runtime.audit.write("shadow_trade", event)
            runtime.recon_log.write_shadow_trade(event)
        for event in runtime.regime_outcomes.close_all(runtime.latest_market, reason="live_canary_session_end"):
            runtime.audit.write("regime_outcome", event)
            runtime.recon_log.write_regime_outcome(event)
        for event in runtime.candidate_outcomes.close_all(runtime.latest_market, reason="live_canary_session_end"):
            runtime.audit.write("candidate_outcome", event)
            runtime.recon_log.write_candidate_outcome(event)

    final = client.live_preflight(symbol=symbol)
    final_position = final["position_risk"][0] if final.get("position_risk") else None
    end_wallet = str(final["account"].get("total_wallet_balance") or "0")
    summary.ok = bool(final.get("ok"))
    summary.ended_wallet_balance = end_wallet
    summary.wallet_delta_usd = _decimal_delta(start_wallet, end_wallet)
    summary.final_position = final_position
    summary.open_orders_count = int(final.get("open_orders_count") or 0)
    runtime.audit.write("live_canary_stop", summary.as_dict())
    return summary.as_dict()


def _decimal_delta(start: object, end: object) -> str:
    return str((Decimal(str(end)) - Decimal(str(start))).normalize())


def _wallet_loss_exceeded(start: object, current: object, max_loss_usd: float) -> bool:
    if current is None:
        return False
    return Decimal(str(current)) - Decimal(str(start)) <= -abs(Decimal(str(max_loss_usd)))


def _live_profit_protection_check(
    settings: Settings,
    *,
    start_wallet: object,
    current_wallet: object,
    state: LiveProfitProtection,
) -> LiveProfitProtectionVerdict:
    current = _decimal_or_none(current_wallet)
    if current is None:
        return LiveProfitProtectionVerdict(
            stop=False,
            reason=None,
            state=state,
            payload={
                "enabled": settings.live_canary_profit_protection_enabled,
                "skipped": "missing_current_wallet",
                "consecutive_losses": state.consecutive_losses,
                "peak_profit_usd": str(state.peak_profit_usd),
            },
        )

    start = Decimal(str(start_wallet))
    trade_delta = current - state.previous_close_wallet
    current_profit = current - start
    peak_profit = max(state.peak_profit_usd, current_profit)
    consecutive_losses = state.consecutive_losses + 1 if trade_delta < 0 else 0
    new_state = LiveProfitProtection(
        previous_close_wallet=current,
        peak_profit_usd=peak_profit,
        consecutive_losses=consecutive_losses,
    )

    max_losses = settings.live_canary_max_consecutive_losses
    min_profit = Decimal(str(settings.live_canary_profit_lock_min_profit_usd))
    max_giveback_fraction = Decimal(str(settings.live_canary_max_profit_giveback_fraction))
    allowed_profit_after_giveback = peak_profit * (Decimal("1") - max_giveback_fraction)
    giveback = peak_profit - current_profit
    stop_reason: str | None = None
    if settings.live_canary_profit_protection_enabled:
        if max_losses > 0 and consecutive_losses >= max_losses:
            stop_reason = "live_consecutive_losses_hit"
        elif peak_profit >= min_profit and current_profit <= allowed_profit_after_giveback:
            stop_reason = "live_profit_giveback_hit"

    payload = {
        "enabled": settings.live_canary_profit_protection_enabled,
        "current_wallet": str(current),
        "previous_close_wallet": str(state.previous_close_wallet),
        "start_wallet": str(start),
        "trade_delta_usd": str(trade_delta.normalize()),
        "current_profit_usd": str(current_profit.normalize()),
        "peak_profit_usd": str(peak_profit.normalize()),
        "giveback_usd": str(giveback.normalize()),
        "allowed_profit_after_giveback_usd": str(allowed_profit_after_giveback.normalize()),
        "max_profit_giveback_fraction": str(max_giveback_fraction),
        "profit_lock_min_profit_usd": str(min_profit),
        "consecutive_losses": consecutive_losses,
        "max_consecutive_losses": max_losses,
        "stop": stop_reason is not None,
        "stop_reason": stop_reason,
    }
    return LiveProfitProtectionVerdict(
        stop=stop_reason is not None,
        reason=stop_reason,
        state=new_state,
        payload=payload,
    )


def _decimal_or_none(value: object) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except Exception:
        return None


def _live_position_loss_check(
    client: BinancePrivateClient,
    symbol: str,
    max_intratrade_loss_usd: float,
) -> dict[str, Any] | None:
    if max_intratrade_loss_usd <= 0:
        return None
    position = next((item for item in client.position_risk(symbol) if item.get("symbol") == symbol), None)
    if not position:
        return None
    unrealized = Decimal(str(position.get("unRealizedProfit") or "0"))
    return {
        "symbol": symbol,
        "positionAmt": position.get("positionAmt"),
        "entryPrice": position.get("entryPrice"),
        "markPrice": position.get("markPrice"),
        "unRealizedProfit": position.get("unRealizedProfit"),
        "max_intratrade_loss_usd": str(max_intratrade_loss_usd),
        "loss_limit_hit": unrealized <= -abs(Decimal(str(max_intratrade_loss_usd))),
    }


def _filled_notional_usd(live_open: dict[str, Any]) -> float | None:
    order = live_open.get("opened_order") or {}
    for key in ("cumQuote", "cummulativeQuoteQty", "quoteQty"):
        value = order.get(key)
        if value is not None:
            parsed = float(value)
            if parsed > 0:
                return parsed
    template = live_open.get("order_template") or {}
    value = template.get("estimated_notional_usd")
    if value is not None:
        parsed = float(value)
        if parsed > 0:
            return parsed
    return None
