from __future__ import annotations

import hashlib
import hmac
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from typing import Any

from futures_lab.config import Settings


LIVE_FAPI_BASE_URL = "https://fapi.binance.com"
LIVE_SAPI_BASE_URL = "https://api.binance.com"
TESTNET_FAPI_BASE_URL = "https://testnet.binancefuture.com"


class BinancePrivateError(RuntimeError):
    pass


@dataclass(frozen=True)
class BinancePrivateClient:
    settings: Settings
    timeout_seconds: float = 10.0

    @property
    def base_url(self) -> str:
        env = self.settings.binance_env.strip().upper()
        if env in {"FUTURES_TESTNET", "TESTNET"}:
            return TESTNET_FAPI_BASE_URL
        return LIVE_FAPI_BASE_URL

    @property
    def api_key(self) -> str:
        if self.settings.binance_api_key:
            return self.settings.binance_api_key
        if self.settings.binance_env.strip().upper() in {"FUTURES_TESTNET", "TESTNET"}:
            if self.settings.binance_futures_testnet_api_key:
                return self.settings.binance_futures_testnet_api_key
        raise BinancePrivateError("Missing BINANCE_API_KEY.")

    @property
    def api_secret(self) -> str:
        if self.settings.binance_api_secret:
            return self.settings.binance_api_secret
        if self.settings.binance_env.strip().upper() in {"FUTURES_TESTNET", "TESTNET"}:
            if self.settings.binance_futures_testnet_api_secret:
                return self.settings.binance_futures_testnet_api_secret
        raise BinancePrivateError("Missing BINANCE_API_SECRET.")

    def server_time_ms(self) -> int:
        payload = self._request_json("GET", "/fapi/v1/time", signed=False)
        try:
            return int(payload["serverTime"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BinancePrivateError(f"Unexpected Binance time payload: {payload}") from exc

    def account(self) -> dict[str, Any]:
        return self._request_json("GET", "/fapi/v2/account", signed=True)

    def balance(self) -> list[dict[str, Any]]:
        payload = self._request_json("GET", "/fapi/v2/balance", signed=True)
        if not isinstance(payload, list):
            raise BinancePrivateError(f"Unexpected Binance balance payload: {payload}")
        return payload

    def open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        symbol = (symbol or self.settings.symbol).strip().upper()
        payload = self._request_json("GET", "/fapi/v1/openOrders", signed=True, params={"symbol": symbol})
        if not isinstance(payload, list):
            raise BinancePrivateError(f"Unexpected Binance openOrders payload: {payload}")
        return payload

    def all_open_orders(self) -> list[dict[str, Any]]:
        payload = self._request_json("GET", "/fapi/v1/openOrders", signed=True)
        if not isinstance(payload, list):
            raise BinancePrivateError(f"Unexpected Binance openOrders payload: {payload}")
        return payload

    def commission_rate(self, symbol: str | None = None) -> dict[str, Any]:
        symbol = (symbol or self.settings.symbol).strip().upper()
        payload = self._request_json("GET", "/fapi/v1/commissionRate", signed=True, params={"symbol": symbol})
        if not isinstance(payload, dict):
            raise BinancePrivateError(f"Unexpected Binance commissionRate payload: {payload}")
        return payload

    def position_risk(self, symbol: str | None = None) -> list[dict[str, Any]]:
        symbol = (symbol or self.settings.symbol).strip().upper()
        payload = self._request_json("GET", "/fapi/v2/positionRisk", signed=True, params={"symbol": symbol})
        if not isinstance(payload, list):
            raise BinancePrivateError(f"Unexpected Binance positionRisk payload: {payload}")
        return payload

    def exchange_info(self) -> dict[str, Any]:
        payload = self._request_json("GET", "/fapi/v1/exchangeInfo", signed=False)
        if not isinstance(payload, dict):
            raise BinancePrivateError(f"Unexpected Binance exchangeInfo payload: {payload}")
        return payload

    def ticker_price(self, symbol: str | None = None) -> Decimal:
        symbol = (symbol or self.settings.symbol).strip().upper()
        payload = self._request_json("GET", "/fapi/v1/ticker/price", signed=False, params={"symbol": symbol})
        try:
            return Decimal(str(payload["price"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise BinancePrivateError(f"Unexpected Binance ticker payload: {payload}") from exc

    def symbol_info(self, symbol: str | None = None) -> dict[str, Any]:
        symbol = (symbol or self.settings.symbol).strip().upper()
        symbols = self.exchange_info().get("symbols") or []
        for item in symbols:
            if item.get("symbol") == symbol:
                return item
        raise BinancePrivateError(f"Symbol {symbol} not found in Binance exchangeInfo.")

    def build_market_order_test(self, side: str, *, symbol: str | None = None, target_notional_usd: float | None = None) -> dict[str, Any]:
        symbol = (symbol or self.settings.symbol).strip().upper()
        side = side.strip().upper()
        if side not in {"BUY", "SELL"}:
            raise BinancePrivateError("side must be BUY or SELL.")
        info = self.symbol_info(symbol)
        filters = {item.get("filterType"): item for item in info.get("filters", [])}
        lot_filter = filters.get("MARKET_LOT_SIZE") or filters.get("LOT_SIZE") or {}
        min_notional_filter = filters.get("MIN_NOTIONAL") or {}
        step_size = Decimal(str(lot_filter.get("stepSize") or "0.001"))
        min_qty = Decimal(str(lot_filter.get("minQty") or "0"))
        min_exchange_notional = Decimal(str(min_notional_filter.get("notional") or self.settings.live_min_notional_usd))
        min_notional = max(Decimal(str(self.settings.live_min_notional_usd)), min_exchange_notional)
        max_notional = Decimal(str(self.settings.live_max_notional_usd))
        target_notional = Decimal(str(target_notional_usd if target_notional_usd is not None else self.settings.live_dust_test_notional_usd))
        if max_notional < min_notional:
            raise BinancePrivateError(f"LIVE_MAX_NOTIONAL_USD {max_notional} is below required minimum notional {min_notional}.")
        if target_notional > max_notional:
            raise BinancePrivateError(f"target notional {target_notional} exceeds max notional {max_notional}.")
        price = self.ticker_price(symbol)
        quantity = _floor_to_step(target_notional / price, step_size)
        if quantity < min_qty or quantity * price < min_notional:
            quantity = _ceil_to_step(max(min_qty, min_notional / price), step_size)
        estimated_notional = quantity * price
        if estimated_notional < min_notional:
            raise BinancePrivateError(
                f"Calculated notional {estimated_notional:.8f} is below required minimum notional {min_notional:.8f}."
            )
        if estimated_notional > max_notional:
            raise BinancePrivateError(
                f"Calculated notional {estimated_notional:.8f} exceeds LIVE_MAX_NOTIONAL_USD {max_notional:.8f}."
            )
        return {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": _format_decimal(quantity),
            "estimated_price": _format_decimal(price),
            "estimated_notional_usd": _format_decimal(estimated_notional),
            "min_notional_usd": _format_decimal(min_notional),
            "max_notional_usd": _format_decimal(max_notional),
            "step_size": _format_decimal(step_size),
            "min_qty": _format_decimal(min_qty),
        }

    def market_order_test_probe(
        self,
        side: str,
        *,
        symbol: str | None = None,
        target_notional_usd: float | None = None,
    ) -> dict[str, Any]:
        order = self.build_market_order_test(side, symbol=symbol, target_notional_usd=target_notional_usd)
        params = {
            "symbol": order["symbol"],
            "side": order["side"],
            "type": order["type"],
            "quantity": order["quantity"],
            "newClientOrderId": f"FL_TEST_{int(time.time() * 1000)}",
        }
        response = self._request_json("POST", "/fapi/v1/order/test", signed=True, params=params)
        return {
            "ok": True,
            "dry_run": True,
            "submitted_to_matching_engine": False,
            "endpoint": "/fapi/v1/order/test",
            "order": order,
            "response": response,
        }

    def place_market_order(
        self,
        side: str,
        *,
        symbol: str | None = None,
        quantity: str,
        reduce_only: bool = False,
        client_order_prefix: str = "FL_LIVE",
    ) -> dict[str, Any]:
        symbol = (symbol or self.settings.symbol).strip().upper()
        side = side.strip().upper()
        if side not in {"BUY", "SELL"}:
            raise BinancePrivateError("side must be BUY or SELL.")
        params = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": quantity,
            "newOrderRespType": "RESULT",
            "newClientOrderId": f"{client_order_prefix}_{int(time.time() * 1000)}",
        }
        if reduce_only:
            params["reduceOnly"] = "true"
        payload = self._request_json("POST", "/fapi/v1/order", signed=True, params=params)
        if not isinstance(payload, dict):
            raise BinancePrivateError(f"Unexpected Binance order payload: {payload}")
        return payload

    def universal_transfer(self, transfer_type: str, *, asset: str = "USDT", amount: Decimal | str | float) -> dict[str, Any]:
        self._require_live_transfer_settings()
        normalized_type = transfer_type.strip().upper()
        if normalized_type not in {"UMFUTURE_FUNDING", "FUNDING_UMFUTURE"}:
            raise BinancePrivateError("transfer_type must be UMFUTURE_FUNDING or FUNDING_UMFUTURE.")
        normalized_asset = asset.strip().upper()
        if normalized_asset != "USDT":
            raise BinancePrivateError("Only USDT treasury transfers are supported.")
        transfer_amount = Decimal(str(amount))
        if transfer_amount <= 0:
            raise BinancePrivateError("Transfer amount must be positive.")
        payload = self._request_json(
            "POST",
            "/sapi/v1/asset/transfer",
            signed=True,
            params={
                "type": normalized_type,
                "asset": normalized_asset,
                "amount": _format_decimal(transfer_amount),
            },
            base_url=LIVE_SAPI_BASE_URL,
        )
        if not isinstance(payload, dict):
            raise BinancePrivateError(f"Unexpected Binance transfer payload: {payload}")
        return payload

    def live_treasury_rebalance(self, *, symbol: str | None = None, reason: str = "manual") -> dict[str, Any]:
        symbol = (symbol or self.settings.symbol).strip().upper()
        if not self.settings.live_treasury_rebalance_enabled:
            account = self.account_probe()
            return {
                "ok": True,
                "enabled": False,
                "submitted_to_matching_engine": False,
                "action": "skipped",
                "reason": "live_treasury_rebalance_disabled",
                "trigger_reason": reason,
                "wallet_balance_before": account.get("total_wallet_balance"),
                "wallet_balance_after": account.get("total_wallet_balance"),
                "transfer": None,
            }
        self._require_live_transfer_settings()
        account_before = self.account_probe()
        open_orders = self.all_open_orders()
        positions = account_before.get("positions") or []
        nonzero_positions = [item for item in positions if _nonzero_number(item.get("positionAmt"))]
        if open_orders or nonzero_positions:
            return {
                "ok": False,
                "enabled": True,
                "submitted_to_matching_engine": False,
                "action": "blocked",
                "reason": "account_not_flat",
                "trigger_reason": reason,
                "wallet_balance_before": account_before.get("total_wallet_balance"),
                "wallet_balance_after": account_before.get("total_wallet_balance"),
                "open_orders_count": len(open_orders),
                "open_orders": [_summarize_open_order(item) for item in open_orders],
                "positions": nonzero_positions,
                "transfer": None,
            }
        wallet = Decimal(str(account_before.get("total_wallet_balance") or "0"))
        target = Decimal(str(self.settings.live_treasury_target_usdt))
        deadband = abs(Decimal(str(self.settings.live_treasury_deadband_usdt)))
        min_transfer = max(Decimal("0"), Decimal(str(self.settings.live_treasury_min_transfer_usdt)))
        max_transfer = max(Decimal("0"), Decimal(str(self.settings.live_treasury_max_transfer_usdt)))
        asset = self.settings.live_treasury_asset.strip().upper()
        amount = Decimal("0")
        transfer_type: str | None = None
        action = "none"
        if wallet > target + deadband:
            amount = wallet - target
            transfer_type = "UMFUTURE_FUNDING"
            action = "sweep_excess_to_funding"
        elif wallet < target - deadband:
            amount = target - wallet
            transfer_type = "FUNDING_UMFUTURE"
            action = "replenish_from_funding"
        if max_transfer > 0 and amount > max_transfer:
            amount = max_transfer
        if amount < min_transfer or transfer_type is None:
            return {
                "ok": True,
                "enabled": True,
                "submitted_to_matching_engine": False,
                "action": "no_op",
                "reason": "within_deadband_or_below_min_transfer",
                "trigger_reason": reason,
                "asset": asset,
                "target_usdt": _format_decimal(target),
                "deadband_usdt": _format_decimal(deadband),
                "wallet_balance_before": account_before.get("total_wallet_balance"),
                "wallet_balance_after": account_before.get("total_wallet_balance"),
                "planned_amount_usdt": _format_decimal(amount),
                "transfer": None,
            }
        transfer = self.universal_transfer(transfer_type, asset=asset, amount=amount)
        account_after = self.account_probe()
        return {
            "ok": True,
            "enabled": True,
            "submitted_to_matching_engine": False,
            "action": action,
            "reason": "transferred",
            "trigger_reason": reason,
            "asset": asset,
            "transfer_type": transfer_type,
            "target_usdt": _format_decimal(target),
            "deadband_usdt": _format_decimal(deadband),
            "amount_usdt": _format_decimal(amount),
            "wallet_balance_before": account_before.get("total_wallet_balance"),
            "wallet_balance_after": account_after.get("total_wallet_balance"),
            "available_balance_after": account_after.get("total_available_balance"),
            "transfer": transfer,
        }

    def live_dust_round_trip(self, side: str, *, symbol: str | None = None) -> dict[str, Any]:
        self._require_live_order_settings()
        symbol = (symbol or self.settings.symbol).strip().upper()
        side = side.strip().upper()
        if side not in {"BUY", "SELL"}:
            raise BinancePrivateError("side must be BUY or SELL.")
        preflight = self.live_preflight(symbol=symbol)
        if not preflight.get("ok"):
            raise BinancePrivateError(f"Live preflight failed: {preflight.get('hard_failures')}")
        template_key = "buy_market_test" if side == "BUY" else "sell_market_test"
        order_template = preflight["order_templates"][template_key]
        quantity = order_template["quantity"]
        before_account = self.account_probe()
        opened_order: dict[str, Any] | None = None
        close_order: dict[str, Any] | None = None
        position_after_open: dict[str, Any] | None = None
        final_position: dict[str, Any] | None = None
        close_attempt_errors: list[str] = []
        try:
            opened_order = self.place_market_order(
                side,
                symbol=symbol,
                quantity=quantity,
                reduce_only=False,
                client_order_prefix="FL_DUST_OPEN",
            )
            position_after_open = self._wait_for_position(symbol, expected_nonzero=True, timeout_seconds=10.0)
            close_side, close_quantity = _close_order_from_position(position_after_open)
            close_order = self.place_market_order(
                close_side,
                symbol=symbol,
                quantity=close_quantity,
                reduce_only=True,
                client_order_prefix="FL_DUST_CLOSE",
            )
        except Exception as exc:
            close_attempt_errors.append(str(exc))
            emergency_position = self._first_position_risk(symbol)
            if _nonzero_number(emergency_position.get("positionAmt")):
                try:
                    close_side, close_quantity = _close_order_from_position(emergency_position)
                    close_order = self.place_market_order(
                        close_side,
                        symbol=symbol,
                        quantity=close_quantity,
                        reduce_only=True,
                        client_order_prefix="FL_DUST_EMERGENCY_CLOSE",
                    )
                except Exception as close_exc:
                    close_attempt_errors.append(f"emergency_close_failed: {close_exc}")
            if close_attempt_errors and close_order is None:
                raise BinancePrivateError("; ".join(close_attempt_errors)) from exc
        final_position = self._wait_for_position(symbol, expected_nonzero=False, timeout_seconds=10.0)
        after_account = self.account_probe()
        open_orders = self.open_orders(symbol)
        final_position_amt = Decimal(str(final_position.get("positionAmt") or "0"))
        flat = final_position_amt == Decimal("0") and not open_orders
        if not flat:
            raise BinancePrivateError(f"Dust round trip did not end flat: position={final_position}, open_orders={open_orders}")
        before_wallet = Decimal(str(before_account.get("total_wallet_balance") or "0"))
        after_wallet = Decimal(str(after_account.get("total_wallet_balance") or "0"))
        return {
            "ok": True,
            "live_order_placed": True,
            "submitted_to_matching_engine": True,
            "symbol": symbol,
            "side": side,
            "order_template": order_template,
            "opened_order": _summarize_order_response(opened_order),
            "position_after_open": position_after_open,
            "close_order": _summarize_order_response(close_order),
            "final_position": final_position,
            "open_orders_count": len(open_orders),
            "wallet_balance_before": before_account.get("total_wallet_balance"),
            "wallet_balance_after": after_account.get("total_wallet_balance"),
            "wallet_balance_delta_usd": _format_decimal(after_wallet - before_wallet),
            "available_balance_after": after_account.get("total_available_balance"),
            "close_attempt_errors": close_attempt_errors,
        }

    def live_dust_open(self, side: str, *, symbol: str | None = None) -> dict[str, Any]:
        self._require_live_order_settings()
        symbol = (symbol or self.settings.symbol).strip().upper()
        side = side.strip().upper()
        if side not in {"BUY", "SELL"}:
            raise BinancePrivateError("side must be BUY or SELL.")
        preflight = self.live_preflight(symbol=symbol)
        if not preflight.get("ok"):
            raise BinancePrivateError(f"Live preflight failed: {preflight.get('hard_failures')}")
        template_key = "buy_market_test" if side == "BUY" else "sell_market_test"
        order_template = preflight["order_templates"][template_key]
        before_account = self.account_probe()
        opened_order = self.place_market_order(
            side,
            symbol=symbol,
            quantity=order_template["quantity"],
            reduce_only=False,
            client_order_prefix="FL_DUST_OPEN_ONLY",
        )
        position_after_open = self._wait_for_position(symbol, expected_nonzero=True, timeout_seconds=10.0)
        after_account = self.account_probe()
        open_orders = self.open_orders(symbol)
        if open_orders:
            raise BinancePrivateError(f"Dust open left open orders: {open_orders}")
        before_wallet = Decimal(str(before_account.get("total_wallet_balance") or "0"))
        after_wallet = Decimal(str(after_account.get("total_wallet_balance") or "0"))
        return {
            "ok": True,
            "live_order_placed": True,
            "submitted_to_matching_engine": True,
            "symbol": symbol,
            "side": side,
            "order_template": order_template,
            "opened_order": _summarize_order_response(opened_order),
            "position_after_open": position_after_open,
            "open_orders_count": len(open_orders),
            "wallet_balance_before": before_account.get("total_wallet_balance"),
            "wallet_balance_after": after_account.get("total_wallet_balance"),
            "wallet_balance_delta_usd": _format_decimal(after_wallet - before_wallet),
            "available_balance_after": after_account.get("total_available_balance"),
        }

    def flatten_position(self, *, symbol: str | None = None) -> dict[str, Any]:
        self._require_live_order_settings()
        symbol = (symbol or self.settings.symbol).strip().upper()
        before_account = self.account_probe()
        position_before = self._first_position_risk(symbol)
        if not _nonzero_number(position_before.get("positionAmt")):
            return {
                "ok": True,
                "live_order_placed": False,
                "submitted_to_matching_engine": False,
                "symbol": symbol,
                "message": "already_flat",
                "position_before": _summarize_position_risk(position_before),
                "final_position": _summarize_position_risk(position_before),
                "open_orders_count": len(self.open_orders(symbol)),
                "wallet_balance_before": before_account.get("total_wallet_balance"),
                "wallet_balance_after": before_account.get("total_wallet_balance"),
                "wallet_balance_delta_usd": "0",
                "available_balance_after": before_account.get("total_available_balance"),
            }
        close_side, close_quantity = _close_order_from_position(position_before)
        close_order = self.place_market_order(
            close_side,
            symbol=symbol,
            quantity=close_quantity,
            reduce_only=True,
            client_order_prefix="FL_FLATTEN",
        )
        final_position = self._wait_for_position(symbol, expected_nonzero=False, timeout_seconds=10.0)
        after_account = self.account_probe()
        open_orders = self.open_orders(symbol)
        final_position_amt = Decimal(str(final_position.get("positionAmt") or "0"))
        if final_position_amt != Decimal("0"):
            raise BinancePrivateError(f"Flatten did not end flat: {final_position}")
        before_wallet = Decimal(str(before_account.get("total_wallet_balance") or "0"))
        after_wallet = Decimal(str(after_account.get("total_wallet_balance") or "0"))
        return {
            "ok": True,
            "live_order_placed": True,
            "submitted_to_matching_engine": True,
            "symbol": symbol,
            "position_before": _summarize_position_risk(position_before),
            "close_order": _summarize_order_response(close_order),
            "final_position": final_position,
            "open_orders_count": len(open_orders),
            "open_orders": [_summarize_open_order(item) for item in open_orders],
            "wallet_balance_before": before_account.get("total_wallet_balance"),
            "wallet_balance_after": after_account.get("total_wallet_balance"),
            "wallet_balance_delta_usd": _format_decimal(after_wallet - before_wallet),
            "available_balance_after": after_account.get("total_available_balance"),
        }

    def live_preflight(self, *, symbol: str | None = None) -> dict[str, Any]:
        symbol = (symbol or self.settings.symbol).strip().upper()
        account_probe = self.account_probe()
        symbol_info = self.symbol_info(symbol)
        filters = {item.get("filterType"): item for item in symbol_info.get("filters", [])}
        commission = self.commission_rate(symbol)
        open_orders = self.open_orders(symbol)
        position_risk = self.position_risk(symbol)
        target_positions = [
            {
                "symbol": item.get("symbol"),
                "positionAmt": item.get("positionAmt"),
                "entryPrice": item.get("entryPrice"),
                "markPrice": item.get("markPrice"),
                "unRealizedProfit": item.get("unRealizedProfit"),
                "liquidationPrice": item.get("liquidationPrice"),
                "leverage": item.get("leverage"),
                "marginType": item.get("marginType"),
                "isolatedMargin": item.get("isolatedMargin"),
                "positionSide": item.get("positionSide"),
            }
            for item in position_risk
            if item.get("symbol") == symbol
        ]
        hard_failures: list[str] = []
        warnings: list[str] = []
        if not account_probe.get("can_trade"):
            hard_failures.append("account_cannot_trade")
        if abs(int(account_probe.get("clock_skew_ms") or 0)) > 2000:
            hard_failures.append("clock_skew_exceeds_2s")
        if open_orders:
            hard_failures.append("symbol_has_open_orders")
        if any(_nonzero_number(item.get("positionAmt")) for item in target_positions):
            hard_failures.append("symbol_has_open_position")
        if float(account_probe.get("total_available_balance") or 0.0) <= 0:
            hard_failures.append("no_available_balance")
        if len(target_positions) != 1:
            warnings.append("unexpected_position_risk_row_count")
        try:
            buy_template = self.build_market_order_test("BUY", symbol=symbol)
            sell_template = self.build_market_order_test("SELL", symbol=symbol)
        except BinancePrivateError as exc:
            hard_failures.append("dust_order_template_invalid")
            buy_template = None
            sell_template = None
            warnings.append(str(exc))
        maker_bps = _rate_to_bps(commission.get("makerCommissionRate"))
        taker_bps = _rate_to_bps(commission.get("takerCommissionRate"))
        return {
            "ok": not hard_failures,
            "dry_run": True,
            "read_only": True,
            "submitted_to_matching_engine": False,
            "symbol": symbol,
            "env": self.settings.binance_env,
            "account": account_probe,
            "commission": {
                "symbol": commission.get("symbol", symbol),
                "makerCommissionRate": commission.get("makerCommissionRate"),
                "takerCommissionRate": commission.get("takerCommissionRate"),
                "maker_bps": _format_decimal(maker_bps) if maker_bps is not None else None,
                "taker_bps": _format_decimal(taker_bps) if taker_bps is not None else None,
                "round_trip_taker_bps": _format_decimal(taker_bps * Decimal("2")) if taker_bps is not None else None,
            },
            "symbol_rules": {
                "status": symbol_info.get("status"),
                "contractType": symbol_info.get("contractType"),
                "pricePrecision": symbol_info.get("pricePrecision"),
                "quantityPrecision": symbol_info.get("quantityPrecision"),
                "marketLotSize": filters.get("MARKET_LOT_SIZE"),
                "lotSize": filters.get("LOT_SIZE"),
                "minNotional": filters.get("MIN_NOTIONAL"),
            },
            "position_risk": target_positions,
            "open_orders_count": len(open_orders),
            "open_orders": [_summarize_open_order(item) for item in open_orders],
            "order_templates": {
                "buy_market_test": buy_template,
                "sell_market_test": sell_template,
            },
            "limits": {
                "live_trading_enabled": self.settings.live_trading_enabled,
                "live_dry_run": self.settings.live_dry_run,
                "live_min_notional_usd": self.settings.live_min_notional_usd,
                "live_max_notional_usd": self.settings.live_max_notional_usd,
                "live_dust_test_notional_usd": self.settings.live_dust_test_notional_usd,
                "live_max_open_positions": self.settings.live_max_open_positions,
                "live_max_trades_per_day": self.settings.live_max_trades_per_day,
                "live_daily_max_loss_usd": self.settings.live_daily_max_loss_usd,
                "live_treasury_rebalance_enabled": self.settings.live_treasury_rebalance_enabled,
                "live_treasury_target_usdt": self.settings.live_treasury_target_usdt,
                "live_treasury_deadband_usdt": self.settings.live_treasury_deadband_usdt,
            },
            "hard_failures": hard_failures,
            "warnings": warnings,
        }

    def _require_live_order_settings(self) -> None:
        if not self.settings.live_trading_enabled:
            raise BinancePrivateError("LIVE_TRADING_ENABLED must be true before placing live orders.")
        if self.settings.live_dry_run:
            raise BinancePrivateError("LIVE_DRY_RUN must be false before placing live orders.")

    def _require_live_transfer_settings(self) -> None:
        if not self.settings.live_treasury_rebalance_enabled:
            raise BinancePrivateError("LIVE_TREASURY_REBALANCE_ENABLED must be true before transferring funds.")
        if not self.settings.live_trading_enabled:
            raise BinancePrivateError("LIVE_TRADING_ENABLED must be true before transferring funds.")
        if self.settings.live_dry_run:
            raise BinancePrivateError("LIVE_DRY_RUN must be false before transferring funds.")

    def _first_position_risk(self, symbol: str) -> dict[str, Any]:
        rows = self.position_risk(symbol)
        for row in rows:
            if row.get("symbol") == symbol:
                return row
        raise BinancePrivateError(f"No position risk row found for {symbol}.")

    def _wait_for_position(self, symbol: str, *, expected_nonzero: bool, timeout_seconds: float) -> dict[str, Any]:
        deadline = time.time() + timeout_seconds
        last: dict[str, Any] | None = None
        while time.time() <= deadline:
            last = self._first_position_risk(symbol)
            is_nonzero = _nonzero_number(last.get("positionAmt"))
            if is_nonzero == expected_nonzero:
                return _summarize_position_risk(last)
            time.sleep(0.5)
        expected = "non-zero" if expected_nonzero else "flat"
        raise BinancePrivateError(f"Timed out waiting for {symbol} position to become {expected}. Last={last}")

    def account_probe(self) -> dict[str, Any]:
        server_time = self.server_time_ms()
        local_time = int(time.time() * 1000)
        account = self.account()
        balances = self.balance()
        usdt = next((item for item in balances if item.get("asset") == "USDT"), {})
        positions = [
            {
                "symbol": item.get("symbol"),
                "positionAmt": item.get("positionAmt"),
                "entryPrice": item.get("entryPrice"),
                "unrealizedProfit": item.get("unrealizedProfit"),
                "leverage": item.get("leverage"),
                "isolated": item.get("isolated"),
                "positionSide": item.get("positionSide"),
            }
            for item in account.get("positions", [])
            if item.get("symbol") == self.settings.symbol
            or _nonzero_number(item.get("positionAmt"))
            or _nonzero_number(item.get("unrealizedProfit"))
        ]
        return {
            "ok": True,
            "env": self.settings.binance_env,
            "base_url": self.base_url,
            "symbol": self.settings.symbol,
            "server_time_ms": server_time,
            "local_time_ms": local_time,
            "clock_skew_ms": local_time - server_time,
            "fee_tier": account.get("feeTier"),
            "can_trade": account.get("canTrade"),
            "can_deposit": account.get("canDeposit"),
            "can_withdraw": account.get("canWithdraw"),
            "multi_assets_margin": account.get("multiAssetsMargin"),
            "trade_group_id": account.get("tradeGroupId"),
            "total_wallet_balance": account.get("totalWalletBalance"),
            "total_margin_balance": account.get("totalMarginBalance"),
            "total_available_balance": account.get("availableBalance"),
            "total_unrealized_profit": account.get("totalUnrealizedProfit"),
            "usdt_balance": {
                "balance": usdt.get("balance"),
                "availableBalance": usdt.get("availableBalance"),
                "crossWalletBalance": usdt.get("crossWalletBalance"),
                "crossUnPnl": usdt.get("crossUnPnl"),
            },
            "positions": positions,
        }

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        signed: bool,
        params: dict[str, Any] | None = None,
        base_url: str | None = None,
    ) -> Any:
        query: dict[str, Any] = dict(params or {})
        if signed:
            query["timestamp"] = int(time.time() * 1000)
            query["recvWindow"] = 5000
            query["signature"] = _sign_query(query, self.api_secret)
        encoded = urllib.parse.urlencode(query)
        url = f"{base_url or self.base_url}{path}"
        if encoded:
            url = f"{url}?{encoded}"
        request = urllib.request.Request(
            url,
            method=method,
            headers={
                "User-Agent": "futures-lab/0.1",
                "X-MBX-APIKEY": self.api_key if signed else "",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise BinancePrivateError(f"Binance HTTP {exc.code} for {path}: {_redact(body)}") from exc
        except urllib.error.URLError as exc:
            raise BinancePrivateError(f"Binance request failed for {path}: {exc.reason}") from exc


def _sign_query(query: dict[str, Any], secret: str) -> str:
    payload = urllib.parse.urlencode(query)
    return hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def _floor_to_step(value: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        return value
    return (value / step).to_integral_value(rounding=ROUND_FLOOR) * step


def _ceil_to_step(value: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        return value
    return (value / step).to_integral_value(rounding=ROUND_CEILING) * step


def _format_decimal(value: Decimal) -> str:
    text = format(value.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _rate_to_bps(value: object) -> Decimal | None:
    if value is None:
        return None
    return Decimal(str(value)) * Decimal("10000")


def _summarize_open_order(order: dict[str, Any]) -> dict[str, Any]:
    return {
        "orderId": order.get("orderId"),
        "symbol": order.get("symbol"),
        "status": order.get("status"),
        "side": order.get("side"),
        "type": order.get("type"),
        "origQty": order.get("origQty"),
        "executedQty": order.get("executedQty"),
        "reduceOnly": order.get("reduceOnly"),
        "positionSide": order.get("positionSide"),
        "time": order.get("time"),
    }


def _close_order_from_position(position: dict[str, Any]) -> tuple[str, str]:
    quantity = Decimal(str(position.get("positionAmt") or "0"))
    if quantity == 0:
        raise BinancePrivateError("Cannot close a flat position.")
    side = "SELL" if quantity > 0 else "BUY"
    return side, _format_decimal(abs(quantity))


def _summarize_position_risk(position: dict[str, Any]) -> dict[str, Any]:
    return {
        "symbol": position.get("symbol"),
        "positionAmt": position.get("positionAmt"),
        "entryPrice": position.get("entryPrice"),
        "markPrice": position.get("markPrice"),
        "unRealizedProfit": position.get("unRealizedProfit"),
        "liquidationPrice": position.get("liquidationPrice"),
        "leverage": position.get("leverage"),
        "marginType": position.get("marginType"),
        "isolatedMargin": position.get("isolatedMargin"),
        "positionSide": position.get("positionSide"),
    }


def _summarize_order_response(order: dict[str, Any] | None) -> dict[str, Any] | None:
    if order is None:
        return None
    return {
        "orderId": order.get("orderId"),
        "symbol": order.get("symbol"),
        "status": order.get("status"),
        "clientOrderId": order.get("clientOrderId"),
        "side": order.get("side"),
        "type": order.get("type"),
        "origQty": order.get("origQty"),
        "executedQty": order.get("executedQty"),
        "avgPrice": order.get("avgPrice"),
        "cumQuote": order.get("cumQuote"),
        "reduceOnly": order.get("reduceOnly"),
        "closePosition": order.get("closePosition"),
        "positionSide": order.get("positionSide"),
        "updateTime": order.get("updateTime"),
    }


def _nonzero_number(value: object) -> bool:
    try:
        return abs(float(value)) > 0
    except (TypeError, ValueError):
        return False


def _redact(text: str) -> str:
    return text.replace("\n", " ")[:500]
