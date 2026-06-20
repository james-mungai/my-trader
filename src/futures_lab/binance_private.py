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

    def _request_json(self, method: str, path: str, *, signed: bool, params: dict[str, Any] | None = None) -> Any:
        query: dict[str, Any] = dict(params or {})
        if signed:
            query["timestamp"] = int(time.time() * 1000)
            query["recvWindow"] = 5000
            query["signature"] = _sign_query(query, self.api_secret)
        encoded = urllib.parse.urlencode(query)
        url = f"{self.base_url}{path}"
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


def _nonzero_number(value: object) -> bool:
    try:
        return abs(float(value)) > 0
    except (TypeError, ValueError):
        return False


def _redact(text: str) -> str:
    return text.replace("\n", " ")[:500]
