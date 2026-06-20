from __future__ import annotations

import hashlib
import hmac
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
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

    def _request_json(self, method: str, path: str, *, signed: bool) -> Any:
        query: dict[str, str | int] = {}
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
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise BinancePrivateError(f"Binance HTTP {exc.code} for {path}: {_redact(body)}") from exc
        except urllib.error.URLError as exc:
            raise BinancePrivateError(f"Binance request failed for {path}: {exc.reason}") from exc


def _sign_query(query: dict[str, str | int], secret: str) -> str:
    payload = urllib.parse.urlencode(query)
    return hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def _nonzero_number(value: object) -> bool:
    try:
        return abs(float(value)) > 0
    except (TypeError, ValueError):
        return False


def _redact(text: str) -> str:
    return text.replace("\n", " ")[:500]
