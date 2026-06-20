import pytest

from futures_lab.binance_private import BinancePrivateClient, BinancePrivateError, _sign_query
from futures_lab.config import Settings


def test_sign_query_uses_hmac_sha256() -> None:
    query = {"timestamp": 1, "recvWindow": 5000}

    assert _sign_query(query, "secret") == "2474045f1dadfba223cdc7e109e4167e97954defbaa72647883f2f8b20e999b2"


def test_client_requires_live_keys() -> None:
    client = BinancePrivateClient(Settings(BINANCE_API_KEY="", BINANCE_API_SECRET=""))

    with pytest.raises(BinancePrivateError, match="Missing BINANCE_API_KEY"):
        _ = client.api_key


def test_testnet_env_uses_testnet_credentials_and_base_url() -> None:
    settings = Settings(
        BINANCE_ENV="FUTURES_TESTNET",
        BINANCE_FUTURES_TESTNET_API_KEY="testnet-key",
        BINANCE_FUTURES_TESTNET_API_SECRET="testnet-secret",
    )
    client = BinancePrivateClient(settings)

    assert client.api_key == "testnet-key"
    assert client.api_secret == "testnet-secret"
    assert client.base_url == "https://testnet.binancefuture.com"


def test_account_probe_redacts_credentials_and_filters_positions() -> None:
    class FakeClient(BinancePrivateClient):
        def server_time_ms(self) -> int:
            return 1000

        def account(self) -> dict:
            return {
                "feeTier": 0,
                "canTrade": True,
                "canDeposit": True,
                "canWithdraw": False,
                "multiAssetsMargin": False,
                "tradeGroupId": -1,
                "totalWalletBalance": "50.00000000",
                "totalMarginBalance": "50.00000000",
                "availableBalance": "49.50000000",
                "totalUnrealizedProfit": "0.00000000",
                "positions": [
                    {
                        "symbol": "ETHUSDT",
                        "positionAmt": "0.000",
                        "entryPrice": "0.00",
                        "unrealizedProfit": "0.00000000",
                        "leverage": "1",
                        "isolated": False,
                        "positionSide": "BOTH",
                    },
                    {
                        "symbol": "BTCUSDT",
                        "positionAmt": "0.001",
                        "entryPrice": "65000.00",
                        "unrealizedProfit": "1.00000000",
                        "leverage": "1",
                        "isolated": False,
                        "positionSide": "BOTH",
                    },
                ],
            }

        def balance(self) -> list[dict]:
            return [
                {
                    "asset": "USDT",
                    "balance": "50.00000000",
                    "availableBalance": "49.50000000",
                    "crossWalletBalance": "50.00000000",
                    "crossUnPnl": "0.00000000",
                }
            ]

    client = FakeClient(Settings(SYMBOL="ETHUSDT", BINANCE_API_KEY="live-key", BINANCE_API_SECRET="live-secret"))

    payload = client.account_probe()

    assert payload["ok"] is True
    assert payload["can_trade"] is True
    assert payload["can_withdraw"] is False
    assert payload["usdt_balance"]["availableBalance"] == "49.50000000"
    assert {position["symbol"] for position in payload["positions"]} == {"ETHUSDT", "BTCUSDT"}
    assert "live-key" not in str(payload)
    assert "live-secret" not in str(payload)


def test_market_order_test_uses_safe_quantity_under_max_notional() -> None:
    class FakeClient(BinancePrivateClient):
        def exchange_info(self) -> dict:
            return {
                "symbols": [
                    {
                        "symbol": "ETHUSDT",
                        "filters": [
                            {"filterType": "MARKET_LOT_SIZE", "minQty": "0.001", "stepSize": "0.001"},
                            {"filterType": "MIN_NOTIONAL", "notional": "20"},
                        ],
                    }
                ]
            }

        def ticker_price(self, symbol: str | None = None):
            from decimal import Decimal

            return Decimal("1722.47")

    client = FakeClient(
        Settings(
            SYMBOL="ETHUSDT",
            BINANCE_API_KEY="live-key",
            BINANCE_API_SECRET="live-secret",
            LIVE_DUST_TEST_NOTIONAL_USD=22,
            LIVE_MAX_NOTIONAL_USD=25,
        )
    )

    order = client.build_market_order_test("BUY")

    assert order["quantity"] == "0.012"
    assert float(order["estimated_notional_usd"]) >= 20
    assert float(order["estimated_notional_usd"]) <= 25


def test_market_order_test_refuses_quantity_that_cannot_clear_min_below_cap() -> None:
    class FakeClient(BinancePrivateClient):
        def exchange_info(self) -> dict:
            return {
                "symbols": [
                    {
                        "symbol": "ETHUSDT",
                        "filters": [
                            {"filterType": "MARKET_LOT_SIZE", "minQty": "0.001", "stepSize": "0.001"},
                            {"filterType": "MIN_NOTIONAL", "notional": "20"},
                        ],
                    }
                ]
            }

        def ticker_price(self, symbol: str | None = None):
            from decimal import Decimal

            return Decimal("1722.47")

    client = FakeClient(
        Settings(
            SYMBOL="ETHUSDT",
            BINANCE_API_KEY="live-key",
            BINANCE_API_SECRET="live-secret",
            LIVE_DUST_TEST_NOTIONAL_USD=22,
            LIVE_MAX_NOTIONAL_USD=19,
        )
    )

    with pytest.raises(BinancePrivateError, match="below required minimum"):
        client.build_market_order_test("BUY")


def test_market_order_test_probe_calls_binance_test_endpoint_only() -> None:
    calls = []

    class FakeClient(BinancePrivateClient):
        def exchange_info(self) -> dict:
            return {
                "symbols": [
                    {
                        "symbol": "ETHUSDT",
                        "filters": [
                            {"filterType": "MARKET_LOT_SIZE", "minQty": "0.001", "stepSize": "0.001"},
                            {"filterType": "MIN_NOTIONAL", "notional": "20"},
                        ],
                    }
                ]
            }

        def ticker_price(self, symbol: str | None = None):
            from decimal import Decimal

            return Decimal("1722.47")

        def _request_json(self, method: str, path: str, *, signed: bool, params: dict | None = None):
            calls.append((method, path, signed, params))
            return {}

    client = FakeClient(
        Settings(
            SYMBOL="ETHUSDT",
            BINANCE_API_KEY="live-key",
            BINANCE_API_SECRET="live-secret",
            LIVE_DUST_TEST_NOTIONAL_USD=22,
            LIVE_MAX_NOTIONAL_USD=25,
        )
    )

    result = client.market_order_test_probe("SELL")

    assert result["ok"] is True
    assert result["submitted_to_matching_engine"] is False
    assert calls[0][0] == "POST"
    assert calls[0][1] == "/fapi/v1/order/test"
    assert calls[0][2] is True
    assert calls[0][3]["side"] == "SELL"
    assert calls[0][3]["quantity"] == "0.012"


def test_live_preflight_reports_clean_read_only_state() -> None:
    class FakeClient(BinancePrivateClient):
        def account_probe(self) -> dict:
            return {
                "ok": True,
                "env": "LIVE",
                "base_url": "https://fapi.binance.com",
                "symbol": "ETHUSDT",
                "clock_skew_ms": 1,
                "can_trade": True,
                "total_available_balance": "100.00000000",
            }

        def symbol_info(self, symbol: str | None = None) -> dict:
            return {
                "symbol": "ETHUSDT",
                "status": "TRADING",
                "contractType": "PERPETUAL",
                "pricePrecision": 2,
                "quantityPrecision": 3,
                "filters": [
                    {"filterType": "MARKET_LOT_SIZE", "minQty": "0.001", "stepSize": "0.001"},
                    {"filterType": "MIN_NOTIONAL", "notional": "20"},
                ],
            }

        def ticker_price(self, symbol: str | None = None):
            from decimal import Decimal

            return Decimal("1722.47")

        def commission_rate(self, symbol: str | None = None) -> dict:
            return {"symbol": "ETHUSDT", "makerCommissionRate": "0.0002", "takerCommissionRate": "0.0004"}

        def open_orders(self, symbol: str | None = None) -> list[dict]:
            return []

        def position_risk(self, symbol: str | None = None) -> list[dict]:
            return [
                {
                    "symbol": "ETHUSDT",
                    "positionAmt": "0.000",
                    "entryPrice": "0.0",
                    "markPrice": "1722.47",
                    "unRealizedProfit": "0.00000000",
                    "liquidationPrice": "0",
                    "leverage": "150",
                    "marginType": "cross",
                    "isolatedMargin": "0.00000000",
                    "positionSide": "BOTH",
                }
            ]

    client = FakeClient(
        Settings(
            SYMBOL="ETHUSDT",
            BINANCE_API_KEY="live-key",
            BINANCE_API_SECRET="live-secret",
            LIVE_DUST_TEST_NOTIONAL_USD=22,
            LIVE_MAX_NOTIONAL_USD=25,
        )
    )

    result = client.live_preflight()

    assert result["ok"] is True
    assert result["read_only"] is True
    assert result["submitted_to_matching_engine"] is False
    assert result["commission"]["taker_bps"] == "4"
    assert result["commission"]["round_trip_taker_bps"] == "8"
    assert result["open_orders_count"] == 0
    assert result["order_templates"]["buy_market_test"]["quantity"] == "0.012"
    assert result["hard_failures"] == []


def test_live_preflight_blocks_dirty_account_state() -> None:
    class FakeClient(BinancePrivateClient):
        def account_probe(self) -> dict:
            return {
                "ok": True,
                "env": "LIVE",
                "symbol": "ETHUSDT",
                "clock_skew_ms": 1,
                "can_trade": True,
                "total_available_balance": "100.00000000",
            }

        def symbol_info(self, symbol: str | None = None) -> dict:
            return {
                "symbol": "ETHUSDT",
                "filters": [
                    {"filterType": "MARKET_LOT_SIZE", "minQty": "0.001", "stepSize": "0.001"},
                    {"filterType": "MIN_NOTIONAL", "notional": "20"},
                ],
            }

        def ticker_price(self, symbol: str | None = None):
            from decimal import Decimal

            return Decimal("1722.47")

        def commission_rate(self, symbol: str | None = None) -> dict:
            return {"symbol": "ETHUSDT", "makerCommissionRate": "0.0002", "takerCommissionRate": "0.0004"}

        def open_orders(self, symbol: str | None = None) -> list[dict]:
            return [{"orderId": 123, "symbol": "ETHUSDT", "side": "BUY", "type": "LIMIT"}]

        def position_risk(self, symbol: str | None = None) -> list[dict]:
            return [
                {
                    "symbol": "ETHUSDT",
                    "positionAmt": "0.012",
                    "entryPrice": "1722.47",
                    "markPrice": "1723.00",
                    "unRealizedProfit": "0.00100000",
                    "liquidationPrice": "1000",
                    "leverage": "150",
                    "marginType": "cross",
                    "isolatedMargin": "0.00000000",
                    "positionSide": "BOTH",
                }
            ]

    client = FakeClient(Settings(SYMBOL="ETHUSDT", BINANCE_API_KEY="live-key", BINANCE_API_SECRET="live-secret"))

    result = client.live_preflight()

    assert result["ok"] is False
    assert "symbol_has_open_orders" in result["hard_failures"]
    assert "symbol_has_open_position" in result["hard_failures"]
    assert result["open_orders"][0]["orderId"] == 123


def test_live_dust_round_trip_requires_explicit_live_settings() -> None:
    client = BinancePrivateClient(Settings(SYMBOL="ETHUSDT", BINANCE_API_KEY="live-key", BINANCE_API_SECRET="live-secret"))

    with pytest.raises(BinancePrivateError, match="LIVE_TRADING_ENABLED"):
        client.live_dust_round_trip("BUY")

    dry_run_client = BinancePrivateClient(
        Settings(
            SYMBOL="ETHUSDT",
            BINANCE_API_KEY="live-key",
            BINANCE_API_SECRET="live-secret",
            LIVE_TRADING_ENABLED=True,
            LIVE_DRY_RUN=True,
        )
    )

    with pytest.raises(BinancePrivateError, match="LIVE_DRY_RUN"):
        dry_run_client.live_dust_round_trip("BUY")


def test_live_dust_round_trip_opens_reduce_only_closes_and_verifies_flat() -> None:
    calls = []

    class FakeClient(BinancePrivateClient):
        account_probe_calls = 0

        def live_preflight(self, *, symbol: str | None = None) -> dict:
            return {
                "ok": True,
                "order_templates": {
                    "buy_market_test": {
                        "symbol": "ETHUSDT",
                        "side": "BUY",
                        "type": "MARKET",
                        "quantity": "0.012",
                        "estimated_notional_usd": "20.72",
                    },
                    "sell_market_test": {
                        "symbol": "ETHUSDT",
                        "side": "SELL",
                        "type": "MARKET",
                        "quantity": "0.012",
                        "estimated_notional_usd": "20.72",
                    },
                },
            }

        def account_probe(self) -> dict:
            self.account_probe_calls += 1
            balance = "100.00000000" if self.account_probe_calls == 1 else "99.98000000"
            return {"total_wallet_balance": balance, "total_available_balance": balance}

        def place_market_order(
            self,
            side: str,
            *,
            symbol: str | None = None,
            quantity: str,
            reduce_only: bool = False,
            client_order_prefix: str = "FL_LIVE",
        ) -> dict:
            calls.append(
                {
                    "side": side,
                    "symbol": symbol,
                    "quantity": quantity,
                    "reduce_only": reduce_only,
                    "client_order_prefix": client_order_prefix,
                }
            )
            return {
                "orderId": len(calls),
                "symbol": symbol,
                "status": "FILLED",
                "side": side,
                "type": "MARKET",
                "origQty": quantity,
                "executedQty": quantity,
                "avgPrice": "1726.50",
                "cumQuote": "20.718",
                "reduceOnly": reduce_only,
                "positionSide": "BOTH",
            }

        def _wait_for_position(self, symbol: str, *, expected_nonzero: bool, timeout_seconds: float) -> dict:
            if expected_nonzero:
                return {
                    "symbol": symbol,
                    "positionAmt": "0.012",
                    "entryPrice": "1726.50",
                    "markPrice": "1726.60",
                    "unRealizedProfit": "0.00100000",
                    "liquidationPrice": "1000",
                    "leverage": "150",
                    "marginType": "cross",
                    "isolatedMargin": "0.00000000",
                    "positionSide": "BOTH",
                }
            return {
                "symbol": symbol,
                "positionAmt": "0.000",
                "entryPrice": "0.0",
                "markPrice": "1726.60",
                "unRealizedProfit": "0.00000000",
                "liquidationPrice": "0",
                "leverage": "150",
                "marginType": "cross",
                "isolatedMargin": "0.00000000",
                "positionSide": "BOTH",
            }

        def open_orders(self, symbol: str | None = None) -> list[dict]:
            return []

    client = FakeClient(
        Settings(
            SYMBOL="ETHUSDT",
            BINANCE_API_KEY="live-key",
            BINANCE_API_SECRET="live-secret",
            LIVE_TRADING_ENABLED=True,
            LIVE_DRY_RUN=False,
            LIVE_MAX_NOTIONAL_USD=25,
        )
    )

    result = client.live_dust_round_trip("BUY")

    assert result["ok"] is True
    assert result["submitted_to_matching_engine"] is True
    assert calls == [
        {
            "side": "BUY",
            "symbol": "ETHUSDT",
            "quantity": "0.012",
            "reduce_only": False,
            "client_order_prefix": "FL_DUST_OPEN",
        },
        {
            "side": "SELL",
            "symbol": "ETHUSDT",
            "quantity": "0.012",
            "reduce_only": True,
            "client_order_prefix": "FL_DUST_CLOSE",
        },
    ]
    assert result["final_position"]["positionAmt"] == "0.000"
    assert result["open_orders_count"] == 0
    assert result["wallet_balance_delta_usd"] == "-0.02"


def test_live_dust_open_places_one_order_and_leaves_position_open() -> None:
    calls = []

    class FakeClient(BinancePrivateClient):
        account_probe_calls = 0

        def live_preflight(self, *, symbol: str | None = None) -> dict:
            return {
                "ok": True,
                "order_templates": {
                    "buy_market_test": {"symbol": "ETHUSDT", "side": "BUY", "type": "MARKET", "quantity": "0.012"},
                    "sell_market_test": {"symbol": "ETHUSDT", "side": "SELL", "type": "MARKET", "quantity": "0.012"},
                },
            }

        def account_probe(self) -> dict:
            self.account_probe_calls += 1
            balance = "100.00000000" if self.account_probe_calls == 1 else "99.99000000"
            return {"total_wallet_balance": balance, "total_available_balance": balance}

        def place_market_order(
            self,
            side: str,
            *,
            symbol: str | None = None,
            quantity: str,
            reduce_only: bool = False,
            client_order_prefix: str = "FL_LIVE",
        ) -> dict:
            calls.append(
                {
                    "side": side,
                    "symbol": symbol,
                    "quantity": quantity,
                    "reduce_only": reduce_only,
                    "client_order_prefix": client_order_prefix,
                }
            )
            return {"orderId": 1, "symbol": symbol, "status": "FILLED", "side": side, "executedQty": quantity}

        def _wait_for_position(self, symbol: str, *, expected_nonzero: bool, timeout_seconds: float) -> dict:
            assert expected_nonzero is True
            return {
                "symbol": symbol,
                "positionAmt": "0.012",
                "entryPrice": "1726.50",
                "markPrice": "1726.60",
                "unRealizedProfit": "0.00100000",
                "liquidationPrice": "1000",
                "leverage": "150",
                "marginType": "cross",
                "isolatedMargin": "0.00000000",
                "positionSide": "BOTH",
            }

        def open_orders(self, symbol: str | None = None) -> list[dict]:
            return []

    client = FakeClient(
        Settings(
            SYMBOL="ETHUSDT",
            BINANCE_API_KEY="live-key",
            BINANCE_API_SECRET="live-secret",
            LIVE_TRADING_ENABLED=True,
            LIVE_DRY_RUN=False,
        )
    )

    result = client.live_dust_open("BUY")

    assert result["ok"] is True
    assert calls == [
        {
            "side": "BUY",
            "symbol": "ETHUSDT",
            "quantity": "0.012",
            "reduce_only": False,
            "client_order_prefix": "FL_DUST_OPEN_ONLY",
        }
    ]
    assert result["position_after_open"]["positionAmt"] == "0.012"


def test_flatten_position_is_noop_when_already_flat() -> None:
    class FakeClient(BinancePrivateClient):
        def account_probe(self) -> dict:
            return {"total_wallet_balance": "100.00000000", "total_available_balance": "100.00000000"}

        def _first_position_risk(self, symbol: str) -> dict:
            return {
                "symbol": symbol,
                "positionAmt": "0.000",
                "entryPrice": "0.0",
                "markPrice": "1726.50",
                "unRealizedProfit": "0.00000000",
                "liquidationPrice": "0",
                "leverage": "150",
                "marginType": "cross",
                "isolatedMargin": "0.00000000",
                "positionSide": "BOTH",
            }

        def open_orders(self, symbol: str | None = None) -> list[dict]:
            return []

    client = FakeClient(
        Settings(
            SYMBOL="ETHUSDT",
            BINANCE_API_KEY="live-key",
            BINANCE_API_SECRET="live-secret",
            LIVE_TRADING_ENABLED=True,
            LIVE_DRY_RUN=False,
        )
    )

    result = client.flatten_position()

    assert result["ok"] is True
    assert result["live_order_placed"] is False
    assert result["message"] == "already_flat"
    assert result["final_position"]["positionAmt"] == "0.000"


def test_flatten_position_closes_existing_position_reduce_only() -> None:
    calls = []

    class FakeClient(BinancePrivateClient):
        account_probe_calls = 0

        def account_probe(self) -> dict:
            self.account_probe_calls += 1
            balance = "100.00000000" if self.account_probe_calls == 1 else "99.99000000"
            return {"total_wallet_balance": balance, "total_available_balance": balance}

        def _first_position_risk(self, symbol: str) -> dict:
            return {
                "symbol": symbol,
                "positionAmt": "-0.012",
                "entryPrice": "1726.50",
                "markPrice": "1726.60",
                "unRealizedProfit": "-0.00100000",
                "liquidationPrice": "10000",
                "leverage": "150",
                "marginType": "cross",
                "isolatedMargin": "0.00000000",
                "positionSide": "BOTH",
            }

        def place_market_order(
            self,
            side: str,
            *,
            symbol: str | None = None,
            quantity: str,
            reduce_only: bool = False,
            client_order_prefix: str = "FL_LIVE",
        ) -> dict:
            calls.append(
                {
                    "side": side,
                    "symbol": symbol,
                    "quantity": quantity,
                    "reduce_only": reduce_only,
                    "client_order_prefix": client_order_prefix,
                }
            )
            return {"orderId": 1, "symbol": symbol, "status": "FILLED", "side": side, "executedQty": quantity}

        def _wait_for_position(self, symbol: str, *, expected_nonzero: bool, timeout_seconds: float) -> dict:
            assert expected_nonzero is False
            return {
                "symbol": symbol,
                "positionAmt": "0.000",
                "entryPrice": "0.0",
                "markPrice": "1726.60",
                "unRealizedProfit": "0.00000000",
                "liquidationPrice": "0",
                "leverage": "150",
                "marginType": "cross",
                "isolatedMargin": "0.00000000",
                "positionSide": "BOTH",
            }

        def open_orders(self, symbol: str | None = None) -> list[dict]:
            return []

    client = FakeClient(
        Settings(
            SYMBOL="ETHUSDT",
            BINANCE_API_KEY="live-key",
            BINANCE_API_SECRET="live-secret",
            LIVE_TRADING_ENABLED=True,
            LIVE_DRY_RUN=False,
        )
    )

    result = client.flatten_position()

    assert result["ok"] is True
    assert calls == [
        {
            "side": "BUY",
            "symbol": "ETHUSDT",
            "quantity": "0.012",
            "reduce_only": True,
            "client_order_prefix": "FL_FLATTEN",
        }
    ]
    assert result["final_position"]["positionAmt"] == "0.000"
