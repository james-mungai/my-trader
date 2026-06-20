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
