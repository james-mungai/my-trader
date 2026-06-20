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
