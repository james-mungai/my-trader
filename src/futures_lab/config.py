from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = Field(default="dev", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    symbol: str = Field(default="BTCUSDT", alias="SYMBOL")
    binance_env: str = Field(default="LIVE", alias="BINANCE_ENV")
    record_raw_ws: bool = Field(default=True, alias="RECORD_RAW_WS")
    data_dir: Path = Field(default=Path("data"), alias="DATA_DIR")
    state_window_seconds: int = Field(default=900, alias="STATE_WINDOW_SECONDS")
    stale_after_seconds: float = Field(default=2.0, alias="STALE_AFTER_SECONDS")

    decision_interval_ms: int = Field(default=500, alias="DECISION_INTERVAL_MS")
    min_warmup_seconds: int = Field(default=180, alias="MIN_WARMUP_SECONDS")
    fast_target_move_pct: float = Field(default=0.005, alias="FAST_TARGET_MOVE_PCT")
    fast_stop_move_pct: float = Field(default=0.002, alias="FAST_STOP_MOVE_PCT")
    slow_target_move_pct: float = Field(default=0.008, alias="SLOW_TARGET_MOVE_PCT")
    slow_stop_move_pct: float = Field(default=0.0035, alias="SLOW_STOP_MOVE_PCT")
    max_spread_bps: float = Field(default=2.0, alias="MAX_SPREAD_BPS")
    min_confidence: float = Field(default=0.72, alias="MIN_CONFIDENCE")

    account_equity_usd: float = Field(default=1000.0, alias="ACCOUNT_EQUITY_USD")
    stake_fraction: float = Field(default=0.15, alias="STAKE_FRACTION")
    fast_leverage: int = Field(default=200, alias="FAST_LEVERAGE")
    slow_leverage: int = Field(default=80, alias="SLOW_LEVERAGE")
    taker_fee_bps: float = Field(default=4.0, alias="TAKER_FEE_BPS")
    daily_target_usd: float = Field(default=150.0, alias="DAILY_TARGET_USD")
    daily_max_loss_usd: float = Field(default=75.0, alias="DAILY_MAX_LOSS_USD")
    max_trades_per_day: int = Field(default=4, alias="MAX_TRADES_PER_DAY")

    api_token: str | None = Field(default=None, alias="API_TOKEN")

    binance_api_key: str | None = Field(default=None, alias="BINANCE_API_KEY")
    binance_api_secret: str | None = Field(default=None, alias="BINANCE_API_SECRET")
    binance_demo_api_key: str | None = Field(default=None, alias="BINANCE_DEMO_API_KEY")
    binance_demo_api_secret: str | None = Field(default=None, alias="BINANCE_DEMO_API_SECRET")
    binance_futures_testnet_api_key: str | None = Field(default=None, alias="BINANCE_FUTURES_TESTNET_API_KEY")
    binance_futures_testnet_api_secret: str | None = Field(default=None, alias="BINANCE_FUTURES_TESTNET_API_SECRET")

    @property
    def symbol_lower(self) -> str:
        return self.symbol.strip().lower()

    @property
    def stake_usd(self) -> float:
        return self.account_equity_usd * self.stake_fraction


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

