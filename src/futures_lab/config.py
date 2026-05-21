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
    consume_depth_stream: bool = Field(default=True, alias="CONSUME_DEPTH_STREAM")
    record_depth_stream: bool = Field(default=False, alias="RECORD_DEPTH_STREAM")
    depth_levels: int = Field(default=5, alias="DEPTH_LEVELS")
    consume_liquidation_stream: bool = Field(default=True, alias="CONSUME_LIQUIDATION_STREAM")
    record_book_ticker_min_interval_ms: int = Field(default=250, alias="RECORD_BOOK_TICKER_MIN_INTERVAL_MS")
    raw_rotation_minutes: int = Field(default=60, alias="RAW_ROTATION_MINUTES")
    compress_rotated_raw: bool = Field(default=True, alias="COMPRESS_ROTATED_RAW")
    write_feature_log: bool = Field(default=True, alias="WRITE_FEATURE_LOG")
    write_decision_log: bool = Field(default=True, alias="WRITE_DECISION_LOG")
    write_paper_trade_log: bool = Field(default=True, alias="WRITE_PAPER_TRADE_LOG")
    write_shadow_trade_log: bool = Field(default=True, alias="WRITE_SHADOW_TRADE_LOG")
    write_regime_outcome_log: bool = Field(default=True, alias="WRITE_REGIME_OUTCOME_LOG")
    data_dir: Path = Field(default=Path("data"), alias="DATA_DIR")
    state_window_seconds: int = Field(default=900, alias="STATE_WINDOW_SECONDS")
    stale_after_seconds: float = Field(default=2.0, alias="STALE_AFTER_SECONDS")
    open_interest_poll_seconds: int = Field(default=30, alias="OPEN_INTEREST_POLL_SECONDS")
    higher_timeframe_enabled: bool = Field(default=True, alias="HIGHER_TIMEFRAME_ENABLED")
    higher_timeframe_intervals: str = Field(default="5m,1h,4h,1d,1w", alias="HIGHER_TIMEFRAME_INTERVALS")
    higher_timeframe_poll_seconds: int = Field(default=300, alias="HIGHER_TIMEFRAME_POLL_SECONDS")
    higher_timeframe_limit: int = Field(default=120, alias="HIGHER_TIMEFRAME_LIMIT")
    higher_timeframe_min_closed_candles: int = Field(default=30, alias="HIGHER_TIMEFRAME_MIN_CLOSED_CANDLES")
    higher_timeframe_score_boost: float = Field(default=0.04, alias="HIGHER_TIMEFRAME_SCORE_BOOST")
    higher_timeframe_score_penalty: float = Field(default=0.02, alias="HIGHER_TIMEFRAME_SCORE_PENALTY")

    decision_interval_ms: int = Field(default=500, alias="DECISION_INTERVAL_MS")
    min_warmup_seconds: int = Field(default=180, alias="MIN_WARMUP_SECONDS")
    strategy_variant: str = Field(default="baseline", alias="STRATEGY_VARIANT")
    enable_markov_state_machine: bool = Field(default=True, alias="ENABLE_MARKOV_STATE_MACHINE")
    markov_state_history: int = Field(default=8, alias="MARKOV_STATE_HISTORY")
    session_bias_side: str = Field(default="neutral", alias="SESSION_BIAS_SIDE")
    session_bias_strength: float = Field(default=0.0, alias="SESSION_BIAS_STRENGTH")
    session_bias_reason: str = Field(default="", alias="SESSION_BIAS_REASON")
    fast_target_move_pct: float = Field(default=0.002, alias="FAST_TARGET_MOVE_PCT")
    fast_stop_move_pct: float = Field(default=0.0015, alias="FAST_STOP_MOVE_PCT")
    slow_target_move_pct: float = Field(default=0.0035, alias="SLOW_TARGET_MOVE_PCT")
    slow_stop_move_pct: float = Field(default=0.0035, alias="SLOW_STOP_MOVE_PCT")
    stateful_target_feasibility_fraction: float = Field(default=0.60, alias="STATEFUL_TARGET_FEASIBILITY_FRACTION")
    stateful_adaptive_target_fraction: float = Field(default=0.50, alias="STATEFUL_ADAPTIVE_TARGET_FRACTION")
    stateful_adaptive_min_sequence_confidence: float = Field(default=0.93, alias="STATEFUL_ADAPTIVE_MIN_SEQUENCE_CONFIDENCE")
    stateful_adaptive_min_score: float = Field(default=0.70, alias="STATEFUL_ADAPTIVE_MIN_SCORE")
    stateful_adaptive_min_quality_long: float = Field(default=0.74, alias="STATEFUL_ADAPTIVE_MIN_QUALITY_LONG")
    stateful_adaptive_min_quality_short: float = Field(default=0.68, alias="STATEFUL_ADAPTIVE_MIN_QUALITY_SHORT")
    stateful_adaptive_target_move_pct: float = Field(default=0.0020, alias="STATEFUL_ADAPTIVE_TARGET_MOVE_PCT")
    stateful_adaptive_stop_move_pct: float = Field(default=0.0015, alias="STATEFUL_ADAPTIVE_STOP_MOVE_PCT")
    higher_timeframe_execution_gate_enabled: bool = Field(default=True, alias="HIGHER_TIMEFRAME_EXECUTION_GATE_ENABLED")
    higher_timeframe_gate_min_strength: float = Field(default=0.35, alias="HIGHER_TIMEFRAME_GATE_MIN_STRENGTH")
    higher_timeframe_countertrend_min_quality: float = Field(default=0.86, alias="HIGHER_TIMEFRAME_COUNTERTREND_MIN_QUALITY")
    higher_timeframe_countertrend_min_score: float = Field(default=0.82, alias="HIGHER_TIMEFRAME_COUNTERTREND_MIN_SCORE")
    higher_timeframe_countertrend_min_sequence_confidence: float = Field(
        default=0.96,
        alias="HIGHER_TIMEFRAME_COUNTERTREND_MIN_SEQUENCE_CONFIDENCE",
    )
    higher_timeframe_exception_target_move_pct: float = Field(default=0.0035, alias="HIGHER_TIMEFRAME_EXCEPTION_TARGET_MOVE_PCT")
    higher_timeframe_strong_exception_target_move_pct: float = Field(
        default=0.0050,
        alias="HIGHER_TIMEFRAME_STRONG_EXCEPTION_TARGET_MOVE_PCT",
    )
    counter_htf_bounce_enabled: bool = Field(default=True, alias="COUNTER_HTF_BOUNCE_ENABLED")
    counter_htf_bounce_symbols: str = Field(default="ETHUSDT", alias="COUNTER_HTF_BOUNCE_SYMBOLS")
    counter_htf_bounce_min_quality: float = Field(default=0.74, alias="COUNTER_HTF_BOUNCE_MIN_QUALITY")
    counter_htf_bounce_min_score: float = Field(default=0.74, alias="COUNTER_HTF_BOUNCE_MIN_SCORE")
    counter_htf_bounce_min_sequence_confidence: float = Field(
        default=0.90,
        alias="COUNTER_HTF_BOUNCE_MIN_SEQUENCE_CONFIDENCE",
    )
    counter_htf_bounce_min_range_pct: float = Field(default=0.0020, alias="COUNTER_HTF_BOUNCE_MIN_RANGE_PCT")
    local_execution_gate_enabled: bool = Field(default=True, alias="LOCAL_EXECUTION_GATE_ENABLED")
    local_5m_countertrend_trend_threshold: float = Field(default=0.45, alias="LOCAL_5M_COUNTERTREND_TREND_THRESHOLD")
    local_reversal_return_15s_pct: float = Field(default=0.00035, alias="LOCAL_REVERSAL_RETURN_15S_PCT")
    local_reversal_return_60s_pct: float = Field(default=0.00060, alias="LOCAL_REVERSAL_RETURN_60S_PCT")
    local_reversal_pressure: float = Field(default=0.08, alias="LOCAL_REVERSAL_PRESSURE")
    fee_edge_quality_gate_enabled: bool = Field(default=True, alias="FEE_EDGE_QUALITY_GATE_ENABLED")
    fee_edge_fast_min_quality: float = Field(default=0.74, alias="FEE_EDGE_FAST_MIN_QUALITY")
    fee_edge_fast_min_sequence_confidence: float = Field(default=0.90, alias="FEE_EDGE_FAST_MIN_SEQUENCE_CONFIDENCE")
    fee_edge_target_fee_buffer: float = Field(default=1.25, alias="FEE_EDGE_TARGET_FEE_BUFFER")
    duplicate_signal_suppression_seconds: int = Field(default=900, alias="DUPLICATE_SIGNAL_SUPPRESSION_SECONDS")
    duplicate_signal_min_quality_improvement: float = Field(default=0.04, alias="DUPLICATE_SIGNAL_MIN_QUALITY_IMPROVEMENT")
    duplicate_signal_min_score_improvement: float = Field(default=0.04, alias="DUPLICATE_SIGNAL_MIN_SCORE_IMPROVEMENT")
    regime_outcome_horizons_seconds: str = Field(default="15,30,60,180,300,900", alias="REGIME_OUTCOME_HORIZONS_SECONDS")
    regime_outcome_target_moves_pct: str = Field(default="0.001,0.002,0.0035,0.005", alias="REGIME_OUTCOME_TARGET_MOVES_PCT")
    regime_outcome_stop_moves_pct: str = Field(default="0.001,0.002", alias="REGIME_OUTCOME_STOP_MOVES_PCT")
    regime_outcome_cooldown_seconds: int = Field(default=900, alias="REGIME_OUTCOME_COOLDOWN_SECONDS")
    regime_outcome_early_horizons_seconds: str = Field(default="15,30,60", alias="REGIME_OUTCOME_EARLY_HORIZONS_SECONDS")
    regime_outcome_min_early_correct: int = Field(default=2, alias="REGIME_OUTCOME_MIN_EARLY_CORRECT")
    max_spread_bps: float = Field(default=2.0, alias="MAX_SPREAD_BPS")
    min_confidence: float = Field(default=0.72, alias="MIN_CONFIDENCE")
    enable_price_stop: bool = Field(default=True, alias="ENABLE_PRICE_STOP")
    emergency_max_adverse_move_pct: float = Field(default=0.004, alias="EMERGENCY_MAX_ADVERSE_MOVE_PCT")
    max_position_seconds: int = Field(default=0, alias="MAX_POSITION_SECONDS")
    max_short_taker_buy_ratio_30s: float = Field(default=0.60, alias="MAX_SHORT_TAKER_BUY_RATIO_30S")
    enable_fast_failure_exit: bool = Field(default=False, alias="ENABLE_FAST_FAILURE_EXIT")
    fast_failure_seconds: int = Field(default=180, alias="FAST_FAILURE_SECONDS")
    fast_failure_min_favorable_move_pct: float = Field(default=0.0005, alias="FAST_FAILURE_MIN_FAVORABLE_MOVE_PCT")
    trade_cooldown_seconds: int = Field(default=1800, alias="TRADE_COOLDOWN_SECONDS")

    account_equity_usd: float = Field(default=1000.0, alias="ACCOUNT_EQUITY_USD")
    stake_fraction: float = Field(default=0.15, alias="STAKE_FRACTION")
    fast_leverage: int = Field(default=200, alias="FAST_LEVERAGE")
    slow_leverage: int = Field(default=80, alias="SLOW_LEVERAGE")
    taker_fee_bps: float = Field(default=4.0, alias="TAKER_FEE_BPS")
    min_gross_target_fee_multiple: float = Field(default=2.0, alias="MIN_GROSS_TARGET_FEE_MULTIPLE")
    daily_target_usd: float = Field(default=150.0, alias="DAILY_TARGET_USD")
    daily_max_loss_usd: float = Field(default=75.0, alias="DAILY_MAX_LOSS_USD")
    max_trades_per_day: int = Field(default=0, alias="MAX_TRADES_PER_DAY")

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
