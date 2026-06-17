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
    binance_stream_profile: str = Field(default="current", alias="BINANCE_STREAM_PROFILE")
    record_raw_ws: bool = Field(default=True, alias="RECORD_RAW_WS")
    consume_book_ticker_stream: bool = Field(default=True, alias="CONSUME_BOOK_TICKER_STREAM")
    consume_depth_stream: bool = Field(default=True, alias="CONSUME_DEPTH_STREAM")
    record_depth_stream: bool = Field(default=False, alias="RECORD_DEPTH_STREAM")
    depth_levels: int = Field(default=5, alias="DEPTH_LEVELS")
    depth_update_speed_ms: int = Field(default=100, alias="DEPTH_UPDATE_SPEED_MS")
    consume_depth_top_book_min_interval_ms: int = Field(default=250, alias="CONSUME_DEPTH_TOP_BOOK_MIN_INTERVAL_MS")
    consume_liquidation_stream: bool = Field(default=True, alias="CONSUME_LIQUIDATION_STREAM")
    consume_book_ticker_min_interval_ms: int = Field(default=250, alias="CONSUME_BOOK_TICKER_MIN_INTERVAL_MS")
    record_book_ticker_min_interval_ms: int = Field(default=250, alias="RECORD_BOOK_TICKER_MIN_INTERVAL_MS")
    record_agg_trade_stream: bool = Field(default=False, alias="RECORD_AGG_TRADE_STREAM")
    record_agg_trade_min_interval_ms: int = Field(default=250, alias="RECORD_AGG_TRADE_MIN_INTERVAL_MS")
    raw_rotation_minutes: int = Field(default=60, alias="RAW_ROTATION_MINUTES")
    compress_rotated_raw: bool = Field(default=True, alias="COMPRESS_ROTATED_RAW")
    write_feature_log: bool = Field(default=True, alias="WRITE_FEATURE_LOG")
    write_decision_log: bool = Field(default=True, alias="WRITE_DECISION_LOG")
    write_paper_trade_log: bool = Field(default=True, alias="WRITE_PAPER_TRADE_LOG")
    write_shadow_trade_log: bool = Field(default=True, alias="WRITE_SHADOW_TRADE_LOG")
    write_regime_outcome_log: bool = Field(default=True, alias="WRITE_REGIME_OUTCOME_LOG")
    write_candidate_outcome_log: bool = Field(default=True, alias="WRITE_CANDIDATE_OUTCOME_LOG")
    data_dir: Path = Field(default=Path("data"), alias="DATA_DIR")
    state_window_seconds: int = Field(default=900, alias="STATE_WINDOW_SECONDS")
    stale_after_seconds: float = Field(default=2.0, alias="STALE_AFTER_SECONDS")
    max_exchange_event_lag_ms: float = Field(default=2_000.0, alias="MAX_EXCHANGE_EVENT_LAG_MS")
    open_interest_poll_seconds: int = Field(default=30, alias="OPEN_INTEREST_POLL_SECONDS")
    shutdown_timeout_seconds: float = Field(default=30.0, alias="SHUTDOWN_TIMEOUT_SECONDS")
    higher_timeframe_enabled: bool = Field(default=True, alias="HIGHER_TIMEFRAME_ENABLED")
    higher_timeframe_intervals: str = Field(default="5m,1h,4h,1d,1w", alias="HIGHER_TIMEFRAME_INTERVALS")
    higher_timeframe_poll_seconds: int = Field(default=300, alias="HIGHER_TIMEFRAME_POLL_SECONDS")
    higher_timeframe_limit: int = Field(default=120, alias="HIGHER_TIMEFRAME_LIMIT")
    higher_timeframe_min_closed_candles: int = Field(default=30, alias="HIGHER_TIMEFRAME_MIN_CLOSED_CANDLES")
    higher_timeframe_score_boost: float = Field(default=0.04, alias="HIGHER_TIMEFRAME_SCORE_BOOST")
    higher_timeframe_score_penalty: float = Field(default=0.02, alias="HIGHER_TIMEFRAME_SCORE_PENALTY")
    cross_market_enabled: bool = Field(default=False, alias="CROSS_MARKET_ENABLED")
    cross_market_anchor_symbol: str = Field(default="BTCUSDT", alias="CROSS_MARKET_ANCHOR_SYMBOL")
    cross_market_max_context_age_seconds: float = Field(default=2.0, alias="CROSS_MARKET_MAX_CONTEXT_AGE_SECONDS")
    cross_market_btc_veto_score_ceiling: float = Field(default=0.86, alias="CROSS_MARKET_BTC_VETO_SCORE_CEILING")
    cross_market_relative_strength_override_score: float = Field(
        default=0.92,
        alias="CROSS_MARKET_RELATIVE_STRENGTH_OVERRIDE_SCORE",
    )
    cross_market_min_relative_strength: float = Field(default=0.72, alias="CROSS_MARKET_MIN_RELATIVE_STRENGTH")

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
    range_bound_target_move_pct: float = Field(default=0.0050, alias="RANGE_BOUND_TARGET_MOVE_PCT")
    range_bound_stop_move_pct: float = Field(default=0.0025, alias="RANGE_BOUND_STOP_MOVE_PCT")
    range_bound_leverage: int = Field(default=200, alias="RANGE_BOUND_LEVERAGE")
    range_bound_timeframes: str = Field(default="15m,30m,1h", alias="RANGE_BOUND_TIMEFRAMES")
    range_bound_min_room_to_target_pct: float = Field(default=0.0055, alias="RANGE_BOUND_MIN_ROOM_TO_TARGET_PCT")
    range_bound_edge_zone: float = Field(default=0.28, alias="RANGE_BOUND_EDGE_ZONE")
    range_bound_max_abs_trend_score: float = Field(default=0.55, alias="RANGE_BOUND_MAX_ABS_TREND_SCORE")
    range_bound_min_average_range_pct: float = Field(default=0.0060, alias="RANGE_BOUND_MIN_AVERAGE_RANGE_PCT")
    range_bound_max_context_age_seconds: float = Field(default=900.0, alias="RANGE_BOUND_MAX_CONTEXT_AGE_SECONDS")
    range_bound_flow_weight: float = Field(default=0.12, alias="RANGE_BOUND_FLOW_WEIGHT")
    range_bound_pressure_weight: float = Field(default=0.10, alias="RANGE_BOUND_PRESSURE_WEIGHT")
    range_bound_paper_live_min_expected_ev_bps: float = Field(
        default=4.0,
        alias="RANGE_BOUND_PAPER_LIVE_MIN_EXPECTED_EV_BPS",
    )
    range_bound_paper_live_min_score: float = Field(default=0.70, alias="RANGE_BOUND_PAPER_LIVE_MIN_SCORE")
    range_bound_paper_live_min_tp_probability: float = Field(
        default=0.70,
        alias="RANGE_BOUND_PAPER_LIVE_MIN_TP_PROBABILITY",
    )
    range_bound_rolling_edge_monitor_enabled: bool = Field(
        default=False,
        alias="RANGE_BOUND_ROLLING_EDGE_MONITOR_ENABLED",
    )
    range_bound_structural_risk_enabled: bool = Field(default=True, alias="RANGE_BOUND_STRUCTURAL_RISK_ENABLED")
    range_bound_structural_exit_enabled: bool = Field(default=True, alias="RANGE_BOUND_STRUCTURAL_EXIT_ENABLED")
    range_bound_structural_buffer_pct: float = Field(default=0.0005, alias="RANGE_BOUND_STRUCTURAL_BUFFER_PCT")
    range_bound_max_structural_adverse_move_pct: float = Field(
        default=0.035,
        alias="RANGE_BOUND_MAX_STRUCTURAL_ADVERSE_MOVE_PCT",
    )
    range_bound_max_account_drawdown_fraction: float = Field(
        default=0.65,
        alias="RANGE_BOUND_MAX_ACCOUNT_DRAWDOWN_FRACTION",
    )
    range_bound_dynamic_leverage_enabled: bool = Field(default=True, alias="RANGE_BOUND_DYNAMIC_LEVERAGE_ENABLED")
    range_bound_min_leverage: int = Field(default=40, alias="RANGE_BOUND_MIN_LEVERAGE")
    range_bound_expected_loss_fraction_of_structural: float = Field(
        default=0.25,
        alias="RANGE_BOUND_EXPECTED_LOSS_FRACTION_OF_STRUCTURAL",
    )
    range_bound_candidate_min_htf_edge_score: float = Field(
        default=0.55,
        alias="RANGE_BOUND_CANDIDATE_MIN_HTF_EDGE_SCORE",
    )
    range_bound_candidate_min_local_edge: float = Field(
        default=0.60,
        alias="RANGE_BOUND_CANDIDATE_MIN_LOCAL_EDGE",
    )
    range_bound_candidate_min_flow_alignment: float = Field(
        default=0.52,
        alias="RANGE_BOUND_CANDIDATE_MIN_FLOW_ALIGNMENT",
    )
    range_bound_candidate_min_pressure_alignment: float = Field(
        default=0.52,
        alias="RANGE_BOUND_CANDIDATE_MIN_PRESSURE_ALIGNMENT",
    )
    range_bound_candidate_min_target_room_multiple: float = Field(
        default=1.20,
        alias="RANGE_BOUND_CANDIDATE_MIN_TARGET_ROOM_MULTIPLE",
    )
    range_bound_candidate_max_account_drawdown_fraction: float = Field(
        default=0.55,
        alias="RANGE_BOUND_CANDIDATE_MAX_ACCOUNT_DRAWDOWN_FRACTION",
    )
    range_bound_time_decay_exit_enabled: bool = Field(
        default=False,
        alias="RANGE_BOUND_TIME_DECAY_EXIT_ENABLED",
    )
    range_bound_time_decay_seconds: int = Field(default=180, alias="RANGE_BOUND_TIME_DECAY_SECONDS")
    range_bound_time_decay_min_mfe_fee_multiple: float = Field(
        default=1.25,
        alias="RANGE_BOUND_TIME_DECAY_MIN_MFE_FEE_MULTIPLE",
    )
    range_bound_time_decay_adverse_move_pct: float = Field(
        default=0.0015,
        alias="RANGE_BOUND_TIME_DECAY_ADVERSE_MOVE_PCT",
    )
    range_bound_time_decay_flow_adverse_move_pct: float = Field(
        default=0.0005,
        alias="RANGE_BOUND_TIME_DECAY_FLOW_ADVERSE_MOVE_PCT",
    )
    range_bound_exit_counterfactual_enabled: bool = Field(
        default=True,
        alias="RANGE_BOUND_EXIT_COUNTERFACTUAL_ENABLED",
    )
    range_bound_exit_counterfactual_horizon_seconds: int = Field(
        default=21_600,
        alias="RANGE_BOUND_EXIT_COUNTERFACTUAL_HORIZON_SECONDS",
    )
    range_bound_exit_counterfactual_cross_wallet_fraction: float = Field(
        default=1.0,
        alias="RANGE_BOUND_EXIT_COUNTERFACTUAL_CROSS_WALLET_FRACTION",
    )
    range_bound_exit_counterfactual_maintenance_margin_pct: float = Field(
        default=0.004,
        alias="RANGE_BOUND_EXIT_COUNTERFACTUAL_MAINTENANCE_MARGIN_PCT",
    )
    stateful_target_feasibility_fraction: float = Field(default=0.60, alias="STATEFUL_TARGET_FEASIBILITY_FRACTION")
    stateful_adaptive_target_fraction: float = Field(default=0.50, alias="STATEFUL_ADAPTIVE_TARGET_FRACTION")
    stateful_adaptive_min_sequence_confidence: float = Field(default=0.93, alias="STATEFUL_ADAPTIVE_MIN_SEQUENCE_CONFIDENCE")
    stateful_adaptive_min_score: float = Field(default=0.70, alias="STATEFUL_ADAPTIVE_MIN_SCORE")
    stateful_adaptive_min_quality_long: float = Field(default=0.74, alias="STATEFUL_ADAPTIVE_MIN_QUALITY_LONG")
    stateful_adaptive_min_quality_short: float = Field(default=0.68, alias="STATEFUL_ADAPTIVE_MIN_QUALITY_SHORT")
    stateful_adaptive_target_move_pct: float = Field(default=0.0020, alias="STATEFUL_ADAPTIVE_TARGET_MOVE_PCT")
    stateful_adaptive_stop_move_pct: float = Field(default=0.0015, alias="STATEFUL_ADAPTIVE_STOP_MOVE_PCT")
    stateful_adaptive_live_enabled: bool = Field(default=False, alias="STATEFUL_ADAPTIVE_LIVE_ENABLED")
    stateful_adaptive_live_min_follow_score: float = Field(
        default=0.88,
        alias="STATEFUL_ADAPTIVE_LIVE_MIN_FOLLOW_SCORE",
    )
    stateful_adaptive_live_min_follow_confirmations: int = Field(
        default=5,
        alias="STATEFUL_ADAPTIVE_LIVE_MIN_FOLLOW_CONFIRMATIONS",
    )
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
    counter_htf_bounce_live_enabled: bool = Field(default=False, alias="COUNTER_HTF_BOUNCE_LIVE_ENABLED")
    counter_htf_bounce_symbols: str = Field(default="ETHUSDT", alias="COUNTER_HTF_BOUNCE_SYMBOLS")
    counter_htf_bounce_min_quality: float = Field(default=0.68, alias="COUNTER_HTF_BOUNCE_MIN_QUALITY")
    counter_htf_bounce_min_score: float = Field(default=0.70, alias="COUNTER_HTF_BOUNCE_MIN_SCORE")
    counter_htf_bounce_min_sequence_confidence: float = Field(
        default=0.85,
        alias="COUNTER_HTF_BOUNCE_MIN_SEQUENCE_CONFIDENCE",
    )
    counter_htf_bounce_min_range_pct: float = Field(default=0.0012, alias="COUNTER_HTF_BOUNCE_MIN_RANGE_PCT")
    counter_htf_bounce_1h_relief_return_pct: float = Field(
        default=0.0030,
        alias="COUNTER_HTF_BOUNCE_1H_RELIEF_RETURN_PCT",
    )
    local_execution_gate_enabled: bool = Field(default=True, alias="LOCAL_EXECUTION_GATE_ENABLED")
    local_5m_countertrend_trend_threshold: float = Field(default=0.45, alias="LOCAL_5M_COUNTERTREND_TREND_THRESHOLD")
    local_reversal_return_15s_pct: float = Field(default=0.00035, alias="LOCAL_REVERSAL_RETURN_15S_PCT")
    local_reversal_return_60s_pct: float = Field(default=0.00060, alias="LOCAL_REVERSAL_RETURN_60S_PCT")
    local_reversal_pressure: float = Field(default=0.08, alias="LOCAL_REVERSAL_PRESSURE")
    fee_edge_quality_gate_enabled: bool = Field(default=True, alias="FEE_EDGE_QUALITY_GATE_ENABLED")
    fee_edge_fast_min_quality: float = Field(default=0.74, alias="FEE_EDGE_FAST_MIN_QUALITY")
    fee_edge_fast_min_sequence_confidence: float = Field(default=0.90, alias="FEE_EDGE_FAST_MIN_SEQUENCE_CONFIDENCE")
    fee_edge_counter_htf_bounce_min_quality: float = Field(default=0.68, alias="FEE_EDGE_COUNTER_HTF_BOUNCE_MIN_QUALITY")
    fee_edge_counter_htf_bounce_min_sequence_confidence: float = Field(
        default=0.85,
        alias="FEE_EDGE_COUNTER_HTF_BOUNCE_MIN_SEQUENCE_CONFIDENCE",
    )
    weak_neutral_short_gate_enabled: bool = Field(default=True, alias="WEAK_NEUTRAL_SHORT_GATE_ENABLED")
    weak_neutral_short_min_quality: float = Field(default=0.90, alias="WEAK_NEUTRAL_SHORT_MIN_QUALITY")
    weak_neutral_short_min_sequence_confidence: float = Field(
        default=0.94,
        alias="WEAK_NEUTRAL_SHORT_MIN_SEQUENCE_CONFIDENCE",
    )
    weak_neutral_short_min_follow_score: float = Field(default=0.95, alias="WEAK_NEUTRAL_SHORT_MIN_FOLLOW_SCORE")
    weak_neutral_long_gate_enabled: bool = Field(default=True, alias="WEAK_NEUTRAL_LONG_GATE_ENABLED")
    weak_neutral_long_min_quality: float = Field(default=0.82, alias="WEAK_NEUTRAL_LONG_MIN_QUALITY")
    weak_neutral_long_min_sequence_confidence: float = Field(
        default=0.94,
        alias="WEAK_NEUTRAL_LONG_MIN_SEQUENCE_CONFIDENCE",
    )
    weak_neutral_long_min_follow_score: float = Field(default=0.98, alias="WEAK_NEUTRAL_LONG_MIN_FOLLOW_SCORE")
    weak_neutral_long_min_1h_trend_score: float = Field(default=0.25, alias="WEAK_NEUTRAL_LONG_MIN_1H_TREND_SCORE")
    weak_neutral_long_min_1h_return_pct: float = Field(default=0.0025, alias="WEAK_NEUTRAL_LONG_MIN_1H_RETURN_PCT")
    fee_edge_target_fee_buffer: float = Field(default=1.25, alias="FEE_EDGE_TARGET_FEE_BUFFER")
    duplicate_signal_suppression_seconds: int = Field(default=900, alias="DUPLICATE_SIGNAL_SUPPRESSION_SECONDS")
    duplicate_signal_min_quality_improvement: float = Field(default=0.04, alias="DUPLICATE_SIGNAL_MIN_QUALITY_IMPROVEMENT")
    duplicate_signal_min_score_improvement: float = Field(default=0.04, alias="DUPLICATE_SIGNAL_MIN_SCORE_IMPROVEMENT")
    entry_follow_through_gate_enabled: bool = Field(default=True, alias="ENTRY_FOLLOW_THROUGH_GATE_ENABLED")
    entry_follow_through_min_score: float = Field(default=0.68, alias="ENTRY_FOLLOW_THROUGH_MIN_SCORE")
    entry_follow_through_min_confirmations: int = Field(default=4, alias="ENTRY_FOLLOW_THROUGH_MIN_CONFIRMATIONS")
    entry_follow_through_return_15s_pct: float = Field(default=0.00035, alias="ENTRY_FOLLOW_THROUGH_RETURN_15S_PCT")
    entry_follow_through_return_60s_pct: float = Field(default=0.00060, alias="ENTRY_FOLLOW_THROUGH_RETURN_60S_PCT")
    entry_follow_through_pressure: float = Field(default=0.08, alias="ENTRY_FOLLOW_THROUGH_PRESSURE")
    entry_follow_through_htf_aligned_override_score: float = Field(
        default=0.80, alias="ENTRY_FOLLOW_THROUGH_HTF_ALIGNED_OVERRIDE_SCORE"
    )
    entry_follow_through_counter_htf_bounce_override_score: float = Field(
        default=0.84, alias="ENTRY_FOLLOW_THROUGH_COUNTER_HTF_BOUNCE_OVERRIDE_SCORE"
    )
    exit_shadow_enabled: bool = Field(default=True, alias="EXIT_SHADOW_ENABLED")
    exit_shadow_fee_trail_activation_fee_multiple: float = Field(
        default=1.50,
        alias="EXIT_SHADOW_FEE_TRAIL_ACTIVATION_FEE_MULTIPLE",
    )
    exit_shadow_fee_trail_floor_fee_multiple: float = Field(
        default=1.05,
        alias="EXIT_SHADOW_FEE_TRAIL_FLOOR_FEE_MULTIPLE",
    )
    exit_shadow_mfe_trail_activation_pct: float = Field(default=0.0015, alias="EXIT_SHADOW_MFE_TRAIL_ACTIVATION_PCT")
    exit_shadow_mfe_trail_distance_pct: float = Field(default=0.0010, alias="EXIT_SHADOW_MFE_TRAIL_DISTANCE_PCT")
    exit_shadow_partial_take_profit_pct: float = Field(default=0.0010, alias="EXIT_SHADOW_PARTIAL_TAKE_PROFIT_PCT")
    exit_shadow_partial_take_profit_alt_pct: float = Field(
        default=0.0015,
        alias="EXIT_SHADOW_PARTIAL_TAKE_PROFIT_ALT_PCT",
    )
    exit_shadow_partial_fraction: float = Field(default=0.50, alias="EXIT_SHADOW_PARTIAL_FRACTION")
    exit_shadow_time_decay_seconds: int = Field(default=180, alias="EXIT_SHADOW_TIME_DECAY_SECONDS")
    exit_shadow_time_decay_min_mfe_fee_multiple: float = Field(
        default=1.25,
        alias="EXIT_SHADOW_TIME_DECAY_MIN_MFE_FEE_MULTIPLE",
    )
    paper_exit_policy: str = Field(default="fixed_tp_stop", alias="PAPER_EXIT_POLICY")
    paper_mfe_trailing_profiles: str = Field(
        default="htf_aligned_fast,eth_counter_htf_bounce",
        alias="PAPER_MFE_TRAILING_PROFILES",
    )
    paper_mfe_trail_activation_pct: float = Field(default=0.0015, alias="PAPER_MFE_TRAIL_ACTIVATION_PCT")
    paper_mfe_trail_distance_pct: float = Field(default=0.0010, alias="PAPER_MFE_TRAIL_DISTANCE_PCT")
    paper_mfe_trail_vol_multiplier: float = Field(default=0.75, alias="PAPER_MFE_TRAIL_VOL_MULTIPLIER")
    regime_outcome_horizons_seconds: str = Field(default="15,30,60,180,300,900", alias="REGIME_OUTCOME_HORIZONS_SECONDS")
    regime_outcome_target_moves_pct: str = Field(default="0.001,0.002,0.0035,0.005", alias="REGIME_OUTCOME_TARGET_MOVES_PCT")
    regime_outcome_stop_moves_pct: str = Field(default="0.001,0.002", alias="REGIME_OUTCOME_STOP_MOVES_PCT")
    regime_outcome_cooldown_seconds: int = Field(default=900, alias="REGIME_OUTCOME_COOLDOWN_SECONDS")
    regime_outcome_early_horizons_seconds: str = Field(default="15,30,60", alias="REGIME_OUTCOME_EARLY_HORIZONS_SECONDS")
    regime_outcome_min_early_correct: int = Field(default=2, alias="REGIME_OUTCOME_MIN_EARLY_CORRECT")
    candidate_outcome_horizons_seconds: str = Field(default="1,3,5,10,30,60", alias="CANDIDATE_OUTCOME_HORIZONS_SECONDS")
    candidate_outcome_cooldown_seconds: int = Field(default=30, alias="CANDIDATE_OUTCOME_COOLDOWN_SECONDS")
    hostile_replay_enabled: bool = Field(default=False, alias="HOSTILE_REPLAY_ENABLED")
    hostile_replay_entry_slippage_bps: float = Field(default=1.0, alias="HOSTILE_REPLAY_ENTRY_SLIPPAGE_BPS")
    hostile_replay_exit_slippage_bps: float = Field(default=1.0, alias="HOSTILE_REPLAY_EXIT_SLIPPAGE_BPS")
    hostile_replay_stop_penalty_bps: float = Field(default=2.0, alias="HOSTILE_REPLAY_STOP_PENALTY_BPS")
    hostile_replay_feed_latency_ms: int = Field(default=100, alias="HOSTILE_REPLAY_FEED_LATENCY_MS")
    hostile_replay_decision_latency_ms: int = Field(default=50, alias="HOSTILE_REPLAY_DECISION_LATENCY_MS")
    hostile_replay_order_latency_ms: int = Field(default=100, alias="HOSTILE_REPLAY_ORDER_LATENCY_MS")
    hostile_replay_latency_penalty_bps: float = Field(default=0.5, alias="HOSTILE_REPLAY_LATENCY_PENALTY_BPS")
    hostile_replay_latency_bps_per_second: float = Field(default=1.0, alias="HOSTILE_REPLAY_LATENCY_BPS_PER_SECOND")
    hostile_replay_max_book_age_ms: int = Field(default=1_000, alias="HOSTILE_REPLAY_MAX_BOOK_AGE_MS")
    hostile_replay_max_event_lag_ms: float = Field(default=1_000.0, alias="HOSTILE_REPLAY_MAX_EVENT_LAG_MS")
    hostile_replay_maker_adverse_selection_bps: float = Field(
        default=2.0,
        alias="HOSTILE_REPLAY_MAKER_ADVERSE_SELECTION_BPS",
    )
    hostile_replay_maker_queue_fill_probability: float = Field(
        default=0.25,
        alias="HOSTILE_REPLAY_MAKER_QUEUE_FILL_PROBABILITY",
    )
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
    maker_fee_bps: float = Field(default=2.0, alias="MAKER_FEE_BPS")
    taker_fee_bps: float = Field(default=4.0, alias="TAKER_FEE_BPS")
    default_entry_order_type: str = Field(default="taker", alias="DEFAULT_ENTRY_ORDER_TYPE")
    default_exit_order_type: str = Field(default="taker", alias="DEFAULT_EXIT_ORDER_TYPE")
    expected_slippage_bps: float = Field(default=0.5, alias="EXPECTED_SLIPPAGE_BPS")
    latency_adverse_selection_bps: float = Field(default=0.5, alias="LATENCY_ADVERSE_SELECTION_BPS")
    min_gross_target_fee_multiple: float = Field(default=2.0, alias="MIN_GROSS_TARGET_FEE_MULTIPLE")
    paper_live_edge_gate_enabled: bool = Field(default=True, alias="PAPER_LIVE_EDGE_GATE_ENABLED")
    paper_live_min_target_cost_multiple: float = Field(default=2.5, alias="PAPER_LIVE_MIN_TARGET_COST_MULTIPLE")
    paper_live_min_expected_ev_bps: float = Field(default=4.0, alias="PAPER_LIVE_MIN_EXPECTED_EV_BPS")
    paper_live_min_score: float = Field(default=0.82, alias="PAPER_LIVE_MIN_SCORE")
    paper_live_min_tp_probability: float = Field(default=0.82, alias="PAPER_LIVE_MIN_TP_PROBABILITY")
    paper_live_rolling_edge_monitor_enabled: bool = Field(
        default=True,
        alias="PAPER_LIVE_ROLLING_EDGE_MONITOR_ENABLED",
    )
    paper_live_rolling_window: int = Field(default=500, alias="PAPER_LIVE_ROLLING_WINDOW")
    paper_live_rolling_min_accepted: int = Field(default=20, alias="PAPER_LIVE_ROLLING_MIN_ACCEPTED")
    paper_live_rolling_min_rejected: int = Field(default=100, alias="PAPER_LIVE_ROLLING_MIN_REJECTED")
    paper_live_rolling_min_target_rate_edge: float = Field(
        default=0.02,
        alias="PAPER_LIVE_ROLLING_MIN_TARGET_RATE_EDGE",
    )
    paper_live_rolling_min_mfe_edge_bps: float = Field(default=1.0, alias="PAPER_LIVE_ROLLING_MIN_MFE_EDGE_BPS")
    edge_router_shadow_enabled: bool = Field(default=True, alias="EDGE_ROUTER_SHADOW_ENABLED")
    edge_router_min_score: float = Field(default=0.70, alias="EDGE_ROUTER_MIN_SCORE")
    edge_router_min_ev_bps: float = Field(default=1.0, alias="EDGE_ROUTER_MIN_EV_BPS")
    edge_router_target_cost_multiple: float = Field(default=2.0, alias="EDGE_ROUTER_TARGET_COST_MULTIPLE")
    edge_router_impulse_max_hold_ms: int = Field(default=20_000, alias="EDGE_ROUTER_IMPULSE_MAX_HOLD_MS")
    edge_router_liquidation_max_hold_ms: int = Field(default=30_000, alias="EDGE_ROUTER_LIQUIDATION_MAX_HOLD_MS")
    edge_router_paper_enabled: bool = Field(default=False, alias="EDGE_ROUTER_PAPER_ENABLED")
    baseline_micro_confirmation_enabled: bool = Field(default=True, alias="BASELINE_MICRO_CONFIRMATION_ENABLED")
    baseline_micro_min_confirmations: int = Field(default=3, alias="BASELINE_MICRO_MIN_CONFIRMATIONS")
    baseline_micro_min_ofi_1s: float = Field(default=0.35, alias="BASELINE_MICRO_MIN_OFI_1S")
    baseline_micro_min_ofi_5s: float = Field(default=0.15, alias="BASELINE_MICRO_MIN_OFI_5S")
    baseline_micro_min_aggression_5s: float = Field(default=0.15, alias="BASELINE_MICRO_MIN_AGGRESSION_5S")
    baseline_micro_min_pressure_bps: float = Field(default=0.0, alias="BASELINE_MICRO_MIN_PRESSURE_BPS")
    baseline_micro_min_depth_pressure: float = Field(default=0.10, alias="BASELINE_MICRO_MIN_DEPTH_PRESSURE")
    maker_reversion_shadow_enabled: bool = Field(default=False, alias="MAKER_REVERSION_SHADOW_ENABLED")
    maker_reversion_target_bps: float = Field(default=8.0, alias="MAKER_REVERSION_TARGET_BPS")
    maker_reversion_stop_bps: float = Field(default=8.0, alias="MAKER_REVERSION_STOP_BPS")
    maker_reversion_max_hold_ms: int = Field(default=15_000, alias="MAKER_REVERSION_MAX_HOLD_MS")
    maker_reversion_max_ofi: float = Field(default=0.20, alias="MAKER_REVERSION_MAX_OFI")
    maker_reversion_max_aggression: float = Field(default=0.20, alias="MAKER_REVERSION_MAX_AGGRESSION")
    maker_reversion_max_vol_60s_pct: float = Field(default=0.0008, alias="MAKER_REVERSION_MAX_VOL_60S_PCT")
    maker_reversion_max_spread_std_bps: float = Field(default=0.25, alias="MAKER_REVERSION_MAX_SPREAD_STD_BPS")
    maker_reversion_max_spread_bps: float = Field(default=1.2, alias="MAKER_REVERSION_MAX_SPREAD_BPS")
    maker_reversion_min_queue_score: float = Field(default=0.55, alias="MAKER_REVERSION_MIN_QUEUE_SCORE")
    taker_impulse_min_ofi_1s: float = Field(default=0.55, alias="TAKER_IMPULSE_MIN_OFI_1S")
    taker_impulse_min_ofi_5s: float = Field(default=0.25, alias="TAKER_IMPULSE_MIN_OFI_5S")
    taker_impulse_min_aggression_1s: float = Field(default=0.55, alias="TAKER_IMPULSE_MIN_AGGRESSION_1S")
    taker_impulse_min_aggression_5s: float = Field(default=0.20, alias="TAKER_IMPULSE_MIN_AGGRESSION_5S")
    taker_impulse_min_pressure_bps: float = Field(default=0.0, alias="TAKER_IMPULSE_MIN_PRESSURE_BPS")
    taker_impulse_min_depth_pressure: float = Field(default=0.20, alias="TAKER_IMPULSE_MIN_DEPTH_PRESSURE")
    taker_impulse_min_refill_pressure: float = Field(default=0.10, alias="TAKER_IMPULSE_MIN_REFILL_PRESSURE")
    taker_impulse_max_spread_bps: float = Field(default=1.5, alias="TAKER_IMPULSE_MAX_SPREAD_BPS")
    taker_impulse_max_spread_std_bps: float = Field(default=0.75, alias="TAKER_IMPULSE_MAX_SPREAD_STD_BPS")
    taker_impulse_max_event_lag_ms: float = Field(default=750.0, alias="TAKER_IMPULSE_MAX_EVENT_LAG_MS")
    taker_impulse_trail_activation_cost_multiple: float = Field(
        default=1.2,
        alias="TAKER_IMPULSE_TRAIL_ACTIVATION_COST_MULTIPLE",
    )
    liquidation_phase_pressure_notional: float = Field(default=50_000.0, alias="LIQUIDATION_PHASE_PRESSURE_NOTIONAL")
    liquidation_phase_impulse_notional: float = Field(default=250_000.0, alias="LIQUIDATION_PHASE_IMPULSE_NOTIONAL")
    liquidation_phase_min_pressure_building: float = Field(
        default=0.62,
        alias="LIQUIDATION_PHASE_MIN_PRESSURE_BUILDING",
    )
    liquidation_phase_min_continuation_pressure: float = Field(
        default=0.70,
        alias="LIQUIDATION_PHASE_MIN_CONTINUATION_PRESSURE",
    )
    liquidation_phase_min_exhaustion_pressure: float = Field(
        default=0.68,
        alias="LIQUIDATION_PHASE_MIN_EXHAUSTION_PRESSURE",
    )
    liquidation_phase_min_refill_pressure: float = Field(default=0.12, alias="LIQUIDATION_PHASE_MIN_REFILL_PRESSURE")
    liquidation_phase_mark_basis_bps: float = Field(default=2.0, alias="LIQUIDATION_PHASE_MARK_BASIS_BPS")
    liquidation_phase_oi_change_pct: float = Field(default=0.0015, alias="LIQUIDATION_PHASE_OI_CHANGE_PCT")
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
