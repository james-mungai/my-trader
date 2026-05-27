import os
from datetime import datetime, timedelta, timezone

from futures_lab.config import Settings
from futures_lab.data_ops import (
    compress_raw,
    prune_raw,
    summarize_candidate_outcomes,
    summarize_data,
    summarize_exit_shadow,
    summarize_regime_outcomes,
)


def test_data_summary_compresses_and_prunes_raw_files(tmp_path):
    raw_dir = tmp_path / "raw_ws"
    raw_dir.mkdir(parents=True)
    old_path = raw_dir / "BTCUSDT_bookTicker_2026-05-11T0700Z.jsonl"
    old_path.write_text('{"ok":true}\n' * 10, encoding="utf-8")
    old_time = (datetime.now(timezone.utc) - timedelta(hours=3)).timestamp()
    os.utime(old_path, (old_time, old_time))

    settings = Settings(DATA_DIR=str(tmp_path))
    summary = summarize_data(settings)
    assert summary.total_bytes > 0
    assert summary.largest_files[0].path.endswith(".jsonl")

    compressed = compress_raw(settings, older_than_minutes=5)
    gz_path = old_path.with_suffix(old_path.suffix + ".gz")
    assert compressed.changed == 1
    assert gz_path.exists()
    assert not old_path.exists()

    pruned_preview = prune_raw(settings, older_than_hours=1, dry_run=True)
    assert pruned_preview.matched == 1
    assert gz_path.exists()

    pruned = prune_raw(settings, older_than_hours=1)
    assert pruned.changed == 1
    assert not gz_path.exists()


def test_regime_outcome_summary_counts_targets_and_horizons(tmp_path):
    outcome_dir = tmp_path / "regime_outcomes"
    outcome_dir.mkdir(parents=True)
    path = outcome_dir / "BTCUSDT_regime_outcomes_2026-05-16.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"event":"open","side":"short","state":"short_continuation_confirmed","quality_score":0.81}',
                '{"event":"close","side":"short","close_reason":"max_horizon_elapsed","target_hits":{"0.001":{"hit":true},"0.002":{"hit":false}},"stop_hits":{"0.001":{"hit":false}},"target_before_stop":{"0.001":{"0.001":true}},"horizons":{"15":{"direction_correct":true},"60":{"direction_correct":false}},"early_follow_through":{"qualified":true}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = summarize_regime_outcomes(Settings(DATA_DIR=str(tmp_path)))

    assert summary.opens == 1
    assert summary.closes == 1
    assert summary.sides == {"short": 1}
    assert summary.target_hits == {"0.001": 1}
    assert summary.target_before_stop == {"0.001": {"0.001": 1}}
    assert summary.horizon_direction_correct["15"] == {"correct": 1}
    assert summary.horizon_direction_correct["60"] == {"wrong": 1}
    assert summary.early_follow_through == {"qualified": 1}
    assert summary.side_early_follow_through == {"short": {"qualified": 1}}
    assert summary.average_quality_score == 0.81


def test_exit_shadow_summary_counts_policy_edges(tmp_path):
    paper_dir = tmp_path / "paper_trades"
    paper_dir.mkdir(parents=True)
    path = paper_dir / "BTCUSDT_paper_trades_2026-05-26.jsonl"
    path.write_text(
        (
            '{"exit_reason":"stop_loss","exit_shadow":{"best_policy":"time_decay","policies":{'
            '"fixed_tp_stop":{"closed":true,"exit_reason":"actual_stop_loss","gross_pnl_usd":-45,"fees_usd":24,"net_pnl_usd":-69,"net_vs_fixed_usd":0},'
            '"time_decay":{"closed":true,"exit_reason":"time_decay","gross_pnl_usd":-3,"fees_usd":24,"net_pnl_usd":-27,"net_vs_fixed_usd":42}'
            "}}}\n"
        ),
        encoding="utf-8",
    )

    summary = summarize_exit_shadow(Settings(DATA_DIR=str(tmp_path)))

    assert summary.rows == 1
    assert summary.rows_with_exit_shadow == 1
    assert summary.best_policy == {"time_decay": 1}
    assert summary.policies["time_decay"].net_vs_fixed_usd == 42
    assert summary.policies["time_decay"].improved_vs_fixed == 1
    assert summary.policies["fixed_tp_stop"].exit_reasons == {"actual_stop_loss": 1}


def test_candidate_outcome_summary_compares_accepted_and_rejected(tmp_path):
    outcome_dir = tmp_path / "candidate_outcomes"
    outcome_dir.mkdir(parents=True)
    path = outcome_dir / "ETHUSDT_candidate_outcomes_2026-05-27.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"event":"open","strategy":"taker_impulse_long","side":"long","accepted":true}',
                '{"event":"close","strategy":"taker_impulse_long","side":"long","accepted":true,"score":0.88,'
                '"outcome_label":"target_first","target_before_stop":{"cost_adjusted":{"gross":true}},'
                '"mfe_mae_horizons":{"60":{"mfe_pct":0.002,"mae_pct":-0.0002}}}',
                '{"event":"close","strategy":"taker_impulse_short","side":"short","accepted":false,"score":0.42,'
                '"outcome_label":"stop_first","target_before_stop":{"cost_adjusted":{"gross":false}},'
                '"mfe_mae_horizons":{"60":{"mfe_pct":0.0004,"mae_pct":-0.001}}}',
            ]
        ),
        encoding="utf-8",
    )

    summary = summarize_candidate_outcomes(Settings(DATA_DIR=str(tmp_path)))

    assert summary.opens == 1
    assert summary.closes == 2
    assert summary.accepted_vs_rejected["accepted"].target_first == 1
    assert summary.accepted_vs_rejected["rejected"].stop_first == 1
    assert summary.target_before_stop == {"cost_adjusted": {"gross": 1}}
    assert summary.score_buckets["score_0.85_plus"].target_first == 1
    assert summary.score_buckets["score_below_0.70"].stop_first == 1
