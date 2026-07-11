import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "notebooks" / "05_exit_laboratory.ipynb"


def markdown(cell_id: str, source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def code(cell_id: str, source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip().splitlines(keepends=True),
    }


cells = [
    markdown(
        "title",
        """
        # Exit laboratory: same entries, different ways out

        Proposal 3 freezes the known side-symmetric squeeze entry rule and asks a narrower question:

        **Can a deterministic exit improve the same trades without borrowing extra opportunities?**

        The reference `100/150/6h` ledger locks non-overlapping entry timestamps first. Every exit
        policy receives that exact cohort. Discovery and calibration choose one policy per family;
        only those locked winners are opened on holdout.
        """,
    ),
    code(
        "setup",
        r"""
        import json
        import os
        from pathlib import Path

        import numpy as np
        import plotly.graph_objects as go
        import plotly.io as pio
        from IPython.display import HTML, display
        from plotly.subplots import make_subplots

        pio.renderers.default = "notebook_connected"
        DATA_ROOT = Path(
            os.environ.get(
                "FUTURES_LAB_RESEARCH_DATA",
                Path.home() / "projects" / "my-trader-research-data" / "20260711",
            )
        )
        REPORT_PATH = DATA_ROOT / "exit-laboratory-local.json"
        study = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        results = study["family_results"]
        result_by_family = {row["family"]: row for row in results}
        candidates = study["selection_candidates"]
        baseline = study["reference_baseline"]
        exposure = study["parameters"]["account_exposure_multiple"]

        GOOD = "#2A9D8F"
        BAD = "#E76F51"
        BLUE = "#457B9D"
        GOLD = "#E9C46A"
        MUTED = "#8D99AE"
        PURPLE = "#7B2CBF"
        TEAL = "#1D7874"
        ORANGE = "#F4A261"
        FAMILY_COLORS = {
            "fixed_barrier": GOOD,
            "conditional_time_decay": ORANGE,
            "fee_breakeven_trail": BLUE,
            "mfe_trailing": TEAL,
            "signal_invalidation": BAD,
            "volatility_scaled": PURPLE,
            "partial_profit": GOLD,
        }
        VERDICT_COLORS = {
            "robust_exit_improvement": GOOD,
            "promising_but_unconfirmed": BLUE,
            "profitable_without_reference_improvement": GOLD,
            "failed_holdout": BAD,
            "selection_failed": MUTED,
        }

        def pretty(value):
            return value.replace("_", " ").title()

        def policy_label(value):
            return value.replace("fixed_", "").replace("mfe_trail_", "MFE ").replace("partial50_", "50% ").replace("_", " ")

        def hours(seconds):
            return seconds / 3600

        print(f"Loaded: {REPORT_PATH}")
        print(f"Coverage: {study['data']['mark_price_start']} to {study['data']['mark_price_end']}")
        print(f"Dense mark-price points: {study['data']['unique_mark_price_points']:,}")
        print(f"Feature snapshots: {study['data']['sampled_feature_rows']:,}")
        print(f"Exit policies: {len(study['policies'])}")
        print(f"Locked holdout opportunities: {study['entry_cohorts']['holdout']['locked_entries']}")
        """,
    ),
    markdown(
        "method-title",
        """
        ## 1. The anti-cheating contract

        A fast exit normally frees the strategy to take more signals. That can be useful in
        production, but it makes exit comparison unfair because each policy sees a different market.
        This laboratory first locks trades with the reference ledger, then applies every exit to the
        same timestamps and directions.
        """,
    ),
    code(
        "method-flow",
        """
        stages = [
            ("Squeeze signals", "Side-symmetric entry rule"),
            ("Reference ledger", "100 target / 150 stop / 6h"),
            ("Locked cohort", "Identical timestamps + directions"),
            ("Exit families", "28 deterministic policies"),
            ("Nested judgment", "Discovery -> calibration -> holdout"),
        ]
        fig = go.Figure()
        for index, (name, note) in enumerate(stages):
            fig.add_trace(
                go.Scatter(
                    x=[index], y=[0], mode="markers+text",
                    marker=dict(size=34, color=[BLUE, GOLD, GOOD, PURPLE, TEAL][index]),
                    text=[f"<b>{name}</b><br>{note}"], textposition="bottom center",
                    hoverinfo="skip", showlegend=False,
                )
            )
            if index < len(stages) - 1:
                fig.add_annotation(x=index + 0.5, y=0, text="->", showarrow=False, font=dict(size=24))
        fig.update_layout(
            title="One entry cohort, then exit competition",
            xaxis=dict(visible=False, range=[-0.45, len(stages) - 0.55]),
            yaxis=dict(visible=False, range=[-0.75, 0.35]),
            height=300,
            margin=dict(l=45, r=45, t=75, b=35),
        )
        fig.show()
        """,
    ),
    markdown(
        "cohort-title",
        """
        ## 2. How many independent opportunities survived?

        Thousands of minute snapshots can describe the same open trade. The reference ledger removes
        overlap and requires a continuous six-hour price path. The resulting cohorts are deliberately
        small but much less misleading.
        """,
    ),
    code(
        "cohort-chart",
        """
        periods = ["discovery", "calibration", "holdout"]
        cohorts = study["entry_cohorts"]
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Signal compression", "Locked direction balance"))
        fig.add_trace(
            go.Bar(
                x=[pretty(period) for period in periods],
                y=[cohorts[period]["signals"] for period in periods],
                name="Raw squeeze signals", marker_color=MUTED,
                text=[f"{cohorts[period]['signals']:,}" for period in periods], textposition="auto",
            ), row=1, col=1,
        )
        fig.add_trace(
            go.Bar(
                x=[pretty(period) for period in periods],
                y=[cohorts[period]["locked_entries"] for period in periods],
                name="Locked entries", marker_color=GOOD,
                text=[cohorts[period]["locked_entries"] for period in periods], textposition="auto",
            ), row=1, col=1,
        )
        fig.add_trace(
            go.Bar(
                x=[pretty(period) for period in periods],
                y=[cohorts[period]["long_entries"] for period in periods],
                name="Long", marker_color=BLUE,
                text=[cohorts[period]["long_entries"] for period in periods], textposition="auto",
            ), row=1, col=2,
        )
        fig.add_trace(
            go.Bar(
                x=[pretty(period) for period in periods],
                y=[cohorts[period]["short_entries"] for period in periods],
                name="Short", marker_color=ORANGE,
                text=[cohorts[period]["short_entries"] for period in periods], textposition="auto",
            ), row=1, col=2,
        )
        fig.update_layout(
            title="Reference-ledger cohorts",
            barmode="group", height=470,
            margin=dict(l=70, r=40, t=90, b=70),
            legend=dict(orientation="h", y=-0.20),
        )
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Locked trades", row=1, col=2)
        fig.show()
        """,
    ),
    markdown(
        "selection-title",
        """
        ## 3. Selection happens before holdout

        Each point is one deterministic exit policy. Policies in the upper-right quadrant made money
        in both discovery and calibration. The labeled point in each family is the one selected before
        holdout. A negative period cannot be repaired by a lucky later result.
        """,
    ),
    code(
        "selection-scatter",
        """
        selected_names = {row["policy"] for row in results}
        selected_text_positions = {
            "fixed_barrier": "top right",
            "conditional_time_decay": "bottom left",
            "fee_breakeven_trail": "top center",
            "mfe_trailing": "bottom right",
            "signal_invalidation": "top left",
            "volatility_scaled": "bottom right",
            "partial_profit": "top left",
        }
        fig = go.Figure()
        for family in sorted({row["family"] for row in candidates}):
            rows = [row for row in candidates if row["family"] == family]
            fig.add_trace(
                go.Scatter(
                    x=[row["discovery"]["average_net_bps_per_trade"] for row in rows],
                    y=[row["calibration"]["average_net_bps_per_trade"] for row in rows],
                    mode="markers+text",
                    name=pretty(family),
                    marker=dict(
                        color=FAMILY_COLORS[family],
                        size=[14 if row["policy"] in selected_names else 8 for row in rows],
                        symbol=["diamond" if row["policy"] in selected_names else "circle" for row in rows],
                        opacity=0.88,
                    ),
                    text=[pretty(family) if row["policy"] in selected_names else "" for row in rows],
                    textposition=[
                        selected_text_positions[family] if row["policy"] in selected_names else "top center"
                        for row in rows
                    ],
                    customdata=[[row["policy"], row["qualifies"]] for row in rows],
                    hovertemplate=(
                        "%{customdata[0]}<br>Discovery: %{x:+.2f} bps"
                        "<br>Calibration: %{y:+.2f} bps<br>Qualified: %{customdata[1]}<extra></extra>"
                    ),
                )
            )
        fig.add_hline(y=0, line_color=BAD, line_width=1)
        fig.add_vline(x=0, line_color=BAD, line_width=1)
        all_x = [row["discovery"]["average_net_bps_per_trade"] for row in candidates]
        all_y = [row["calibration"]["average_net_bps_per_trade"] for row in candidates]
        fig.update_layout(
            title="All 28 policies before holdout",
            xaxis_title="Discovery average net bps/trade",
            yaxis_title="Calibration average net bps/trade",
            height=650,
            margin=dict(l=80, r=45, t=85, b=135),
            legend=dict(orientation="h", y=-0.22),
        )
        fig.update_xaxes(range=[min(all_x) - 4, max(all_x) + 5])
        fig.update_yaxes(range=[min(all_y) - 5, max(all_y) + 6])
        fig.show()
        """,
    ),
    markdown(
        "period-title",
        """
        ## 4. What survived the chronological handoff?

        The fixed reference remained the strongest pre-holdout choice. MFE trailing and partial
        profit-taking remained profitable in every period, but they did not improve expectancy on
        the final locked cohort.
        """,
    ),
    code(
        "period-bars",
        """
        ordered = sorted(results, key=lambda row: row["robust_floor_bps"], reverse=True)
        labels = [pretty(row["family"]) for row in ordered]
        fig = go.Figure()
        for period, color in (("discovery", BLUE), ("calibration", GOLD), ("holdout", GOOD)):
            fig.add_trace(
                go.Bar(
                    x=labels,
                    y=[row[period]["average_net_bps_per_trade"] for row in ordered],
                    name=pretty(period), marker_color=color,
                    text=[f"{row[period]['average_net_bps_per_trade']:+.1f}" for row in ordered],
                    textposition="auto",
                    customdata=[[row["policy"]] for row in ordered],
                    hovertemplate="%{x}<br>%{customdata[0]}<br>%{y:+.2f} bps/trade<extra></extra>",
                )
            )
        fig.add_hline(y=0, line_color=BAD, line_width=1)
        fig.update_layout(
            title="Selected policy per family through time",
            yaxis_title="Average net bps per locked trade",
            barmode="group", height=560,
            margin=dict(l=80, r=40, t=80, b=150),
            legend=dict(orientation="h", y=-0.28),
        )
        fig.update_xaxes(tickangle=-24)
        fig.show()
        """,
    ),
    markdown(
        "paired-title",
        """
        ## 5. Did any exit beat the reference trade by trade?

        This is the decisive comparison. Zero means identical average expectancy to `100/150/6h`.
        Whiskers are 95% moving-block bootstrap intervals over paired holdout trades. An upgrade needs
        its entire interval to clear zero.
        """,
    ),
    code(
        "paired-forest",
        """
        paired_rows = sorted(results, key=lambda row: row["paired_vs_reference"]["mean_delta_bps"], reverse=True)
        values = [row["paired_vs_reference"]["mean_delta_bps"] for row in paired_rows]
        lows = [row["paired_vs_reference"]["bootstrap_p025_bps"] for row in paired_rows]
        highs = [row["paired_vs_reference"]["bootstrap_p975_bps"] for row in paired_rows]
        fig = go.Figure(
            go.Scatter(
                x=values,
                y=[pretty(row["family"]) for row in paired_rows],
                mode="markers+text",
                text=[f"{value:+.1f}" for value in values],
                textposition="middle right",
                marker=dict(
                    size=13,
                    color=[VERDICT_COLORS[row["verdict"]] for row in paired_rows],
                    symbol="diamond",
                ),
                error_x=dict(
                    type="data", symmetric=False,
                    array=[high - value for high, value in zip(highs, values)],
                    arrayminus=[value - low for value, low in zip(values, lows)],
                    thickness=2, width=6, color=MUTED,
                ),
                customdata=[[row["policy"], low, high] for row, low, high in zip(paired_rows, lows, highs)],
                hovertemplate=(
                    "%{y}<br>%{customdata[0]}<br>Mean delta: %{x:+.2f} bps"
                    "<br>95% interval: %{customdata[1]:+.2f} to %{customdata[2]:+.2f}<extra></extra>"
                ),
                showlegend=False,
            )
        )
        fig.add_vline(x=0, line_color=BAD, line_width=2)
        fig.update_layout(
            title="Paired holdout delta versus the fixed reference",
            xaxis_title="Average net bps improvement per identical trade",
            height=530,
            margin=dict(l=190, r=55, t=80, b=70),
        )
        fig.show()
        """,
    ),
    markdown(
        "risk-title",
        """
        ## 6. The trade-off: smoother is not automatically richer

        Some policies reduce drawdown and holding time. That is useful engineering information, but
        a lower-risk policy is not an expectancy upgrade when it gives away too much of the winning
        tail. Bubble size is average holding time.
        """,
    ),
    code(
        "risk-return",
        """
        fig = go.Figure(
            go.Scatter(
                x=[row["holdout"]["max_additive_account_drawdown_pct"] for row in results],
                y=[row["holdout"]["average_net_bps_per_trade"] for row in results],
                mode="markers+text",
                text=[pretty(row["family"]) for row in results],
                textposition=[
                    {
                        "fixed_barrier": "top center",
                        "conditional_time_decay": "middle left",
                        "fee_breakeven_trail": "bottom left",
                        "mfe_trailing": "top right",
                        "signal_invalidation": "bottom left",
                        "volatility_scaled": "middle right",
                        "partial_profit": "top right",
                    }[row["family"]]
                    for row in results
                ],
                marker=dict(
                    size=[10 + min(20, hours(row["holdout"]["average_duration_seconds"]) * 3) for row in results],
                    color=[FAMILY_COLORS[row["family"]] for row in results],
                    line=dict(width=1),
                ),
                customdata=[
                    [row["policy"], hours(row["holdout"]["average_duration_seconds"]), row["holdout"]["positive_net_trade_rate"]]
                    for row in results
                ],
                hovertemplate=(
                    "%{text}<br>%{customdata[0]}<br>Expectancy: %{y:+.2f} bps"
                    "<br>Max additive DD: %{x:.1f}%<br>Avg hold: %{customdata[1]:.2f}h"
                    "<br>Positive trades: %{customdata[2]:.1%}<extra></extra>"
                ),
                showlegend=False,
            )
        )
        fig.add_hline(y=0, line_color=BAD, line_width=1)
        fig.update_layout(
            title="Holdout expectancy versus modeled drawdown",
            xaxis_title="Maximum additive account drawdown at 4.95x exposure (%)",
            yaxis_title="Average net bps per locked trade",
            height=600,
            margin=dict(l=85, r=45, t=80, b=75),
        )
        fig.update_xaxes(range=[4, 20])
        fig.update_yaxes(range=[-10, 23])
        fig.show()
        """,
    ),
    markdown(
        "curve-title",
        """
        ## 7. Same trades, different equity paths

        The reference preserves the largest winners. MFE trailing and partial profit-taking smooth
        portions of the path, but their cumulative lines finish below it. These are modeled additive
        returns at 4.95x exposure, not a promise of compound live returns.
        """,
    ),
    code(
        "equity-curves",
        """
        comparison_families = ["fixed_barrier", "mfe_trailing", "partial_profit", "fee_breakeven_trail"]
        fig = go.Figure()
        for family in comparison_families:
            row = result_by_family[family]
            trades = row["holdout_trades"]
            cumulative = np.cumsum([trade["net_bps"] / 100 * exposure for trade in trades])
            fig.add_trace(
                go.Scatter(
                    x=[trade["entry"] for trade in trades], y=cumulative,
                    mode="lines+markers", name=pretty(family),
                    line=dict(color=FAMILY_COLORS[family], width=3 if family == "fixed_barrier" else 2),
                    marker=dict(size=5),
                    customdata=[[trade["exit_reason"], trade["net_bps"]] for trade in trades],
                    hovertemplate=(
                        "%{x}<br>Cumulative: %{y:+.2f}%<br>Trade: %{customdata[1]:+.1f} bps"
                        "<br>%{customdata[0]}<extra></extra>"
                    ),
                )
            )
        fig.add_hline(y=0, line_color=BAD, line_width=1)
        fig.update_layout(
            title="Locked holdout cohort: additive account path",
            xaxis_title="Entry time (UTC)", yaxis_title="Cumulative modeled account return (%)",
            height=570,
            margin=dict(l=85, r=40, t=80, b=80),
            legend=dict(orientation="h", y=-0.20),
        )
        fig.show()
        """,
    ),
    markdown(
        "reasons-title",
        """
        ## 8. What each policy actually did

        MFE trailing exited 29 of 36 trades through its trail. Partial profit touched its first scale
        on most trades. Signal invalidation closed every trade at the next qualifying minute snapshot,
        which mostly converted movement noise plus fees into losses.
        """,
    ),
    code(
        "reason-stack",
        """
        reason_groups = [
            ("Target", lambda reason: reason in {"take_profit", "partial_then_target"}),
            ("Stop", lambda reason: reason in {"price_stop", "partial_then_stop", "partial_then_breakeven"}),
            ("Trail", lambda reason: "trail" in reason),
            ("Time decay", lambda reason: reason == "time_decay"),
            ("Invalidation", lambda reason: reason == "signal_invalidation"),
            ("Timeout", lambda reason: "timeout" in reason),
        ]
        labels = [pretty(row["family"]) for row in results]
        fig = go.Figure()
        group_colors = [GOOD, BAD, BLUE, ORANGE, PURPLE, MUTED]
        for (label, matcher), color in zip(reason_groups, group_colors):
            values = []
            for row in results:
                values.append(sum(count for reason, count in row["holdout"]["exit_reasons"].items() if matcher(reason)))
            fig.add_trace(go.Bar(x=labels, y=values, name=label, marker_color=color, text=values, textposition="auto"))
        fig.update_layout(
            title="Terminal exit reasons on the same 36 holdout trades",
            yaxis_title="Trades", barmode="stack", height=560,
            margin=dict(l=70, r=40, t=80, b=145),
            legend=dict(orientation="h", y=-0.27),
        )
        fig.update_xaxes(tickangle=-24)
        fig.show()
        """,
    ),
    markdown(
        "coin-title",
        """
        ## 9. Did direction still matter after changing exits?

        Diamonds are the squeeze strategy. Whiskers show the 5th-95th percentile from 32 random
        long/short assignments at the same timestamps using the same exit. Profitable exit families
        still rely on the squeeze direction; the exit alone does not manufacture the edge.
        """,
    ),
    code(
        "coin-controls",
        """
        coin_rows = sorted(results, key=lambda row: row["holdout"]["average_net_bps_per_trade"], reverse=True)
        centers = [row["matched_coin_control"]["median_average_net_bps"] for row in coin_rows]
        lows = [row["matched_coin_control"]["p05_average_net_bps"] for row in coin_rows]
        highs = [row["matched_coin_control"]["p95_average_net_bps"] for row in coin_rows]
        strategy_values = [row["holdout"]["average_net_bps_per_trade"] for row in coin_rows]
        labels = [pretty(row["family"]) for row in coin_rows]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=centers, y=labels, mode="markers", name="Matched coin median and 5-95%",
                marker=dict(color=MUTED, size=9),
                error_x=dict(
                    type="data", symmetric=False,
                    array=[high - center for high, center in zip(highs, centers)],
                    arrayminus=[center - low for center, low in zip(centers, lows)],
                    thickness=2, width=6, color=MUTED,
                ),
                hovertemplate="%{y}<br>Coin median: %{x:+.2f} bps<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=strategy_values, y=labels, mode="markers+text", name="Squeeze direction",
                text=[f"{value:+.1f}" for value in strategy_values], textposition="middle right",
                marker=dict(color=[FAMILY_COLORS[row["family"]] for row in coin_rows], size=14, symbol="diamond"),
                hovertemplate="%{y}<br>Strategy: %{x:+.2f} bps/trade<extra></extra>",
            )
        )
        fig.add_vline(x=0, line_color=BAD, line_width=1)
        fig.update_layout(
            title="Holdout strategy versus matched direction luck",
            xaxis_title="Average net bps per locked trade",
            height=540,
            margin=dict(l=190, r=55, t=80, b=75),
            legend=dict(orientation="h", y=-0.16),
        )
        fig.update_xaxes(range=[min(lows) - 5, max(strategy_values) + 10])
        fig.show()
        """,
    ),
    markdown(
        "verdict-title",
        """
        ## 10. Exit-laboratory verdict
        """,
    ),
    code(
        "verdict",
        """
        fixed = result_by_family["fixed_barrier"]
        trail = result_by_family["mfe_trailing"]
        partial = result_by_family["partial_profit"]
        invalidation = result_by_family["signal_invalidation"]
        display(
            HTML(
                f'''
                <div style="border-left: 5px solid {GOOD}; padding: 8px 18px; max-width: 980px;">
                  <h3 style="margin-top: 0;">Keep the fixed exit as control; shadow two alternatives</h3>
                  <ul>
                    <li><strong>Fixed 100/150/6h remains the winner.</strong> It made
                        <strong>{fixed['holdout']['average_net_bps_per_trade']:+.2f} net bps/trade</strong>
                        on 36 locked holdout opportunities, with longs at
                        {fixed['holdout']['long']['average_net_bps_per_trade']:+.2f} and shorts at
                        {fixed['holdout']['short']['average_net_bps_per_trade']:+.2f}.</li>
                    <li><strong>MFE 40/20 trailing</strong> remained profitable at
                        {trail['holdout']['average_net_bps_per_trade']:+.2f} bps/trade, but lost
                        {abs(trail['paired_vs_reference']['mean_delta_bps']):.2f} bps versus the same
                        reference trades. Its paired interval crosses zero.</li>
                    <li><strong>50% partial at 40 bps</strong> made
                        {partial['holdout']['average_net_bps_per_trade']:+.2f} bps/trade, but also
                        finished below reference by {abs(partial['paired_vs_reference']['mean_delta_bps']):.2f} bps/trade.</li>
                    <li><strong>Minute-sampled signal invalidation is rejected.</strong> It scored
                        {invalidation['discovery']['average_net_bps_per_trade']:+.2f},
                        {invalidation['calibration']['average_net_bps_per_trade']:+.2f}, and
                        {invalidation['holdout']['average_net_bps_per_trade']:+.2f} bps/trade.</li>
                    <li>No exit family produced a statistically clear paired improvement over the fixed
                        reference. The honest strategy change today is therefore <strong>none</strong>.</li>
                  </ul>
                </div>
                '''
            )
        )
        """,
    ),
    markdown(
        "next-title",
        """
        ## 11. What proposal 3 hands forward

        The forward candidate is now a three-ledger shadow comparison on new data:

        1. `fixed_100_150_6h` as control,
        2. `mfe_trail_activate40_retrace20`,
        3. `partial50_at40_original_stop`.

        They should share each squeeze timestamp exactly, just as this lab did. Recycled entry
        throughput can be measured separately after the paired result is known. The current AWS
        `60/60` strategy arena remains untouched by this retrospective analysis.
        """,
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Futures Lab Research",
            "language": "python",
            "name": "futures-lab-research",
        },
        "language_info": {"name": "python", "version": "3.13"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(OUTPUT)
