import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "notebooks" / "04_strategy_family_tournament.ipynb"


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
        # ETH strategy-family tournament

        Nine deterministic strategy families enter. Each is allowed to generate variants and choose
        one exit profile using only **discovery plus calibration**. The chosen configuration is then
        locked and judged once on the newest **holdout**.

        For every finalist, 32 matched fair coins trade at the exact same signal timestamps but pick
        direction randomly. This asks whether the strategy found direction, rather than merely finding
        moments when ETH was likely to move.
        """,
    ),
    code(
        "setup",
        r"""
        import json
        import os
        from datetime import datetime
        from pathlib import Path

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
        REPORT_PATH = DATA_ROOT / "strategy-tournament-local.json"
        PREAUDIT_PATH = DATA_ROOT / "strategy-tournament-prebaseline-audit.json"
        study = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        preaudit = json.loads(PREAUDIT_PATH.read_text(encoding="utf-8")) if PREAUDIT_PATH.exists() else None

        GOOD = "#2A9D8F"
        BAD = "#E76F51"
        NEUTRAL = "#457B9D"
        GOLD = "#E9C46A"
        MUTED = "#8D99AE"
        PURPLE = "#7B2CBF"
        FAMILY_COLORS = {
            "volatility_squeeze_breakout": GOOD,
            "btc_flow_lead": PURPLE,
            "oi_price_expansion": GOLD,
            "range_edge_reversion": "#F4A261",
            "flow_exhaustion_reversal": BAD,
            "book_pressure_continuation": "#A44A3F",
            "micro_impulse_continuation": NEUTRAL,
            "htf_momentum": "#264653",
            "trend_pullback_continuation": MUTED,
        }
        VERDICT_COLORS = {
            "generalized_candidate": GOOD,
            "promising_but_unconfirmed": GOLD,
            "indistinguishable_from_direction_luck": PURPLE,
            "failed_holdout": BAD,
            "selection_failed": MUTED,
        }
        SHORT_NAMES = {
            "volatility_squeeze_breakout": "squeeze breakout",
            "btc_flow_lead": "BTC flow lead",
            "oi_price_expansion": "OI expansion",
            "range_edge_reversion": "range reversion",
            "flow_exhaustion_reversal": "flow exhaustion",
            "book_pressure_continuation": "book pressure",
            "micro_impulse_continuation": "micro impulse",
            "htf_momentum": "HTF momentum",
            "trend_pullback_continuation": "trend pullback",
        }
        results = study["family_results"]
        result_by_family = {row["family"]: row for row in results}
        family_meta = {row["family"]: row for row in study["families"]}

        def pretty(value):
            return value.replace("_", " ")

        def short_name(value):
            return SHORT_NAMES.get(value, pretty(value))

        def horizon_label(seconds):
            return f"{seconds // 3600}h" if seconds >= 3600 else f"{seconds // 60}m"

        print(f"Loaded: {REPORT_PATH}")
        print(f"Coverage: {study['data']['mark_price_start']} to {study['data']['mark_price_end']}")
        print(f"Signal variants: {study['data']['signal_variants']}")
        print(f"Variant/profile candidates: {study['data']['candidate_configurations']}")
        print(f"Holdout begins: {study['data']['holdout_start']}")
        """,
    ),
    markdown(
        "method-title",
        """
        ## 1. Three locked rooms

        Candidate definitions are fixed before the holdout door opens. A full six-hour purge is
        applied at each boundary so a trade label cannot reach across periods.
        """,
    ),
    code(
        "period-timeline",
        """
        start = datetime.fromisoformat(study["data"]["mark_price_start"])
        calibration = datetime.fromisoformat(study["data"]["calibration_start"])
        holdout = datetime.fromisoformat(study["data"]["holdout_start"])
        end = datetime.fromisoformat(study["data"]["mark_price_end"])
        periods = [
            ("Discovery", start, calibration, NEUTRAL),
            ("Calibration", calibration, holdout, GOLD),
            ("Holdout", holdout, end, GOOD),
        ]
        fig = go.Figure()
        for name, left, right, color in periods:
            fig.add_trace(
                go.Bar(
                    x=[(right - left).total_seconds() / 86400],
                    y=["Archive"],
                    base=[(left - start).total_seconds() / 86400],
                    orientation="h",
                    name=name,
                    marker_color=color,
                    text=f"{name}<br>{left:%d %b} to {right:%d %b}",
                    textposition="inside",
                    hovertemplate=f"{name}: {left:%Y-%m-%d} to {right:%Y-%m-%d}<extra></extra>",
                )
            )
        fig.update_layout(
            title="Chronological tournament split",
            xaxis_title="Days from archive start",
            barmode="overlay",
            height=300,
            margin=dict(l=70, r=30, t=70, b=60),
            legend=dict(orientation="h", y=-0.35),
        )
        fig.show()
        """,
    ),
    markdown(
        "families-title",
        """
        ## 2. The contestants and the selection funnel

        Every variant is crossed with 11 predeclared exit profiles. A candidate qualifies only when
        discovery and calibration are both profitable, both have enough sequential trades, both are
        positive across at least half their chronological blocks, and neither side is nearly absent.
        """,
    ),
    code(
        "family-funnel",
        """
        family_order = [row["family"] for row in results]
        variants = [family_meta[name]["variants"] for name in family_order]
        total_candidates = [result_by_family[name]["family_candidate_count"] for name in family_order]
        qualified = [result_by_family[name]["qualified_candidate_count"] for name in family_order]
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Signal variants", "Qualified variant/profile cells"))
        fig.add_trace(
            go.Bar(
                x=variants,
                y=[pretty(name) for name in family_order],
                orientation="h",
                marker_color=[FAMILY_COLORS[name] for name in family_order],
                text=variants,
                textposition="auto",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=qualified,
                y=[pretty(name) for name in family_order],
                orientation="h",
                marker_color=[GOOD if value else MUTED for value in qualified],
                text=[f"{value} / {total}" for value, total in zip(qualified, total_candidates)],
                textposition="auto",
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig.update_xaxes(title_text="Count")
        fig.update_layout(height=600, margin=dict(l=220, r=40, t=80, b=60))
        fig.show()
        """,
    ),
    markdown(
        "selection-title",
        """
        ## 3. What selection saw before holdout

        Each dot is a variant/profile candidate. The selected finalist for each family is outlined.
        The desirable region is upper-right; a candidate green in only one period is unstable.
        """,
    ),
    code(
        "selection-scatter",
        """
        selected_keys = {(row["variant"], row["profile"]) for row in results}
        fig = go.Figure()
        for family in family_order:
            rows = [row for row in study["selection_candidates"] if row["family"] == family]
            fig.add_trace(
                go.Scatter(
                    x=[row["discovery"]["average_net_bps_per_trade"] for row in rows],
                    y=[row["calibration"]["average_net_bps_per_trade"] for row in rows],
                    mode="markers",
                    name=pretty(family),
                    marker=dict(
                        color=FAMILY_COLORS[family],
                        opacity=0.42,
                        size=[13 if (row["variant"], row["profile"]) in selected_keys else 7 for row in rows],
                        line=dict(
                            width=1.8,
                            color=["#111111" if (row["variant"], row["profile"]) in selected_keys else FAMILY_COLORS[family] for row in rows],
                        ),
                    ),
                    customdata=[
                        [row["profile"], row["discovery"]["sequential_trades"], row["calibration"]["sequential_trades"], row["qualifies"]]
                        for row in rows
                    ],
                    hovertemplate=(
                        "%{fullData.name}<br>Profile: %{customdata[0]}<br>Discovery: %{x:+.2f} bps"
                        "<br>Calibration: %{y:+.2f} bps<br>Trades: %{customdata[1]}/%{customdata[2]}"
                        "<br>Qualified: %{customdata[3]}<extra></extra>"
                    ),
                )
            )
        fig.add_hline(y=0, line_color=BAD, line_width=1)
        fig.add_vline(x=0, line_color=BAD, line_width=1)
        fig.update_layout(
            title="Discovery versus calibration expectancy",
            xaxis_title="Discovery average net bps/trade",
            yaxis_title="Calibration average net bps/trade",
            height=720,
            margin=dict(l=80, r=30, t=70, b=70),
            legend=dict(orientation="h", y=-0.22),
        )
        fig.show()
        """,
    ),
    markdown(
        "profiles-title",
        """
        ## 4. The locked finalists

        The profile below was selected without reading holdout. Families marked `selection failed`
        still receive a diagnostic holdout score, but they are not promoted merely because that later
        period happens to be green.
        """,
    ),
    code(
        "profile-map",
        """
        profile_labels = [short_name(row["family"]) for row in results]
        fig = go.Figure()
        for index, row in enumerate(results):
            fig.add_shape(
                type="line",
                x0=row["target_bps"],
                x1=row["stop_bps"],
                y0=index,
                y1=index,
                line=dict(color=MUTED, width=2),
            )
        fig.add_trace(
            go.Scatter(
                x=[row["target_bps"] for row in results],
                y=list(range(len(results))),
                mode="markers+text",
                name="Target",
                text=[f"{row['target_bps']:.0f}" for row in results],
                textposition="middle left",
                marker=dict(color=GOOD, size=13, symbol="circle"),
                customdata=[[horizon_label(row["horizon_seconds"]), row["verdict"]] for row in results],
                hovertemplate=(
                    "%{y}<br>Target: %{x:.0f} bps<br>Horizon: %{customdata[0]}"
                    "<br>Verdict: %{customdata[1]}<extra></extra>"
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[row["stop_bps"] for row in results],
                y=list(range(len(results))),
                mode="markers+text",
                name="Stop",
                text=[f"{row['stop_bps']:.0f}" for row in results],
                textposition="middle right",
                marker=dict(color=BAD, size=13, symbol="x"),
                customdata=[[horizon_label(row["horizon_seconds"]), row["verdict"]] for row in results],
                hovertemplate=(
                    "%{y}<br>Stop: %{x:.0f} bps<br>Horizon: %{customdata[0]}"
                    "<br>Verdict: %{customdata[1]}<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            title="Target-to-stop span chosen before holdout",
            xaxis_title="Barrier distance (bps)",
            yaxis=dict(tickmode="array", tickvals=list(range(len(results))), ticktext=profile_labels),
            xaxis_range=[0, 170],
            height=620,
            margin=dict(l=220, r=60, t=80, b=70),
            legend=dict(orientation="h", y=-0.14),
        )
        fig.show()
        """,
    ),
    markdown(
        "holdout-title",
        """
        ## 5. Holdout against matched direction luck

        Diamonds are strategy results. Circles and whiskers show the median and 5th-95th percentile
        range of 32 fair coins entering at that strategy's exact signal timestamps. Clearing the
        whisker is stronger evidence than merely making money.
        """,
    ),
    code(
        "holdout-controls",
        """
        labels = [pretty(row["family"]) for row in results]
        medians = [row["matched_coin_control"]["median_average_net_bps"] for row in results]
        p05 = [row["matched_coin_control"]["p05_average_net_bps"] for row in results]
        p95 = [row["matched_coin_control"]["p95_average_net_bps"] for row in results]
        strategy_values = [row["holdout"]["average_net_bps_per_trade"] for row in results]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=medians,
                y=labels,
                mode="markers",
                name="Matched coin median and 5-95%",
                marker=dict(color=MUTED, size=9),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=[high - center for high, center in zip(p95, medians)],
                    arrayminus=[center - low for center, low in zip(medians, p05)],
                    color=MUTED,
                    thickness=2,
                    width=6,
                ),
                hovertemplate="%{y}<br>Coin median: %{x:+.2f} bps<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=strategy_values,
                y=labels,
                mode="markers+text",
                name="Selected strategy",
                text=[f"{value:+.1f}" for value in strategy_values],
                textposition=["middle left" if value > 20 else "middle right" for value in strategy_values],
                marker=dict(
                    color=[VERDICT_COLORS[row["verdict"]] for row in results],
                    size=14,
                    symbol="diamond",
                    line=dict(width=1),
                ),
                hovertemplate="%{y}<br>Strategy: %{x:+.2f} bps/trade<extra></extra>",
            )
        )
        fig.add_vline(x=0, line_color=BAD, line_width=1)
        chart_min = min([*p05, *strategy_values]) - 6
        chart_max = max([*p95, *strategy_values]) + 8
        fig.update_layout(
            title="Holdout average net bps versus matched fair coins",
            xaxis_title="Average net bps per sequential trade",
            height=620,
            margin=dict(l=220, r=60, t=80, b=70),
            legend=dict(orientation="h", y=-0.14),
        )
        fig.update_xaxes(range=[chart_min, chart_max])
        fig.show()
        """,
    ),
    markdown(
        "symmetry-title",
        """
        ## 6. Did both directions work?

        Points in the upper-right quadrant made money on both longs and shorts. This prevents a
        bearish or bullish month from disguising a one-sided rule as a general strategy.
        """,
    ),
    code(
        "side-scatter",
        """
        fig = go.Figure(
            go.Scatter(
                x=[row["holdout"]["long"]["average_net_bps_per_trade"] for row in results],
                y=[row["holdout"]["short"]["average_net_bps_per_trade"] for row in results],
                mode="markers+text",
                text=[short_name(row["family"]) for row in results],
                textposition=[
                    "top left"
                    if row["holdout"]["long"]["average_net_bps_per_trade"] > 35
                    else "top right"
                    if row["holdout"]["long"]["average_net_bps_per_trade"] < -5
                    else "top center"
                    for row in results
                ],
                marker=dict(
                    size=[10 + min(12, row["holdout"]["sequential_trades"] / 8) for row in results],
                    color=[VERDICT_COLORS[row["verdict"]] for row in results],
                    line=dict(width=1),
                ),
                customdata=[
                    [row["holdout"]["long"]["trades"], row["holdout"]["short"]["trades"], row["verdict"]]
                    for row in results
                ],
                hovertemplate=(
                    "%{text}<br>Long: %{x:+.2f} bps (%{customdata[0]} trades)"
                    "<br>Short: %{y:+.2f} bps (%{customdata[1]} trades)"
                    "<br>%{customdata[2]}<extra></extra>"
                ),
                showlegend=False,
            )
        )
        fig.add_hline(y=0, line_color=BAD, line_width=1)
        fig.add_vline(x=0, line_color=BAD, line_width=1)
        fig.update_layout(
            title="Holdout expectancy by selected trade direction",
            xaxis_title="Long average net bps/trade",
            yaxis_title="Short average net bps/trade",
            xaxis_range=[-16, 50],
            yaxis_range=[-30, 25],
            height=640,
            margin=dict(l=80, r=40, t=80, b=70),
        )
        fig.show()
        """,
    ),
    markdown(
        "winner-title",
        """
        ## 7. The one family that cleared every gate
        """,
    ),
    code(
        "winner-detail",
        """
        winner = result_by_family["volatility_squeeze_breakout"]
        period_values = [
            winner["discovery"]["average_net_bps_per_trade"],
            winner["calibration"]["average_net_bps_per_trade"],
            winner["holdout"]["average_net_bps_per_trade"],
        ]
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Expectancy through time", "Holdout outcomes"))
        fig.add_trace(
            go.Bar(
                x=["Discovery", "Calibration", "Holdout"],
                y=period_values,
                marker_color=[NEUTRAL, GOLD, GOOD],
                text=[f"{value:+.2f}" for value in period_values],
                textposition="auto",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=["Target", "Stop", "Timeout"],
                y=[winner["holdout"]["targets"], winner["holdout"]["stops"], winner["holdout"]["timeouts"]],
                marker_color=[GOOD, BAD, MUTED],
                text=[winner["holdout"]["targets"], winner["holdout"]["stops"], winner["holdout"]["timeouts"]],
                textposition="auto",
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig.add_hline(y=0, line_color=BAD, line_width=1, row=1, col=1)
        fig.update_yaxes(title_text="Average net bps/trade", row=1, col=1)
        fig.update_yaxes(title_text="Sequential trades", row=1, col=2)
        fig.update_layout(
            title=f"Existing squeeze rule | {winner['target_bps']:.0f}/{winner['stop_bps']:.0f} bps | {horizon_label(winner['horizon_seconds'])}",
            height=480,
            margin=dict(l=80, r=40, t=90, b=70),
        )
        fig.show()
        """,
    ),
    markdown(
        "audit-title",
        """
        ## 8. Why preserving the implementation audit matters

        The first engineering pass omitted the exact pre-existing baseline rule and selected a
        stricter squeeze variant. After the known baseline was restored, it won selection. Because
        holdout had already been inspected during that audit, this entire tournament remains
        **retrospective evidence**, not a fresh forward result.
        """,
    ),
    code(
        "audit-chart",
        """
        if preaudit is not None:
            before = next(row for row in preaudit["family_results"] if row["family"] == "volatility_squeeze_breakout")
            after = winner
            labels = ["Initial implementation audit", "Baseline-complete tournament"]
            values = [before["holdout"]["average_net_bps_per_trade"], after["holdout"]["average_net_bps_per_trade"]]
            fig = go.Figure(
                go.Bar(
                    x=labels,
                    y=values,
                    marker_color=[BAD, GOOD],
                    text=[f"{value:+.2f} bps" for value in values],
                    textposition="auto",
                    customdata=[
                        [before["variant"], before["profile"]],
                        [after["variant"], after["profile"]],
                    ],
                    hovertemplate="%{x}<br>%{customdata[0]}<br>%{customdata[1]}<br>%{y:+.2f} bps/trade<extra></extra>",
                )
            )
            fig.add_hline(y=0, line_color=MUTED, line_width=1)
            fig.update_layout(
                title="Same family, different pre-holdout candidate set",
                yaxis_title="Holdout average net bps/trade",
                height=430,
                margin=dict(l=80, r=40, t=80, b=80),
            )
            fig.show()
        """,
    ),
    markdown(
        "failed-title",
        """
        ## 9. Green holdout does not rescue failed selection

        Micro impulse and HTF momentum happened to turn positive later, but their discovery and/or
        calibration evidence was negative. Promoting them now would be textbook hindsight fitting.
        """,
    ),
    code(
        "failed-periods",
        """
        examples = ["micro_impulse_continuation", "htf_momentum", "trend_pullback_continuation"]
        fig = go.Figure()
        for period, color in (("discovery", NEUTRAL), ("calibration", GOLD), ("holdout", GOOD)):
            fig.add_trace(
                go.Bar(
                    x=[pretty(name) for name in examples],
                    y=[result_by_family[name][period]["average_net_bps_per_trade"] for name in examples],
                    name=period.title(),
                    marker_color=color,
                    text=[f"{result_by_family[name][period]['average_net_bps_per_trade']:+.1f}" for name in examples],
                    textposition="auto",
                )
            )
        fig.add_hline(y=0, line_color=BAD, line_width=1)
        fig.update_layout(
            title="Selected diagnostics for families that failed pre-holdout selection",
            yaxis_title="Average net bps/trade",
            barmode="group",
            height=500,
            margin=dict(l=80, r=40, t=80, b=110),
            legend=dict(orientation="h", y=-0.24),
        )
        fig.show()
        """,
    ),
    markdown(
        "verdict-title",
        """
        ## 10. Tournament verdict
        """,
    ),
    code(
        "verdict",
        """
        btc = result_by_family["btc_flow_lead"]
        oi = result_by_family["oi_price_expansion"]
        coin_gap = winner["holdout"]["average_net_bps_per_trade"] - winner["matched_coin_control"]["p95_average_net_bps"]
        coin_beats = int(round(winner["matched_coin_control"]["seed_fraction_at_or_above_strategy"] * winner["matched_coin_control"]["seeds"]))
        display(
            HTML(
                f'''
                <div style="border-left: 5px solid {GOOD}; padding: 8px 18px; max-width: 980px;">
                  <h3 style="margin-top: 0;">One candidate advances, but it is not yet a live-trading verdict</h3>
                  <ul>
                    <li><strong>Volatility squeeze breakout</strong> selected the pre-existing rule with a
                        <strong>{winner['target_bps']:.0f}/{winner['stop_bps']:.0f} bps, {horizon_label(winner['horizon_seconds'])}</strong>
                        profile. It scored <strong>{winner['discovery']['average_net_bps_per_trade']:+.2f}</strong>,
                        <strong>{winner['calibration']['average_net_bps_per_trade']:+.2f}</strong>, and
                        <strong>{winner['holdout']['average_net_bps_per_trade']:+.2f} bps/trade</strong>.</li>
                    <li>Holdout contained <strong>{winner['holdout']['sequential_trades']} trades</strong>:
                        {winner['holdout']['targets']} targets, {winner['holdout']['stops']} stops, and
                        {winner['holdout']['timeouts']} timeouts. Long and short results were both positive.</li>
                    <li>It exceeded the matched-coin 95th percentile by only <strong>{coin_gap:+.2f} bps</strong>;
                        {coin_beats} of {winner['matched_coin_control']['seeds']} random directions still matched or beat it.
                        That is encouraging, but borderline rather than overwhelming.</li>
                    <li><strong>BTC-flow lead</strong> was positive on both sides at
                        {btc['holdout']['average_net_bps_per_trade']:+.2f} bps/trade, but did not clear its matched-coin
                        range. Its threshold saturated at 1.0, so feed representation also needs investigation.</li>
                    <li><strong>OI expansion</strong> was positive overall, but longs made
                        {oi['holdout']['long']['average_net_bps_per_trade']:+.2f} while shorts made
                        {oi['holdout']['short']['average_net_bps_per_trade']:+.2f} bps/trade. It is not side-general.</li>
                    <li>No genuinely fast profile won. The advancing profile is a six-hour first-touch setup with a
                        150 bps adverse barrier. Position sizing and liquidation economics must be studied before any
                        live interpretation.</li>
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
        ## 11. What proposal 2 hands to proposal 3

        The tournament nominates one entry family and exposes its unresolved problem: exits. Proposal
        **3, the exit laboratory**, should compare fixed barriers, signal invalidation, MFE trailing,
        time decay, volatility scaling, and partial profit-taking using the squeeze entries and matched
        controls.

        The currently deployed AWS arena remains unchanged at symmetric `60/60`. This retrospective
        tournament does not silently alter that clean forward run.
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
