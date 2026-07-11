import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "notebooks" / "03_market_behavior_atlas.ipynb"


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
        # ETH market behavior atlas

        This chapter deliberately forgets the `60/60` thesis for a moment. It asks three broader
        questions:

        1. How far does ETH naturally move, and how long does it take?
        2. How do fees reshape the required hit rate for many target/stop combinations?
        3. Which deterministic signal families survive both an older discovery period and a newer
           validation period?

        Nearby timestamps are not treated as independent trades. Strategy results use sequential,
        non-overlapping positions and a horizon-sized purge between periods.
        """,
    ),
    code(
        "setup",
        r"""
        import json
        import math
        import os
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
        REPORT_PATH = DATA_ROOT / "market-atlas-local.json"
        study = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

        GOOD = "#2A9D8F"
        BAD = "#E76F51"
        NEUTRAL = "#457B9D"
        GOLD = "#E9C46A"
        MUTED = "#8D99AE"
        PURPLE = "#7B2CBF"
        STRATEGY_COLORS = {
            "fair_coin_control": MUTED,
            "range_reversion": GOLD,
            "micro_momentum": NEUTRAL,
            "micro_persistence": "#1D7874",
            "impulse_pullback": "#F4A261",
            "flow_exhaustion_reversal": BAD,
            "volatility_squeeze_breakout": GOOD,
            "strategy_family_router": PURPLE,
            "htf_momentum": "#264653",
            "regime_router": "#A44A3F",
        }

        natural = study["natural_first_passage"]
        strategy = study["strategy_surface"]
        robust = [row for row in study["robust_candidate_shortlist"] if row["qualifies"]]
        print(f"Loaded: {REPORT_PATH}")
        print(f"Coverage: {study['data']['mark_price_start']} to {study['data']['mark_price_end']}")
        print(f"Unique mark-price points: {study['data']['unique_mark_price_points']:,}")
        print(f"Minute feature samples: {study['data']['sampled_feature_rows']:,}")
        print(f"Strategy-period cells tested: {len(strategy):,}")
        print(f"Robust shortlist: {len(robust)} cells")

        def horizon_label(seconds):
            if seconds < 60:
                return f"{seconds}s"
            if seconds < 3600:
                return f"{seconds // 60}m"
            return f"{seconds // 3600}h"

        def matrix(rows, y_values, x_values, value_key, *, y_key, x_key, multiplier=1.0):
            lookup = {(row[y_key], row[x_key]): row.get(value_key) for row in rows}
            return [
                [None if lookup.get((y, x)) is None else lookup[(y, x)] * multiplier for x in x_values]
                for y in y_values
            ]

        def heat_text(values, suffix="", digits=1):
            return [["" if value is None else f"{value:.{digits}f}{suffix}" for value in row] for row in values]
        """,
    ),
    markdown(
        "natural-title",
        """
        ## 1. How much clock does each barrier need?

        Each cell asks whether either the upper or lower barrier was reached before the stated
        timeout. No trading direction is selected here. This is simply ETH's movement capacity.
        """,
    ),
    code(
        "resolution-heatmap",
        """
        barriers = sorted({row["barrier_bps"] for row in natural})
        horizons = sorted({row["horizon_seconds"] for row in natural})
        resolved = matrix(
            natural, barriers, horizons, "resolution_rate",
            y_key="barrier_bps", x_key="horizon_seconds", multiplier=100,
        )
        fig = go.Figure(
            go.Heatmap(
                x=[horizon_label(value) for value in horizons],
                y=[f"{value:.0f}" for value in barriers],
                z=resolved,
                text=heat_text(resolved, "%"),
                texttemplate="%{text}",
                colorscale="Viridis",
                zmin=0,
                zmax=100,
                colorbar_title="Resolved %",
                hovertemplate="Barrier: %{y} bps<br>Horizon: %{x}<br>Resolved: %{z:.1f}%<extra></extra>",
            )
        )
        fig.update_layout(
            title="Probability either side reaches the barrier before timeout",
            xaxis_title="Maximum holding time",
            yaxis_title="Symmetric barrier (bps)",
            height=540,
            margin=dict(l=90, r=40, t=70, b=70),
        )
        fig.show()
        """,
    ),
    markdown(
        "direction-title",
        """
        ## 2. Which side arrived first?

        Zero means the two directions split evenly. Positive means the upper barrier arrived first
        more often; negative means the lower barrier did. This is descriptive drift, not a strategy.
        """,
    ),
    code(
        "direction-heatmap",
        """
        skew = matrix(
            natural, barriers, horizons, "directional_skew_from_half",
            y_key="barrier_bps", x_key="horizon_seconds", multiplier=100,
        )
        fig = go.Figure(
            go.Heatmap(
                x=[horizon_label(value) for value in horizons],
                y=[f"{value:.0f}" for value in barriers],
                z=skew,
                text=heat_text(skew, " pp", digits=1),
                texttemplate="%{text}",
                colorscale=[[0, BAD], [0.5, "#F7F7F7"], [1, GOOD]],
                zmid=0,
                colorbar_title="Up-first edge",
                hovertemplate="Barrier: %{y} bps<br>Horizon: %{x}<br>Up-first minus 50%%: %{z:.1f} pp<extra></extra>",
            )
        )
        fig.update_layout(
            title="Upper-first probability minus a fair 50% split",
            xaxis_title="Maximum holding time",
            yaxis_title="Symmetric barrier (bps)",
            height=540,
            margin=dict(l=90, r=40, t=70, b=70),
        )
        fig.show()
        """,
    ),
    markdown(
        "speed-title",
        """
        ## 3. How long did successful movement take?

        The median below considers only entries where one barrier was reached. Blank or slow cells
        are a warning against calling a profile a scalp merely because its target is small.
        """,
    ),
    code(
        "touch-time-heatmap",
        """
        touch_minutes = matrix(
            natural, barriers, horizons, "median_touch_seconds",
            y_key="barrier_bps", x_key="horizon_seconds", multiplier=1 / 60,
        )
        fig = go.Figure(
            go.Heatmap(
                x=[horizon_label(value) for value in horizons],
                y=[f"{value:.0f}" for value in barriers],
                z=touch_minutes,
                text=heat_text(touch_minutes, "m", digits=1),
                texttemplate="%{text}",
                colorscale="Cividis",
                colorbar_title="Minutes",
                hovertemplate="Barrier: %{y} bps<br>Horizon: %{x}<br>Median touch: %{z:.1f} min<extra></extra>",
            )
        )
        fig.update_layout(
            title="Median time until the first barrier touch",
            xaxis_title="Maximum holding time",
            yaxis_title="Symmetric barrier (bps)",
            height=540,
            margin=dict(l=90, r=40, t=70, b=70),
        )
        fig.show()
        """,
    ),
    markdown(
        "fees-title",
        """
        ## 4. Fees bend the supposedly fair coin

        The chart shows the hit rate required merely to break even with a modeled 10 bps round trip.
        Small targets and wide stops can demand an intimidating win rate even before extra slippage.
        """,
    ),
    code(
        "fee-heatmap",
        """
        cost = study["parameters"]["round_trip_cost_bps"]
        fee_rows = [row for row in study["economics_surface"] if row["cost_bps"] == cost]
        fee_targets = sorted({row["target_bps"] for row in fee_rows})
        fee_stops = sorted({row["stop_bps"] for row in fee_rows})
        required = matrix(
            fee_rows, fee_stops, fee_targets, "break_even_win_rate",
            y_key="stop_bps", x_key="target_bps", multiplier=100,
        )
        fig = go.Figure(
            go.Heatmap(
                x=[f"{value:.0f}" for value in fee_targets],
                y=[f"{value:.0f}" for value in fee_stops],
                z=required,
                text=heat_text(required, "%"),
                texttemplate="%{text}",
                colorscale="YlOrRd",
                zmin=40,
                zmax=100,
                colorbar_title="Break-even %",
                hovertemplate="Target: %{x} bps<br>Stop: %{y} bps<br>Required wins: %{z:.1f}%<extra></extra>",
            )
        )
        fig.update_layout(
            title=f"Fee-adjusted break-even win rate at {cost:.0f} bps round-trip cost",
            xaxis_title="Target (bps)",
            yaxis_title="Stop (bps)",
            height=620,
            margin=dict(l=90, r=40, t=70, b=70),
        )
        fig.show()
        """,
    ),
    markdown(
        "tournament-title",
        """
        ## 5. The strategy tournament

        Every dot is one target, stop, horizon, and signal-family combination with at least 20
        sequential trades in both periods. The upper-right quadrant is desirable. Dots that are
        green only on one axis are regime-dependent or lucky.
        """,
    ),
    code(
        "tournament-scatter",
        """
        pair_lookup = {}
        for row in strategy:
            key = (row["strategy"], row["target_bps"], row["stop_bps"], row["horizon_seconds"])
            pair_lookup.setdefault(key, {})[row["period"]] = row

        pairs = []
        for key, periods in pair_lookup.items():
            if "discovery" not in periods or "validation" not in periods:
                continue
            discovery = periods["discovery"]
            validation = periods["validation"]
            if min(discovery["sequential_trades"], validation["sequential_trades"]) < 20:
                continue
            pairs.append((key, discovery, validation))

        fig = go.Figure()
        for family in study["parameters"]["strategies"]:
            selected = [row for row in pairs if row[0][0] == family]
            if not selected:
                continue
            fig.add_trace(
                go.Scatter(
                    x=[row[1]["average_net_bps_per_trade"] for row in selected],
                    y=[row[2]["average_net_bps_per_trade"] for row in selected],
                    mode="markers",
                    name=family.replace("_", " "),
                    marker=dict(
                        color=STRATEGY_COLORS.get(family, NEUTRAL),
                        size=[6 + min(12, math.sqrt(row[2]["sequential_trades"])) for row in selected],
                        opacity=0.58,
                    ),
                    customdata=[
                        [row[0][1], row[0][2], horizon_label(row[0][3]), row[1]["sequential_trades"], row[2]["sequential_trades"]]
                        for row in selected
                    ],
                    hovertemplate=(
                        "%{fullData.name}<br>Target/stop: %{customdata[0]:.0f}/%{customdata[1]:.0f} bps"
                        "<br>Horizon: %{customdata[2]}<br>Discovery: %{x:+.2f} bps"
                        "<br>Validation: %{y:+.2f} bps<br>Trades: %{customdata[3]}/%{customdata[4]}<extra></extra>"
                    ),
                )
            )
        fig.add_hline(y=0, line_color=BAD, line_width=1)
        fig.add_vline(x=0, line_color=BAD, line_width=1)
        fig.update_layout(
            title="Did net expectancy survive chronological validation?",
            xaxis_title="Discovery average net bps/trade",
            yaxis_title="Validation average net bps/trade",
            height=720,
            margin=dict(l=80, r=30, t=70, b=70),
            legend=dict(orientation="h", y=-0.22),
        )
        fig.show()
        """,
    ),
    markdown(
        "shortlist-title",
        """
        ## 6. What survived the stability filter?

        A cell qualifies only when both periods are positive, at least half its neighboring
        target/stop settings are also positive, and at least half its validation time blocks are
        positive. This reduces isolated lucky pixels, but does not eliminate multiple-testing risk.
        """,
    ),
    code(
        "shortlist-bars",
        """
        labels = [
            f"{row['strategy'].replace('_', ' ')} | {row['target_bps']:.0f}/{row['stop_bps']:.0f} | {horizon_label(row['horizon_seconds'])}"
            for row in robust
        ]
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                y=labels,
                x=[row["discovery_average_net_bps"] for row in robust],
                name="Discovery",
                orientation="h",
                marker_color=NEUTRAL,
                text=[f"{row['discovery_average_net_bps']:+.1f}" for row in robust],
                textposition="auto",
            )
        )
        fig.add_trace(
            go.Bar(
                y=labels,
                x=[row["validation_average_net_bps"] for row in robust],
                name="Validation",
                orientation="h",
                marker_color=GOOD,
                text=[f"{row['validation_average_net_bps']:+.1f}" for row in robust],
                textposition="auto",
            )
        )
        fig.update_layout(
            title="Average net bps per sequential trade after modeled cost",
            xaxis_title="Net bps/trade",
            barmode="group",
            height=max(470, 65 * len(robust)),
            margin=dict(l=315, r=40, t=70, b=70),
            legend=dict(orientation="h", y=-0.15),
        )
        fig.show()
        """,
    ),
    markdown(
        "surface-title",
        """
        ## 7. Is the winning cell surrounded by support?

        These are the six-hour squeeze-breakout surfaces. A broad region that remains green in
        both periods is more credible than one perfect cell.
        """,
    ),
    code(
        "squeeze-surface",
        """
        squeeze = [
            row for row in strategy
            if row["strategy"] == "volatility_squeeze_breakout" and row["horizon_seconds"] == 21600
        ]
        surface_targets = sorted({row["target_bps"] for row in squeeze})
        surface_stops = sorted({row["stop_bps"] for row in squeeze})
        discovery_surface = matrix(
            [row for row in squeeze if row["period"] == "discovery"],
            surface_stops, surface_targets, "average_net_bps_per_trade",
            y_key="stop_bps", x_key="target_bps",
        )
        validation_surface = matrix(
            [row for row in squeeze if row["period"] == "validation"],
            surface_stops, surface_targets, "average_net_bps_per_trade",
            y_key="stop_bps", x_key="target_bps",
        )
        bound = max(abs(value) for grid in (discovery_surface, validation_surface) for row in grid for value in row if value is not None)
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Discovery", "Validation"), shared_yaxes=True)
        for column, values in enumerate((discovery_surface, validation_surface), start=1):
            fig.add_trace(
                go.Heatmap(
                    x=[f"{value:.0f}" for value in surface_targets],
                    y=[f"{value:.0f}" for value in surface_stops],
                    z=values,
                    text=heat_text(values, "", digits=1),
                    texttemplate="%{text}",
                    colorscale=[[0, BAD], [0.5, "#F7F7F7"], [1, GOOD]],
                    zmin=-bound,
                    zmax=bound,
                    showscale=column == 2,
                    colorbar=dict(title="Net bps") if column == 2 else None,
                    hovertemplate="Target: %{x} bps<br>Stop: %{y} bps<br>Net: %{z:+.2f} bps/trade<extra></extra>",
                ),
                row=1,
                col=column,
            )
        fig.update_xaxes(title_text="Target (bps)")
        fig.update_yaxes(title_text="Stop (bps)", row=1, col=1)
        fig.update_layout(
            title="Volatility squeeze breakout: six-hour target/stop surface",
            height=560,
            margin=dict(l=80, r=60, t=90, b=70),
        )
        fig.show()
        """,
    ),
    markdown(
        "control-title",
        """
        ## 8. A lucky coin can look intelligent for one period

        The comparison uses the strongest robust squeeze profile. The fair coin control is included
        to show why one green period is insufficient.
        """,
    ),
    code(
        "control-bars",
        """
        winner = robust[0]
        comparison_families = [
            "volatility_squeeze_breakout",
            "fair_coin_control",
            "micro_momentum",
            "range_reversion",
        ]
        selected = [
            row for row in strategy
            if row["strategy"] in comparison_families
            and row["target_bps"] == winner["target_bps"]
            and row["stop_bps"] == winner["stop_bps"]
            and row["horizon_seconds"] == winner["horizon_seconds"]
        ]
        fig = go.Figure()
        for period, color in (("discovery", NEUTRAL), ("validation", GOOD)):
            period_rows = {row["strategy"]: row for row in selected if row["period"] == period}
            fig.add_trace(
                go.Bar(
                    x=[family.replace("_", " ") for family in comparison_families],
                    y=[period_rows[family]["average_net_bps_per_trade"] for family in comparison_families],
                    name=period.title(),
                    marker_color=color,
                    text=[f"{period_rows[family]['average_net_bps_per_trade']:+.1f}" for family in comparison_families],
                    textposition="auto",
                )
            )
        fig.add_hline(y=0, line_color=BAD, line_width=1)
        fig.update_layout(
            title=f"Same {winner['target_bps']:.0f}/{winner['stop_bps']:.0f} bps, {horizon_label(winner['horizon_seconds'])} profile",
            yaxis_title="Average net bps/trade",
            barmode="group",
            height=500,
            margin=dict(l=80, r=40, t=80, b=120),
            legend=dict(orientation="h", y=-0.28),
        )
        fig.show()
        """,
    ),
    markdown(
        "regime-title",
        """
        ## 9. Volatility changes speed more reliably than direction

        The six-hour `60 bps` example below shows how often a move resolves and how long it takes.
        High volatility makes barriers arrive faster; it does not automatically tell us which side
        will win.
        """,
    ),
    code(
        "regime-bars",
        """
        conditional = [
            row for row in study["conditional_first_passage"]
            if row["barrier_bps"] == 60 and row["horizon_seconds"] == 21600
        ]
        vol_order = ["low", "middle", "high"]
        volatility_rows = {row["bucket"]: row for row in conditional if row["dimension"] == "volatility"}
        session_order = ["00-08 Asia", "08-16 Europe", "16-24 Americas"]
        session_rows = {row["bucket"]: row for row in conditional if row["dimension"] == "session_utc"}
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Resolution and speed by volatility", "Upper barrier arrived first by UTC window"),
        )
        fig.add_trace(
            go.Bar(
                x=vol_order,
                y=[volatility_rows[name]["resolution_rate"] * 100 for name in vol_order],
                marker_color=[MUTED, NEUTRAL, GOOD],
                text=[
                    f"{volatility_rows[name]['resolution_rate'] * 100:.1f}%<br>{volatility_rows[name]['median_touch_seconds'] / 60:.0f}m"
                    for name in vol_order
                ],
                textposition="auto",
                name="Resolved",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=session_order,
                y=[session_rows[name]["up_first_rate"] * 100 for name in session_order],
                marker_color=[GOLD, NEUTRAL, PURPLE],
                text=[f"{session_rows[name]['up_first_rate'] * 100:.1f}%" for name in session_order],
                textposition="auto",
                name="Up first",
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig.add_hline(y=50, line_dash="dash", line_color=BAD, row=1, col=2)
        fig.update_yaxes(title_text="Resolved within six hours (%)", row=1, col=1, range=[0, 105])
        fig.update_yaxes(title_text="Upper-first share (%)", row=1, col=2, range=[30, 60])
        fig.update_layout(height=500, margin=dict(l=80, r=40, t=80, b=100))
        fig.show()
        """,
    ),
    markdown(
        "verdict-title",
        """
        ## 10. What the atlas changes
        """,
    ),
    code(
        "verdict",
        """
        top = robust[0]
        discovery = next(
            row for row in strategy
            if row["strategy"] == top["strategy"]
            and row["target_bps"] == top["target_bps"]
            and row["stop_bps"] == top["stop_bps"]
            and row["horizon_seconds"] == top["horizon_seconds"]
            and row["period"] == "discovery"
        )
        validation = next(
            row for row in strategy
            if row["strategy"] == top["strategy"]
            and row["target_bps"] == top["target_bps"]
            and row["stop_bps"] == top["stop_bps"]
            and row["horizon_seconds"] == top["horizon_seconds"]
            and row["period"] == "validation"
        )
        coin_validation_best = max(
            (row for row in strategy if row["strategy"] == "fair_coin_control" and row["period"] == "validation" and row["sequential_trades"] >= 20),
            key=lambda row: row["average_net_bps_per_trade"],
        )
        coin_discovery_match = next(
            row for row in strategy
            if row["strategy"] == "fair_coin_control"
            and row["period"] == "discovery"
            and row["target_bps"] == coin_validation_best["target_bps"]
            and row["stop_bps"] == coin_validation_best["stop_bps"]
            and row["horizon_seconds"] == coin_validation_best["horizon_seconds"]
        )
        display(
            HTML(
                f'''
                <div style="border-left: 5px solid {GOOD}; padding: 8px 18px; max-width: 980px;">
                  <h3 style="margin-top: 0;">The strongest hypothesis is slower and wider than our starting thesis</h3>
                  <ul>
                    <li>Only <strong>volatility squeeze breakout</strong> produced settings that passed every
                        stability rule.</li>
                    <li>The strongest cell was <strong>{top['target_bps']:.0f} bps target / {top['stop_bps']:.0f}
                        bps stop / {horizon_label(top['horizon_seconds'])}</strong>, not a rapid symmetric scalp.</li>
                    <li>Discovery: <strong>{discovery['sequential_trades']} trades</strong>,
                        <strong>{discovery['average_net_bps_per_trade']:+.2f} bps/trade</strong>.
                        Validation: <strong>{validation['sequential_trades']} trades</strong>,
                        <strong>{validation['average_net_bps_per_trade']:+.2f} bps/trade</strong>.</li>
                    <li>Validation outcomes were <strong>{validation['targets']} targets, {validation['stops']} stops,
                        and {validation['timeouts']} timeouts</strong>. Long and short averages were both positive.</li>
                    <li>The best-looking validation coin cell made
                        <strong>{coin_validation_best['average_net_bps_per_trade']:+.2f} bps/trade</strong>, but the
                        exact same coin profile made <strong>{coin_discovery_match['average_net_bps_per_trade']:+.2f}</strong>
                        in discovery. That is why one green period is not evidence.</li>
                    <li>We searched <strong>{len(strategy):,} strategy-period cells</strong>. The result is a candidate
                        for forward shadow testing, not permission to deploy a wide-stop live trade.</li>
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
        ## 11. Next research chapters

        1. Add **parallel squeeze exit profiles** (`60/60`, `60/80`, `80/150`, and `100/150`) to a
           future shadow arena while sharing identical entry timestamps.
        2. Build an **exit and excursion atlas**: MFE, MAE, time-to-peak, and signal decay for each
           entry family.
        3. Run **block bootstrap and risk-of-ruin simulations** using complete trade sequences,
           including fee and slippage stress.
        4. Repeat the atlas on new months and radically different volatility regimes. A durable edge
           should survive time, not just neighboring parameter cells.

        The currently deployed arena remains the clean forward test of the original symmetric
        hypothesis. This notebook does not change that running container.
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
        "language_info": {
            "name": "python",
            "version": "3.13",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(OUTPUT)
