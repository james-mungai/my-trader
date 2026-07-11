import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "notebooks" / "06_regime_atlas.ipynb"


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
        # Regime atlas: when should the squeeze strategy be allowed?

        Proposal 4 keeps the side-symmetric squeeze entry and fixed `100/150/6h` exit unchanged.
        It asks whether a deterministic **permission layer** can improve those trades by recognizing
        volatility, liquidity, trend alignment, flow strength, extension, or session context.

        Thresholds are frozen from discovery data. Gates are selected using discovery plus
        calibration only. Holdout is opened once. Long/short side is an audit dimension and is
        explicitly forbidden from gate selection.
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
        REPORT_PATH = DATA_ROOT / "regime-atlas-local.json"
        study = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        gate_results = study["gate_results"]
        gate_by_dimension = {row["dimension"]: row for row in gate_results}
        dimensions = study["dimensions"]
        dimension_by_name = {row["name"]: row for row in dimensions}
        cells = study["cell_results"]
        candidates = study["selection_candidates"]
        baseline = study["baseline"]

        GOOD = "#2A9D8F"
        BAD = "#E76F51"
        BLUE = "#457B9D"
        GOLD = "#E9C46A"
        MUTED = "#8D99AE"
        PURPLE = "#7B2CBF"
        TEAL = "#1D7874"
        ORANGE = "#F4A261"
        DIMENSION_COLORS = {
            "volatility": BAD,
            "spread": BLUE,
            "htf_alignment": GOOD,
            "flow_strength": TEAL,
            "breakout_extension": PURPLE,
            "utc_session": GOLD,
            "liquidity_volatility": ORANGE,
            "entry_side": MUTED,
        }
        VERDICT_COLORS = {
            "robust_permission_candidate": GOOD,
            "promising_but_unconfirmed": BLUE,
            "not_side_general": GOLD,
            "failed_holdout": BAD,
            "selection_failed": MUTED,
        }

        def pretty(value):
            return value.replace("_", " ").title()

        def compact_categories(values):
            return ", ".join(value.replace("_", " ") for value in values)

        print(f"Loaded: {REPORT_PATH}")
        print(f"Coverage: {study['data']['mark_price_start']} to {study['data']['mark_price_end']}")
        print(f"Dense mark-price points: {study['data']['unique_mark_price_points']:,}")
        print(f"Locked cohorts: " + ", ".join(f"{name}={row['locked_entries']}" for name, row in study['entry_cohorts'].items()))
        print(f"Regime dimensions: {len(dimensions)} (7 gate-eligible + side audit)")
        print(f"Permission candidates: {len(candidates)}")
        print(f"Promoted gates: {len(study['promotion_candidates'])}")
        """,
    ),
    markdown(
        "contract-title",
        """
        ## 1. The regime contract

        The atlas is not allowed to invent a new entry, direction, or exit. It may only say **trade**
        or **stand aside** for a squeeze opportunity. This keeps the question measurable and prevents
        a bearish month from turning into a hard-coded short preference.
        """,
    ),
    code(
        "contract-flow",
        """
        labels = [
            ("Locked squeeze", "Same entry timestamps"),
            ("Frozen context", "Discovery-only thresholds"),
            ("Permission gate", "Trade or stand aside"),
            ("Fixed exit", "100 / 150 bps / 6h"),
            ("Nested verdict", "Discovery -> calibration -> holdout"),
        ]
        colors = [BLUE, GOLD, PURPLE, GOOD, TEAL]
        fig = go.Figure()
        for index, ((title, note), color) in enumerate(zip(labels, colors)):
            fig.add_trace(
                go.Scatter(
                    x=[index], y=[0], mode="markers+text",
                    marker=dict(size=34, color=color),
                    text=[f"<b>{title}</b><br>{note}"], textposition="bottom center",
                    hoverinfo="skip", showlegend=False,
                )
            )
            if index < len(labels) - 1:
                fig.add_annotation(x=index + 0.5, y=0, text="->", showarrow=False, font=dict(size=24))
        fig.update_layout(
            title="One permission decision around an unchanged strategy",
            xaxis=dict(visible=False, range=[-0.45, 4.45]),
            yaxis=dict(visible=False, range=[-0.75, 0.35]),
            height=300, margin=dict(l=45, r=45, t=75, b=35),
        )
        fig.show()
        """,
    ),
    markdown(
        "threshold-title",
        """
        ## 2. What was frozen before outcomes?

        Numeric cut points come only from discovery-period distributions. HTF alignment uses a fixed
        `+/-0.20` trend-score band, and sessions use fixed eight-hour UTC windows. These thresholds
        are context definitions, not fitted profit targets.
        """,
    ),
    code(
        "threshold-rulers",
        """
        thresholds = study["thresholds_frozen_from_discovery"]
        numeric = [
            ("Realized volatility", thresholds["volatility_bps"]["low_max"], thresholds["volatility_bps"]["high_min"], "bps"),
            ("Spread", thresholds["spread_bps"]["low_max"], thresholds["spread_bps"]["high_min"], "bps"),
            ("Flow strength", thresholds["flow_strength"]["low_max"], thresholds["flow_strength"]["high_min"], "score"),
            ("Breakout extension", thresholds["breakout_extension_bps"]["low_max"], thresholds["breakout_extension_bps"]["high_min"], "bps"),
        ]
        fig = go.Figure()
        for index, (label, low, high, unit) in enumerate(numeric):
            fig.add_trace(
                go.Scatter(
                    x=[1 / 3, 2 / 3], y=[index, index], mode="lines+markers+text",
                    line=dict(color=DIMENSION_COLORS[["volatility", "spread", "flow_strength", "breakout_extension"][index]], width=8),
                    marker=dict(size=13),
                    text=[f"{low:.3f} {unit}", f"{high:.3f} {unit}"], textposition=["middle left", "middle right"],
                    customdata=[[low], [high]],
                    hovertemplate=f"{label}<br>%{{customdata[0]:.4f}} {unit}<extra></extra>",
                    showlegend=False,
                )
            )
        fig.add_vrect(x0=0, x1=1 / 3, fillcolor="rgba(69,123,157,0.10)", line_width=0)
        fig.add_vrect(x0=1 / 3, x1=2 / 3, fillcolor="rgba(233,196,106,0.10)", line_width=0)
        fig.add_vrect(x0=2 / 3, x1=1, fillcolor="rgba(231,111,81,0.10)", line_width=0)
        for x, label in ((1 / 6, "Low"), (1 / 2, "Normal"), (5 / 6, "High")):
            fig.add_annotation(x=x, y=3.55, text=label, showarrow=False)
        fig.update_layout(
            title="Discovery-frozen low/normal/high boundaries",
            xaxis=dict(visible=False, range=[0, 1]),
            yaxis=dict(tickmode="array", tickvals=list(range(len(numeric))), ticktext=[row[0] for row in numeric]),
            height=430, margin=dict(l=170, r=80, t=90, b=45),
        )
        fig.show()
        """,
    ),
    markdown(
        "cell-title",
        """
        ## 3. The complete regime map

        Every row partitions the same baseline trades. Text shows average net bps and trade count.
        A stable permission regime should keep its sign as we move from discovery to calibration to
        holdout. Most cells are visibly noisy because the independent cohorts are small.
        """,
    ),
    code(
        "cell-heatmap",
        """
        row_order = []
        for definition in dimensions:
            for category in definition["categories"]:
                row_order.append((definition["name"], category))
        lookup = {(row["dimension"], row["category"], row["period"]): row for row in cells}
        periods = ["discovery", "calibration", "holdout"]
        z = []
        text = []
        labels = []
        for dimension, category in row_order:
            values = [lookup[(dimension, category, period)] for period in periods]
            z.append([row["average_net_bps_per_trade"] if row["trades"] else None for row in values])
            text.append([f"{row['average_net_bps_per_trade']:+.1f}<br>n={row['trades']}" if row["trades"] else "n=0" for row in values])
            labels.append(f"{pretty(dimension)} | {pretty(category)}")
        bound = max(abs(value) for row in z for value in row if value is not None)
        fig = go.Figure(
            go.Heatmap(
                x=[pretty(period) for period in periods], y=labels, z=z, text=text,
                texttemplate="%{text}", colorscale="RdYlGn", zmid=0, zmin=-bound, zmax=bound,
                colorbar_title="Net bps/trade",
                hovertemplate="%{y}<br>%{x}<br>%{z:+.2f} net bps/trade<extra></extra>",
            )
        )
        fig.update_layout(
            title="Regime-cell expectancy through chronological periods",
            xaxis_title="Period", yaxis_title="Regime cell",
            height=980, margin=dict(l=245, r=45, t=80, b=70),
        )
        fig.show()
        """,
    ),
    markdown(
        "selection-title",
        """
        ## 4. Which gates looked useful before holdout?

        Each point is an allowed category set. The upper-right quadrant improved expectancy versus
        trading every squeeze in both discovery and calibration. Diamonds are the pre-holdout choice
        for each dimension. Qualification also requires adequate coverage, both directions, and
        positive chronological blocks.
        """,
    ),
    code(
        "gate-selection",
        """
        selected = {row["dimension"]: tuple(row["accepted_categories"]) for row in gate_results}
        selected_labels = {
            "htf_alignment": "HTF neutral",
            "flow_strength": "Flow low + medium",
            "volatility": "High volatility",
        }
        selected_label_positions = {
            "htf_alignment": "top right",
            "flow_strength": "top right",
            "volatility": "bottom left",
        }
        fig = go.Figure()
        for dimension in sorted({row["dimension"] for row in candidates}):
            rows = [row for row in candidates if row["dimension"] == dimension]
            is_selected = [tuple(row["accepted_categories"]) == selected[dimension] for row in rows]
            fig.add_trace(
                go.Scatter(
                    x=[row["discovery"]["accepted_delta_vs_all_bps"] for row in rows],
                    y=[row["calibration"]["accepted_delta_vs_all_bps"] for row in rows],
                    mode="markers+text", name=pretty(dimension),
                    marker=dict(
                        color=DIMENSION_COLORS[dimension],
                        size=[14 if flag else 8 for flag in is_selected],
                        symbol=["diamond" if flag else "circle" for flag in is_selected],
                        opacity=0.88,
                    ),
                    text=[selected_labels.get(dimension, "") if flag else "" for flag in is_selected],
                    textposition=[selected_label_positions.get(dimension, "top center") for _ in is_selected],
                    customdata=[[compact_categories(row["accepted_categories"]), row["qualifies"]] for row in rows],
                    hovertemplate=(
                        "%{customdata[0]}<br>Discovery delta: %{x:+.2f} bps"
                        "<br>Calibration delta: %{y:+.2f} bps<br>Qualified: %{customdata[1]}<extra></extra>"
                    ),
                )
            )
        all_x = [row["discovery"]["accepted_delta_vs_all_bps"] for row in candidates]
        all_y = [row["calibration"]["accepted_delta_vs_all_bps"] for row in candidates]
        fig.add_hline(y=0, line_color=BAD, line_width=1)
        fig.add_vline(x=0, line_color=BAD, line_width=1)
        fig.update_layout(
            title="Permission candidates before holdout",
            xaxis_title="Discovery improvement versus all squeezes (bps/trade)",
            yaxis_title="Calibration improvement versus all squeezes (bps/trade)",
            height=680, margin=dict(l=85, r=45, t=85, b=135),
            legend=dict(orientation="h", y=-0.22),
        )
        fig.update_xaxes(range=[min(all_x) - 8, max(all_x) + 10])
        fig.update_yaxes(range=[min(all_y) - 8, max(all_y) + 10])
        fig.show()
        """,
    ),
    markdown(
        "holdout-title",
        """
        ## 5. Holdout broke the apparent gates

        Bars compare the selected gate's accepted and rejected holdout trades. The dashed line is the
        unchanged all-trade baseline. A useful gate needs accepted trades above the line and rejected
        trades below it for reasons that survive uncertainty.
        """,
    ),
    code(
        "holdout-bars",
        """
        ordered = sorted(gate_results, key=lambda row: row["robust_gate_score_bps"], reverse=True)
        labels = [pretty(row["dimension"]) for row in ordered]
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=labels, y=[row["holdout"]["accepted"]["average_net_bps_per_trade"] for row in ordered],
                name="Accepted", marker_color=GOOD,
                text=[f"{row['holdout']['accepted']['average_net_bps_per_trade']:+.1f}" for row in ordered],
                textposition="auto",
            )
        )
        fig.add_trace(
            go.Bar(
                x=labels, y=[row["holdout"]["rejected"]["average_net_bps_per_trade"] for row in ordered],
                name="Rejected", marker_color=BAD,
                text=[f"{row['holdout']['rejected']['average_net_bps_per_trade']:+.1f}" for row in ordered],
                textposition="auto",
            )
        )
        fig.add_hline(
            y=baseline["holdout"]["average_net_bps_per_trade"], line_color=BLUE, line_dash="dash",
            annotation_text=f"All squeezes {baseline['holdout']['average_net_bps_per_trade']:+.1f}",
            annotation_position="top left",
        )
        fig.add_hline(y=0, line_color=MUTED, line_width=1)
        fig.update_layout(
            title="Selected gates on the untouched holdout",
            yaxis_title="Average net bps per trade", barmode="group", height=570,
            margin=dict(l=80, r=45, t=85, b=145),
            legend=dict(orientation="h", y=-0.24),
        )
        fig.update_xaxes(tickangle=-22)
        fig.show()
        """,
    ),
    markdown(
        "bootstrap-title",
        """
        ## 6. Uncertainty is wider than the story

        Diamonds show accepted minus rejected expectancy. Whiskers are moving-block bootstrap 95%
        intervals. HTF-neutral's point estimate is nearly zero and its interval spans roughly
        `-63` to `+76` bps. None of the pre-holdout selected gates clears zero reliably.
        """,
    ),
    code(
        "bootstrap-forest",
        """
        forest = sorted(gate_results, key=lambda row: row["bootstrap_accepted_minus_rejected"]["mean_delta_bps"], reverse=True)
        values = [row["bootstrap_accepted_minus_rejected"]["mean_delta_bps"] for row in forest]
        lows = [row["bootstrap_accepted_minus_rejected"]["bootstrap_p025_bps"] for row in forest]
        highs = [row["bootstrap_accepted_minus_rejected"]["bootstrap_p975_bps"] for row in forest]
        fig = go.Figure(
            go.Scatter(
                x=values, y=[pretty(row["dimension"]) for row in forest], mode="markers+text",
                text=[f"{value:+.1f}" for value in values], textposition="middle right",
                marker=dict(
                    size=13, symbol="diamond",
                    color=[VERDICT_COLORS[row["verdict"]] for row in forest],
                ),
                error_x=dict(
                    type="data", symmetric=False,
                    array=[high - value for high, value in zip(highs, values)],
                    arrayminus=[value - low for value, low in zip(values, lows)],
                    thickness=2, width=6, color=MUTED,
                ),
                customdata=[[compact_categories(row["accepted_categories"]), low, high] for row, low, high in zip(forest, lows, highs)],
                hovertemplate=(
                    "%{y}<br>Allow: %{customdata[0]}<br>Delta: %{x:+.2f} bps"
                    "<br>95% interval: %{customdata[1]:+.2f} to %{customdata[2]:+.2f}<extra></extra>"
                ), showlegend=False,
            )
        )
        fig.add_vline(x=0, line_color=BAD, line_width=2)
        fig.update_layout(
            title="Holdout accepted-minus-rejected uncertainty",
            xaxis_title="Average net bps difference",
            height=550, margin=dict(l=190, r=60, t=80, b=70),
        )
        fig.show()
        """,
    ),
    markdown(
        "symmetry-title",
        """
        ## 7. Side-neutral means both sides must survive

        The upper-right quadrant is required. HTF-neutral accepted longs remained strong, but shorts
        turned negative. High-volatility accepted trades showed the same directional asymmetry. That
        is why an apparently profitable aggregate gate still fails generalization.
        """,
    ),
    code(
        "side-scatter",
        """
        side_label_positions = {
            "htf_alignment": "top right", "flow_strength": "bottom left", "volatility": "bottom right",
            "spread": "top right", "breakout_extension": "top center",
            "utc_session": "top left", "liquidity_volatility": "bottom left",
        }
        fig = go.Figure(
            go.Scatter(
                x=[row["holdout"]["accepted"]["long"]["average_net_bps_per_trade"] for row in gate_results],
                y=[row["holdout"]["accepted"]["short"]["average_net_bps_per_trade"] for row in gate_results],
                mode="markers+text", text=[pretty(row["dimension"]) for row in gate_results],
                textposition=[side_label_positions[row["dimension"]] for row in gate_results],
                marker=dict(
                    size=[10 + row["holdout"]["accepted"]["trades"] / 2 for row in gate_results],
                    color=[DIMENSION_COLORS[row["dimension"]] for row in gate_results], line=dict(width=1),
                ),
                customdata=[[row["holdout"]["accepted"]["long"]["trades"], row["holdout"]["accepted"]["short"]["trades"], row["verdict"]] for row in gate_results],
                hovertemplate=(
                    "%{text}<br>Long: %{x:+.2f} bps (%{customdata[0]} trades)"
                    "<br>Short: %{y:+.2f} bps (%{customdata[1]} trades)<br>%{customdata[2]}<extra></extra>"
                ), showlegend=False,
            )
        )
        fig.add_hline(y=0, line_color=BAD, line_width=1)
        fig.add_vline(x=0, line_color=BAD, line_width=1)
        fig.update_layout(
            title="Accepted holdout expectancy by direction",
            xaxis_title="Long average net bps/trade", yaxis_title="Short average net bps/trade",
            height=610, margin=dict(l=85, r=50, t=80, b=75),
        )
        fig.show()
        """,
    ),
    markdown(
        "temptation-title",
        """
        ## 8. The tempting hindsight traps

        The upper-left quadrant contains gates that look excellent only after holdout was seen.
        Extended breakouts, Europe/Americas sessions, and wider spreads all produced attractive final
        results, but each had already failed at least one pre-holdout period. Promoting them now would
        be regime overfitting.
        """,
    ),
    code(
        "temptation-scatter",
        """
        temptation_label_positions = {
            "htf_alignment": "top left", "flow_strength": "bottom left", "volatility": "top right",
            "spread": "top right", "breakout_extension": "middle right",
            "utc_session": "top left", "liquidity_volatility": "bottom right",
        }
        fig = go.Figure(
            go.Scatter(
                x=[row["robust_gate_score_bps"] for row in gate_results],
                y=[row["holdout"]["accepted_delta_vs_all_bps"] for row in gate_results],
                mode="markers+text", text=[pretty(row["dimension"]) for row in gate_results],
                textposition=[temptation_label_positions[row["dimension"]] for row in gate_results],
                marker=dict(
                    size=14, color=[DIMENSION_COLORS[row["dimension"]] for row in gate_results],
                    symbol=["diamond" if row["selection_qualified"] else "circle" for row in gate_results],
                ),
                customdata=[[compact_categories(row["accepted_categories"]), row["selection_qualified"], row["verdict"]] for row in gate_results],
                hovertemplate=(
                    "%{text}<br>Allow: %{customdata[0]}<br>Pre-holdout robust score: %{x:+.2f}"
                    "<br>Holdout improvement: %{y:+.2f}<br>Selection qualified: %{customdata[1]}"
                    "<br>%{customdata[2]}<extra></extra>"
                ), showlegend=False,
            )
        )
        fig.add_hline(y=0, line_color=BAD, line_width=1)
        fig.add_vline(x=0, line_color=BAD, line_width=1)
        fig.add_annotation(
            x=-5, y=37, text="Looks good only after holdout", showarrow=False,
            bgcolor="rgba(231,111,81,0.15)", bordercolor=BAD,
        )
        fig.update_layout(
            title="Pre-holdout evidence versus holdout temptation",
            xaxis_title="Robust discovery/calibration gate score (bps)",
            yaxis_title="Holdout accepted improvement versus all trades (bps)",
            height=600, margin=dict(l=85, r=50, t=80, b=75),
        )
        temptation_x = [row["robust_gate_score_bps"] for row in gate_results]
        temptation_y = [row["holdout"]["accepted_delta_vs_all_bps"] for row in gate_results]
        fig.update_xaxes(range=[min(temptation_x) - 8, max(temptation_x) + 8])
        fig.update_yaxes(range=[min(temptation_y) - 8, max(temptation_y) + 12])
        fig.show()
        """,
    ),
    markdown(
        "coin-title",
        """
        ## 9. Regime filtering does not replace directional edge

        Diamonds show accepted squeeze trades. Whiskers show 32 random directions at those exact
        accepted timestamps using the same fixed exit. Even the selected HTF-neutral subset does not
        clear the random-direction 95th percentile.
        """,
    ),
    code(
        "coin-controls",
        """
        coin_rows = sorted(gate_results, key=lambda row: row["holdout"]["accepted"]["average_net_bps_per_trade"], reverse=True)
        centers = [row["matched_coin_control"]["median_average_net_bps"] for row in coin_rows]
        lows = [row["matched_coin_control"]["p05_average_net_bps"] for row in coin_rows]
        highs = [row["matched_coin_control"]["p95_average_net_bps"] for row in coin_rows]
        strategy_values = [row["holdout"]["accepted"]["average_net_bps_per_trade"] for row in coin_rows]
        labels = [pretty(row["dimension"]) for row in coin_rows]
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
                x=strategy_values, y=labels, mode="markers+text", name="Accepted squeeze directions",
                text=[f"{value:+.1f}" for value in strategy_values], textposition="middle right",
                marker=dict(color=[DIMENSION_COLORS[row["dimension"]] for row in coin_rows], size=14, symbol="diamond"),
                hovertemplate="%{y}<br>Strategy: %{x:+.2f} bps/trade<extra></extra>",
            )
        )
        fig.add_vline(x=0, line_color=BAD, line_width=1)
        fig.update_layout(
            title="Accepted holdout trades versus matched direction luck",
            xaxis_title="Average net bps per trade", height=550,
            margin=dict(l=190, r=60, t=80, b=75), legend=dict(orientation="h", y=-0.17),
        )
        fig.update_xaxes(range=[min(lows) - 8, max([*highs, *strategy_values]) + 14])
        fig.show()
        """,
    ),
    markdown(
        "verdict-title",
        """
        ## 10. Regime-atlas verdict
        """,
    ),
    code(
        "verdict",
        """
        htf = gate_by_dimension["htf_alignment"]
        flow = gate_by_dimension["flow_strength"]
        volatility = gate_by_dimension["volatility"]
        extension = gate_by_dimension["breakout_extension"]
        session = gate_by_dimension["utc_session"]
        display(
            HTML(
                f'''
                <div style="border-left: 5px solid {GOLD}; padding: 8px 18px; max-width: 980px;">
                  <h3 style="margin-top: 0;">No permission gate is ready for strategy code</h3>
                  <ul>
                    <li><strong>HTF-neutral</strong> was the strongest legitimate pre-holdout candidate:
                        discovery and calibration improved by {htf['discovery']['accepted_delta_vs_all_bps']:+.2f}
                        and {htf['calibration']['accepted_delta_vs_all_bps']:+.2f} bps/trade. On holdout it improved
                        by only <strong>{htf['holdout']['accepted_delta_vs_all_bps']:+.2f}</strong>, retained just
                        {htf['holdout']['accepted']['trades']} trades, and shorts made
                        {htf['holdout']['accepted']['short']['average_net_bps_per_trade']:+.2f} bps/trade.</li>
                    <li><strong>Lower/medium flow strength</strong> and <strong>high volatility</strong> both qualified
                        before holdout, then reversed: accepted holdout expectancy was
                        {flow['holdout']['accepted']['average_net_bps_per_trade']:+.2f} and
                        {volatility['holdout']['accepted']['average_net_bps_per_trade']:+.2f} bps/trade.</li>
                    <li><strong>Extended breakouts</strong> look spectacular in holdout at
                        {extension['holdout']['accepted']['average_net_bps_per_trade']:+.2f} bps/trade, but calibration
                        had already put their gate score below zero. They are a forward hypothesis, not a retrofit.</li>
                    <li><strong>Europe/Americas sessions</strong> also look attractive only after holdout. Their
                        discovery improvement was {session['discovery']['accepted_delta_vs_all_bps']:+.2f} bps/trade.</li>
                    <li>The promotion list is empty. The correct engineering action is
                        <strong>leave the current strategy ungated</strong> and carry these labels into fresh forward data.</li>
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
        ## 11. What proposal 4 hands to proposal 5

        The atlas adds no live or paper permission gate. It does provide frozen labels that proposal
        **5, feature-value analysis**, can use to ask a more granular question: which individual
        features actually separate winning from losing squeeze entries after costs, and are their
        signs stable across periods and both directions?

        Forward research should continue logging HTF-neutral, extension, session, spread, and joint
        liquidity/volatility labels in shadow. They are hypotheses awaiting new regimes, not switches
        to turn on today.
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
