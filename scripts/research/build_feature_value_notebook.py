import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "notebooks" / "07_feature_value_analysis.ipynb"


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
        # Feature-value analysis: which inputs actually improve the squeeze trade?

        Proposal 5 keeps the same side-neutral squeeze entries and the unchanged `100/150/6h`
        reference exit. It tests **39 interpretable inputs** without changing timestamps, direction,
        or exit policy.

        Directional features are mirrored by trade side, so a positive value always means "supports
        the chosen direction" for both longs and shorts. Discovery alone chooses feature orientation
        and tercile thresholds. Calibration must confirm the relationship before holdout is opened.
        Two feed-lag fields are diagnostic-only and are forbidden from becoming alpha.
        """,
    ),
    code(
        "setup",
        r"""
        import json
        import math
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
        REPORT_PATH = DATA_ROOT / "feature-value-local.json"
        study = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        features = study["feature_results"]
        groups = study["group_results"]
        baseline = study["baseline"]
        definitions = {row["name"]: row for row in study["features"]}
        selected_names = {row["feature"] for row in groups}
        stable_names = {row["feature"] for row in features if row["stable_preholdout"]}

        GOOD = "#2A9D8F"
        BAD = "#E76F51"
        BLUE = "#457B9D"
        GOLD = "#E9C46A"
        MUTED = "#8D99AE"
        PURPLE = "#7B2CBF"
        TEAL = "#1D7874"
        ORANGE = "#F4A261"
        PINK = "#D45087"
        GROUP_COLORS = {
            "local_order_flow": TEAL,
            "book_pressure": BLUE,
            "price_action": PURPLE,
            "volatility_liquidity": ORANGE,
            "higher_timeframe": GOOD,
            "cross_market": GOLD,
            "derivatives": PINK,
            "data_quality": MUTED,
        }
        VERDICT_COLORS = {
            "robust_feature_candidate": GOOD,
            "promising_but_unconfirmed": BLUE,
            "not_side_general": GOLD,
            "failed_holdout": BAD,
            "selection_failed": MUTED,
        }

        def pretty(value):
            return value.replace("_", " ").title()

        print(f"Loaded: {REPORT_PATH}")
        print(f"Coverage: {study['data']['mark_price_start']} to {study['data']['mark_price_end']}")
        print(f"Dense mark-price points: {study['data']['unique_mark_price_points']:,}")
        print(f"Sampled feature rows: {study['data']['sampled_feature_rows']:,}")
        print("Locked cohorts: " + ", ".join(f"{name}={row['locked_entries']}" for name, row in study['entry_cohorts'].items()))
        print(f"Features: {len(features)} total, {sum(row['alpha_eligible'] for row in features)} alpha-eligible")
        print(f"Stable before holdout: {len(stable_names)}")
        print(f"Promoted after holdout: {len(study['promotion_candidates'])}")
        """,
    ),
    markdown(
        "contract-title",
        """
        ## 1. The anti-overfitting contract

        The analysis is nested. Holdout is not allowed to choose a sign, threshold, or winning
        feature. This matters because testing dozens of noisy inputs will always produce a few
        beautiful accidents.
        """,
    ),
    code(
        "contract-flow",
        """
        labels = [
            ("Locked entries", "Same squeeze trades"),
            ("Discovery", "Choose sign + terciles"),
            ("Calibration", "Must confirm"),
            ("One per family", "Pre-holdout selection"),
            ("Holdout", "Judge once"),
            ("Forward shadow", "Required before code"),
        ]
        colors = [BLUE, GOLD, ORANGE, PURPLE, GOOD, TEAL]
        fig = go.Figure()
        for index, ((title, note), color) in enumerate(zip(labels, colors)):
            fig.add_trace(
                go.Scatter(
                    x=[index], y=[0], mode="markers+text", marker=dict(size=32, color=color),
                    text=[f"<b>{title}</b><br>{note}"], textposition="bottom center",
                    hoverinfo="skip", showlegend=False,
                )
            )
            if index < len(labels) - 1:
                fig.add_annotation(x=index + 0.5, y=0, text="->", showarrow=False, font=dict(size=23))
        fig.update_layout(
            title="A feature cannot learn from the period that judges it",
            xaxis=dict(visible=False, range=[-0.45, 5.45]),
            yaxis=dict(visible=False, range=[-0.82, 0.35]),
            height=310, margin=dict(l=40, r=40, t=75, b=35),
        )
        fig.show()
        """,
    ),
    markdown(
        "coverage-title",
        """
        ## 2. Did we actually observe each feature?

        Coverage below 80% disqualifies a feature. Several BTC cross-market fields are absent from
        the older archive, and liquidation events are naturally sparse. Missing data means
        **unjudgeable**, not automatically useless.
        """,
    ),
    code(
        "coverage-heatmap",
        """
        ordered = sorted(features, key=lambda row: (row["group"], row["feature"]))
        periods = ["discovery", "calibration", "holdout"]
        z = [[100 * row[period]["coverage"] for period in periods] for row in ordered]
        labels = [f"{pretty(row['group'])} | {pretty(row['feature'])}" for row in ordered]
        fig = go.Figure(
            go.Heatmap(
                x=[pretty(period) for period in periods], y=labels, z=z,
                text=[[f"{value:.0f}%" for value in values] for values in z],
                texttemplate="%{text}", colorscale=[[0, BAD], [0.799, GOLD], [0.80, GOOD], [1, TEAL]],
                zmin=0, zmax=100, colorbar_title="Coverage",
                hovertemplate="%{y}<br>%{x}: %{z:.1f}%<extra></extra>",
            )
        )
        fig.update_layout(
            title="Feature availability by chronological period",
            xaxis_title="Period", yaxis_title="Feature family and input",
            height=1120, margin=dict(l=315, r=50, t=80, b=70),
        )
        fig.show()
        """,
    ),
    markdown(
        "selection-title",
        """
        ## 3. What looked repeatable before holdout?

        Each point compares top-tercile minus bottom-tercile expectancy. The upper-right quadrant is
        necessary: the feature must separate trades in the same direction in discovery and
        calibration. Diamonds are the single pre-holdout choice from each economic family. Large
        outlined points passed all non-FDR stability rules.
        """,
    ),
    code(
        "selection-scatter",
        """
        fig = go.Figure()
        for group_name in sorted({row["group"] for row in features if row["alpha_eligible"]}):
            rows = [row for row in features if row["group"] == group_name and row["alpha_eligible"]]
            text = [pretty(row["feature"]) if row["feature"] in stable_names else "" for row in rows]
            fig.add_trace(
                go.Scatter(
                    x=[row["discovery"]["top_minus_bottom_bps"] for row in rows],
                    y=[row["calibration"]["top_minus_bottom_bps"] for row in rows],
                    mode="markers+text", name=pretty(group_name), text=text, textposition="top center",
                    marker=dict(
                        color=GROUP_COLORS[group_name],
                        size=[16 if row["stable_preholdout"] else 11 if row["feature"] in selected_names else 8 for row in rows],
                        symbol=["diamond" if row["feature"] in selected_names else "circle" for row in rows],
                        line=dict(width=[2 if row["stable_preholdout"] else 0 for row in rows], color="#222"),
                        opacity=0.86,
                    ),
                    customdata=[[
                        pretty(row["feature"]), row["stable_preholdout"],
                        row["robust_preholdout_score_bps"], row["calibration"]["fdr_q_value"],
                    ] for row in rows],
                    hovertemplate=(
                        "%{customdata[0]}<br>Discovery tail effect: %{x:+.2f} bps"
                        "<br>Calibration tail effect: %{y:+.2f} bps"
                        "<br>Robust score: %{customdata[2]:+.2f}"
                        "<br>Stable before holdout: %{customdata[1]}"
                        "<br>Calibration FDR q: %{customdata[3]:.3f}<extra></extra>"
                    ),
                )
            )
        fig.add_hline(y=0, line_color=BAD, line_width=1)
        fig.add_vline(x=0, line_color=BAD, line_width=1)
        fig.update_layout(
            title="Discovery versus calibration feature separation",
            xaxis_title="Discovery top-minus-bottom net bps/trade",
            yaxis_title="Calibration top-minus-bottom net bps/trade",
            height=720, margin=dict(l=85, r=50, t=85, b=140),
            legend=dict(orientation="h", y=-0.20),
        )
        fig.show()
        """,
    ),
    markdown(
        "period-title",
        """
        ## 4. Stability is harder than one green cell

        The heatmap shows every feature's frozen top-minus-bottom effect through time. A real feature
        should stay green across columns. Instead, many relationships change sign. This is the visual
        signature of regime dependence, small samples, and multiple-testing luck.
        """,
    ),
    code(
        "period-heatmap",
        """
        ranked = sorted(
            features,
            key=lambda row: (row["stable_preholdout"], row["robust_preholdout_score_bps"]),
            reverse=True,
        )
        periods = ["discovery", "calibration", "holdout"]
        z = [[row[period]["top_minus_bottom_bps"] for period in periods] for row in ranked]
        text = [[
            f"{row[period]['top_minus_bottom_bps']:+.1f}<br>q={row[period]['fdr_q_value']:.2f}"
            for period in periods
        ] for row in ranked]
        bound = max(abs(value) for values in z for value in values)
        labels = [
            ("* " if row["stable_preholdout"] else "") + pretty(row["feature"])
            for row in ranked
        ]
        fig = go.Figure(
            go.Heatmap(
                x=[pretty(period) for period in periods], y=labels, z=z, text=text,
                texttemplate="%{text}", colorscale="RdYlGn", zmid=0, zmin=-bound, zmax=bound,
                colorbar_title="Tail effect bps",
                hovertemplate="%{y}<br>%{x}: %{z:+.2f} bps<extra></extra>",
            )
        )
        fig.update_layout(
            title="Frozen top-tercile advantage by period (* = pre-holdout stable)",
            xaxis_title="Period", yaxis_title="Feature",
            height=1100, margin=dict(l=260, r=50, t=85, b=70),
        )
        fig.show()
        """,
    ),
    markdown(
        "fdr-title",
        """
        ## 5. Thirty-seven alpha tests demand a multiple-testing penalty

        The horizontal line is the relaxed `q <= 0.20` threshold in both discovery and calibration.
        No feature clears it. The apparent winners are not statistically distinguishable from the
        best accidents expected when many noisy inputs are tried.
        """,
    ),
    code(
        "fdr-scatter",
        """
        eligible = [row for row in features if row["alpha_eligible"]]
        significance = [
            -math.log10(max(row["discovery"]["fdr_q_value"], row["calibration"]["fdr_q_value"], 1e-6))
            for row in eligible
        ]
        fig = go.Figure(
            go.Scatter(
                x=[row["robust_preholdout_score_bps"] for row in eligible], y=significance,
                mode="markers+text",
                text=[pretty(row["feature"]) if row["stable_preholdout"] else "" for row in eligible],
                textposition="top center",
                marker=dict(
                    size=[15 if row["stable_preholdout"] else 9 for row in eligible],
                    color=[GROUP_COLORS[row["group"]] for row in eligible],
                    symbol=["diamond" if row["feature"] in selected_names else "circle" for row in eligible],
                ),
                customdata=[[
                    pretty(row["feature"]), pretty(row["group"]),
                    row["discovery"]["fdr_q_value"], row["calibration"]["fdr_q_value"],
                ] for row in eligible],
                hovertemplate=(
                    "%{customdata[0]} (%{customdata[1]})<br>Robust score: %{x:+.2f} bps"
                    "<br>Discovery q: %{customdata[2]:.3f}<br>Calibration q: %{customdata[3]:.3f}<extra></extra>"
                ), showlegend=False,
            )
        )
        fig.add_vline(x=0, line_color=BAD, line_width=1)
        fig.add_hline(
            y=-math.log10(0.20), line_color=GOOD, line_dash="dash",
            annotation_text="Both periods q <= 0.20 required", annotation_position="top left",
        )
        fig.update_layout(
            title="Pre-holdout effect versus multiplicity-adjusted evidence",
            xaxis_title="Worst discovery/calibration advantage (net bps/trade)",
            yaxis_title="-log10(worse discovery/calibration FDR q)",
            height=620, margin=dict(l=85, r=45, t=85, b=75),
        )
        fig.show()
        """,
    ),
    markdown(
        "holdout-title",
        """
        ## 6. Even the strongest family representatives failed holdout

        One diagnostic representative per economic family was frozen before holdout. No family met
        the full FDR-qualified selection rule. Diamonds show holdout top-minus-bottom expectancy;
        whiskers are moving-block bootstrap 95% intervals. Every holdout value in this chart is
        therefore diagnostic only.
        """,
    ),
    code(
        "holdout-forest",
        """
        ordered_groups = sorted(groups, key=lambda row: row["robust_preholdout_score_bps"], reverse=True)
        values = [row["holdout"]["top_minus_bottom_bps"] for row in ordered_groups]
        intervals = [row["holdout"]["top_minus_bottom_bootstrap_95"] for row in ordered_groups]
        lows = [interval[0] if interval else value for interval, value in zip(intervals, values)]
        highs = [interval[1] if interval else value for interval, value in zip(intervals, values)]
        labels = [f"{pretty(row['group'])}<br>{pretty(row['feature'])}" for row in ordered_groups]
        fig = go.Figure(
            go.Scatter(
                x=values, y=labels, mode="markers+text",
                text=[f"{value:+.1f}" for value in values], textposition="middle right",
                marker=dict(
                    size=14, symbol="diamond",
                    color=[VERDICT_COLORS[row["verdict"]] for row in ordered_groups],
                ),
                error_x=dict(
                    type="data", symmetric=False,
                    array=[high - value for high, value in zip(highs, values)],
                    arrayminus=[value - low for value, low in zip(values, lows)],
                    thickness=2, width=6, color=MUTED,
                ),
                customdata=[[
                    row["selection_qualified"], row["holdout"]["top"]["trades"], row["verdict"],
                ] for row in ordered_groups],
                hovertemplate=(
                    "%{y}<br>Holdout tail effect: %{x:+.2f} bps"
                    "<br>Top-tercile trades: %{customdata[1]}"
                    "<br>Pre-holdout qualified: %{customdata[0]}<br>%{customdata[2]}<extra></extra>"
                ), showlegend=False,
            )
        )
        fig.add_vline(x=0, line_color=BAD, line_width=2)
        fig.update_layout(
            title="Selected family features on untouched holdout",
            xaxis_title="Top-minus-bottom average net bps/trade",
            height=610, margin=dict(l=285, r=70, t=85, b=75),
        )
        fig.show()
        """,
    ),
    markdown(
        "side-title",
        """
        ## 7. Side neutrality is a real constraint

        The left panel shows the combined discovery/calibration tail effect by direction. Only the
        upper-right quadrant can pass. The right panel shows holdout top-tercile expectancy for each
        selected family feature. A bearish or bullish episode cannot justify a permanent one-sided
        rule.
        """,
    ),
    code(
        "side-panels",
        """
        group_features = [next(row for row in features if row["feature"] == group["feature"]) for group in groups]
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Pre-holdout top-minus-bottom", "Holdout top-tercile expectancy"),
            horizontal_spacing=0.14,
        )
        for row in group_features:
            color = GROUP_COLORS[row["group"]]
            fig.add_trace(
                go.Scatter(
                    x=[row["preholdout_side_effect"]["long"]["top_minus_bottom_bps"]],
                    y=[row["preholdout_side_effect"]["short"]["top_minus_bottom_bps"]],
                    mode="markers", name=pretty(row["group"]),
                    marker=dict(size=13, color=color, symbol="diamond"), showlegend=True,
                    customdata=[[pretty(row["feature"])]],
                    hovertemplate="%{customdata[0]}<br>Long effect: %{x:+.2f}<br>Short effect: %{y:+.2f}<extra></extra>",
                ), row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=[row["holdout"]["top"]["long"]["average_net_bps_per_trade"]],
                    y=[row["holdout"]["top"]["short"]["average_net_bps_per_trade"]],
                    mode="markers",
                    marker=dict(size=13, color=color, symbol="diamond"), showlegend=False,
                    customdata=[[pretty(row["feature"])]],
                    hovertemplate="%{customdata[0]}<br>Long expectancy: %{x:+.2f}<br>Short expectancy: %{y:+.2f}<extra></extra>",
                ), row=1, col=2,
            )
        for column in (1, 2):
            fig.add_hline(y=0, line_color=BAD, line_width=1, row=1, col=column)
            fig.add_vline(x=0, line_color=BAD, line_width=1, row=1, col=column)
        fig.update_xaxes(title_text="Long net bps/trade", row=1, col=1)
        fig.update_yaxes(title_text="Short net bps/trade", row=1, col=1)
        fig.update_xaxes(title_text="Long net bps/trade", row=1, col=2)
        fig.update_yaxes(title_text="Short net bps/trade", row=1, col=2)
        fig.update_layout(
            title="Does the same feature help both chosen directions?",
            height=690, margin=dict(l=85, r=50, t=105, b=135),
            legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
        )
        fig.show()
        """,
    ),
    markdown(
        "temptation-title",
        """
        ## 8. The hindsight-trap quadrant

        Upper-left points look good only after holdout is revealed. `Side wall advantage`, fast
        aggression, and longer local returns are examples. Adding them now would convert a diagnostic
        observation into an overfit strategy rule.
        """,
    ),
    code(
        "temptation-scatter",
        """
        top_hindsight = {
            row["feature"] for row in sorted(
                [row for row in features if not row["stable_preholdout"]],
                key=lambda row: row["holdout"]["top_minus_bottom_bps"], reverse=True,
            )[:6]
        }
        fig = go.Figure(
            go.Scatter(
                x=[row["robust_preholdout_score_bps"] for row in features if row["alpha_eligible"]],
                y=[row["holdout"]["top_minus_bottom_bps"] for row in features if row["alpha_eligible"]],
                mode="markers+text",
                text=[
                    pretty(row["feature"]) if row["feature"] in top_hindsight or row["stable_preholdout"] else ""
                    for row in features if row["alpha_eligible"]
                ],
                textposition="top center",
                marker=dict(
                    size=[14 if row["stable_preholdout"] else 9 for row in features if row["alpha_eligible"]],
                    color=[GROUP_COLORS[row["group"]] for row in features if row["alpha_eligible"]],
                    symbol=["diamond" if row["stable_preholdout"] else "circle" for row in features if row["alpha_eligible"]],
                ),
                customdata=[[
                    pretty(row["feature"]), pretty(row["group"]), row["stable_preholdout"],
                ] for row in features if row["alpha_eligible"]],
                hovertemplate=(
                    "%{customdata[0]} (%{customdata[1]})<br>Pre-holdout robust score: %{x:+.2f}"
                    "<br>Holdout tail effect: %{y:+.2f}<br>Pre-holdout stable: %{customdata[2]}<extra></extra>"
                ), showlegend=False,
            )
        )
        fig.add_hline(y=0, line_color=BAD, line_width=1)
        fig.add_vline(x=0, line_color=BAD, line_width=1)
        fig.add_annotation(
            x=-38, y=55, text="Tempting only after holdout", showarrow=False,
            bgcolor="rgba(231,111,81,0.14)", bordercolor=BAD,
        )
        fig.update_layout(
            title="Pre-holdout robustness versus holdout temptation",
            xaxis_title="Worst discovery/calibration advantage (net bps/trade)",
            yaxis_title="Holdout top-minus-bottom net bps/trade",
            height=690, margin=dict(l=85, r=45, t=85, b=75),
        )
        fig.show()
        """,
    ),
    markdown(
        "correlation-title",
        """
        ## 9. Different names can measure the same pressure

        Correlated features do not provide independent votes. Discovery data shows depth imbalance
        and VAMP displacement above `|rho| = 0.80`; a later multivariate model should avoid counting
        both as separate confirmations without regularization.
        """,
    ),
    code(
        "correlation-heatmap",
        """
        correlation = study["correlation"]
        labels = [pretty(name) for name in correlation["features"]]
        fig = go.Figure(
            go.Heatmap(
                x=labels, y=labels, z=correlation["spearman_matrix"],
                colorscale="RdBu", reversescale=True, zmin=-1, zmax=1, zmid=0,
                text=[[f"{value:+.2f}" for value in row] for row in correlation["spearman_matrix"]],
                texttemplate="%{text}", colorbar_title="Spearman rho",
                hovertemplate="%{y}<br>%{x}<br>rho=%{z:+.3f}<extra></extra>",
            )
        )
        fig.update_layout(
            title="Discovery-period redundancy among leading features",
            height=830, margin=dict(l=230, r=45, t=85, b=230),
        )
        fig.update_xaxes(tickangle=-45)
        fig.show()
        """,
    ),
    markdown(
        "coin-title",
        """
        ## 10. A good filter must beat direction luck at the same timestamps

        Diamonds are the selected feature's top-tercile holdout trades. Whiskers show the 5th to
        95th percentile from 32 deterministic random-direction controls at those exact timestamps.
        None of the selected feature subsets clears its matched coin-control ceiling.
        """,
    ),
    code(
        "coin-controls",
        """
        coin_rows = sorted(groups, key=lambda row: row["holdout"]["top"]["average_net_bps_per_trade"], reverse=True)
        centers = [row["matched_coin_control"]["median_average_net_bps"] for row in coin_rows]
        lows = [row["matched_coin_control"]["p05_average_net_bps"] for row in coin_rows]
        highs = [row["matched_coin_control"]["p95_average_net_bps"] for row in coin_rows]
        strategy_values = [row["holdout"]["top"]["average_net_bps_per_trade"] for row in coin_rows]
        labels = [f"{pretty(row['group'])}<br>{pretty(row['feature'])}" for row in coin_rows]
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
                x=strategy_values, y=labels, mode="markers+text", name="Feature top tercile",
                text=[f"{value:+.1f}" for value in strategy_values], textposition="middle right",
                marker=dict(
                    color=[GROUP_COLORS[row["group"]] for row in coin_rows],
                    size=14, symbol="diamond",
                ),
                hovertemplate="%{y}<br>Feature subset: %{x:+.2f} bps/trade<extra></extra>",
            )
        )
        fig.add_vline(x=0, line_color=BAD, line_width=1)
        fig.update_layout(
            title="Selected feature subsets versus matched direction luck",
            xaxis_title="Average net bps per trade", height=650,
            margin=dict(l=285, r=60, t=85, b=100), legend=dict(orientation="h", y=-0.12),
        )
        fig.show()
        """,
    ),
    markdown(
        "verdict-title",
        """
        ## 11. Feature-value verdict
        """,
    ),
    code(
        "verdict",
        """
        flow = next(row for row in groups if row["group"] == "local_order_flow")
        htf = next(row for row in groups if row["group"] == "higher_timeframe")
        funding = next(row for row in groups if row["group"] == "derivatives")
        btc = next(row for row in groups if row["group"] == "cross_market")
        display(
            HTML(
                f'''
                <div style="border-left: 5px solid {GOLD}; padding: 8px 18px; max-width: 1000px;">
                  <h3 style="margin-top: 0;">No single-feature gate is ready for strategy code</h3>
                  <ul>
                    <li><strong>39 features tested; 37 were alpha-eligible.</strong> Four passed the
                        non-FDR pre-holdout stability rules, none passed pre-holdout FDR, and the
                        promotion list is empty.</li>
                    <li><strong>15-second aggression</strong> was the best legitimate flow candidate:
                        discovery/calibration separation was {flow['discovery']['top_minus_bottom_bps']:+.2f} /
                        {flow['calibration']['top_minus_bottom_bps']:+.2f} bps, then holdout reversed to
                        <strong>{flow['holdout']['top_minus_bottom_bps']:+.2f}</strong>.</li>
                    <li><strong>HTF return alignment</strong> and <strong>funding carry</strong> repeated
                        before holdout, then reversed to {htf['holdout']['top_minus_bottom_bps']:+.2f} and
                        {funding['holdout']['top_minus_bottom_bps']:+.2f} bps.</li>
                    <li><strong>BTC aggression</strong> stayed positive across periods, but its selection
                        failed robustness requirements and its holdout top subset did not beat matched
                        direction luck ({btc['holdout']['top']['average_net_bps_per_trade']:+.2f} versus
                        coin p95 {btc['matched_coin_control']['p95_average_net_bps']:+.2f} bps/trade).</li>
                    <li>Fast aggression and wall pressure look strong only after holdout. They remain
                        forward hypotheses, not retrofit confirmations.</li>
                    <li><strong>Engineering action:</strong> add no feature gate. Preserve the clean
                        squeeze arena, continue collecting fresh independent regimes, and carry a small
                        frozen shortlist into forward shadow scoring.</li>
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
        ## 12. What proposal 5 hands to proposal 6

        The useful output is a disciplined shortlist, not a new rule:

        - Forward-score `signed_aggression_15s`, `signed_taker_ratio_10s`,
          `htf_return_alignment_bps`, and `favorable_funding_carry_bps` with their discovery-frozen
          orientations and thresholds.
        - Keep `signed_btc_aggression_1s` as an explicitly unconfirmed cross-market hypothesis.
        - Never use feed latency as alpha; it remains a safety/diagnostic signal.
        - Do not combine correlated book features as independent votes.

        Proposal 6 should test **calibration and ensemble construction**: whether a tiny, frozen,
        regularized combination can rank opportunity quality better than any individual feature,
        while preserving side neutrality and matched-control comparisons. It must be developed on
        discovery/calibration and then validated on fresh forward data rather than reopening this
        holdout to tune weights.
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
