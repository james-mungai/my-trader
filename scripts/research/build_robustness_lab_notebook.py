import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "notebooks" / "09_robustness_promotion_lab.ipynb"


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
        # Robustness and promotion lab: what would justify a tiny live canary?

        Proposal 7 is the final exploratory proposal. It stops fitting models and stress-tests the
        strategy that remains.

        The historical lane uses 94 locked side-neutral squeeze trades with the fixed `100/150/6h`
        exit. Those trades are development evidence, not independent proof. The forward lane uses
        the separate AWS `+60/-60` shadow arena and a frozen promotion scorecard. The two contracts
        are shown together but never pooled as one experiment.
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
        REPORT_PATH = DATA_ROOT / "robustness-lab-local.json"
        study = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        historical = study["historical_combined"]
        periods = study["historical_periods"]
        blocks = study["chronological_blocks"]
        stress = study["stress_grid"]
        power = study["forward_power_plan"]
        scorecard = study["forward_promotion_scorecard"]

        GOOD = "#2A9D8F"
        BAD = "#E76F51"
        BLUE = "#457B9D"
        GOLD = "#E9C46A"
        MUTED = "#8D99AE"
        PURPLE = "#7B2CBF"
        TEAL = "#1D7874"
        ORANGE = "#F4A261"
        STATUS_COLORS = {"pass": GOOD, "pending": GOLD, "fail": BAD}

        def pretty(value):
            return value.replace("_", " ").title()

        def scenario(cost, missed):
            return next(
                row for row in stress
                if row["total_cost_bps"] == cost and row["missed_winner_rate"] == missed
            )

        def exposure(row, multiple):
            return next(item for item in row["exposures"] if item["exposure_multiple"] == multiple)

        print(f"Loaded: {REPORT_PATH}")
        print(f"Coverage: {study['data']['mark_price_start']} to {study['data']['mark_price_end']}")
        print(f"Historical locked trades: {study['data']['historical_locked_trades']}")
        print(f"Bootstrap paths: {study['parameters']['bootstrap_paths']:,} x {study['parameters']['bootstrap_path_trades']} trades")
        print(f"Stress cells: {len(stress)} across {len(study['parameters']['stress_exposure_multiples'])} exposures")
        print(f"Forward snapshot: {scorecard.get('snapshot_at')}")
        print(f"Forward scorecard: {scorecard['overall_status']}")
        print(f"Promotion candidates: {len(study['promotion_candidates'])}")
        """,
    ),
    markdown(
        "contract-title",
        """
        ## 1. Two evidence lanes, one decision standard

        Historical stress can reject a fragile strategy, but it cannot promote one because the
        archive shaped earlier proposals. Only fresh forward evidence can nominate a separately
        authorized canary.
        """,
    ),
    code(
        "contract-flow",
        """
        fig = make_subplots(rows=2, cols=1, vertical_spacing=0.28, subplot_titles=("Historical development lane", "Fresh forward lane"))
        lanes = [
            (
                1,
                [
                    ("94 locked trades", "100/150/6h"),
                    ("Paired path stress", "fees + missed wins"),
                    ("Exposure ladder", "drawdown + ruin"),
                    ("Reject fragility", "cannot promote"),
                ],
                [BLUE, ORANGE, PURPLE, BAD],
            ),
            (
                2,
                [
                    ("AWS shadow arena", "60/60/6h"),
                    ("100+ trades", "minimum checkpoint"),
                    ("Frozen scorecard", "edge + sides + health"),
                    ("Tiny canary review", "explicit approval"),
                ],
                [TEAL, GOLD, GOOD, BLUE],
            ),
        ]
        for row_number, labels, colors in lanes:
            for index, ((title, note), color) in enumerate(zip(labels, colors)):
                fig.add_trace(
                    go.Scatter(
                        x=[index], y=[0], mode="markers+text", marker=dict(size=30, color=color),
                        text=[f"<b>{title}</b><br>{note}"], textposition="bottom center",
                        hoverinfo="skip", showlegend=False,
                    ), row=row_number, col=1,
                )
                if index < len(labels) - 1:
                    fig.add_annotation(
                        x=index + 0.5, y=0, text="->", showarrow=False, font=dict(size=22),
                        row=row_number, col=1,
                    )
        for row_number in (1, 2):
            fig.update_xaxes(visible=False, range=[-0.45, 3.45], row=row_number, col=1)
            fig.update_yaxes(visible=False, range=[-0.78, 0.32], row=row_number, col=1)
        fig.update_layout(
            title="Historical evidence may veto; only forward evidence may promote",
            height=560, margin=dict(l=45, r=45, t=105, b=35),
        )
        fig.show()
        """,
    ),
    markdown(
        "historical-title",
        """
        ## 2. Historical expectancy survived moderate cost, but confidence remains wide

        Average net expectancy stayed positive in discovery, calibration, and holdout under both 10
        and 15 bps total costs. The combined positive-trade rate was 64.9%, but its 95% Wilson interval
        spans roughly 54.8% to 73.8%. The archive is profitable-looking, not independently conclusive.
        """,
    ),
    code(
        "historical-periods",
        """
        period_names = ["discovery", "calibration", "holdout"]
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Average net expectancy", "Positive-trade rate"))
        fig.add_trace(
            go.Bar(
                x=[pretty(name) for name in period_names],
                y=[periods[name]["reference"]["average_net_bps_per_trade"] for name in period_names],
                name="10 bps total cost", marker_color=BLUE,
                text=[f"{periods[name]['reference']['average_net_bps_per_trade']:+.1f}" for name in period_names],
                textposition="auto",
            ), row=1, col=1,
        )
        fig.add_trace(
            go.Bar(
                x=[pretty(name) for name in period_names],
                y=[periods[name]["hostile"]["average_net_bps_per_trade"] for name in period_names],
                name="15 bps total cost", marker_color=ORANGE,
                text=[f"{periods[name]['hostile']['average_net_bps_per_trade']:+.1f}" for name in period_names],
                textposition="auto",
            ), row=1, col=1,
        )
        fig.add_trace(
            go.Bar(
                x=[pretty(name) for name in period_names],
                y=[100 * periods[name]["reference"]["positive_net_trade_rate"] for name in period_names],
                marker_color=[BLUE, GOLD, GOOD], showlegend=False,
                text=[f"{100 * periods[name]['reference']['positive_net_trade_rate']:.1f}%" for name in period_names],
                textposition="auto",
            ), row=1, col=2,
        )
        fig.add_hline(y=0, line_color=BAD, line_width=1, row=1, col=1)
        fig.update_yaxes(title_text="Average net bps/trade", row=1, col=1)
        fig.update_yaxes(title_text="Positive trades (%)", range=[0, 82], row=1, col=2)
        fig.update_layout(
            title="Locked historical squeeze entries by chronological period",
            barmode="group", height=520, margin=dict(l=75, r=45, t=100, b=105),
            legend=dict(orientation="h", y=-0.16),
        )
        fig.show()
        """,
    ),
    markdown(
        "blocks-title",
        """
        ## 3. Chronological blocks reveal the weak edges of the story

        Seven of eight blocks were positive at 10 bps cost, but the final block lost money. At 15 bps
        cost the first block also becomes negative. The strategy is not a smooth return machine; path
        ordering matters.
        """,
    ),
    code(
        "block-bars",
        """
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=[f"Block {row['block']}" for row in blocks],
                y=[row["reference_average_net_bps"] for row in blocks],
                name="10 bps cost", marker_color=BLUE,
                text=[f"{row['reference_average_net_bps']:+.1f}" for row in blocks], textposition="auto",
            )
        )
        fig.add_trace(
            go.Bar(
                x=[f"Block {row['block']}" for row in blocks],
                y=[row["hostile_average_net_bps"] for row in blocks],
                name="15 bps cost", marker_color=ORANGE,
                text=[f"{row['hostile_average_net_bps']:+.1f}" for row in blocks], textposition="auto",
            )
        )
        fig.add_hline(y=0, line_color=BAD, line_width=2)
        fig.update_layout(
            title="Eight chronological blocks of 11-12 trades",
            xaxis_title="Historical trade block", yaxis_title="Average net bps/trade",
            barmode="group", height=560, margin=dict(l=80, r=45, t=85, b=100),
            legend=dict(orientation="h", y=-0.16),
        )
        fig.show()
        """,
    ),
    markdown(
        "side-title",
        """
        ## 4. Both directions were positive, but the short edge was thin

        Side neutrality does not mean equal performance in every sample. Historical longs averaged
        about 33 bps after cost; shorts averaged only 7 bps and are more vulnerable to execution
        degradation.
        """,
    ),
    code(
        "side-bars",
        """
        side = study["side_robustness"]
        fig = go.Figure()
        for cost_label, key, color in (("10 bps cost", "reference", BLUE), ("15 bps cost", "hostile", ORANGE)):
            fig.add_trace(
                go.Bar(
                    x=["Long", "Short"],
                    y=[side[name][key]["average_net_bps_per_trade"] for name in ("long", "short")],
                    name=cost_label, marker_color=color,
                    text=[f"{side[name][key]['average_net_bps_per_trade']:+.1f}" for name in ("long", "short")],
                    textposition="auto",
                    customdata=[[side[name][key]["trades"]] for name in ("long", "short")],
                    hovertemplate="%{x}<br>%{fullData.name}<br>%{y:+.2f} bps/trade<br>n=%{customdata[0]}<extra></extra>",
                )
            )
        fig.add_hline(y=0, line_color=BAD, line_width=1)
        fig.update_layout(
            title="Historical expectancy by chosen direction",
            yaxis_title="Average net bps/trade", barmode="group", height=470,
            margin=dict(l=75, r=45, t=85, b=95), legend=dict(orientation="h", y=-0.18),
        )
        fig.show()
        """,
    ),
    markdown(
        "stress-title",
        """
        ## 5. Execution asymmetry is the real edge killer

        Each cell uses the same 5,000 bootstrapped paths. The left panel shows how often a 100-trade
        path finishes positive. The right panel shows its 5th-percentile average trade. Missing only
        favorable fills is intentionally hostile: it models being present for losses and late for wins.
        """,
    ),
    code(
        "stress-heatmaps",
        """
        costs = study["parameters"]["stress_total_cost_bps"]
        misses = study["parameters"]["stress_missed_winner_rates"]
        positive = [[100 * scenario(cost, missed)["positive_path_fraction"] for cost in costs] for missed in misses]
        downside = [[scenario(cost, missed)["path_average_net_bps"]["p05"] for cost in costs] for missed in misses]
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Positive 100-trade paths", "5th-percentile average trade"), horizontal_spacing=0.13)
        fig.add_trace(
            go.Heatmap(
                x=costs, y=[f"{100 * value:.0f}%" for value in misses], z=positive,
                text=[[f"{value:.1f}%" for value in row] for row in positive], texttemplate="%{text}",
                colorscale="RdYlGn", zmin=0, zmax=100, colorbar=dict(title="Positive %", x=0.44),
                hovertemplate="Cost %{x} bps<br>Missed winners %{y}<br>Positive paths %{z:.1f}%<extra></extra>",
            ), row=1, col=1,
        )
        bound = max(abs(value) for row in downside for value in row)
        fig.add_trace(
            go.Heatmap(
                x=costs, y=[f"{100 * value:.0f}%" for value in misses], z=downside,
                text=[[f"{value:+.1f}" for value in row] for row in downside], texttemplate="%{text}",
                colorscale="RdYlGn", zmid=0, zmin=-bound, zmax=bound,
                colorbar=dict(title="Net bps", x=1.02),
                hovertemplate="Cost %{x} bps<br>Missed winners %{y}<br>5th percentile %{z:+.2f} bps/trade<extra></extra>",
            ), row=1, col=2,
        )
        fig.update_xaxes(
            title_text="Total round-trip cost (bps)", tickmode="array", tickvals=costs,
            ticktext=[f"{cost:g}" for cost in costs], row=1, col=1,
        )
        fig.update_xaxes(
            title_text="Total round-trip cost (bps)", tickmode="array", tickvals=costs,
            ticktext=[f"{cost:g}" for cost in costs], row=1, col=2,
        )
        fig.update_yaxes(title_text="Missed-winner rate", row=1, col=1)
        fig.update_yaxes(title_text="Missed-winner rate", row=1, col=2)
        fig.update_layout(
            title="Paired execution stress over 100 opportunity paths",
            height=570, margin=dict(l=85, r=75, t=100, b=85),
        )
        fig.show()
        """,
    ),
    markdown(
        "drawdown-title",
        """
        ## 6. Exposure determines whether ordinary variance becomes account trauma

        Lines show median and 95th-percentile maximum drawdown on identical signal paths. At reference
        execution, 4.95x exposure reaches roughly 31% at the 95th percentile. At 15x it reaches about
        75%, with a 64% probability of losing at least half the account before recovery.
        """,
    ),
    code(
        "drawdown-lines",
        """
        reference = scenario(10.0, 0.0)
        hostile = scenario(15.0, 0.10)
        fig = go.Figure()
        for row, label, color, dash in (
            (reference, "Reference median", BLUE, "solid"),
            (reference, "Reference p95", BLUE, "dash"),
            (hostile, "15 bps + 10% misses median", BAD, "solid"),
            (hostile, "15 bps + 10% misses p95", BAD, "dash"),
        ):
            quantile = "median" if "median" in label else "p95"
            fig.add_trace(
                go.Scatter(
                    x=[item["exposure_multiple"] for item in row["exposures"]],
                    y=[item["max_drawdown_pct"][quantile] for item in row["exposures"]],
                    mode="lines+markers+text", name=label,
                    text=[f"{item['max_drawdown_pct'][quantile]:.0f}%" for item in row["exposures"]],
                    textposition="top center", line=dict(color=color, dash=dash, width=2), marker=dict(size=9),
                    hovertemplate="%{fullData.name}<br>Exposure %{x}x<br>Max drawdown %{y:.2f}%<extra></extra>",
                )
            )
        fig.add_hline(y=30, line_color=GOLD, line_dash="dot", annotation_text="30% drawdown")
        fig.add_hline(y=50, line_color=BAD, line_dash="dot", annotation_text="50% drawdown")
        fig.update_layout(
            title="Maximum drawdown versus effective account exposure",
            xaxis_title="Position notional / account equity", yaxis_title="Maximum drawdown (%)",
            height=620, margin=dict(l=80, r=55, t=85, b=145),
            legend=dict(orientation="h", y=-0.22),
        )
        fig.show()
        """,
    ),
    markdown(
        "terminal-title",
        """
        ## 7. Positive expectation does not make 15x a sensible canary

        Whiskers show the 5th to 95th percentile compounded return after 100 opportunities. High
        exposure magnifies upside, but path-dependent volatility and compounding make its downside
        increasingly nonlinear. A canary exists to validate plumbing and edge, not maximize return.
        """,
    ),
    code(
        "terminal-forest",
        """
        fig = go.Figure()
        for row, label, color, offset in (
            (reference, "Reference execution", BLUE, -0.12),
            (hostile, "15 bps + 10% misses", BAD, 0.12),
        ):
            x = [item["terminal_compounded_return_pct"]["median"] for item in row["exposures"]]
            low = [item["terminal_compounded_return_pct"]["p05"] for item in row["exposures"]]
            high = [item["terminal_compounded_return_pct"]["p95"] for item in row["exposures"]]
            y = [item["exposure_multiple"] + offset for item in row["exposures"]]
            fig.add_trace(
                go.Scatter(
                    x=x, y=y, mode="markers", name=label,
                    marker=dict(color=color, size=11, symbol="diamond"),
                    error_x=dict(
                        type="data", symmetric=False,
                        array=[upper - center for upper, center in zip(high, x)],
                        arrayminus=[center - lower for center, lower in zip(x, low)],
                        thickness=2, width=6, color=color,
                    ),
                    customdata=[[lower, upper] for lower, upper in zip(low, high)],
                    hovertemplate=(
                        "%{fullData.name}<br>Exposure %{y:.2f}x<br>Median %{x:+.1f}%"
                        "<br>5-95%: %{customdata[0]:+.1f}% to %{customdata[1]:+.1f}%<extra></extra>"
                    ),
                )
            )
        fig.add_vline(x=0, line_color=BAD, line_width=1)
        fig.update_layout(
            title="Compounded 100-opportunity return distribution",
            xaxis_title="Terminal account return (%)", yaxis_title="Effective exposure multiple",
            height=570, margin=dict(l=80, r=55, t=85, b=105),
            legend=dict(orientation="h", y=-0.15),
        )
        fig.update_yaxes(tickvals=study["parameters"]["stress_exposure_multiples"], ticksuffix="x")
        fig.show()
        """,
    ),
    markdown(
        "power-title",
        """
        ## 8. One hundred trades is a checkpoint, not the law of large numbers arriving

        With symmetric `+60/-60` barriers and 10 bps cost, break-even is 58.33%. If the true win rate
        is 65%, approximately 331 resolved trades are needed for 80% power at one-sided 5% alpha.
        At 100 trades, at least 67 targets are needed merely to put the one-sided 95% Wilson lower
        bound above break-even.
        """,
    ),
    code(
        "power-panels",
        """
        true_rows = power["true_rate_scenarios"]
        checkpoints = power["confidence_checkpoints"]
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Power-based sample size", "Observed win rate needed for confidence"))
        fig.add_trace(
            go.Bar(
                x=[f"{100 * row['true_win_rate']:.0f}%" for row in true_rows],
                y=[row["approximate_required_resolved_trades"] for row in true_rows],
                marker_color=[ORANGE, GOLD, GOOD, TEAL],
                text=[f"{row['approximate_required_resolved_trades']:,}" for row in true_rows],
                textposition="outside", showlegend=False,
                customdata=[[row["expected_net_bps_per_trade"]] for row in true_rows],
                hovertemplate="True win rate %{x}<br>Required n %{y:,}<br>Expected edge %{customdata[0]:+.1f} bps/trade<extra></extra>",
            ), row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[row["resolved_trades"] for row in checkpoints],
                y=[100 * row["minimum_observed_win_rate"] for row in checkpoints],
                mode="lines+markers+text", line=dict(color=PURPLE, width=2), marker=dict(size=10),
                text=[f"{100 * row['minimum_observed_win_rate']:.1f}%" for row in checkpoints],
                textposition="top center", showlegend=False,
                customdata=[[row["minimum_targets_for_one_sided_95pct_lower_bound_above_break_even"]] for row in checkpoints],
                hovertemplate="n=%{x}<br>Required rate %{y:.1f}%<br>Targets %{customdata[0]}<extra></extra>",
            ), row=1, col=2,
        )
        fig.add_hline(y=100 * power["break_even_win_rate"], line_color=BAD, line_dash="dash", row=1, col=2)
        fig.update_yaxes(
            type="log", title_text="Approximate resolved trades",
            tickmode="array", tickvals=[100, 300, 1000, 3000, 6000],
            ticktext=["100", "300", "1,000", "3,000", "6,000"], row=1, col=1,
        )
        fig.update_xaxes(title_text="Assumed true win rate", row=1, col=1)
        fig.update_xaxes(title_text="Resolved forward trades", row=1, col=2)
        fig.update_yaxes(title_text="Minimum observed win rate (%)", range=[57, 70], row=1, col=2)
        fig.update_layout(
            title="Evidence needed for the symmetric forward contract",
            height=570, margin=dict(l=80, r=45, t=100, b=85),
        )
        fig.show()
        """,
    ),
    markdown(
        "forward-title",
        """
        ## 9. Forward snapshot: encouraging, almost information-free

        At the frozen snapshot the squeeze arm had three targets and one stop, while its matched coin
        control was negative. But the 95% win-rate interval spans about 30% to 95%, and forward shorts
        were negative. The only scorecard gate ready to judge was feed health.
        """,
    ),
    code(
        "forward-panels",
        """
        strategy = scorecard["strategy_arm"]
        control = scorecard["matched_control_arm"]
        interval = scorecard["observed_wilson_95"]
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Forward win-rate uncertainty", "After-cost net bps"))
        observed = 100 * strategy["resolved_win_rate"]
        low = 100 * interval[0]
        high = 100 * interval[1]
        fig.add_trace(
            go.Scatter(
                x=[observed], y=["Squeeze"], mode="markers+text",
                text=[f"{observed:.1f}% (n={strategy['targets'] + strategy['stops']})"], textposition="top center",
                marker=dict(color=GOOD, size=15, symbol="diamond"),
                error_x=dict(type="data", symmetric=False, array=[high - observed], arrayminus=[observed - low], width=7, thickness=2),
                hovertemplate="Observed %{x:.1f}%<br>95% interval " + f"{low:.1f}% to {high:.1f}%<extra></extra>",
                showlegend=False,
            ), row=1, col=1,
        )
        fig.add_vline(x=100 * scorecard["break_even_win_rate"], line_color=BAD, line_dash="dash", row=1, col=1)
        fig.add_trace(
            go.Bar(
                x=["Squeeze", "Matched coin", "Squeeze long", "Squeeze short"],
                y=[
                    strategy["total_net_bps"], control["total_net_bps"],
                    strategy["by_side"]["long"]["net_bps"], strategy["by_side"]["short"]["net_bps"],
                ],
                marker_color=[GOOD, MUTED, BLUE, ORANGE],
                text=[
                    f"{strategy['total_net_bps']:+.1f}", f"{control['total_net_bps']:+.1f}",
                    f"{strategy['by_side']['long']['net_bps']:+.1f}", f"{strategy['by_side']['short']['net_bps']:+.1f}",
                ], textposition="auto", showlegend=False,
            ), row=1, col=2,
        )
        fig.add_hline(y=0, line_color=BAD, line_width=1, row=1, col=2)
        fig.update_xaxes(title_text="Resolved win rate (%)", range=[20, 100], row=1, col=1)
        fig.update_yaxes(title_text="Total net bps", row=1, col=2)
        fig.update_layout(
            title=f"AWS shadow arena snapshot at {scorecard['snapshot_at']}",
            height=520, margin=dict(l=75, r=45, t=100, b=100),
        )
        fig.show()
        """,
    ),
    markdown(
        "scorecard-title",
        """
        ## 10. The frozen promotion scorecard is mostly waiting for data

        Pending is not failure. It means the condition is deliberately withheld until its minimum
        sample is available. This prevents `3/4` from being celebrated and prevents an early losing
        streak from killing a potentially valid experiment.
        """,
    ),
    code(
        "scorecard-chart",
        """
        gates = scorecard["gates"]
        status_value = {"fail": -1, "pending": 0, "pass": 1}
        fig = go.Figure(
            go.Bar(
                x=[status_value[row["status"]] for row in gates],
                y=[pretty(row["gate"]) for row in gates], orientation="h",
                marker_color=[STATUS_COLORS[row["status"]] for row in gates],
                text=[row["status"].upper() for row in gates], textposition="outside",
                customdata=[[json.dumps(row["observed"]), json.dumps(row["required"])] for row in gates],
                hovertemplate="%{y}<br>Status %{text}<br>Observed %{customdata[0]}<br>Required %{customdata[1]}<extra></extra>",
            )
        )
        fig.add_vline(x=0, line_color=MUTED, line_width=1)
        fig.update_layout(
            title=f"Forward scorecard status: {pretty(scorecard['overall_status'])}",
            xaxis=dict(range=[-1.25, 1.45], tickvals=[-1, 0, 1], ticktext=["Fail", "Pending", "Pass"]),
            yaxis_title="Promotion gate", height=580, margin=dict(l=225, r=90, t=85, b=65),
        )
        fig.show()
        """,
    ),
    markdown(
        "verdict-title",
        """
        ## 11. Final research verdict
        """,
    ),
    code(
        "verdict",
        """
        ref = scenario(10.0, 0.0)
        hostile = scenario(15.0, 0.10)
        ref_495 = exposure(ref, 4.95)
        ref_15 = exposure(ref, 15.0)
        hostile_495 = exposure(hostile, 4.95)
        rate_65 = next(row for row in power["true_rate_scenarios"] if row["true_win_rate"] == 0.65)
        checkpoint_100 = next(row for row in power["confidence_checkpoints"] if row["resolved_trades"] == 100)
        display(
            HTML(
                f'''
                <div style="border-left: 5px solid {GOLD}; padding: 8px 18px; max-width: 1030px;">
                  <h3 style="margin-top: 0;">Promising research strategy, no live promotion yet</h3>
                  <ul>
                    <li><strong>Historical expectancy:</strong> {historical['reference']['average_net_bps_per_trade']:+.2f}
                        bps/trade at 10 bps cost and {historical['hostile']['average_net_bps_per_trade']:+.2f}
                        at 15 bps. Seven of eight chronological blocks were positive at reference cost.</li>
                    <li><strong>Directional robustness:</strong> longs averaged
                        {study['side_robustness']['long']['reference']['average_net_bps_per_trade']:+.2f}
                        bps; shorts only {study['side_robustness']['short']['reference']['average_net_bps_per_trade']:+.2f}.
                        Both are positive, but the short cushion is thin.</li>
                    <li><strong>4.95x exposure:</strong> reference 95th-percentile drawdown is
                        {ref_495['max_drawdown_pct']['p95']:.1f}%. Under 15 bps cost plus 10% missed winners it rises to
                        {hostile_495['max_drawdown_pct']['p95']:.1f}%, and the 5th-percentile terminal result is
                        {hostile_495['terminal_compounded_return_pct']['p05']:+.1f}%.</li>
                    <li><strong>15x exposure is not justified:</strong> reference median / p95 drawdown is
                        {ref_15['max_drawdown_pct']['median']:.1f}% / {ref_15['max_drawdown_pct']['p95']:.1f}%, with a
                        {100 * ref_15['probability_drawdown_gte_50pct']:.1f}% chance of a 50% drawdown.</li>
                    <li><strong>Forward evidence:</strong> squeeze is {scorecard['strategy_arm']['targets']}/
                        {scorecard['strategy_arm']['targets'] + scorecard['strategy_arm']['stops']} targets and stops,
                        {scorecard['strategy_arm']['total_net_bps']:+.1f} net bps, versus matched coin
                        {scorecard['matched_control_arm']['total_net_bps']:+.1f}. At n=4 this is nearly information-free.</li>
                    <li><strong>Law-of-large-numbers reality:</strong> a true 65% win rate needs about
                        {rate_65['approximate_required_resolved_trades']} resolved trades for 80% power. At n=100 the
                        one-sided confidence checkpoint needs at least {checkpoint_100['minimum_targets_for_one_sided_95pct_lower_bound_above_break_even']}
                        targets, not merely 59.</li>
                    <li><strong>Engineering action:</strong> keep the AWS arena running unchanged and aggregate future
                        independent runs under this scorecard. Do not launch a live canary yet. If every scorecard gate
                        eventually passes, review a first canary near <strong>2x effective exposure</strong>, not 15x.</li>
                  </ul>
                </div>
                '''
            )
        )
        """,
    ),
    markdown(
        "close-title",
        """
        ## 12. The seven-proposal chapter is complete

        The local archive has now been used for:

        1. market-behavior mapping;
        2. deterministic strategy-family competition;
        3. exit-policy testing;
        4. regime permission analysis;
        5. individual feature-value analysis;
        6. ensemble calibration;
        7. robustness and promotion planning.

        The honest outcome is narrower than "we solved trading," but much more useful than another
        tuned backtest: the ungated squeeze remains a plausible hypothesis, feature gates did not add
        stable value, execution asymmetry can erase the edge, and high exposure can turn a positive
        strategy into an intolerable account path.

        From here, the research loop changes. **Freeze the strategy and collect independent forward
        trades.** Reopen model development only after materially new regimes and enough sample size
        exist, not because four fresh trades feel exciting.
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
