import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "notebooks" / "08_ensemble_calibration_lab.ipynb"


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
        # Ensemble calibration lab: can weak clues become a useful quality gate?

        Proposal 6 does **not** invent a new direction. It keeps the same side-neutral squeeze entry
        and fixed `100/150/6h` reference exit, then asks whether a tiny ensemble can identify which
        already-chosen trades are worth taking.

        Four Proposal 5 inputs and their orientations are frozen before this study. Discovery fits
        models with expanding chronological validation. Calibration may choose one acceptance
        threshold. The previously opened historical holdout is audit-only, and fresh forward shadow
        evidence is mandatory before any strategy change.
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
        REPORT_PATH = DATA_ROOT / "ensemble-lab-local.json"
        study = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        candidates = study["candidates"]
        candidate_by_name = {row["name"]: row for row in candidates}
        baseline = study["baseline"]

        GOOD = "#2A9D8F"
        BAD = "#E76F51"
        BLUE = "#457B9D"
        GOLD = "#E9C46A"
        MUTED = "#8D99AE"
        PURPLE = "#7B2CBF"
        TEAL = "#1D7874"
        ORANGE = "#F4A261"
        CANDIDATE_COLORS = {
            "single_aggression_benchmark": MUTED,
            "equal_weight_core": BLUE,
            "consensus_vote_core": GOLD,
            "ridge_logistic_core": PURPLE,
        }
        PERIOD_COLORS = {"discovery": BLUE, "calibration": GOLD, "holdout": GOOD}
        SHORT_NAMES = {
            "single_aggression_benchmark": "Single aggression",
            "equal_weight_core": "Equal weight",
            "consensus_vote_core": "Consensus vote",
            "ridge_logistic_core": "Ridge logistic",
        }

        def pretty(value):
            return value.replace("_", " ").title()

        print(f"Loaded: {REPORT_PATH}")
        print(f"Coverage: {study['data']['mark_price_start']} to {study['data']['mark_price_end']}")
        print(f"Dense mark-price points: {study['data']['unique_mark_price_points']:,}")
        print("Locked cohorts: " + ", ".join(f"{name}={row['locked_entries']}" for name, row in study['entry_cohorts'].items()))
        print(f"Frozen ensemble features: {len(study['frozen_shortlist'])}")
        print(f"Eligible ensemble candidates: {sum(row['selection_eligible'] for row in candidates)}")
        print(f"Qualified pre-holdout candidates: {sum(row['selection_qualified'] for row in candidates)}")
        print(f"Forward-shadow candidates: {len(study['forward_shadow_candidates'])}")
        """,
    ),
    markdown(
        "contract-title",
        """
        ## 1. What the ensemble is allowed to do

        The squeeze strategy still chooses long or short. The ensemble sees only side-canonical
        context and may say **take** or **skip**. It cannot reverse the trade, prefer a permanent side,
        change the exit, or learn from the historical audit period.
        """,
    ),
    code(
        "contract-flow",
        """
        labels = [
            ("Squeeze signal", "Chooses long or short"),
            ("Four frozen clues", "Mirrored by chosen side"),
            ("Quality ensemble", "Estimates positive trade"),
            ("Probability gate", "Take or skip"),
            ("Fixed exit", "100 / 150 bps / 6h"),
            ("Fresh shadow", "Required for promotion"),
        ]
        colors = [BLUE, TEAL, PURPLE, GOLD, ORANGE, GOOD]
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
            title="A quality gate around an unchanged side-neutral strategy",
            xaxis=dict(visible=False, range=[-0.45, 5.45]),
            yaxis=dict(visible=False, range=[-0.82, 0.35]),
            height=310, margin=dict(l=40, r=40, t=75, b=35),
        )
        fig.show()
        """,
    ),
    markdown(
        "features-title",
        """
        ## 2. The inputs were frozen, including their counterintuitive signs

        Proposal 5 found that lower chosen-side aggression, lower chosen-side taker ratio, and lower
        HTF return alignment looked better in discovery. Those signs are not re-optimized here.
        Funding carry kept its intuitive positive orientation. None passed Proposal 5 FDR, so the
        ensemble starts as an explicitly exploratory hypothesis.
        """,
    ),
    code(
        "feature-orientation",
        """
        frozen = study["frozen_shortlist"]
        labels = [pretty(row["feature"]) for row in frozen]
        values = [row["orientation"] for row in frozen]
        fig = go.Figure(
            go.Bar(
                x=values, y=labels, orientation="h",
                marker_color=[GOOD if value > 0 else ORANGE for value in values],
                text=["Higher raw value favored" if value > 0 else "Lower raw value favored" for value in values],
                textposition="outside",
                customdata=[[
                    row["discovery_median_oriented"], row["discovery_robust_scale"], row["discovery_coverage"]
                ] for row in frozen],
                hovertemplate=(
                    "%{y}<br>%{text}<br>Oriented median: %{customdata[0]:.4f}"
                    "<br>Robust scale: %{customdata[1]:.4f}<br>Coverage: %{customdata[2]:.0%}<extra></extra>"
                ),
            )
        )
        fig.add_vline(x=0, line_color=MUTED, line_width=1)
        fig.update_layout(
            title="Proposal 5 orientations carried forward unchanged",
            xaxis=dict(title="Frozen orientation", range=[-1.15, 1.95], tickvals=[-1, 1], ticktext=["Lower better", "Higher better"]),
            yaxis_title="Feature", height=470, margin=dict(l=245, r=180, t=80, b=75),
        )
        fig.show()
        """,
    ),
    markdown(
        "chronology-title",
        """
        ## 3. Discovery predictions are genuinely forward within discovery

        The first 16 locked trades seed the model. Each later eight-trade block is predicted using
        only earlier trades. These 24 out-of-fold predictions estimate discovery behavior. The final
        discovery fit then scores calibration, where thresholds are judged independently.
        """,
    ),
    code(
        "chronology-heatmap",
        """
        splits = study["validation_contract"]["expanding_discovery_splits"]
        total = study["entry_cohorts"]["discovery"]["locked_entries"]
        matrix = []
        text = []
        labels = []
        for index, split in enumerate(splits, start=1):
            row = []
            row_text = []
            for trade_index in range(total):
                if split["train"][0] <= trade_index <= split["train"][1]:
                    row.append(1)
                    row_text.append("Train")
                elif split["validate"][0] <= trade_index <= split["validate"][1]:
                    row.append(2)
                    row_text.append("Validate")
                else:
                    row.append(0)
                    row_text.append("Future")
            matrix.append(row)
            text.append(row_text)
            labels.append(f"Fold {index}")
        fig = go.Figure(
            go.Heatmap(
                x=list(range(1, total + 1)), y=labels, z=matrix, text=text,
                colorscale=[[0, "#F1F3F5"], [0.49, "#F1F3F5"], [0.50, BLUE], [0.74, BLUE], [0.75, GOLD], [1, GOLD]],
                zmin=0, zmax=2, showscale=False,
                hovertemplate="%{y}<br>Discovery trade %{x}<br>%{text}<extra></extra>",
            )
        )
        fig.update_layout(
            title="Expanding chronological validation inside discovery",
            xaxis_title="Locked discovery trade number", yaxis_title="Validation fold",
            height=360, margin=dict(l=95, r=45, t=80, b=70),
        )
        fig.show()
        """,
    ),
    markdown(
        "baseline-title",
        """
        ## 4. The ungated squeeze baseline is already the hard benchmark

        The fixed `100/150` barrier has a simple resolved-trade break-even rate near 64% after 10 bps
        cost, although six-hour timeouts make realized expectancy the final judge. Baseline average
        net bps stayed positive in discovery, calibration, and holdout. A gate must improve this,
        not merely inherit it.
        """,
    ),
    code(
        "baseline-panels",
        """
        periods = ["discovery", "calibration", "holdout"]
        fig = make_subplots(rows=1, cols=2, subplot_titles=("After-cost expectancy", "Positive-trade rate"))
        fig.add_trace(
            go.Bar(
                x=[pretty(period) for period in periods],
                y=[baseline[period]["average_net_bps_per_trade"] for period in periods],
                marker_color=[PERIOD_COLORS[period] for period in periods],
                text=[f"{baseline[period]['average_net_bps_per_trade']:+.1f}" for period in periods],
                textposition="auto", showlegend=False,
            ), row=1, col=1,
        )
        fig.add_trace(
            go.Bar(
                x=[pretty(period) for period in periods],
                y=[100 * baseline[period]["positive_net_trade_rate"] for period in periods],
                marker_color=[PERIOD_COLORS[period] for period in periods],
                text=[f"{100 * baseline[period]['positive_net_trade_rate']:.1f}%" for period in periods],
                textposition="auto", showlegend=False,
            ), row=1, col=2,
        )
        fig.add_hline(y=0, line_color=BAD, line_width=1, row=1, col=1)
        fig.add_hline(
            y=64, line_color=BAD, line_dash="dash", row=1, col=2,
            annotation_text="Approx. 64% resolved-trade break-even", annotation_position="top left",
        )
        fig.update_yaxes(title_text="Average net bps/trade", row=1, col=1)
        fig.update_yaxes(title_text="Positive trades (%)", range=[0, 82], row=1, col=2)
        fig.update_layout(
            title="Unfiltered squeeze entries under the locked exit",
            height=510, margin=dict(l=75, r=45, t=100, b=75),
        )
        fig.show()
        """,
    ),
    markdown(
        "quality-title",
        """
        ## 5. None of the models ranked winners reliably

        AUC `0.50` is random ordering. Positive Brier skill means probabilities improve on always
        forecasting the discovery win rate. Every candidate failed one or both tests on calibration;
        ridge was actively inverted there with AUC `0.247`.
        """,
    ),
    code(
        "quality-panels",
        """
        periods = ["discovery", "calibration", "holdout"]
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Ranking quality", "Probability skill"))
        for candidate in candidates:
            color = CANDIDATE_COLORS[candidate["name"]]
            fig.add_trace(
                go.Scatter(
                    x=[pretty(period) for period in periods],
                    y=[candidate[f"{period}_quality"]["auc"] for period in periods],
                    mode="lines+markers", name=SHORT_NAMES[candidate["name"]],
                    line=dict(color=color, width=2), marker=dict(size=9),
                    hovertemplate="%{fullData.name}<br>%{x}<br>AUC %{y:.3f}<extra></extra>",
                ), row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=[pretty(period) for period in periods],
                    y=[candidate[f"{period}_quality"]["brier_skill_vs_discovery_base"] for period in periods],
                    mode="lines+markers", name=SHORT_NAMES[candidate["name"]], showlegend=False,
                    line=dict(color=color, width=2), marker=dict(size=9),
                    hovertemplate="%{fullData.name}<br>%{x}<br>Brier skill %{y:+.3f}<extra></extra>",
                ), row=1, col=2,
            )
        fig.add_hline(y=0.50, line_color=BAD, line_dash="dash", row=1, col=1)
        fig.add_hline(y=0, line_color=BAD, line_dash="dash", row=1, col=2)
        fig.update_yaxes(title_text="AUC", range=[0.15, 0.68], row=1, col=1)
        fig.update_yaxes(title_text="Brier skill vs discovery base", row=1, col=2)
        fig.update_layout(
            title="Probability models through chronological periods",
            height=560, margin=dict(l=75, r=45, t=100, b=125),
            legend=dict(orientation="h", y=-0.20, x=0.5, xanchor="center"),
        )
        fig.show()
        """,
    ),
    markdown(
        "collapse-title",
        """
        ## 6. Transparent score relationships disappeared through time

        The monotonic calibrator is forbidden from reversing a frozen orientation. It may use a
        positive slope or admit that the score has no information by choosing zero. All three
        transparent scores ended at zero after later discovery folds contradicted the first fold.
        """,
    ),
    code(
        "slope-collapse",
        """
        transparent = [candidate for candidate in candidates if candidate["kind"] != "ridge"]
        fig = go.Figure()
        for candidate in transparent:
            folds = candidate["model"]["expanding_fold_score_calibrators"]
            values = [row["nonnegative_slope"] for row in folds]
            values.append(candidate["model"]["final_score_calibrator"]["nonnegative_slope"])
            fig.add_trace(
                go.Scatter(
                    x=["Fold 1", "Fold 2", "Fold 3", "Final discovery fit"], y=values,
                    mode="lines+markers+text", name=SHORT_NAMES[candidate["name"]],
                    text=[f"{value:.2f}" for value in values], textposition="top center",
                    line=dict(color=CANDIDATE_COLORS[candidate["name"]], width=2), marker=dict(size=10),
                    hovertemplate="%{fullData.name}<br>%{x}<br>Nonnegative slope %{y:.2f}<extra></extra>",
                )
            )
        fig.add_hline(y=0, line_color=BAD, line_width=1)
        fig.update_layout(
            title="Frozen-score calibration slope across expanding folds",
            xaxis_title="Discovery fit", yaxis_title="Calibrator slope",
            height=520, margin=dict(l=75, r=45, t=85, b=115),
            legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
        )
        fig.show()
        """,
    ),
    markdown(
        "ridge-title",
        """
        ## 7. Ridge chose near-total shrinkage

        Stronger shrinkage won expanding discovery validation. `C=0.01` leaves every standardized
        coefficient close to zero. The model is effectively saying that a fitted combination is less
        trustworthy than the discovery base rate.
        """,
    ),
    code(
        "ridge-panels",
        """
        ridge = candidate_by_name["ridge_logistic_core"]
        grid = ridge["model"]["regularization_grid"]
        coefficients = ridge["model"]["standardized_coefficients"]
        coefficient_labels = {
            "signed_aggression_15s": "Aggression 15s",
            "signed_taker_ratio_10s": "Taker ratio 10s",
            "htf_return_alignment_bps": "HTF return",
            "favorable_funding_carry_bps": "Funding carry",
        }
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Regularization choice", "Final standardized weights"))
        fig.add_trace(
            go.Scatter(
                x=[row["c"] for row in grid], y=[row["oof_log_loss"] for row in grid],
                mode="lines+markers+text", text=[f"{row['oof_log_loss']:.3f}" for row in grid],
                textposition="top center", line=dict(color=PURPLE, width=2), marker=dict(size=10),
                hovertemplate="C=%{x}<br>OOF log loss %{y:.4f}<extra></extra>", showlegend=False,
            ), row=1, col=1,
        )
        fig.add_trace(
            go.Bar(
                x=[row["coefficient"] for row in coefficients],
                y=[coefficient_labels[row["feature"]] for row in coefficients], orientation="h",
                marker_color=[GOOD if row["coefficient"] > 0 else BAD for row in coefficients],
                text=[f"{row['coefficient']:+.4f}" for row in coefficients], textposition="auto",
                hovertemplate="%{y}<br>Coefficient %{x:+.5f}<extra></extra>", showlegend=False,
            ), row=1, col=2,
        )
        fig.update_xaxes(
            type="log", title_text="Logistic C (smaller = more shrinkage)",
            tickvals=[0.01, 0.03, 0.10, 0.30, 1.0], ticktext=["0.01", "0.03", "0.10", "0.30", "1.0"],
            row=1, col=1,
        )
        fig.update_yaxes(title_text="Expanding OOF log loss", row=1, col=1)
        fig.update_xaxes(title_text="Coefficient", range=[-0.04, 0.04], row=1, col=2)
        fig.update_layout(
            title=f"Ridge selected C={ridge['model']['selected_c']}",
            height=560, margin=dict(l=75, r=80, t=100, b=90),
        )
        fig.show()
        """,
    ),
    markdown(
        "threshold-title",
        """
        ## 8. No fixed threshold improved both development periods

        Every point is a probability threshold frozen before the audit. A useful gate belongs in the
        upper-right quadrant: accepted trades improve on taking every squeeze in both discovery OOF
        and calibration. None qualified. Points on zero simply accepted every trade.
        """,
    ),
    code(
        "threshold-landscape",
        """
        fig = go.Figure()
        for candidate in candidates:
            rows = candidate["threshold_candidates"]
            fig.add_trace(
                go.Scatter(
                    x=[row["discovery"]["accepted_delta_vs_all_bps"] for row in rows],
                    y=[row["calibration"]["accepted_delta_vs_all_bps"] for row in rows],
                    mode="markers", name=SHORT_NAMES[candidate["name"]],
                    marker=dict(
                        size=[10 + row["calibration"]["accepted"]["trades"] / 2 for row in rows],
                        color=CANDIDATE_COLORS[candidate["name"]],
                        symbol=["diamond" if row["threshold"] == candidate["selected_threshold"] else "circle" for row in rows],
                    ),
                    customdata=[[
                        row["threshold"], row["discovery"]["accepted"]["trades"],
                        row["calibration"]["accepted"]["trades"], row["basic_qualified"]
                    ] for row in rows],
                    hovertemplate=(
                        "%{fullData.name}<br>Threshold %{customdata[0]:.2f}<br>Discovery improvement %{x:+.2f} bps"
                        "<br>Calibration improvement %{y:+.2f} bps"
                        "<br>Accepted n: %{customdata[1]} / %{customdata[2]}"
                        "<br>Basic qualified: %{customdata[3]}<extra></extra>"
                    ),
                )
            )
        fig.add_hline(y=0, line_color=BAD, line_width=1)
        fig.add_vline(x=0, line_color=BAD, line_width=1)
        fig.add_annotation(
            x=4, y=4, text="Required quadrant", showarrow=False,
            bgcolor="rgba(42,157,143,0.15)", bordercolor=GOOD,
        )
        fig.update_layout(
            title="Threshold improvement before historical audit",
            xaxis_title="Discovery OOF accepted improvement (net bps/trade)",
            yaxis_title="Calibration accepted improvement (net bps/trade)",
            height=680, margin=dict(l=85, r=45, t=85, b=130),
            legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
        )
        fig.show()
        """,
    ),
    markdown(
        "gate-title",
        """
        ## 9. The selected gates mostly became "take everything"

        Acceptance fractions reveal the failure directly. On calibration, every selected threshold
        accepted all 18 trades. Ridge also accepted all 24 discovery OOF rows. A filter that passes
        everything cannot claim the baseline's profit as ensemble value.
        """,
    ),
    code(
        "gate-panels",
        """
        periods = ["discovery", "calibration", "holdout"]
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Accepted opportunity fraction", "Accepted improvement vs all"))
        for candidate in candidates:
            color = CANDIDATE_COLORS[candidate["name"]]
            gates = [
                candidate["discovery_selected_gate"],
                candidate["calibration_selected_gate"],
                candidate["holdout_selected_gate"],
            ]
            fig.add_trace(
                go.Bar(
                    x=[pretty(period) for period in periods],
                    y=[100 * gate["accepted_fraction"] for gate in gates],
                    name=SHORT_NAMES[candidate["name"]], marker_color=color,
                    hovertemplate="%{fullData.name}<br>%{x}<br>Accepted %{y:.1f}%<extra></extra>",
                ), row=1, col=1,
            )
            fig.add_trace(
                go.Bar(
                    x=[pretty(period) for period in periods],
                    y=[gate["accepted_delta_vs_all_bps"] for gate in gates],
                    name=SHORT_NAMES[candidate["name"]], marker_color=color, showlegend=False,
                    hovertemplate="%{fullData.name}<br>%{x}<br>Improvement %{y:+.2f} bps<extra></extra>",
                ), row=1, col=2,
            )
        fig.add_hline(y=0, line_color=BAD, line_width=1, row=1, col=2)
        fig.update_yaxes(title_text="Accepted (%)", range=[0, 112], row=1, col=1)
        fig.update_yaxes(title_text="Net bps/trade versus all", row=1, col=2)
        fig.update_layout(
            title="Frozen selected threshold behavior",
            barmode="group", height=570, margin=dict(l=75, r=45, t=100, b=135),
            legend=dict(orientation="h", y=-0.20, x=0.5, xanchor="center"),
        )
        fig.show()
        """,
    ),
    markdown(
        "calibration-title",
        """
        ## 10. Confidence did not track realized accuracy

        Perfect calibration lies on the diagonal. Circles are calibration and diamonds are the
        historical holdout audit. Transparent ensembles issued the same 65%
        discovery-base forecast for calibration, while actual positive-trade rate was 61.1%. Ridge
        varied probabilities but ranked outcomes backward. Small calibration error from a constant
        forecast is not useful selectivity.
        """,
    ),
    code(
        "calibration-scatter",
        """
        fig = go.Figure()
        for candidate in candidates:
            for period, symbol in (("calibration", "circle"), ("holdout", "diamond")):
                quality = candidate[f"{period}_quality"]
                fig.add_trace(
                    go.Scatter(
                        x=[100 * quality["average_probability"]],
                        y=[100 * quality["positive_rate"]],
                        mode="markers", name=SHORT_NAMES[candidate["name"]],
                        legendgroup=candidate["name"], showlegend=period == "calibration",
                        marker=dict(
                            size=14, color=CANDIDATE_COLORS[candidate["name"]], symbol=symbol,
                            line=dict(width=1, color="#333"),
                        ),
                        customdata=[[pretty(period), quality["auc"], quality["brier_skill_vs_discovery_base"], quality["expected_calibration_error"]]],
                        hovertemplate=(
                            "%{fullData.name} | %{customdata[0]}<br>Mean forecast %{x:.1f}%<br>Actual positive rate %{y:.1f}%"
                            "<br>AUC %{customdata[1]:.3f}<br>Brier skill %{customdata[2]:+.3f}"
                            "<br>ECE %{customdata[3]:.3f}<extra></extra>"
                        ),
                    )
                )
        fig.add_trace(
            go.Scatter(
                x=[45, 75], y=[45, 75], mode="lines", name="Perfect average calibration",
                line=dict(color=MUTED, dash="dash"),
            )
        )
        fig.update_layout(
            title="Average forecast versus realized positive-trade rate",
            xaxis_title="Average predicted probability (%)",
            yaxis_title="Actual positive-trade rate (%)",
            height=620, margin=dict(l=80, r=45, t=85, b=170),
            legend=dict(orientation="h", y=-0.28),
        )
        fig.show()
        """,
    ),
    markdown(
        "coin-title",
        """
        ## 11. Matched-coin performance belongs to the squeeze entry, not the gate

        Because selected gates accepted every historical holdout trade, their diamonds equal the
        ungated squeeze result. Beating some random-direction controls here supports the underlying
        squeeze direction at those timestamps; it provides zero evidence that the ensemble filtered
        opportunity quality.
        """,
    ),
    code(
        "coin-controls",
        """
        centers = [row["matched_coin_control"]["median_average_net_bps"] for row in candidates]
        lows = [row["matched_coin_control"]["p05_average_net_bps"] for row in candidates]
        highs = [row["matched_coin_control"]["p95_average_net_bps"] for row in candidates]
        strategy = [row["holdout_selected_gate"]["accepted"]["average_net_bps_per_trade"] for row in candidates]
        labels = [SHORT_NAMES[row["name"]] for row in candidates]
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
                hovertemplate="%{y}<br>Coin median %{x:+.2f} bps<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=strategy, y=labels, mode="markers+text", name="Accepted squeeze directions",
                text=[f"{value:+.1f}" for value in strategy], textposition="middle right",
                marker=dict(
                    color=[CANDIDATE_COLORS[row["name"]] for row in candidates],
                    size=14, symbol="diamond",
                ),
                hovertemplate="%{y}<br>Accepted squeeze %{x:+.2f} bps/trade<extra></extra>",
            )
        )
        fig.add_vline(x=0, line_color=BAD, line_width=1)
        fig.update_layout(
            title="Historical accepted trades versus matched random directions",
            xaxis_title="Average net bps per trade", height=510,
            margin=dict(l=175, r=65, t=85, b=100), legend=dict(orientation="h", y=-0.16),
        )
        fig.show()
        """,
    ),
    markdown(
        "verdict-title",
        """
        ## 12. Ensemble-calibration verdict
        """,
    ),
    code(
        "verdict",
        """
        equal = candidate_by_name["equal_weight_core"]
        vote = candidate_by_name["consensus_vote_core"]
        ridge = candidate_by_name["ridge_logistic_core"]
        display(
            HTML(
                f'''
                <div style="border-left: 5px solid {GOLD}; padding: 8px 18px; max-width: 1000px;">
                  <h3 style="margin-top: 0;">No ensemble quality gate is ready</h3>
                  <ul>
                    <li><strong>The ungated squeeze baseline remained positive:</strong>
                        {baseline['discovery']['average_net_bps_per_trade']:+.2f},
                        {baseline['calibration']['average_net_bps_per_trade']:+.2f}, and
                        {baseline['holdout']['average_net_bps_per_trade']:+.2f} net bps/trade.</li>
                    <li><strong>Equal weight</strong> and <strong>consensus vote</strong> lost their
                        score relationship in later discovery folds. Final monotonic slope was zero,
                        calibration AUC was {equal['calibration_quality']['auc']:.3f}, and every
                        calibration trade was accepted.</li>
                    <li><strong>Ridge logistic</strong> chose C={ridge['model']['selected_c']}, shrinking
                        coefficients nearly to zero. Its discovery OOF AUC was
                        {ridge['discovery_quality']['auc']:.3f} and calibration AUC fell to
                        <strong>{ridge['calibration_quality']['auc']:.3f}</strong>.</li>
                    <li>No threshold improved expectancy in both discovery OOF and calibration.
                        Bootstrap support stayed below 80% for every candidate.</li>
                    <li>The selected diagnostic fallback was <code>{study['selected_preholdout_candidate']['name']}</code>
                        at probability {study['selected_preholdout_candidate']['selected_threshold']:.2f}; it accepted
                        everything and added exactly {study['selected_preholdout_candidate']['robust_preholdout_score_bps']:+.2f}
                        bps of robust pre-holdout value.</li>
                    <li><strong>Engineering action:</strong> add no ensemble, probability gate, or
                        feature weights to strategy code. The AWS shadow arena remains unchanged.</li>
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
        ## 13. What proposal 6 hands to proposal 7

        Proposal 6 closes the model-fitting branch for this archive. More combinations of the same
        94 locked trades would be optimization theater.

        Proposal 7 should be the **robustness and promotion laboratory** for the strategy that remains:

        - stress the ungated squeeze across chronological bootstrap paths and regime blocks;
        - inject worse fees, slippage, and missed fills;
        - map drawdown and losing-streak distributions at realistic account exposure;
        - measure how many fresh forward trades are needed to distinguish the observed edge from luck;
        - combine the live 48-hour arena with future independent runs under a frozen scorecard.

        That final proposal should answer a practical question: not "can we fit a prettier model?",
        but **what evidence and risk budget would justify a tiny live canary, if any?**
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
