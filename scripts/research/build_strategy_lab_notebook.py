import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "notebooks" / "02_symmetric_strategy_lab.ipynb"


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
        # Symmetric first-touch strategy lab

        This chapter keeps the experiment clean: every strategy predicts which **symmetric
        `+60 bps` or `-60 bps` barrier** comes first. The stop is not widened to manufacture a
        higher win rate.

        Data is split chronologically into **fit**, **calibration**, and **final holdout** periods,
        with a six-hour purge between them. Green bars are hypotheses worth retesting, not trophies.
        """,
    ),
    code(
        "setup",
        r"""
        import json
        import os
        from pathlib import Path

        import polars as pl
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
        REPORT_PATH = DATA_ROOT / "strategy-lab-local.json"
        study = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

        GOOD = "#2A9D8F"
        BAD = "#E76F51"
        NEUTRAL = "#457B9D"
        MUTED = "#8D99AE"
        GOLD = "#E9C46A"
        PERIODS = ["fit", "calibration", "holdout"]

        print(f"Loaded: {REPORT_PATH}")
        print(f"Coverage: {study['data']['mark_price_start']} to {study['data']['mark_price_end']}")
        print(f"Symmetric barrier: +/-{study['parameters']['barrier_bps']:.0f} bps")
        print(f"Fee-adjusted break-even: {study['economics']['fee_adjusted_break_even_win_rate']:.2%}")
        """,
    ),
    markdown(
        "split-title",
        """
        ## 1. The three locked rooms

        The model first learns from old data, then calibrates its probabilities on a later period,
        and is judged only on the newest holdout. The six-hour gaps prevent labels near a boundary
        from seeing prices on the other side.
        """,
    ),
    code(
        "split-chart",
        """
        split_counts = {
            "Fit after purge": study["data"]["fit_samples_after_purge"],
            "Calibration": study["data"]["calibration_samples"],
            "Final holdout": study["data"]["holdout_samples"],
        }
        fig = go.Figure(
            go.Bar(
                x=list(split_counts.keys()),
                y=list(split_counts.values()),
                marker_color=[NEUTRAL, GOLD, GOOD],
                text=[f"{value:,}" for value in split_counts.values()],
                textposition="auto",
            )
        )
        fig.update_layout(
            title="Chronological sample split",
            yaxis_title="Labelable minute samples",
            height=380,
            margin=dict(l=70, r=40, t=60, b=60),
        )
        fig.show()
        """,
    ),
    markdown(
        "coverage-title",
        """
        ## 2. Which instruments are actually plugged in?

        Features below the 80% coverage line are excluded. This prevents missing BTC or liquidation
        fields from acting as fingerprints for the historical software version that produced a row.
        """,
    ),
    code(
        "coverage-chart",
        """
        coverage = pl.DataFrame(study["feature_selection"]["coverage"])
        coverage = coverage.with_columns(
            pl.min_horizontal("fit_coverage", "calibration_coverage", "holdout_coverage").alias("minimum_coverage")
        ).sort("minimum_coverage")
        fig = go.Figure(
            go.Bar(
                x=coverage["minimum_coverage"] * 100,
                y=coverage["feature"],
                orientation="h",
                marker_color=[GOOD if selected else MUTED for selected in coverage["selected"]],
                text=[f"{value * 100:.0f}%" for value in coverage["minimum_coverage"]],
                textposition="auto",
            )
        )
        fig.add_vline(x=80, line_dash="dash", line_color=BAD)
        fig.update_layout(
            title="Minimum feature coverage across all three periods",
            xaxis_title="Coverage (%)",
            height=760,
            margin=dict(l=230, r=40, t=60, b=60),
        )
        fig.show()
        """,
    ),
    markdown(
        "strategy-guide",
        """
        ## 3. What the five new strategies are trying to see

        - **Micro persistence:** several 1s, 5s, and 15s flow measurements must point the same way.
          A single excited update is not enough.
        - **Impulse pullback:** price makes a clear move, briefly pauses or retraces, while order flow
          still supports continuation in the original direction.
        - **Flow exhaustion reversal:** aggressive traders keep pushing one way, but the order book
          absorbs them and pressure flips. The strategy predicts a reversal.
        - **Volatility squeeze breakout:** the recent range is compressed, price reaches one edge,
          and persistent microstructure pressure agrees with a breakout through that edge.
        - **Strategy-family router:** exhaustion gets first priority, then pullback, squeeze, and
          persistent flow. It is deliberately deterministic rather than LLM-directed.

        Every one of these emits only **long**, **short**, or **no trade**. All are judged with the
        same symmetric `+60/-60 bps` barriers.
        """,
    ),
    markdown(
        "period-title",
        """
        ## 4. Did a strategy work through time?

        Each cell is average net bps per sequential trade after modeled costs. A strategy that turns
        green only in the final column may be regime-specific or lucky; it is not yet generalizable.
        """,
    ),
    code(
        "period-heatmap",
        """
        period_rows = []
        for strategy, periods in study["deterministic_strategy_periods"].items():
            row = {"strategy": strategy.replace("_", " ")}
            row.update({period: periods[period]["average_net_bps_per_trade"] for period in PERIODS})
            period_rows.append(row)
        period_df = pl.DataFrame(period_rows).sort("holdout", descending=True)
        z = [period_df[period].to_list() for period in PERIODS]
        z = list(map(list, zip(*z)))
        limit = max(abs(value) for row in z for value in row)
        fig = go.Figure(
            go.Heatmap(
                z=z,
                x=["Fit", "Calibration", "Holdout"],
                y=period_df["strategy"].to_list(),
                colorscale="RdYlGn",
                zmid=0,
                zmin=-limit,
                zmax=limit,
                text=[[f"{value:+.1f}" for value in row] for row in z],
                texttemplate="%{text}",
                colorbar_title="Net bps/trade",
            )
        )
        fig.update_layout(
            title="Strategy stability across chronological periods",
            height=610,
            margin=dict(l=275, r=60, t=60, b=60),
        )
        fig.show()
        """,
    ),
    markdown(
        "holdout-title",
        """
        ## 5. The final holdout scoreboard

        The dashed gray line is random direction accuracy. The red line is the higher threshold
        required to pay the modeled round-trip cost with symmetric barriers.
        """,
    ),
    code(
        "holdout-win-chart",
        """
        holdout_rows = []
        for strategy, metrics in study["holdout_deterministic_strategies"].items():
            holdout_rows.append(
                {
                    "strategy": strategy.replace("_", " "),
                    "win_rate": metrics["barrier_win_rate"] * 100,
                    "trades": metrics["sequential_trades"],
                    "net_bps": metrics["average_net_bps_per_trade"],
                }
            )
        holdout_df = pl.DataFrame(holdout_rows).sort("win_rate")
        break_even = study["economics"]["fee_adjusted_break_even_win_rate"] * 100
        fig = go.Figure(
            go.Bar(
                x=holdout_df["win_rate"],
                y=holdout_df["strategy"],
                orientation="h",
                marker_color=[GOOD if value >= break_even else BAD for value in holdout_df["win_rate"]],
                text=[f"{value:.1f}%" for value in holdout_df["win_rate"]],
                textposition="auto",
                customdata=holdout_df["trades"],
                hovertemplate="%{y}<br>Win rate: %{x:.2f}%<br>Sequential trades: %{customdata}<extra></extra>",
            )
        )
        fig.add_vline(x=50, line_dash="dot", line_color=MUTED)
        fig.add_vline(x=break_even, line_dash="dash", line_color=BAD)
        fig.add_annotation(
            x=50,
            y=1.04,
            xref="x",
            yref="paper",
            text="random 50%",
            showarrow=False,
            xanchor="right",
            font_color=MUTED,
        )
        fig.add_annotation(
            x=break_even,
            y=1.04,
            xref="x",
            yref="paper",
            text=f"break-even {break_even:.1f}%",
            showarrow=False,
            xanchor="left",
            font_color=BAD,
        )
        fig.update_layout(
            title="Symmetric +/-60 bps holdout win rates",
            xaxis_title="Resolved barrier win rate (%)",
            height=610,
            margin=dict(l=275, r=70, t=85, b=60),
        )
        fig.show()
        """,
    ),
    code(
        "holdout-net-chart",
        """
        net_df = holdout_df.sort("net_bps")
        fig = go.Figure(
            go.Bar(
                x=net_df["net_bps"],
                y=net_df["strategy"],
                orientation="h",
                marker_color=[GOOD if value > 0 else BAD for value in net_df["net_bps"]],
                text=[f"{value:+.1f}" for value in net_df["net_bps"]],
                textposition="auto",
            )
        )
        fig.add_vline(x=0, line_color=MUTED)
        fig.update_layout(
            title="Holdout economics after modeled cost",
            xaxis_title="Average net bps per sequential trade",
            height=610,
            margin=dict(l=275, r=70, t=60, b=60),
        )
        fig.show()
        """,
    ),
    markdown(
        "squeeze-title",
        """
        ## 6. The green candidate, under interrogation

        Squeeze breakout was the only deterministic holdout strategy above fee-adjusted break-even.
        It must work on both sides and through time; otherwise a market trend can masquerade as skill.
        """,
    ),
    code(
        "squeeze-chart",
        """
        squeeze = study["holdout_deterministic_strategies"]["volatility_squeeze_breakout"]
        sides = ["long", "short"]
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Barrier win rate", "Net economics"))
        fig.add_trace(
            go.Bar(
                x=sides,
                y=[squeeze["by_side"][side]["barrier_win_rate"] * 100 for side in sides],
                marker_color=[NEUTRAL, GOLD],
                text=[f"{squeeze['by_side'][side]['barrier_win_rate'] * 100:.1f}%" for side in sides],
                textposition="auto",
                name="Win rate",
            ),
            row=1,
            col=1,
        )
        fig.add_hline(y=break_even, line_dash="dash", line_color=BAD, row=1, col=1)
        fig.add_trace(
            go.Bar(
                x=sides,
                y=[squeeze["by_side"][side]["average_net_bps_per_trade"] for side in sides],
                marker_color=[NEUTRAL, GOLD],
                text=[f"{squeeze['by_side'][side]['average_net_bps_per_trade']:+.1f}" for side in sides],
                textposition="auto",
                name="Net bps/trade",
            ),
            row=1,
            col=2,
        )
        fig.add_hline(y=0, line_color=MUTED, row=1, col=2)
        fig.update_yaxes(title_text="Win rate (%)", row=1, col=1)
        fig.update_yaxes(title_text="Net bps/trade", row=1, col=2)
        fig.update_layout(title="Squeeze-breakout holdout balance", showlegend=False, height=420)
        fig.show()
        """,
    ),
    markdown(
        "model-title",
        """
        ## 7. Can the models recognize their own good guesses?

        Perfect calibration follows the diagonal: predictions called 65% confident should be right
        about 65% of the time. Falling below the line means the model is overconfident.
        """,
    ),
    code(
        "calibration-chart",
        """
        fig = go.Figure()
        for model_name, model in study["holdout_models"].items():
            rows = model["confidence_calibration"]
            fig.add_trace(
                go.Scatter(
                    x=[row["average_confidence"] * 100 for row in rows],
                    y=[row["actual_accuracy"] * 100 for row in rows],
                    mode="lines+markers",
                    name=model_name.replace("_", " "),
                    customdata=[row["count"] for row in rows],
                    hovertemplate="Confidence: %{x:.1f}%<br>Accuracy: %{y:.1f}%<br>Rows: %{customdata}<extra></extra>",
                )
            )
        fig.add_trace(
            go.Scatter(x=[50, 100], y=[50, 100], mode="lines", line=dict(color=MUTED, dash="dash"), name="perfect")
        )
        fig.update_layout(
            title="Holdout confidence calibration",
            xaxis_title="Predicted side confidence (%)",
            yaxis_title="Actual direction accuracy (%)",
            height=470,
        )
        fig.show()
        """,
    ),
    code(
        "selectivity-chart",
        """
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        for model_name, model in study["holdout_models"].items():
            thresholds = []
            net_values = []
            trades = []
            for key, metrics in model["thresholds"].items():
                thresholds.append(float(key.rsplit("_", 1)[1]) * 100)
                net_values.append(metrics["average_net_bps_per_trade"])
                trades.append(metrics["sequential_trades"])
            label = model_name.replace("_", " ")
            fig.add_trace(
                go.Scatter(x=thresholds, y=net_values, mode="lines+markers", name=f"{label} net"),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(x=thresholds, y=trades, mode="lines", line_dash="dot", name=f"{label} trades"),
                secondary_y=True,
            )
        fig.add_hline(y=0, line_color=MUTED, secondary_y=False)
        fig.update_xaxes(title_text="Minimum predicted-side confidence (%)")
        fig.update_yaxes(title_text="Average net bps/trade", secondary_y=False)
        fig.update_yaxes(title_text="Sequential trades", secondary_y=True)
        fig.update_layout(title="Does model selectivity improve economics?", height=470)
        fig.show()
        """,
    ),
    code(
        "summary",
        r"""
        squeeze_periods = study["deterministic_strategy_periods"]["volatility_squeeze_breakout"]
        logistic = study["holdout_models"]["regularized_logistic"]
        boosting = study["holdout_models"]["hist_gradient_boosting"]
        display(
            HTML(
                f'''
                <h2>What this chapter actually found</h2>
                <ul>
                  <li><strong>No strategy was positive in fit, calibration, and holdout.</strong>
                      That is the most important result.</li>
                  <li>Squeeze breakout was promising only in the newest regime:
                      <strong>{squeeze['targets']} TP, {squeeze['stops']} stops,
                      {squeeze['timeouts']} timeouts</strong>, and
                      <strong>{squeeze['average_net_bps_per_trade']:+.2f} bps/trade</strong> on holdout.</li>
                  <li>Its earlier economics were
                      <strong>{squeeze_periods['fit']['average_net_bps_per_trade']:+.2f}</strong> and
                      <strong>{squeeze_periods['calibration']['average_net_bps_per_trade']:+.2f} bps/trade</strong>.
                      That is regime dependence, not yet a universal edge.</li>
                  <li>Regularized logistic holdout accuracy was
                      <strong>{logistic['direction_accuracy']:.1%}</strong>; gradient boosting was
                      <strong>{boosting['direction_accuracy']:.1%}</strong>. Neither model's confidence ranking
                      generalized reliably.</li>
                  <li>The correct next move is a <strong>forward shadow test of squeeze breakout</strong> while
                      collecting more independent regimes. It should not replace the running strategy merely because
                      one holdout is green.</li>
                </ul>
                '''
            )
        )
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
