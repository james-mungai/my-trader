import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "notebooks" / "01_first_touch_overview.ipynb"


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
        # ETH first-touch strategy: the picture-book version

        This notebook asks one simple question: **which side reaches its price barrier first?**

        Every result below uses chronological holdout data, non-overlapping trades, a `+0.60%`
        target, a `-0.70%` stop, and `10 bps` modeled round-trip cost unless a chart says otherwise.
        The goal is not to find a pretty backtest. It is to find an edge that survives fees,
        drawdown, both trade directions, and future data.
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
        from IPython.display import Markdown, display
        from plotly.subplots import make_subplots

        pio.renderers.default = "notebook_connected"

        DATA_ROOT = Path(
            os.environ.get(
                "FUTURES_LAB_RESEARCH_DATA",
                Path.home() / "projects" / "my-trader-research-data" / "20260711",
            )
        )
        REPORT_PATH = DATA_ROOT / "first-touch-study-local.json"
        study = json.loads(REPORT_PATH.read_text(encoding="utf-8-sig"))

        GOOD = "#2A9D8F"
        BAD = "#E76F51"
        NEUTRAL = "#457B9D"
        MUTED = "#8D99AE"

        print(f"Loaded: {REPORT_PATH}")
        print(f"Coverage: {study['data']['mark_price_start']} to {study['data']['mark_price_end']}")
        print(f"Unique mark points: {study['data']['unique_mark_price_points']:,}")
        print(f"Labelable minute samples: {study['data']['labelable_samples']:,}")
        """,
    ),
    markdown(
        "data-map",
        """
        ## 1. Do we have enough data?

        The bars below separate raw observations from usable research samples. Millions of price
        updates are useful for locating barrier touches, while the smaller minute sample is what
        prevents us from pretending every millisecond is an independent trade.
        """,
    ),
    code(
        "data-chart",
        """
        data_counts = {
            "Unique 1s mark points": study["data"]["unique_mark_price_points"],
            "Sampled feature rows": study["data"]["sampled_feature_rows"],
            "Labelable samples": study["data"]["labelable_samples"],
            "Purged training rows": study["data"]["train_samples_after_purge"],
            "Holdout rows": study["data"]["test_samples"],
        }
        fig = go.Figure(
            go.Bar(
                x=list(data_counts.values()),
                y=list(data_counts.keys()),
                orientation="h",
                marker_color=[MUTED, NEUTRAL, NEUTRAL, GOOD, GOOD],
                text=[f"{value:,}" for value in data_counts.values()],
                textposition="auto",
            )
        )
        fig.update_layout(
            title="From market observations to honest holdout samples",
            xaxis_title="Rows (log scale)",
            xaxis_type="log",
            yaxis_autorange="reversed",
            height=380,
            margin=dict(l=170, r=70, t=60, b=50),
        )
        fig.show()
        """,
    ),
    markdown(
        "data-explain",
        """
        **How to read it:** we have far more than 100 observations. More importantly, the simulator
        can form over 100 sequential, non-overlapping trades in the holdout period. Gaps are split
        into separate segments, so a target is never credited by jumping across missing data.
        """,
    ),
    markdown(
        "strategy-title",
        """
        ## 2. Which simple strategy survived fees?

        A bar above zero earned money after the modeled 10 bps round-trip cost. A bar below zero
        lost money even if its raw direction accuracy looked respectable.
        """,
    ),
    code(
        "strategy-chart",
        """
        stop_key = "70.0"
        strategy_rows = []
        for name, scenarios in study["holdout_deterministic_strategies"].items():
            row = scenarios[stop_key]
            strategy_rows.append(
                {
                    "strategy": name.replace("_", " "),
                    "average_net_bps": row["average_net_bps_per_trade"],
                    "trades": row["sequential_trades"],
                    "drawdown_pct": row["max_additive_account_drawdown_pct"],
                }
            )
        strategy_df = pl.DataFrame(strategy_rows).sort("average_net_bps")
        values = strategy_df["average_net_bps"].to_list()

        fig = go.Figure(
            go.Bar(
                x=values,
                y=strategy_df["strategy"].to_list(),
                orientation="h",
                marker_color=[GOOD if value > 0 else BAD for value in values],
                text=[f"{value:+.1f} bps" for value in values],
                textposition="outside",
                customdata=strategy_df["trades"].to_list(),
                hovertemplate="%{y}<br>Net: %{x:.2f} bps/trade<br>Trades: %{customdata}<extra></extra>",
            )
        )
        fig.add_vline(x=0, line_width=1, line_color=MUTED)
        fig.update_layout(
            title="Holdout economics at a 60 bps target / 70 bps stop",
            xaxis_title="Average net bps per sequential trade",
            height=430,
            margin=dict(l=210, r=80, t=60, b=50),
        )
        fig.show()
        """,
    ),
    markdown(
        "strategy-explain",
        """
        **What the picture says:** micro-momentum is the only simple 60/70 rule above zero in this
        holdout. That makes it our leading hypothesis, not a guaranteed winner. Looking at this
        result means this holdout is now part of model development; the running forward paper test
        remains important independent evidence.
        """,
    ),
    markdown(
        "outcomes-title",
        """
        ## 3. Wins, losses, and timeouts

        A profitable strategy does not need to win every trade. It needs enough targets to pay for
        stops, timeouts, and fees.
        """,
    ),
    code(
        "outcomes-chart",
        """
        outcome_rows = []
        for name, scenarios in study["holdout_deterministic_strategies"].items():
            row = scenarios[stop_key]
            outcome_rows.append(
                {
                    "strategy": name.replace("_", " "),
                    "targets": row["targets"],
                    "stops": row["stops"],
                    "timeouts": row["timeouts"],
                }
            )
        outcome_df = pl.DataFrame(outcome_rows)
        fig = go.Figure()
        fig.add_bar(name="Target first", x=outcome_df["strategy"], y=outcome_df["targets"], marker_color=GOOD)
        fig.add_bar(name="Stop first", x=outcome_df["strategy"], y=outcome_df["stops"], marker_color=BAD)
        fig.add_bar(name="Timeout", x=outcome_df["strategy"], y=outcome_df["timeouts"], marker_color=MUTED)
        fig.update_layout(
            barmode="stack",
            title="Sequential holdout outcomes",
            yaxis_title="Trades",
            xaxis_title="",
            height=430,
            margin=dict(l=60, r=30, t=60, b=130),
        )
        fig.update_xaxes(tickangle=-28)
        fig.show()
        """,
    ),
    markdown(
        "threshold-explain",
        """
        At 60 bps target, 70 bps stop, and 10 bps cost, a resolved trade needs a **61.5% barrier win
        rate** to break even. The historical micro-momentum holdout cleared that level, but its
        confidence interval still overlaps break-even. More independent trades are needed.
        """,
    ),
    markdown(
        "grid-title",
        """
        ## 4. Does the idea survive different stop widths?

        If only one precise stop works, we should suspect luck. A broad positive region is healthier
        than a single isolated peak.
        """,
    ),
    code(
        "stop-grid-chart",
        """
        micro_scenarios = study["holdout_deterministic_strategies"]["micro_momentum"]
        stop_rows = []
        for stop, row in micro_scenarios.items():
            stop_rows.append(
                {
                    "stop_bps": float(stop),
                    "average_net_bps": row["average_net_bps_per_trade"],
                    "barrier_win_rate": row["barrier_win_rate"] * 100,
                    "break_even_rate": row["break_even_win_rate"] * 100,
                    "trades": row["sequential_trades"],
                }
            )
        stop_df = pl.DataFrame(stop_rows).sort("stop_bps")
        visible = stop_df.filter(pl.col("stop_bps") <= 150)

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(
                x=visible["stop_bps"],
                y=visible["average_net_bps"],
                name="Net bps/trade",
                marker_color=[GOOD if value > 0 else BAD for value in visible["average_net_bps"]],
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=visible["stop_bps"],
                y=visible["barrier_win_rate"],
                name="Observed win rate",
                mode="lines+markers",
                line_color=NEUTRAL,
            ),
            secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(
                x=visible["stop_bps"],
                y=visible["break_even_rate"],
                name="Required win rate",
                mode="lines+markers",
                line=dict(color=MUTED, dash="dash"),
            ),
            secondary_y=True,
        )
        fig.update_layout(
            title="Micro-momentum sensitivity to stop width",
            xaxis_title="Stop width (bps)",
            height=450,
            margin=dict(l=60, r=70, t=60, b=60),
        )
        fig.update_yaxes(title_text="Average net bps/trade", secondary_y=False)
        fig.update_yaxes(title_text="Barrier win rate (%)", secondary_y=True)
        fig.show()
        """,
    ),
    markdown(
        "risk-title",
        """
        ## 5. Profit is not enough: what did it cost in drawdown?

        The upper-left area is attractive: positive edge with smaller drawdown. Strategies farther
        right demand more emotional and financial survival to realize their average result.
        """,
    ),
    code(
        "risk-chart",
        """
        fig = go.Figure(
            go.Scatter(
                x=strategy_df["drawdown_pct"],
                y=strategy_df["average_net_bps"],
                mode="markers+text",
                text=strategy_df["strategy"],
                textposition="top center",
                marker=dict(
                    size=[max(10, min(28, trades / 6)) for trades in strategy_df["trades"]],
                    color=[GOOD if value > 0 else BAD for value in strategy_df["average_net_bps"]],
                ),
                hovertemplate="%{text}<br>Drawdown: %{x:.1f}%<br>Net: %{y:+.2f} bps/trade<extra></extra>",
            )
        )
        fig.add_hline(y=0, line_width=1, line_color=MUTED)
        fig.update_layout(
            title="Return versus modeled account drawdown at 4.95x exposure",
            xaxis_title="Maximum additive account drawdown (%)",
            yaxis_title="Average net bps per trade",
            height=470,
            margin=dict(l=70, r=50, t=60, b=60),
        )
        fig.show()
        """,
    ),
    markdown(
        "blocks-title",
        """
        ## 6. Did the edge persist through time?

        We divide the sequential holdout trades into chronological blocks. A strategy carried by one
        lucky burst is weaker than one that stays near or above zero across several blocks.
        """,
    ),
    code(
        "blocks-chart",
        """
        micro = study["holdout_deterministic_strategies"]["micro_momentum"][stop_key]
        blocks = pl.DataFrame(micro["chronological_trade_blocks"])
        block_labels = [f"Block {value}" for value in blocks["block"]]
        block_values = blocks["average_net_bps_per_trade"].to_list()
        fig = go.Figure(
            go.Bar(
                x=block_labels,
                y=block_values,
                marker_color=[GOOD if value > 0 else BAD for value in block_values],
                text=[f"{value:+.1f}" for value in block_values],
                textposition="outside",
                customdata=list(zip(blocks["trades"].to_list(), blocks["start"].to_list(), blocks["end"].to_list())),
                hovertemplate="%{x}<br>Net: %{y:+.2f} bps/trade<br>Trades: %{customdata[0]}<br>%{customdata[1]} to %{customdata[2]}<extra></extra>",
            )
        )
        fig.add_hline(y=0, line_width=1, line_color=MUTED)
        fig.update_layout(
            title="Micro-momentum through chronological holdout blocks",
            xaxis_title="Oldest to newest",
            yaxis_title="Average net bps/trade",
            height=400,
            margin=dict(l=70, r=40, t=60, b=60),
        )
        fig.show()
        """,
    ),
    code(
        "summary",
        """
        display(
            Markdown(
                f'''
                ## What we know, in plain English

                - The completed local archive produced **{micro['sequential_trades']} independent holdout trades**.
                - Micro-momentum recorded **{micro['targets']} targets, {micro['stops']} stops, and {micro['timeouts']} timeouts**.
                - Its average result was **{micro['average_net_bps_per_trade']:+.2f} bps per trade after modeled costs**.
                - Its modeled maximum account drawdown was **{micro['max_additive_account_drawdown_pct']:.1f}% at 4.95x exposure**.
                - This is promising historical evidence, but the current forward paper run is the next reality check.

                **Decision rule:** do not promote a strategy because one chart is green. Require positive walk-forward
                blocks, balanced long/short behavior, a confidence margin above break-even, and survival under worse fees
                and slippage.
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
