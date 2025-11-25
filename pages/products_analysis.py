import dash
from dash import html, dcc, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd

# Register page
dash.register_page(__name__, name="Product", path="/product-analysis", order=3)

# --- Constants ---
CUTOFF_RANK = 21385

# --- Data ---
cumulative_df = pd.read_csv("metrics_cumulative_plt.csv")

# --- Helpers ---
def compute_stats(df: pd.DataFrame) -> dict:
    d = df.sort_values("product_rank")
    total_profit   = float(d["profits"].iloc[-1])
    total_revenue  = float(d["revenues"].iloc[-1]) if "revenues" in d else 0.0

    if d["product_rank"].max() >= CUTOFF_RANK:
        top_profit_at_cutoff  = float(d.loc[d["product_rank"] <= CUTOFF_RANK, "profits"].iloc[-1])
        top_revenue_at_cutoff = float(d.loc[d["product_rank"] <= CUTOFF_RANK, "revenues"].iloc[-1]) if "revenues" in d else 0.0
    else:
        top_profit_at_cutoff  = total_profit
        top_revenue_at_cutoff = total_revenue

    tail_profit  = total_profit  - top_profit_at_cutoff
    tail_revenue = total_revenue - top_revenue_at_cutoff

    return {
        "total_profit": total_profit,
        "tail_profit": tail_profit,
        "total_revenue": total_revenue,
        "tail_revenue": tail_revenue,
    }

def fmt_money(x) -> str:
    return f"{x:,.0f} BRL"

BASE = compute_stats(cumulative_df)

# --- Reusable KPI card ---
def kpi_card(title, value_id=None, delta_id=None, fixed_value=None):
    title_el = html.Div(
        title,
        className="text-capitalize text-dark",
    )
    value_el = (
        html.H4(fmt_money(fixed_value), className="mb-1")
        if fixed_value is not None
        else html.H4(id=value_id, className="mb-1")
    )
    delta_el = html.Div(id=delta_id) if delta_id else html.Div("\u00A0")

    return dbc.Card(
        dbc.CardBody(
            [title_el, value_el, delta_el],
            className="d-flex flex-column justify-content-between",
        ),
        className="shadow-sm",
        style={
            "minHeight": "90px",
            "borderColor": "#e5e7eb",
        },
    )

# --- UI ---
filter_switch = html.Div(
    dbc.Checklist(
        id="cutoff_switch",
        options=[{"label": "With / Without non profitable products", "value": "cutoff"}],
        value=[],
        switch=True,
        label_style={"marginLeft": "0.5rem", "fontWeight": "600"},
    ),
    className="d-flex justify-content-end mb-2",
)

kpi_row = dbc.Row(
    [
        dbc.Col(kpi_card("Total profit", value_id="kpi_total_profit", delta_id="kpi_total_profit_delta"), md=3),
        dbc.Col(kpi_card("Tail profit", fixed_value=BASE["tail_profit"]), md=3),
        dbc.Col(kpi_card("Total revenues", value_id="kpi_total_revenue", delta_id="kpi_total_revenue_delta"), md=3),
        dbc.Col(kpi_card("Tail revenues", fixed_value=BASE["tail_revenue"]), md=3),
    ],
    className="gy-3 mb-3",
)

# Graphs with reduced height (75% of ~450px → 337px)
cumulative_metrics_plot = dcc.Graph(id="cumulative_metrics_plot", style={"height": "337px"})
cumulative_profit_plot  = dcc.Graph(id="cumulative_profit_plot",  style={"height": "337px"})

layout = dbc.Container(
    [
        html.H1("Product Profitability Analysis", className="text-center my-3 mb-4"),
        filter_switch,
        kpi_row,
        dbc.Row(
            [
                dbc.Col(cumulative_metrics_plot, md=6),
                dbc.Col(cumulative_profit_plot,  md=6),
            ]
        ),
    ],
    fluid=False,
)

# --- Figures ---
def make_figs(df: pd.DataFrame):
    fig_metrics = px.line(
        df,
        x="product_rank",
        y=["revenues", "cost_of_reviews", "profits"],
        title="Cumulative Revenues, Costs, and Profits",
        labels={"value": "Cumulative Value", "product_rank": "Number of Products", "variable": "Metric"},
    )
    fig_profit = (
        px.line(
            df,
            x="product_rank",
            y="profits",
            title="Cumulative Profit by Product Rank",
            labels={"product_rank": "Number of Products", "profits": "Cumulative Profits"},
        )
        .update_traces(line=dict(color="#00CC96"))
        .add_hline(y=0, line_dash="dash", line_color="#ef4444")
    )
    if df["product_rank"].min() <= CUTOFF_RANK <= df["product_rank"].max():
        fig_profit = (
            fig_profit
            .add_vline(x=CUTOFF_RANK, line_dash="dot", line_color="#219eeb")
            .add_annotation(
                x=CUTOFF_RANK, y=0,
                text=f"Non profitable products start at Rank #{CUTOFF_RANK:,}",
                showarrow=True, arrowhead=2, yshift=30,
            )
        )
    return fig_metrics, fig_profit

# --- Callback ---
@callback(
    Output("cumulative_metrics_plot", "figure"),
    Output("cumulative_profit_plot", "figure"),
    Output("kpi_total_profit", "children"),
    Output("kpi_total_profit_delta", "children"),
    Output("kpi_total_revenue", "children"),
    Output("kpi_total_revenue_delta", "children"),
    Input("cutoff_switch", "value"),
)
def update_page(switch_value):
    apply_cut = ("cutoff" in (switch_value or []))
    view = cumulative_df.loc[cumulative_df["product_rank"] <= CUTOFF_RANK].copy() if apply_cut else cumulative_df

    stats = compute_stats(view)
    fig_metrics, fig_profit = make_figs(view)

    total_profit_txt  = fmt_money(stats["total_profit"])
    total_revenue_txt = fmt_money(stats["total_revenue"])

    if apply_cut:
        diff_p = stats["total_profit"] - BASE["total_profit"]
        pct_p  = (diff_p / abs(BASE["total_profit"])) * 100 if BASE["total_profit"] != 0 else None
        total_profit_badge = dbc.Badge(
            f"{pct_p:+.2f}%" if pct_p is not None else "",
            color=("success" if diff_p >= 0 else "danger"),
            pill=True,
        )

        diff_r = stats["total_revenue"] - BASE["total_revenue"]
        pct_r  = (diff_r / abs(BASE["total_revenue"])) * 100 if BASE["total_revenue"] != 0 else None
        total_revenue_badge = dbc.Badge(
            f"{pct_r:+.2f}%" if pct_r is not None else "",
            color=("success" if diff_r >= 0 else "danger"),
            pill=True,
        )
    else:
        total_profit_badge  = html.Span("\u00A0")
        total_revenue_badge = html.Span("\u00A0")

    return (
        fig_metrics,
        fig_profit,
        total_profit_txt,
        total_profit_badge,
        total_revenue_txt,
        total_revenue_badge,
    )
