import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
from functools import lru_cache
from pathlib import Path
import numpy as np

dash.register_page(__name__, name="Executive Overview", path="/overview", order=1)

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parents[1]  # /pages -> project root
SUMMARY_CSV = ROOT / "summary_metrics.csv"
PROFITS_CSV = ROOT / "sellers_profits.csv"

# ---------- Palette ----------
PALETTE = {
    "revenue_light": "#A7C7E7",
    "revenue_mid":   "#4A90E2",
    "revenue_dark":  "#1F4F82",
    "cost_light":    "#F5B7B1",
    "cost_mid":      "#E74C3C",
    "cost_dark":     "#7B241C",
    "profit_light":  "#ABEBC6",
    "profit_dark":   "#1E8449",
}

# ---------- Data Loading ----------
@lru_cache(maxsize=1)
def load_summary() -> pd.Series:
    df = pd.read_csv(SUMMARY_CSV)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="ignore")
    return df.iloc[0]

@lru_cache(maxsize=1)
def load_profits() -> pd.DataFrame:
    df = pd.read_csv(PROFITS_CSV)
    if "profits" not in df.columns:
        raise ValueError("Expected a 'profits' column in sellers_profits.csv")
    df["profits"] = pd.to_numeric(df["profits"], errors="coerce").fillna(0)
    return df

# ---------- Helpers ----------
def kpi_card(label, value, sub=None, color="dark"):
    return dbc.Card(
        dbc.CardBody([
            html.Div(label, className="text-muted small"),
            html.H4(value, className=f" mb-0"),
            html.Div(sub, className="text-muted small mt-1") if sub else None,
        ]),
        className="shadow-sm",
        style={"minHeight": "130px"}
    )

def make_distribution_metrics(profits_df: pd.DataFrame):
    s = profits_df["profits"]
    prof = s[s > 0]
    n_profitable = int(len(prof))
    top10_share = 0.0
    n_top10 = 0
    if n_profitable > 0:
        k = max(1, int(np.ceil(0.10 * n_profitable)))
        n_top10 = k
        top_slice = prof.sort_values(ascending=False).head(k)
        denom = prof.sum()
        top10_share = float(top_slice.sum() / denom) if denom != 0 else 0.0
    return dict(top10_share=top10_share, n_profitable=n_profitable, n_top10=n_top10)

def make_violin(profits_df: pd.DataFrame, trim: bool = False, q_low: float = 0.02, q_high: float = 0.98):
    """
    Violin plot of seller profits.
    - trim=False: show ALL points & full range
    - trim=True : keep only inner [q_low, q_high] quantiles and zoom to that range
    Hover is disabled to avoid clutter.
    """
    df = profits_df.copy()
    df["profits"] = pd.to_numeric(df["profits"], errors="coerce").fillna(0.0)
    df["group"] = "All Sellers"  # align violin & points at same x

    if trim:
        lo = float(df["profits"].quantile(q_low))
        hi = float(df["profits"].quantile(q_high))
        df_used = df[(df["profits"] >= lo) & (df["profits"] <= hi)].copy()
        y_range = [lo, hi]
        title_suffix = f" (focused {int(q_low*100)}–{int(q_high*100)}th pct)"
    else:
        df_used = df
        y_range = None
        title_suffix = ""

    fig = px.violin(
        df_used, x="group", y="profits",
        box=True,
        points="all",      # dots ON in both modes
        title=f"Distribution of Seller Profits (Violin Plot){title_suffix}"
    )

    fig.update_traces(
        selector=dict(type="violin"),
        meanline_visible=True,
        scalemode="count",
        spanmode="soft",
        pointpos=0,          # center points on the violin
        jitter=0.25,
        marker_opacity=0.5,
        marker_size=5,
        hoverinfo="skip",    # disable tooltips on the trace
        hovertemplate=None,
        hoveron=None,
    )

    fig.update_layout(
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis_title=None,
        yaxis_title="Profit (BRL)",
        yaxis_tickformat=",",
        showlegend=False,
        autosize=True        # container controls final size
    )
    fig.update_xaxes(showticklabels=False)
    if y_range is not None:
        fig.update_yaxes(range=y_range)

    # Zero-profit ref line (only if inside current view)
    yr = y_range or [df_used["profits"].min(), df_used["profits"].max()]
    if yr[0] <= 0 <= yr[1]:
        fig.add_hline(y=0, line_dash="dot", line_color="grey")

    return fig

def make_main_figures(m: pd.Series, profits_df: pd.DataFrame):
    # Revenue bar
    rev_df = pd.DataFrame({
        "Category": ["Sales fees", "Subscriptions", "Total Revenues"],
        "Value": [m["revenues_sales"], m["revenues_subscription"], m["revenues_total"]],
    })
    rev_fig = px.bar(
        rev_df, x="Category", y="Value", color="Category",
        color_discrete_map={
            "Sales fees": PALETTE["revenue_light"],
            "Subscriptions": PALETTE["revenue_mid"],
            "Total Revenues": PALETTE["revenue_dark"],
        },
        text_auto=".2s",
        title="Revenue Breakdown (BRL)",
    )
    rev_fig.update_traces(marker_line_width=0, hovertemplate="%{x}<br>%{y:,} BRL<extra></extra>")
    rev_fig.update_layout(showlegend=False, yaxis_tickformat=",", margin=dict(l=10, r=10, t=60, b=10), autosize=True)

    # Cost bar
    cost_df = pd.DataFrame({
        "Category": ["Review Costs", "IT Costs", "Total Costs"],
        "Value": [m["costs_reviews"], m["costs_it"], m["costs_total"]],
    })
    cost_fig = px.bar(
        cost_df, x="Category", y="Value", color="Category",
        color_discrete_map={
            "Review Costs": PALETTE["cost_light"],
            "IT Costs": PALETTE["cost_mid"],
            "Total Costs": PALETTE["cost_dark"],
        },
        text_auto=".2s",
        title="Cost Breakdown (BRL)",
    )
    cost_fig.update_traces(marker_line_width=0, hovertemplate="%{x}<br>%{y:,} BRL<extra></extra>")
    cost_fig.update_layout(showlegend=False, yaxis_tickformat=",", margin=dict(l=10, r=10, t=60, b=10), autosize=True)

    # Profit bar
    profit_df = pd.DataFrame({
        "Category": ["Gross Profit", "Net Profit"],
        "Value": [m["profits_gross"], m["profits_net"]],
    })
    prof_fig = px.bar(
        profit_df, x="Category", y="Value", color="Category",
        color_discrete_map={
            "Gross Profit": PALETTE["profit_light"],
            "Net Profit": PALETTE["profit_dark"],
        },
        text_auto=".2s",
        title="Profit Before/After IT Costs",
    )
    prof_fig.update_traces(marker_line_width=0, hovertemplate="%{x}<br>%{y:,} BRL<extra></extra>")
    prof_fig.update_layout(showlegend=False, yaxis_tickformat=",", margin=dict(l=10, r=10, t=60, b=10), autosize=True)

    return rev_fig, cost_fig, prof_fig

# ---------- Layout ----------
def layout():
    m = load_summary()
    profits_df = load_profits()
    dist = make_distribution_metrics(profits_df)

    rev_total = float(m.get("revenues_total", 0.0))
    rev_sales = float(m.get("revenues_sales", 0.0))
    rev_subs  = float(m.get("revenues_subscription", 0.0))
    sales_pct = (rev_sales / rev_total * 100) if rev_total else 0.0
    subs_pct  = (rev_subs  / rev_total * 100) if rev_total else 0.0

    cost_total = float(m.get("costs_total", 0.0))
    cost_rev   = float(m.get("costs_reviews", 0.0))
    cost_it    = float(m.get("costs_it", 0.0))
    rev_cost_pct = (cost_rev / cost_total * 100) if cost_total else 0.0
    it_cost_pct  = (cost_it  / cost_total * 100) if cost_total else 0.0

    rev_fig, cost_fig, profit_fig = make_main_figures(m, profits_df)

    # Initial violin: full range, all dots
    violin_fig = make_violin(profits_df, trim=False)

    return dbc.Container([
        html.H1("Olist P&L Overview", className="text-center my-3"),

        dbc.Row([
            dbc.Col(kpi_card("Net Profit", f"{float(m['profits_net']):,.0f} BRL",
                             sub=f"Gross: {float(m['profits_gross']):,.0f} BRL"), md=2),
            dbc.Col(kpi_card("Total Revenues", f"{rev_total:,.0f} BRL",
                             sub=f"Fees {sales_pct:.1f}% · Subs {subs_pct:.1f}%"), md=3),
            dbc.Col(kpi_card("Total Costs", f"{cost_total:,.0f} BRL",
                             sub=f"Reviews {rev_cost_pct:.1f}% · IT {it_cost_pct:.1f}%"), md=3),
            dbc.Col(kpi_card("% Losing Sellers", f"{float(m['pct_negative']):.1f}%",
                             sub=f"{int(m['n_negative']):,} / {int(m['n_sellers']):,}"), md=2),
            dbc.Col(kpi_card("Top 10% Profit Share",
                             f"{dist['top10_share']*100:,.1f}%",
                             sub=f"Top 10% sellers: {dist['n_top10']:,} of {dist['n_profitable']:,} profitable"), md=2),
        ], className="g-3 my-2", justify="evenly"),

        dbc.Row([
            dbc.Col(dcc.Graph(id="big_rev_fig", figure=rev_fig, style={"height": "360px"},
                              config={"responsive": False, "displayModeBar": False}), md=6),
            dbc.Col(dcc.Graph(id="big_cost_fig", figure=cost_fig, style={"height": "360px"},
                              config={"responsive": False, "displayModeBar": False}), md=6),
        ], className="my-2"),

        dbc.Row([
            dbc.Col(dcc.Graph(id="big_profit_fig", figure=profit_fig, style={"height": "360px"},
                              config={"responsive": False, "displayModeBar": False}), md=6),

            # Right column: violin + bottom-right toggle
            dbc.Col([
                dcc.Graph(id="big_profit_violin", figure=violin_fig, style={"height": "420px"},
                          config={"responsive": False, "displayModeBar": False}),
                html.Div(
                    dbc.Checklist(
                        id="toggle_violin_focus",
                        options=[{"label": " Focus (trim outliers)", "value": "trim"}],
                        value=[],  # default OFF = full range
                        switch=True,
                    ),
                    className="text-end mt-1"
                )
            ], md=6),
        ], className="my-2"),
    ], fluid=True)

layout = layout

# ---------- Callback: toggle focus (trim + zoom) ----------
@dash.callback(
    Output("big_profit_violin", "figure"),
    Input("toggle_violin_focus", "value")
)
def update_violin(toggle_vals):
    trim = "trim" in (toggle_vals or [])
    profits_df = load_profits()
    # inner 2–98% is a good “zoom a little” default; tweak to 0.01/0.99 if needed
    return make_violin(profits_df, trim=trim, q_low=0.02, q_high=0.98)
