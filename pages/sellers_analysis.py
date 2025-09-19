import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path

dash.register_page(__name__, name="Sellers", path="/sellers", order=2)

# =======================
# CONFIG
# =======================
CSV_PATH = Path(__file__).resolve().parent.parent / "sellers_metrics.csv"
ALPHA = 3157.27
BETA  = 978.23

# =======================
# LOAD & PREP BASE DATA
# =======================
if not CSV_PATH.exists():
    raise FileNotFoundError(f"Could not find {CSV_PATH}")

base = pd.read_csv(CSV_PATH)
required = ["revenues", "cost_of_reviews", "profits", "quantity"]
for c in required:
    base[c] = pd.to_numeric(base[c], errors="coerce")
base = base.dropna(subset=required).reset_index(drop=True)

# Sort sellers by individual profit (desc)
metrics_ordered = (
    base[["revenues", "cost_of_reviews", "profits", "quantity"]]
    .sort_values(by="profits", ascending=False)
    .reset_index(drop=True)
)
# For cumulative visuals, use negative review costs
metrics_ordered["cost_of_reviews"] = -metrics_ordered["cost_of_reviews"]
metrics_ordered["n_sellers"] = 1

def cost_of_it(df_cum: pd.DataFrame, alpha: float, beta: float) -> pd.Series:
    """Return POSITIVE IT cost for cumulative rows."""
    return alpha * (df_cum["n_sellers"] ** 0.5) + beta * (df_cum["quantity"] ** 0.5)

# Full-sweep cumulative (for global curves & KPIs)
cum_all = metrics_ordered.cumsum()
cum_all["it_costs_pos"] = cost_of_it(cum_all, ALPHA, BETA)
cum_all["it_costs"] = -cum_all["it_costs_pos"]
cum_all["profits_after_it"] = cum_all["profits"] + cum_all["it_costs"]

# Optima indices
opt_no_it_idx = int(cum_all["profits"].argmax())
opt_with_it_idx = int(cum_all["profits_after_it"].argmax())

# =======================
# UI ELEMENTS
# =======================
mode_selector = html.Div([
    html.Label("Seller Selection", className="fw-semibold mb-2"),
    dcc.RadioItems(
        id="mode",
        options=[
            {"label": " All Sellers", "value": "all"},
            {"label": " Optimum Sellers (Profit only)", "value": "no_it"},
            {"label": " Optimum Sellers (Profit incl. IT costs)", "value": "with_it"},
        ],
        value="all",
        inputStyle={"margin-right": "6px"},
        labelStyle={"margin-right": "16px", "display": "inline-block"}
    )
], className="my-2")

def kpi_card(label, value, color="dark", id_=None):
    h4_props = {"className": f" mb-0"}
    if id_ is not None:
        h4_props["id"] = id_
    return dbc.Card(
        dbc.CardBody([
            html.H6(label, className="text-muted"),
            html.H4(value, **h4_props)
        ]),
        className="shadow-sm h-100"
    )

# =======================
# LAYOUT
# =======================
layout = dbc.Container([
    html.H1("Seller Profitability Analysis", className="text-center my-3"),

    dbc.Row([dbc.Col(mode_selector, md=12)], className="g-3 my-1"),

    # KPI row 1 (3 cards)
    dbc.Row([
        dbc.Col(kpi_card("Revenues (Selected)", "—", id_="kpi_rev"), md=4),
        dbc.Col(kpi_card("Total Costs (Reviews + IT)", "—", id_="kpi_costs"), md=4),
        dbc.Col(kpi_card("Net Profit (incl. IT)", "—", id_="kpi_net"), md=4),
    ], className="g-3 my-2"),

    # KPI row 2 (3 cards)
    dbc.Row([
        dbc.Col(kpi_card("Profit Margin", "—", id_="kpi_margin"), md=4),
        dbc.Col(kpi_card("Δ Net Profit vs All", "—", id_="kpi_delta"), md=4),
        dbc.Col(kpi_card("Sellers in View", "—", id_="kpi_n"), md=4),
    ], className="g-3 my-2"),

    # Row 2: Cumulative
    dbc.Row([
        dbc.Col(dcc.Graph(id="fig_cum", config={"displayModeBar": False}), md=12),
    ], className="my-2"),

    # Row 3: Profit curves
    dbc.Row([
        dbc.Col(dcc.Graph(id="fig_curves", config={"displayModeBar": False}), md=12),
    ], className="my-2"),

    # Row 4: Impact
    dbc.Row([
        dbc.Col(dcc.Graph(id="fig_impact", config={"displayModeBar": False}), md=12),
    ], className="my-2"),

    # Static context
    dbc.Row([
        dbc.Col(html.Small(f"Optimum Sellers (Profit only): {opt_no_it_idx + 1:,}", className="text-muted"), md=6),
        dbc.Col(html.Small(f"Optimum Sellers (incl. IT costs): {opt_with_it_idx + 1:,}", className="text-muted text-md-end"), md=6),
    ]),
    # html.Small(f"Data source: {CSV_PATH.name}", className="text-muted"),
], fluid=True)

# =======================
# HELPERS
# =======================
def subset_upto(idx_inclusive: int) -> pd.DataFrame:
    sub = metrics_ordered.iloc[: idx_inclusive + 1].cumsum()
    sub["it_costs_pos"] = cost_of_it(sub, ALPHA, BETA)
    sub["it_costs"] = -sub["it_costs_pos"]
    sub["profits_after_it"] = sub["profits"] + sub["it_costs"]
    return sub

def table_for_mode(mode: str) -> pd.DataFrame:
    if mode == "all":
        return subset_upto(len(metrics_ordered) - 1)
    if mode == "no_it":
        return subset_upto(opt_no_it_idx)
    if mode == "with_it":
        return subset_upto(opt_with_it_idx)
    return subset_upto(len(metrics_ordered) - 1)

# =======================
# CALLBACK
# =======================
@dash.callback(
    Output("fig_cum", "figure"),
    Output("fig_impact", "figure"),
    Output("fig_curves", "figure"),
    Output("kpi_rev", "children"),
    Output("kpi_costs", "children"),
    Output("kpi_net", "children"),
    Output("kpi_margin", "children"),
    Output("kpi_delta", "children"),
    Output("kpi_n", "children"),
    Input("mode", "value"),
    prevent_initial_call=False
)
def update_figs(mode):
    df_cum = table_for_mode(mode)
    n_sellers_view = len(df_cum)

    # Totals for selected set
    totals_sel = df_cum.iloc[-1]
    revenue_sel = float(totals_sel["revenues"])
    review_costs_pos_sel = float(-totals_sel["cost_of_reviews"])  # flip back
    it_costs_pos_sel = float(totals_sel.get("it_costs_pos", 0.0))
    total_costs_sel = review_costs_pos_sel + it_costs_pos_sel
    net_sel = float(totals_sel["profits_after_it"])
    margin_sel = (net_sel / revenue_sel) if revenue_sel > 0 else 0.0

    # Totals for all sellers
    totals_all = cum_all.iloc[-1]
    net_all = float(totals_all["profits_after_it"])

    # KPI strings
    kpi_rev = f"{revenue_sel:,.0f} BRL"
    kpi_costs = f"{total_costs_sel:,.0f} BRL"
    kpi_net = f"{net_sel:,.0f} BRL"
    kpi_margin = f"{(margin_sel*100):.1f}%"
    kpi_delta = f"{(net_sel - net_all):+,.0f} BRL"
    kpi_n = f"{n_sellers_view:,}"

    # --- Cumulative chart ---
    fig_cum = px.line(
        df_cum,
        x=df_cum.index,
        y=["revenues", "cost_of_reviews", "profits_after_it"],
        labels={"index": "# Sellers Included", "value": "BRL"},
        title="Cumulative Revenues, Costs & Profits (Selected Seller Set, With IT Costs)",
    )
    fig_cum.add_vline(x=opt_with_it_idx, line_dash="dash", line_color="red",
                      annotation_text="Optimum (incl. IT costs)", annotation_position="top right")
    fig_cum.add_vline(x=opt_no_it_idx, line_dash="dot", line_color="grey",
                      annotation_text="Optimum (Profit only)", annotation_position="top right")
    fig_cum.update_layout(
        margin=dict(l=10, r=10, t=60, b=10),
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig_cum.update_yaxes(tickformat=",")

    # --- Impact vs keeping all ---
    impact = (totals_sel - totals_all)[["quantity", "profits", "cost_of_reviews", "revenues"]]
    impact_df = impact.rename_axis("Metric").reset_index(name="Impact")
    fig_imp = px.bar(
        impact_df, y="Metric", x="Impact", orientation="h",
        title="Impact of Trimming Sellers (Difference vs Keeping All)",
        color="Metric",
        color_discrete_map={
            "revenues": "#4A90E2",
            "cost_of_reviews": "#E74C3C",
            "profits": "#1E8449",
            "quantity": "#7F8C8D",
        },
    )
    fig_imp.update_layout(
        showlegend=False,
        margin=dict(l=10, r=10, t=60, b=10),
        height=520,
    )
    fig_imp.update_yaxes(tickformat=",")
    fig_imp.update_traces(marker_line_width=0, hovertemplate="%{y}: %{x:,}<extra></extra>")

    # --- Profit curves ---
    fig_curves = go.Figure()
    fig_curves.add_trace(go.Scatter(
        x=cum_all.index, y=cum_all["profits"], mode="lines", name="Profit (no IT)"))
    fig_curves.add_trace(go.Scatter(
        x=cum_all.index, y=cum_all["profits_after_it"], mode="lines", name="Profit (incl. IT costs)"))
    fig_curves.add_vline(x=opt_no_it_idx, line_dash="dot", line_color="grey")
    fig_curves.add_vline(x=opt_with_it_idx, line_dash="dash", line_color="red")
    fig_curves.add_vrect(x0=0, x1=n_sellers_view - 1, fillcolor="LightGreen", opacity=0.15, line_width=0)
    fig_curves.update_layout(
        title=dict(
            text="Global Profit Curves by # Sellers Included (Profit vs Profit After IT)",
            y=0.98, x=0.01, xanchor="left"
        ),
        xaxis_title="# Sellers Included (sorted by individual profit)",
        yaxis_title="BRL",
        margin=dict(l=10, r=10, t=70, b=60),
        height=520,
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="right", x=1),
    )
    fig_curves.update_yaxes(tickformat=",")

    return fig_cum, fig_imp, fig_curves, kpi_rev, kpi_costs, kpi_net, kpi_margin, kpi_delta, kpi_n
