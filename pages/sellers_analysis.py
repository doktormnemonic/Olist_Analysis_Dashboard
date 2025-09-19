import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
import math

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
# For cumulative visuals, make review costs negative to stack with revenues/profits
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

def kpi_card(title, value_id, delta_id=None):
    """Card shell: title, H4 value, and (optional) small delta area."""
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(title, className="text-dark"),
                html.H4("—", id=value_id, className="mb-1"),
                (html.Div("\u00A0", id=delta_id) if delta_id else html.Div("\u00A0")),
            ],
            className="d-flex flex-column justify-content-between",
        ),
        className="shadow-sm",
        style={"minHeight": "90px", "borderColor": "#e5e7eb"},
    )

def make_badge_pct(pct: float | None, *, invert: bool = False):
    """Return a dbc.Badge with +/-xx.xx% (green/red).
       invert=True → green when pct < 0 (good for costs or lower counts).
    """
    if pct is None or math.isinf(pct) or math.isnan(pct):
        return html.Span("\u00A0")
    is_good = (pct > 0 and not invert) or (pct < 0 and invert)
    color = "success" if is_good else ("danger" if pct != 0 else "secondary")
    sign = "+" if pct > 0 else ""
    return dbc.Badge(f"{sign}{pct:.2f}%", color=color, pill=True)

def make_badge_pp(delta_pp: float | None):
    """Return a dbc.Badge with +/-x.x pp."""
    if delta_pp is None or math.isinf(delta_pp) or math.isnan(delta_pp):
        return html.Span("\u00A0")
    color = "success" if delta_pp > 0 else ("danger" if delta_pp < 0 else "secondary")
    sign = "+" if delta_pp > 0 else ""
    return dbc.Badge(f"{sign}{delta_pp:.1f} pp", color=color, pill=True)

def make_badge_neutral(pct: float | None):
    """Always neutral info-colored badge for sellers percentage."""
    if pct is None or math.isinf(pct) or math.isnan(pct):
        return html.Span("\u00A0")
    sign = "+" if pct > 0 else ""
    return dbc.Badge(f"{sign}{pct:.2f}%", color="info", pill=True)

def make_badge_info(text: str):
    """Neutral info badge for arbitrary text (e.g., removed count)."""
    return dbc.Badge(text, color="info", pill=True)

# =======================
# LAYOUT (graphs area is dynamic)
# =======================
layout = dbc.Container([
    html.H1("Seller Profitability Analysis", className="text-center my-3"),

    dbc.Row([dbc.Col(mode_selector, md=12)], className="g-3 my-1"),

    # KPI row 1 (3 cards)
    dbc.Row([
        dbc.Col(kpi_card("Revenues (Selected)", "kpi_rev", "kpi_rev_sub"), md=4),
        dbc.Col(kpi_card("Total Costs (Reviews + IT)", "kpi_costs", "kpi_costs_sub"), md=4),
        dbc.Col(kpi_card("Net Profit (incl. IT)", "kpi_net", "kpi_net_sub"), md=4),
    ], className="gy-3 my-2"),

    # KPI row 2 (3 cards) — Sellers in View will show two blue badges side-by-side
    dbc.Row([
        dbc.Col(kpi_card("Profit Margin", "kpi_margin", "kpi_margin_sub"), md=4),
        dbc.Col(kpi_card("Δ Net Profit vs All", "kpi_delta", "kpi_delta_sub"), md=4),
        dbc.Col(kpi_card("Sellers in View", "kpi_n", "kpi_n_sub"), md=4),
    ], className="gy-3 my-2"),

    # --- Dynamic plot area (built by callback) ---
    html.Div(id="plots_area"),

    # Static context
    dbc.Row([
        dbc.Col(html.Small(f"Optimum Sellers (Profit only): {opt_no_it_idx + 1:,}", className="text-muted"), md=6),
        dbc.Col(html.Small(f"Optimum Sellers (incl. IT costs): {opt_with_it_idx + 1:,}", className="text-muted text-md-end"), md=6),
    ]),
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
    Output("plots_area", "children"),   # dynamic layout for plots
    Output("kpi_rev", "children"),
    Output("kpi_costs", "children"),
    Output("kpi_net", "children"),
    Output("kpi_margin", "children"),
    Output("kpi_delta", "children"),
    Output("kpi_n", "children"),
    Output("kpi_rev_sub", "children"),
    Output("kpi_costs_sub", "children"),
    Output("kpi_net_sub", "children"),
    Output("kpi_margin_sub", "children"),
    Output("kpi_delta_sub", "children"),
    Output("kpi_n_sub", "children"),
    Input("mode", "value"),
    prevent_initial_call=False
)
def update_figs(mode):
    df_cum = table_for_mode(mode)
    n_sellers_view = len(df_cum)

    # Totals for selected set
    totals_sel = df_cum.iloc[-1]
    revenue_sel = float(totals_sel["revenues"])
    review_costs_pos_sel = float(-totals_sel["cost_of_reviews"])  # flip back to positive
    it_costs_pos_sel = float(totals_sel.get("it_costs_pos", 0.0))
    total_costs_sel = review_costs_pos_sel + it_costs_pos_sel
    net_sel = float(totals_sel["profits_after_it"])
    margin_sel = (net_sel / revenue_sel) if revenue_sel > 0 else 0.0

    # Totals for all sellers
    totals_all = cum_all.iloc[-1]
    revenue_all = float(totals_all["revenues"])
    review_costs_pos_all = float(-totals_all["cost_of_reviews"])
    it_costs_pos_all = float(totals_all.get("it_costs_pos", 0.0))
    total_costs_all = review_costs_pos_all + it_costs_pos_all
    net_all = float(totals_all["profits_after_it"])
    margin_all = (net_all / revenue_all) if revenue_all > 0 else 0.0

    # KPI main values
    kpi_rev = f"{revenue_sel:,.0f} BRL"
    kpi_costs = f"{total_costs_sel:,.0f} BRL"
    kpi_net = f"{net_sel:,.0f} BRL"
    kpi_margin = f"{(margin_sel*100):.1f}%"
    kpi_delta_abs = net_sel - net_all
    kpi_delta = f"{kpi_delta_abs:+,.0f} BRL"
    kpi_n = f"{n_sellers_view:,}"

    # Compute deltas vs All (percent or pp)
    rev_pct = ((revenue_sel - revenue_all) / revenue_all * 100) if revenue_all else None
    costs_pct = ((total_costs_sel - total_costs_all) / total_costs_all * 100) if total_costs_all else None
    net_pct = ((net_sel - net_all) / abs(net_all) * 100) if net_all != 0 else None
    margin_pp = ((margin_sel - margin_all) * 100) if revenue_all else None
    delta_pct = ((net_sel - net_all) / abs(net_all) * 100) if net_all != 0 else None

    # Sellers % vs all (neutral blue) and removed count (neutral blue)
    N_all = len(cum_all)
    sellers_pct = ((n_sellers_view - N_all) / N_all * 100) if N_all else None
    removed_n = max(N_all - n_sellers_view, 0)

    # Build badges
    if mode == "all":
        rev_sub = html.Span("\u00A0")
        costs_sub = html.Span("\u00A0")
        net_sub = html.Span("\u00A0")
        margin_sub = html.Span("\u00A0")
        delta_sub = html.Span("\u00A0")
        # Keep previous behavior (blank subs when 'all')
        n_sub = html.Span("\u00A0")
    else:
        rev_sub = make_badge_pct(rev_pct, invert=False)
        costs_sub = make_badge_pct(costs_pct, invert=True)   # invert for costs
        net_sub = make_badge_pct(net_pct, invert=False)
        margin_sub = make_badge_pp(margin_pp)
        delta_sub = make_badge_pct(delta_pct, invert=False)

        # two neutral blue badges side-by-side: % and number removed
        pct_badge = make_badge_neutral(sellers_pct)
        removed_badge = make_badge_info(f"−{removed_n:,}")
        n_sub = html.Div(
            [pct_badge, removed_badge],
            className="d-flex align-items-center gap-2 flex-wrap"
        )

    # --- Cumulative chart (full; green area shows removed sellers) ---
    N = len(cum_all)
    cut_x = n_sellers_view - 1  # last included seller index in current view

    fig_cum = px.line(
        cum_all,
        x=cum_all.index,
        y=["revenues", "cost_of_reviews", "profits_after_it"],
        labels={"index": "# Sellers Included", "value": "BRL"},
        title="Cumulative Revenues, Review Costs & Net Profit (Incl. IT)<br><br>",
    )
    # Optimum markers
    fig_cum.add_vline(
        x=opt_with_it_idx, line_dash="dash", line_color="red",
        annotation_text="Optimum <br>(incl. IT costs)", annotation_position="top left"
    )
    fig_cum.add_vline(x=opt_no_it_idx, line_dash="dot", line_color="grey")
    fig_cum.add_annotation(
        text="Optimum <br>(excl. IT costs)",
        x=opt_no_it_idx, y=0.5,
        xref="x", yref="paper",
        showarrow=False,
        font=dict(color="black", size=12),
        xanchor="left", align="left"
    )
    # Shade removed sellers region (only when not 'all')
    if mode != "all":
        fig_cum.add_vrect(
            x0=cut_x + 0.5, x1=N - 1 + 0.5,
            fillcolor="LightGreen", opacity=0.15, line_width=0,
            annotation_text="Removed sellers", annotation_position="top right"
        )
    fig_cum.update_layout(
        legend_title_text="",
        margin=dict(l=10, r=10, t=60, b=60),
        height=520,
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="right", x=1),
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

    # --- Profit curves (global) ---
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

    # --- Build dynamic layout ---
    if mode == "all":
        # Stack: cumulative over curves; hide impact
        plots_children = [
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_cum, config={"displayModeBar": False}), md=12)], className="my-2"),
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_curves, config={"displayModeBar": False}), md=12)], className="my-2"),
        ]
    else:
        # Reflow: cumulative + curves side-by-side, impact below
        plots_children = [
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_cum, config={"displayModeBar": False}), md=6),
                dbc.Col(dcc.Graph(figure=fig_curves, config={"displayModeBar": False}), md=6),
            ], className="my-2"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_imp, config={"displayModeBar": False}), md=12),
            ], className="my-2"),
        ]

    return (plots_children,
            kpi_rev, kpi_costs, kpi_net, kpi_margin, kpi_delta, kpi_n,
            rev_sub, costs_sub, net_sub, margin_sub, delta_sub, n_sub)
