import dash
from dash import html, dcc, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# ---------- Page registration ----------
dash.register_page(__name__, name="Products & Sellers", path="/cross-analysis", order=5)

# ---------- Load data ----------
df = pd.read_csv("loss_matrix.csv")
for col in ("Unnamed: 0", "index"):
    if col in df.columns:
        df = df.drop(columns=[col])

required = {"category", "seller_id", "allocated_profit", "items_sold"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"loss_matrix.csv missing columns: {missing}")

# Color range now uses ITEMS (for heatmap coloring)
ITEMS_RANGE = [float(df["items_sold"].min()), float(df["items_sold"].max())]

# Category totals to classify category profitability (for the category scope buttons)
cat_totals = (
    df.groupby("category", as_index=False)["allocated_profit"]
      .sum()
      .rename(columns={"allocated_profit": "category_profit"})
)
df = df.merge(cat_totals, on="category", how="left")

DEFAULT_CATEGORY = (
    cat_totals.assign(abs_profit=lambda x: x["category_profit"].abs())
              .sort_values("abs_profit", ascending=False)["category"]
              .iloc[0]
    if not cat_totals.empty else None
)

# ---------- Controls ----------
lm_cat_mode = dbc.RadioItems(
    id="lm_cat_mode",
    options=[
        {"label": "All categories",      "value": "all"},
        {"label": "Only profitable",     "value": "profit"},
        {"label": "Non profitable only", "value": "loss"},
    ],
    value="all",
    inline=True,
    inputClassName="me-1",
    labelStyle={"marginRight": "1rem"},
)

lm_cat_filter = dcc.Dropdown(
    id="lm_cat_filter",
    options=[],
    value=None,        # single category at a time
    multi=False,
    placeholder="Select one category",
    maxHeight=300,
)

lm_seller_mode = dbc.RadioItems(
    id="lm_seller_mode",
    options=[
        {"label": "All sellers",         "value": "all"},
        {"label": "Only profitable",     "value": "profit"},
        {"label": "Non profitable only", "value": "loss"},
    ],
    value="all",
    inline=True,
    inputClassName="me-1",
    labelStyle={"marginRight": "1rem"},
)

controls = dbc.Container(
    [
        html.H1("Cross Analysis — Sellers vs Categories", className="text-center my-3 mb-4"),
        # dbc.Row([dbc.Col(html.H1("Cross Analysis — Sellers vs Categories"), md=12)]),
        dbc.Row([dbc.Col(html.Div([html.Small("Category scope:"), lm_cat_mode]), md=12, className="mb-2")]),
        dbc.Row([dbc.Col(lm_cat_filter, md=12, className="mb-2")]),
        dbc.Row([dbc.Col(html.Div([html.Small("Seller scope (filters bars):"), lm_seller_mode]), md=12, className="mb-2")]),
    ],
    fluid=False,
)

# ---------- Graph ----------
lm_plot = dcc.Graph(id="lm_plot")

layout = dbc.Container([controls, 
                        dbc.Row([dbc.Col(lm_plot, width=12)])], 
                        fluid=False)

# ---------- Helpers ----------
def filter_categories_by_mode(dframe: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "profit":
        return dframe[dframe["category_profit"] > 0]
    elif mode == "loss":
        return dframe[dframe["category_profit"] <= 0]
    return dframe

def filter_sellers_by_mode(dframe: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "profit":
        return dframe[dframe["allocated_profit"] > 0]
    elif mode == "loss":
        return dframe[dframe["allocated_profit"] <= 0]
    return dframe

def pick_default_category(available_categories: list[str]) -> str | None:
    if not available_categories:
        return None
    if DEFAULT_CATEGORY in available_categories:
        return DEFAULT_CATEGORY
    return sorted(available_categories)[0]

def make_empty_fig(msg: str, height: int = 420) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        height=height,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[dict(text=msg, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)],
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig

def build_ranked_bar(dframe: pd.DataFrame) -> go.Figure:
    if dframe.empty:
        return make_empty_fig("No sellers match the current filters.")

    d = dframe.sort_values("allocated_profit", ascending=False).copy()
    n = len(d)
    d["rank"] = np.arange(1, n + 1)

    # label thinning (height fixed at 420px)
    if n <= 40:
        text = d["items_sold"].map(lambda x: f"{int(x):,}").tolist()
    else:
        if n <= 120:
            step = 5
        elif n <= 240:
            step = 10
        else:
            step = 20
        text = []
        for i, val in enumerate(d["items_sold"]):
            if i < 8 or i >= n - 8 or (i % step == 0):
                text.append(f"{int(val):,}")
            else:
                text.append("")

    # COLOR BY ITEMS SOLD (heatmap)
    fig = px.bar(
        d,
        x="rank",
        y="allocated_profit",
        color="items_sold",
        color_continuous_scale=[
            (0.0, "red"),
            (0.25, "yellow"),
            (1.0, "green")
        ],  
     # sequential scale fits quantity
        range_color=ITEMS_RANGE,
        labels={"rank": "Seller Rank", "allocated_profit": "Allocated Profit", "items_sold": "Items Sold"},
        title="Seller Profitability (ranked)",
        text=text,
    )
    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
        texttemplate="%{text}",
        hovertemplate=(
            "<b>Rank:</b> %{x}<br>"
            "<b>Items (color):</b> %{marker.color:,}<br>"
            "<b>Profit:</b> %{y:,}<br>"
            "<b>Seller:</b> %{customdata[0]}<br>"
            "<b>Category:</b> %{customdata[1]}<extra></extra>"
        ),
        customdata=np.stack([d["seller_id"].values, d["category"].values], axis=-1),
    )

    if n <= 40:
        tick_step = 1
    elif n <= 120:
        tick_step = 5
    elif n <= 240:
        tick_step = 10
    else:
        tick_step = 20

    fig.update_layout(
        height=420,
        margin=dict(l=60, r=20, t=60, b=60),
        coloraxis_colorbar=dict(title="Items Sold"),
        xaxis=dict(
            title="Seller Rank",
            tickmode="array",
            tickvals=list(range(1, n + 1, tick_step)),
            ticktext=[str(v) for v in range(1, n + 1, tick_step)],
        ),
        yaxis=dict(title="Allocated Profit"),
    )
    return fig

# ---------- Callback ----------
@callback(
    Output("lm_cat_filter", "options"),
    Output("lm_cat_filter", "value"),
    Output("lm_plot", "figure"),
    Input("lm_cat_mode", "value"),
    Input("lm_cat_filter", "value"),
    Input("lm_seller_mode", "value"),
)
def sync_controls_and_plot(cat_mode_value, selected_category, seller_mode_value):
    df_cat_scope = filter_categories_by_mode(df, cat_mode_value)
    mode_categories = sorted(df_cat_scope["category"].unique().tolist())
    options = [{"label": c, "value": c} for c in mode_categories]

    if selected_category not in mode_categories:
        new_value = pick_default_category(mode_categories)
    else:
        new_value = selected_category

    if new_value is None:
        return options, None, make_empty_fig("No categories available in this mode.")

    df_view = df_cat_scope[df_cat_scope["category"] == new_value]
    df_view = filter_sellers_by_mode(df_view, seller_mode_value)
    fig = build_ranked_bar(df_view)

    return options, new_value, fig
