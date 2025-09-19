import dash
from dash import html, dcc, Input, Output, callback, no_update
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Register page
dash.register_page(__name__, name="Category", path="/category-analysis", order=4)

# Load data
cat_overview = pd.read_csv("cat_overview.csv")

# --- constants for dynamic height ---
MAX_HEIGHT = 1200  # height when showing all categories (within current mode)
MIN_HEIGHT = 450   # reduced height when showing few categories

# Stable color range across all filters/modes
COLOR_RANGE = [
    float(cat_overview["total_profits"].min()),
    float(cat_overview["total_profits"].max()),
]

# --- Controls ---
profit_mode = dbc.RadioItems(
    id="profit_mode",
    options=[
        {"label": "All categories",      "value": "all"},
        {"label": "Only profitable",     "value": "profit"},
        {"label": "Non profitable only", "value": "loss"},
    ],
    value="all",
    inline=True,
    inputClassName="me-1",
    labelStyle={"marginRight": "1rem"},
    className="mb-2",
)

category_filter = dcc.Dropdown(
    id="cat_filter",
    options=[],           # set dynamically below
    value=[],             # [] => show all in the current mode
    multi=True,
    placeholder="Select categories (leave empty for all)",
    maxHeight=300,
    closeOnSelect=False,  # keep menu open while selecting multiple
)

controls = dbc.Row(
    [
        dbc.Col(profit_mode, md=12),
        dbc.Col(category_filter, md=12, className="mb-3"),
    ]
)

# Graph placeholder
cat_overview_plot = dcc.Graph(id="cat_overview_plot")

# Layout
layout = dbc.Container(
    [
        html.H1("Category Analysis", className="text-center my-3 mb-4"),
        controls,
        dbc.Row([dbc.Col(cat_overview_plot, width=12)]),
    ],
    fluid=False,
)

# --- helper functions ---
def filter_by_mode(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "profit":
        return df[df["total_profits"] > 0]
    elif mode == "loss":
        return df[df["total_profits"] <= 0]
    return df

def make_empty_fig(height: int, msg: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        height=height,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[dict(text=msg, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)],
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig

def make_bar(df: pd.DataFrame, height: int) -> go.Figure:
    if df.empty:
        return make_empty_fig(height, "No categories match the current filters.")

    d = df.sort_values("total_quantity")
    fig = px.bar(
        d,
        x="total_quantity",
        y="category",
        color="total_profits",
        color_continuous_scale=[(0.0, "red"), (0.8, "yellow"), (1.0, "green")],
        range_color=COLOR_RANGE,  # keep consistent even when filtered
        title="Quantity of Items Sold per Category (colored by Profit)",
        labels={"total_quantity": "Items Sold", "total_profits": "Profit"},
        text="total_quantity",  # show values on bars
    ).update_traces(
        textposition="outside",
        cliponaxis=False,
        texttemplate="%{text:,}",  # 12,345 formatting
        hovertemplate="<b>%{y}</b><br>Items: %{x:,}<br>Profit: %{marker.color:,}<extra></extra>",
    ).update_layout(
        yaxis=dict(tickfont=dict(size=10)),
        height=height,
        coloraxis_colorbar=dict(title="Profit"),
        margin=dict(l=80, r=20, t=60, b=40),
    )
    return fig

# --- callback: dynamic options + figure (keep selection in sync) ---
@callback(
    Output("cat_filter", "options"),
    Output("cat_filter", "value"),
    Output("cat_overview_plot", "figure"),
    Input("profit_mode", "value"),
    Input("cat_filter", "value"),
)
def update_controls_and_chart(mode, selected_categories):
    # 1) Filter dataset by profit mode
    df_mode = filter_by_mode(cat_overview, mode)

    # 2) Dynamic dropdown options for the current mode
    mode_categories = sorted(df_mode["category"].unique().tolist())
    options = [{"label": c, "value": c} for c in mode_categories]

    # 3) Reconcile current selections with available options
    selected_set = set(selected_categories or [])
    valid_set = selected_set.intersection(mode_categories)

    if selected_categories is None:
        new_value = []
    elif not valid_set and selected_set:
        # Previously chosen categories no longer available -> clear selection (show all)
        new_value = []
    else:
        new_value = sorted(valid_set)

    # 4) Apply category filter (if any)
    if new_value:
        df_view = df_mode[df_mode["category"].isin(new_value)]
        n_selected = len(new_value)
    else:
        df_view = df_mode
        n_selected = len(mode_categories)

    # 5) Dynamic height scales within the current mode
    n_total_mode = max(len(mode_categories), 1)
    frac = n_selected / n_total_mode
    height = int(MIN_HEIGHT + (MAX_HEIGHT - MIN_HEIGHT) * frac)

    # 6) Build figure
    fig = make_bar(df_view, height=height)

    return options, new_value, fig
