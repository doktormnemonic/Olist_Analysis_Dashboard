# main.py
import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc

# If any page defines callbacks at import-time, import them here (optional, helpful):
# import pages.some_page_with_callbacks  # <- add as needed

app = Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,  # ✅ allow callbacks whose targets aren't in root layout yet
)

# --- order helper (keeps your desired order if pages set `order=` in dash.register_page) ---
def ordered_pages():
    return sorted(
        [p for p in dash.page_registry.values() if p.get("path") is not None],
        key=lambda x: x.get("order", 999)
    )

# Build simple nav links (no dropdown). Home is explicit; others come from page registry.
nav_links = [
    dbc.NavItem(dbc.NavLink("Home", href="/")),
    *[
        dbc.NavItem(dbc.NavLink(page["name"], href=page["relative_path"]))
        for page in ordered_pages()
        if page["relative_path"] != "/"  # avoid duplicating Home
    ]
]

navbar = dbc.NavbarSimple(
    children=nav_links,
    brand="Olist Analysis",
    brand_href="#",
    color="#442976",
    dark=True,
)

# Root layout
app.layout = html.Div([
    dcc.Location(id="url"),  # helps routing + some callbacks
    navbar,
    dash.page_container
])

# -------- Validation layout (superset of IDs used by callbacks anywhere) --------
# Add one stub per ID with the right component type. These make Dash validation happy at startup.
validation_stubs = html.Div([
    # IDs used on the Big Picture page (figures):
    dcc.Graph(id="big_rev_fig"),
    dcc.Graph(id="big_cost_fig"),
    dcc.Graph(id="big_profit_fig"),
    dcc.Graph(id="big_profit_hist"),

    # If other pages/callbacks reference other IDs, add them here too, e.g.:
    # dcc.Graph(id="cumulative_profit_plot"),
    # dcc.Dropdown(id="category_dropdown"),
    # dcc.Slider(id="fee_percent_slider"),
    # dcc.Checklist(id="toggle_subscriptions"),
    # dcc.Store(id="filters_store"),
])

# Teach Dash how to validate callbacks across pages
app.validation_layout = html.Div([dash.page_container, validation_stubs])
# -------------------------------------------------------------------------------

if __name__ == '__main__':
    app.run(debug=False)
