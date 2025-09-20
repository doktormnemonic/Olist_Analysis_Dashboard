import dash
from dash import Dash, html, dcc, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd

# Register page
dash.register_page(__name__, name="About", path="/", order=0)

# Create components
header = html.H1("Olist Analysis Dashboard", style={"textAlign": "center"})

intro = dbc.Card(
    dbc.CardBody([
        # html.H4("About this dashboard", className="card-title mb-3"),
        html.P(
            "This interactive dashboard analyzes Olist’s marketplace profitability from multiple perspectives — "
            "starting with a global overview, then drilling down into sellers, products, categories, and customer sentiment."
        ),
        html.Ul([
            html.Li(html.Span([
                html.B("Executive Overview — "),
                "Revenue and cost breakdowns, gross vs. net profit, and the distribution of seller performance."
            ])),
            html.Li(html.Span([
                html.B("Sellers Analysis — "),
                "Cumulative profit curves, the impact of IT costs, and identification of optimal vs. underperforming sellers."
            ])),
            html.Li(html.Span([
                html.B("Product Analysis — "),
                "Product-level sales and profitability, highlighting best- and worst-performing items."
            ])),
            html.Li(html.Span([
                html.B("Category Analysis — "),
                "Comparison of item volumes with profitability across categories, revealing where high sales don’t always mean high profit."
            ])),
            html.Li(html.Span([
                html.B("Cross Analysis (Product × Seller) — "),
                "Combined view of how products and sellers interact, showing concentration, dependencies, and diversification opportunities."
            ])),
            html.Li(html.Span([
                html.B("Negative Reviews Word Cloud — "),
                "Themes from customer complaints, providing qualitative context to the quantitative analysis."
            ])),
        ], className="mb-3"),
        html.P(
            "Together, these pages turn raw transaction and review data into actionable insights — clarifying where value is created, "
            "where costs erode margins, and where strategic focus can deliver the greatest impact."
        ),
    ]),
    className="shadow-sm"
)

layout = dbc.Container([
    dbc.Row([
        dbc.Col(header, md=12)
    ], className="my-4"),
    dbc.Row([
        dbc.Col(intro, md=10, lg=8, className="mx-auto")
    ], className="mb-5"),
], fluid=True)
