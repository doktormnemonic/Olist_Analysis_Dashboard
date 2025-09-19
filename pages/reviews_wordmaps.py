import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS

# --------- Register this page at a custom route ---------
dash.register_page(__name__, name="Reviews Word Cloud", path="/reviews-wordcloud", order=6)

# Create this file in your notebook pipeline and place next to your CSVs.
REVIEWS_PATH_BR = "bad_reviews_br.txt"
try:
    with open(REVIEWS_PATH_BR, "r", encoding="utf-8") as f:
        BAD_REVIEWS_TEXT_BR = f.read()
except FileNotFoundError:
    BAD_REVIEWS_TEXT_BR = ""  # empty triggers placeholder text below

REVIEWS_PATH_EN = "bad_reviews_content_translated.txt"
try:
    with open(REVIEWS_PATH_EN, "r", encoding="utf-8") as f:
        BAD_REVIEWS_TEXT_EN = f.read()
except FileNotFoundError:
    BAD_REVIEWS_TEXT_EN = ""  # empty triggers placeholder text below

# --------- Helper: build word cloud as a Plotly figure ---------
def make_wordcloud_figure(
    text: str,
    extra_stops: set | None = None,
    width: int = 1000,
    height: int = 700,
    bg: str = "white",
    colormap: str = "viridis"  # try "viridis", "plasma", "cividis", "magma", etc.
):
    # English base stopwords + ecommerce review noise terms
    review_stops = {
        "product", "item", "buy", "bought", "purchase",
        "order", "ordered", "get", "got", "use", "used",
        "really", "also", "one", "seller", "category", "receive",
        "received", "delivery", "arrived", "didn", "dont", "doesn",
        "im", "ive", "theyre", "youre", "cant", "wont"
        }
    pt_stops = {
    "de","do","da","em","no","na","para","com","os","as",
    "um","uma","uns","umas","ao","aos","às","e","que","se",
    "por","pela","pelas","pelos","como","mais","não","nem",
    "já","quando","onde","até","porque","ser","tem","têm",
    "está","estão","foi","era","são","nan"
        }
    stops = STOPWORDS.union(pt_stops) | review_stops | (extra_stops or set())
    # stops = STOPWORDS | review_stops | (extra_stops or set())

    # fallback if text is empty
    text = text if isinstance(text, str) and text.strip() else "no data"

    wc = WordCloud(
        width=width,
        height=height,
        background_color=bg,
        stopwords=stops,
        colormap=colormap,
        prefer_horizontal=0.9,
        random_state=42,
        normalize_plurals=True,
    ).generate(text)

    img = wc.to_array()  # NumPy array (H, W, 3)
    fig = px.imshow(img)
    fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_showscale=False,
        title=dict(text="Bad Reviews Word Cloud", x=0.5, xanchor="center", font=dict(size=22)),
    )
    return fig

# --------- Components ---------
wordcloud_graph = dcc.Graph(
    id="bad_reviews_wordcloud",
    figure=make_wordcloud_figure(BAD_REVIEWS_TEXT_BR, height=700),
    style={"height": "720px", "width": "100%"}
)

wordcloud_graph_2 = dcc.Graph(
    id="bad_reviews_wordcloud",
    figure=make_wordcloud_figure(BAD_REVIEWS_TEXT_EN, height=700),
    style={"height": "720px", "width": "100%"}
)

# --------- Page layout ---------
layout = dbc.Container(
    [
        dbc.Row(
            [dbc.Col(html.H1("Negative Reviews — Word Cloud", className="text-center mb-4"), width=12)]
        ),
        dbc.Row([dbc.Col(wordcloud_graph, width=12)]),
        dbc.Row([dbc.Col(wordcloud_graph_2, width=12)])
    ],
    fluid=False,
)