import dash_bootstrap_components as dbc
import dash_html_components as html

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    'position': 'fixed',
    'top': 0,
    'left': 0,
    'bottom': 0,
    'width': '16rem',
    'padding': '2rem 1rem',
    'background-color': '#f8f9fa',
}

# the styles for the main content position it to the right of the side bar and add some padding
CONTENT_STYLE = {
    'margin-left': '18rem',
    'margin-right': '2rem',
    'padding': '2rem 1rem',
}

NAVBAR_STYLE = {
    'position': 'relative',
    'left': 0,
    'right': 12,
    'margin-left': '16rem', # left margin starts at 16rem width of sidebar
    'padding': '1rem 1rem',
}

sidebar = html.Div(
    [
        html.H2("WAMEX", className="display-4"),  # sidebar header(?)
        html.Hr(),
        html.P(
            "Natural Language Processing Dashboard UWA MDS", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Reports", href="/page-1", id="page-1-link"),
                dbc.NavLink("Events", href="/page-2", id="page-2-link"),
                dbc.NavLink("Clusters", href="/page-3", id="page-3-link"),
                dbc.NavLink("Events Map", href="/page-4", id="page-4-link"),
            ],
            vertical=True,  # ?
            pills=True,  # ?
        ),
    ],
    style=SIDEBAR_STYLE
)