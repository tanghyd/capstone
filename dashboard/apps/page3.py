# import dash_core_components as dcc
# import dash_html_components as html
# from dash.dependencies import Input, Output

# # import our main dash app variable from the app.py file
# from app import app

# # specify an html layout for this app's page
# layout = html.Div([
#     html.H3('Page 3'),  # header name
#     dcc.Dropdown(
#         id='page-3-dropdown',
#         options=[{'label': f'Page 3 - {i}', 'value': i} for i in ['NYC', 'MTL', 'LA']],
#     ),
#     html.Div(id='page-3-display-value'),  # we will output the result from our dropdown here
#     #dcc.Link('Go to App 2', href='/page-3')  # html link to /apps/app2
#     ]
# )

# # handle the user interactivity for our dropdown
# @app.callback(
#     Output('page-3-display-value', 'children'),
#     [Input('page-3-dropdown', 'value')])
# def display_value(value):  # define the function that will compute the output of the dropdown
#     return f'You have selected "{value}"'

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from sklearn import datasets
from sklearn.cluster import KMeans

# import our main dash app variable from the app.py file
from app import app

iris_raw = datasets.load_iris()
iris = pd.DataFrame(iris_raw["data"], columns=iris_raw["feature_names"])

controls = dbc.Card(
    [
        dbc.FormGroup(
            [
                dbc.Label("X variable"),
                dcc.Dropdown(
                    id="x-variable",
                    options=[
                        {"label": col, "value": col} for col in iris.columns
                    ],
                    value="sepal length (cm)",
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Y variable"),
                dcc.Dropdown(
                    id="y-variable",
                    options=[
                        {"label": col, "value": col} for col in iris.columns
                    ],
                    value="sepal width (cm)",
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Cluster count"),
                dbc.Input(id="cluster-count", type="number", value=3),
            ]
        ),
    ],
    body=True,
)

layout = dbc.Container(
    [
        html.H1("Iris k-means clustering"),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(controls, md=4),
                dbc.Col(dcc.Graph(id="cluster-graph"), md=8),
            ],
            align="center",
        ),
        html.Div("Note: Dashboard will error if Cluster ccount field left empty."),

    ],
    fluid=True,
)

@app.callback(
    Output("cluster-graph", "figure"),
    [Input("x-variable", "value"),
    Input("y-variable", "value"),
    Input("cluster-count", "value")])
def make_graph(x, y, n_clusters):
    # minimal input validation, make sure there's at least one cluster
    km = KMeans(n_clusters=max(n_clusters, 1))
    df = iris.loc[:, [x, y]]
    km.fit(df.values)
    df["cluster"] = km.labels_

    centers = km.cluster_centers_
    
    # specify data as three graph objects (scatter plot) with x and y data for each cluster
    data = [
        go.Scatter(
            x=df.loc[df.cluster == c, x],
            y=df.loc[df.cluster == c, y],
            mode="markers",
            marker={"size": 8},
            name="Cluster {}".format(c),
        )
        for c in range(n_clusters)
    ]

    # append the cluster centers as a scatter plot to the data list
    data.append(
        go.Scatter(
            x=centers[:, 0],
            y=centers[:, 1],
            mode="markers",
            marker={"color": "#000", "size": 12, "symbol": "diamond"},
            name="Cluster centers",
        )
    )

    layout = {"xaxis": {"title": x}, "yaxis": {"title": y}}

    # return a graph object (a "Figure") with data (list of 4 scatter plots) and a layout
    return go.Figure(data=data, layout=layout)


# make sure that x and y values can't be the same variable
def filter_options(v):
    """Disable option v"""
    return [
        {"label": col, "value": col, "disabled": col == v}
        for col in iris.columns
    ]

# functionality is the same for both dropdowns, so we reuse filter_options
app.callback(Output("x-variable", "options"), [Input("y-variable", "value")])(
    filter_options
)
app.callback(Output("y-variable", "options"), [Input("x-variable", "value")])(
    filter_options
)