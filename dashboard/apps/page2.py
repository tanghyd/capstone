import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd

# import our main dash app variable from the app.py file
from app import app

# # specify an html layout for this app's page
# layout = html.Div([
#     html.H3('Page 2'),  # header name
#     dcc.Dropdown(
#         id='page-2-dropdown',
#         options=[{'label': f'Page 2 - {i}', 'value': i} for i in ['NYC', 'MTL', 'LA']],
#     ),
#     html.Div(id='page-2-display-value'),  # we will output the result from our dropdown here
#    # dcc.Link('Go to App 1', href='/apps/app1')  # html link to /apps/app2
#     ]
# )
# # handle the user interactivity for our dropdown
# @app.callback(
#     Output('app-2-display-value', 'children'),
#     [Input('app-2-dropdown', 'value')])
# def display_value(value):  # define the function that will compute the output of the dropdown

df = pd.read_csv(
    'https://gist.githubusercontent.com/chriddyp/c78bf172206ce24f77d6363a2d754b59/raw/c353e8ef842413cae56ae3920b8fd78468aa4cb2/usa-agricultural-exports-2011.csv',
    index_col=0)

def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

layout = html.Div(children=[
    html.H4(children='US Agriculture Exports (2011)'),
    generate_table(df)
])