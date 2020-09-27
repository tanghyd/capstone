import dash_core_components as dcc
import dash_bootstrap_components as dbc
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

# this should also be loaded once, and then is subsetted when called back.
# it is important to only read what is required to display -- reading all then subsetting will not reduce load time
df = pd.read_csv('data/group_all_labelled.csv', usecols=['group','filename', 'Near Miss Event','event_text', 'reviewed'], nrows=50)
df['label'] = df['Near Miss Event'].astype(int)
df = df.loc[df.reviewed, ['group', 'filename', 'event_text', 'label']]  # only show reviewed events but drop column after subset


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
    html.H4(children='Extracted Labelled Events'),
    #generate_table(df),
    dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True)
])