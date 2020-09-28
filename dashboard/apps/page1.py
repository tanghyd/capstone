import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import pandas as pd

# import our main dash app variable from the app.py file
from app import app

# this should also be loaded once, and then is subsetted when called back.
# it is important to only read what is required to display -- reading all then subsetting will not reduce load time
df = pd.read_csv('data/group_all_labelled.csv', usecols=['group','filename','reviewed'], nrows=50)
df = df.loc[df.reviewed, ['group', 'filename']]

options = [{'label': filename, 'value': filename} for filename in df.filename.unique()]

# specify an html layout for this app's page
layout = html.Div([
    html.H3('Page 1'),  # header name
    dcc.Dropdown(
        id='page-1-dropdown',
        options=options,
    ),
    html.Div(id='page-1-display-value'),  # we will output the result from our dropdown here
    #dcc.Link('Go to App 2', href='/apps/app2')  # html link to /apps/app2
    ]
)

# handle the user interactivity for our dropdown
@app.callback(
    Output('page-1-display-value', 'children'),
    [Input('page-1-dropdown', 'value')])
def display_value(value):  # define the function that will compute the output of the dropdown
    return f'You have selected "{value}"'