"""
Dynamically Create a Layout for Multi-Page App Validation
Sourced from: https://dash.plotly.com/urls

Dash applies validation to your callbacks, which performs checks such as 
validating the types of callback arguments and checking to see whether the 
specified Input and Output components actually have the specified properties.

For full validation, all components within your callback must therefore appear
in the initial layout of your app, and you will see an error if they do not. 
 
However, in the case of more complex Dash apps that involve dynamic modification
of the layout (such as multi-page apps), not every component appearing in your
callbacks will be included in the initial layout.

New in Dash 1.12 You can set app.validation_layout to a "complete" layout that
contains all the components you'll use in any of the pages / sections.
 
app.validation_layout must be a Dash component, not a function. 
Then set app.layout to just the index layout. 

In previous Dash versions there was a trick you could use to achieve the same result checking 
flask.has_request_context inside a layout function - that will still work but is no longer
recommended.
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import flask

app = dash.Dash(__name__)

url_bar_and_content_div = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# html layout for index (home) page
layout_index = html.Div([
    dcc.Link('Navigate to "/page-1"', href='/page-1'), # create link object with (text, href)
    html.Br(),  # html line break
    dcc.Link('Navigate to "/page-2"', href="/page-2")  # create link object with (text, href)
])

# html layout for page 1
layout_page_1 = html.Div([
    html.H2('Page 1'),  # html header title
    # create text input fields with ids, variable type, default value
    dcc.Input(id='input-1-state', type='text', value='Montreal'),
    dcc.Input(id='input-2-state', type='text', value='Canada'),
    html.Button(id='submit-button', n_clicks=0, children='Submit'),  # create button object

    # this html object has id output-state, used in later callback to print text here
    html.Div(id='output-state'),
    html.Br(),
    dcc.Link('Navigate to "/"', href='/'), # create link object with (text, href)
    html.Br(),
    dcc.Link('Navigate to "/page-2"', href='/page-2'), # create link object with (text, href)
])

# html layout for page 2
layout_page_2 = html.Div([
    html.H2('Page 2'),
    dcc.Dropdown(
        id='page-2-dropdown',
        options=[{'label': i, 'value': i} for i in ['LA', 'NYC', 'MTL']],
        value='LA'
    ),
    html.Div(id='page-2-display-value'),  # html object for "page-2-display-value" output
    html.Br(),
    dcc.Link('Navigate to "/"', href='/'), # create link object with (text, href)
    html.Br(),
    dcc.Link('Navigate to "/page-1"', href='/page-1'), # create link object with (text, href)
])

# index layout
app.layout = url_bar_and_content_div

# "complete" layout - see text at top of file
app.validation_layout = html.Div([
    url_bar_and_content_div,
    layout_index,
    layout_page_1,
    layout_page_2,
])

# index callbacks (interactivity on index homepage)

# outputs the return item ("children") to the html.Div with id page-content 
# the input will be from "url" as the href argument, and passed to the function as "pathname"
@app.callback(Output('page-content', 'children'), 
              [Input('url', 'pathname')])
def display_page(pathname):   # pathname <- href
    if pathname == "/page-1":  # if we click the link ('Navigate to "/page-1"', href='/page-1')
        return layout_page_1   # return the html layout defined in layout_page_1
    elif pathname == "/page-2": # if we click the link ('Navigate to "/page-2"', href='/page-2')
        return layout_page_2  # return the html layout defined in layout_page_2
    else:
        return layout_index  # return the html layout for the index home page

# Page 1 callbacks
@app.callback(Output('output-state', 'children'),  # output-state display object (html.Div)
            [Input('submit-button', 'n_clicks')],  # actual input when user clicks submit button
            [State('input-1-state', 'value'),  # stored state of text typed in text field
            State('input-2-state', 'value')])  # stored state of text typed in text field
def update_output(n_clicks, input1, input2):  # display this text in html.Div output-state
    return (f'The Button has been pressed {n_clicks} times,'
            f'Input 1 is "{input1}",'
            f'and Input 2 is "{input2}"')

# page 2 callbacks
@app.callback(Output('page-2-display-value', 'children'),
              [Input('page-2-dropdown', 'value')])  # input from dropdown passed as value
def display_value(value):
    print('display_value')
    return f'You have selected"{value}"'

if __name__ == '__main__':
    app.run_server(debug=True)