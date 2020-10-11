"""
This file specifies how to handle the index (home) page for the dash application.

First we import our packages and then application directory packages.

This file will then specify the layout attribute of the app (app.layout) with an HTML <div> tag.
i.e. https://www.w3schools.com/Tags/tag_div.asp

In python (specifically dash_html_components.html) we specify an html div tag as html.Div.
The list inside the html.Div is the sequential order that we insert objects into the structure
of the web page.

After specifying the html (front-end) layout, we then handle any interactivity between the user
and the webpage with 'callbacks'.

We write functions that define how this interactivity is handled (i.e. `display_page` below),
and then we use a Python "decorator function" to handle the callback functionality.

Specifically this decorator function (@app.callback) will enable us to:
    - Anytime the html object with id='url' receives input we pass it to the function as a 'pathname'
    - The output returned by this function is passed to the html object with id='page-content'
    
Finally, we run the server in debug mode if the file is run explicitly with app.run_server().
"""

# # import dash modules
# import dash_bootstrap_components as dbc
# import dash_core_components as dcc
# import dash_html_components as html
# from dash.dependencies import Input, Output

# # import application files from directory
# from app import app  # from the app.py file import the app variable
# from apps import app1, app2  # from the apps folder import app1.py and app2.py

# url_bar_and_content_div = html.Div([
#     dcc.Location(id='url', refresh=False),
#     html.Div(id='page-content')
# ])

# #write callback function to handle input and output to dash web app
# @app.callback(Output('page-content', 'children'),
#               [Input('url', 'pathname')])
# def display_page(pathname):
#     if pathname == '/apps/app1':  # if input href is /apps/app1
#         return app1.layout  # display the layout for app1
#     elif pathname == '/apps/app2': 
#         return app2.layout
#     else:
#         return '404'

# if __name__ == '__main__':
#     app.run_server(debug=True, port=8051)

# import dash modules
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# import application files from directory
from dashboard.app import app  # from the app.py file import the app variable
from dashboard.apps import page1, page2, page3, page4 # from the apps folder import app1.py and app2.py
from dashboard.apps.sidebar import sidebar, CONTENT_STYLE
from dashboard.apps.navbar import navbar

# url_bar_and_content_div = html.Div([
#     dcc.Location(id='url', refresh=False),
#     html.Div(id='page-content'),
#     style=CONTENT_STYLE)
# ])


#nav = Navbar(style=NAVBAR_STYLE)

content = html.Div(id='page-content', style=CONTENT_STYLE)

app.layout = html.Div(
    [dcc.Location(id="url", refresh=False),
    sidebar,
    navbar,
    content,
    # Hidden div inside the app that stores the intermediate value
    html.Div(id='intermediate-value', style={'display': 'none'})
])

# this calllback uses the current pathname to set the activate state of the
# corresponding nav link to true, allowing users to tell which page they are on
@app.callback(
    [Output(f"page-{i}-link", "active") for i in range(1,5)],
    [Input("url", "pathname")])
def toggle_active_links(pathname):
    if pathname == "/":
        # treat page 1 as the homepage / index
        return True, False, False, False 
    return [pathname == f"/page-{i}" for i in range(1,5)]

#write callback function to take page pathname as input, and outputs the layout to page-content
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname in ["/", '/page-1', "/home"]:
        return page1.layout  # display the layout for app1
    elif pathname == '/page-2': 
        return page2.layout
    elif pathname == '/page-3': 
        return page3.layout
    elif pathname == '/page-4': 
        return page4.layout
    else:
        return dbc.Jumbotron(
            [
                html.H1("404: Not found", className="text-danger"),
                html.Hr(),
                html.P(f"The pathname {pathname} was not recognised...")
            ]
        )

if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
