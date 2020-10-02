import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import plotly.express as px

import pandas as pd
import geopandas as gpd
import json

# import our main dash app variable from the app.py file
from app import app

# this should also be loaded once, and then is subsetted when called back.
# it is important to only read what is required to display -- reading all then subsetting will not reduce load time
df = pd.read_csv('data/group_all_labelled.csv')
df = df.loc[df.reviewed]

mapdict = {True: 1, False: 0}
df["Near Miss Event_int"] = df["Near Miss Event"].map(mapdict)

reports_grouped = pd.DataFrame(df.groupby("filename").agg({'Near Miss Event_int': 'sum'}).reset_index())
reports_grouped["a_number_text"] = reports_grouped["filename"].str.split("_").str[0]
reports_grouped["a_number_int"] = reports_grouped["a_number_text"].str.replace("a","").astype(int)

zipfile = "zip://data/geoview/Exploration_Reports_GDA2020_shp.zip"
geoview = gpd.read_file(zipfile)

df = geoview.merge(reports_grouped, left_on='ANUMBER',right_on="a_number_int")

#plotlydf = df[["ANUMBER","geometry","Near Miss Event_int"]]
#plotlydf.to_file("data/geofile.json",driver="GeoJSON") # this part shouldn't need to happen every time we run the app

with open("data/geofile.json") as geofile:
    j_file = json.load(geofile)

options = [{'label': filename, 'value': filename} for filename in df.TARGET_COM.unique()]

# specify an html layout for this app's page
layout = html.Div([
    html.H3('Page 4'),  # header name
    dcc.Dropdown(
        id='page-4-dropdown',
        options=options,
    ),
    html.Div(id='page-4-display-value'),
    dcc.Graph(id="graph", style={"width": "75%", "display": "inline-block"}),  # we will output the result from our dropdown here
    #dcc.Link('Go to App 2', href='/apps/app2')  # html link to /apps/app2
    ]
)

# handle the user interactivity for our dropdown
@app.callback(
    Output('page-4-display-value', 'children'),
    [Input('page-4-dropdown', 'value')])
def display_value(value):  # define the function that will compute the output of the dropdown
    return f'You have selected "{value}"'

@app.callback(
    Output("graph","figure"),
    [Input('page-4-dropdown', 'value')])
def make_map(value):
    fdf = df[df["TARGET_COM"]==value]
    return px.choropleth_mapbox(fdf, geojson=j_file, locations=fdf["ANUMBER"],
                           color='Near Miss Event_int',
                           featureidkey="properties.ANUMBER",
                           color_continuous_scale="Viridis",
                           mapbox_style="carto-positron",
                           zoom=3, 
                           center = {"lat": -27, "lon": 121.5},
                           opacity=0.5,
                           labels={'Near Miss Event_int':'# of near miss events'}
                          )