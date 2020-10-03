import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_table

import plotly.express as px

import pandas as pd
import geopandas as gpd
import json

# import our main dash app variable from the app.py file
from app import app

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

# this should also be loaded once, and then is subsetted when called back.
# it is important to only read what is required to display -- reading all then subsetting will not reduce load time
events = pd.read_csv('data/group_all_labelled.csv')
events = events.loc[events.reviewed]

mapdict = {True: 1, False: 0}
events["Near Miss Event_int"] = events["Near Miss Event"].map(mapdict)
events["a_number_text"] = events["filename"].str.split("_").str[0]
events["a_number_int"] = events["a_number_text"].str.replace("a","").astype(int)

reports_grouped = pd.DataFrame(events.groupby(["a_number_int","filename"]).agg({'Near Miss Event_int': 'sum'}).reset_index())

zipfile = "zip://data/geoview/Exploration_Reports_GDA2020_shp.zip"
geoview = gpd.read_file(zipfile)

df = geoview.merge(reports_grouped, left_on='ANUMBER',right_on="a_number_int")

events_timeline = events.loc[events["Near Miss Event"]]
events_timeline = geoview.merge(events_timeline, left_on='ANUMBER',right_on="a_number_int")
events_timeline = events_timeline[["REPORT_YEA","TITLE","ANUMBER","MINERAL","event_text"]]

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
    dcc.Graph(id="graph", style={"width": "75%", "display": "inline-block"}),  
    #dcc.Link('Go to App 2', href='/apps/app2')  # html link to /apps/app2
 html.Div([
            dcc.Markdown("""
                **Selection Data**

                Choose the lasso or rectangle tool in the map's menu
                bar and then select points in the map.
            """),
            html.Div(id='selected-data', style=styles['pre']),
            dash_table.DataTable(id='selected-table',columns=[{"name": i, "id": i} for i in events_timeline.columns]),
        ], className='three columns'),
    ]
)

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

@app.callback(
    Output('selected-table', 'data'),
    [Input('graph', 'selectedData')])
def display_selected_data(selectedData):
    if selectedData is not None:
        selectedANumbers = []
        for i in range(len(selectedData["points"])):
            selectedANumbers.append(selectedData["points"][i]["location"])
        selectedANumbers = pd.DataFrame({"a_number_int":selectedANumbers})
        metadf = events_timeline.loc[:, events_timeline.columns != "geometry"] # remove the geometry info to get only the metadata
        return metadf.merge(selectedANumbers,left_on="ANUMBER",right_on="a_number_int").to_dict("records")
    else:
        return None
# want to return all the events in the selected region by timeline.