### WAMEX XML ETL FOR REPORT METADATA ##
import os
from datetime import datetime

import geopandas as gpd
import pandas as pd

from pipeline.preprocessing.text import load_files

default_headers = ['ANUMBER', 'TITLE', 'REPORT_YEA', 'AUTHOR_NAM', 'AUTHOR_COM', 'REPORT_TYP',
                   'DATE_FROM', 'DATE_TO', 'PROJECT', 'OPERATOR', 'ABSTRACT', 'KEYWORDS',
                   'TARGET_COM', 'DATE_RELEA', 'geometry']

# ETL from reports zip file to dimension table for geoview
def get_geoview_data(data_file='Exploration_Reports_GDA2020_shp.zip',
                     data_folder='data',
                     zip=True,
                     headers=None,
                     all=False):
    # pathlib did not work with zip://

    if zip:
        # read geoview metadata from zip
        file_path = os.path.join('zip://',data_folder, 'geoview', data_file)
    else:
        file_path = os.path.join(data_folder, 'geoview', data_file)

    headers = default_headers if headers is None else headers

    # load file and subset on headers
    geoview = gpd.read_file(file_path)
    geoview = geoview.loc[:, headers]

    if not all:  # subset for capstone group reports only
        capstone_files = get_report_data(data_folder=data_folder)  # run other utils function for files
        geoview = geoview.loc[geoview.ANUMBER.isin(
                capstone_files.anumber)]  # at this stage we have not lower-cased geoview's ANUMBER column

    # DATA CLEANING
    geoview = geoview.loc[geoview.REPORT_YEA.notna()]
    geoview.loc[geoview.REPORT_YEA == 9877, 'REPORT_YEA'] = 1977  # 9877 is invalid year - text data said 1977

    # convert year as int/floats (will be float if NA's are present in column) to datetime
    # DATE_TO Column has (5) NA values
    geoview.insert(len(geoview.columns) - 2, 'report_year',
                   geoview.REPORT_YEA.apply(lambda x: datetime.strptime(str(int(x)), '%Y')))
    geoview.insert(len(geoview.columns) - 1, 'date_from',
                   geoview.DATE_FROM.apply(lambda x: datetime.strptime(x, '%Y-%M-%d')))
    geoview.insert(len(geoview.columns) - 1, 'date_to', geoview.DATE_TO.apply(
            lambda x: datetime.strptime(x, '%Y-%M-%d') if pd.notna(x) else pd.NaT))

    # aggregate date_from and date_to into year buckets
    geoview.insert(len(geoview.columns) - 1, 'year_from', geoview.date_from.map(lambda x: x.year).apply(
            lambda x: datetime.strptime(str(x), "%Y")))
    geoview.insert(len(geoview.columns) - 1, 'year_to', geoview.date_to.map(lambda x: x.year).apply(
            lambda x: datetime.strptime(str(int(x)), "%Y") if pd.notna(x) else pd.NaT))

    geoview = geoview.rename(columns={
        'AUTHOR_NAM': 'AUTHOR', 'AUTHOR_COM': 'COMPANY', 'REPORT_TYP': 'REPORT_TYPE',
        'DATE_RELEA': 'DATE_RELEASED', 'TARGET_COM': 'COMMODITY'
    })
    geoview.drop(columns=['DATE_FROM', 'DATE_TO', 'REPORT_YEA'], inplace=True)
    geoview.columns = geoview.columns.str.lower()

    return geoview


# create dataframe for filename and a-number data
def get_report_data(data_folder='data', count_sentences=False, return_files=False):
    # get filenames in wamex data folder directory and save to filenames - does not load .json files
    wamex_folder_path = os.path.join(data_folder, 'wamex_xml')
    filenames = [file.split('/', 4)[-1] for file in os.listdir(wamex_folder_path) if file.split('.', 1)[-1] == 'json']
    anumbers = [int(file.split("_", 1)[0].replace("a", "")) for file in filenames] # get anumbers from file string
    #tenements
    data = {'filename': filenames, 'anumber': anumbers}
   
    if return_files or count_sentences:  
        # then we need to load files
        files = load_files(filenames, data_path=wamex_folder_path)

        # get number of sentences in json
        if count_sentences:
            sentence_count = [len(sentences) for sentences in files.values()]
            data = {**data, 'sentence_count': sentence_count}

        if return_files:
            return pd.DataFrame.from_dict(data), files
        else:
            return pd.DataFrame.from_dict(data)

    return pd.DataFrame.from_dict(data)



