### WAMEX XML ETL FOR REPORT METADATA ###
print('metadata.py')

import os
import pandas as pd

# ETL from reports zip file to dimension table for geoview
def get_geoview_data(data_file='Exploration_Reports_GDA2020_shp.zip', data_folder='data', zip=True, headers=None, all=False):
    import geopandas as gpd
    from datetime import datetime
    import os  # pathlib did not work with zip://

    if zip:
        # read geoview metadata from zip
        file_path = os.path.join('zip://', *(str(data_folder).split('/')), 'geoview', data_file)
    else:
        file_path = os.path.join(*(str(data_folder).split('/')), 'geoview', data_file)

    # only include specified headers
    if headers == None:
        headers = ['ANUMBER', 'TITLE', 'REPORT_YEA', 'AUTHOR_NAM', 'AUTHOR_COM', 'REPORT_TYP', 
                'DATE_FROM', 'DATE_TO', 'PROJECT', 'OPERATOR', 'ABSTRACT', 'KEYWORDS', 
                'TARGET_COM', 'DATE_RELEA', 'geometry']

    # load file and subset on headers
    geoview = gpd.read_file(file_path)
    geoview = geoview.loc[:, headers]

    if not all: # subset for capstone group reports only
        capstone_files = get_report_anumbers(data_folder=data_folder)  # run other utils function for files
        geoview = geoview.loc[geoview.ANUMBER.isin(capstone_files.anumber)]  # at this stage we have not lower-cased geoview's ANUMBER column

    # DATA CLEANING
    geoview = geoview.loc[geoview.REPORT_YEA.notna()]
    geoview.loc[geoview.REPORT_YEA == 9877, 'REPORT_YEA'] = 1977  # 9877 is invalid year - text data said 1977

    # convert year as int/floats (will be float if NA's are present in column) to datetime
    # DATE_TO Column has (5) NA values
    geoview.insert(len(geoview.columns)-2,'report_year', geoview.REPORT_YEA.apply(lambda x : datetime.strptime(str(int(x)), '%Y')))
    geoview.insert(len(geoview.columns)-1, 'date_from', geoview.DATE_FROM.apply(lambda x: datetime.strptime(x, '%Y-%M-%d')))
    geoview.insert(len(geoview.columns)-1, 'date_to', geoview.DATE_TO.apply(
        lambda x: datetime.strptime(x, '%Y-%M-%d') if pd.notna(x) else pd.NaT))

    # aggregate date_from and date_to into year buckets
    geoview.insert(len(geoview.columns)-1, 'year_from', geoview.date_from.map(lambda x : x.year).apply(
        lambda x : datetime.strptime(str(x), "%Y")))
    geoview.insert(len(geoview.columns)-1, 'year_to', geoview.date_to.map(lambda x : x.year).apply(
        lambda x : datetime.strptime(str(int(x)), "%Y") if pd.notna(x) else pd.NaT))

    geoview = geoview.rename(columns={
        'AUTHOR_NAM': 'AUTHOR', 'AUTHOR_COM': 'COMPANY', 'REPORT_TYP': 'REPORT_TYPE',
        'DATE_RELEA': 'DATE_RELEASED', 'TARGET_COM': 'COMMODITY'
    })
    geoview.drop(columns=['DATE_FROM', 'DATE_TO','REPORT_YEA'], inplace=True)
    geoview.columns = geoview.columns.str.lower()

    return geoview

# create dataframe for filename and a-number data
def get_report_anumbers(data_folder='data'):
    import os
    #filenames = [file.split('/',4)[-1] for file in glob.glob(f'{data_folder}/wamex_xml/*.json')]  # does not work on windows due to file path
    wamex_folder_path = wamex_folder_path = os.path.join(*(str(data_folder).split('/')), 'wamex_xml')
    filenames = [file.split('/',4)[-1] for file in os.listdir(wamex_folder_path) if file.split('.',1)[-1] == 'json']
    anumbers = [int(file.split("_",1)[0].replace("a","")) for file in filenames]
    files = pd.DataFrame.from_dict({'filename': filenames, 'anumber': anumbers})
    return files

def get_metadata(data_folder='data'):
    ''' Function to load wamex_xml file names and geoview metadata and return dataframe '''
    report_anumbers = get_report_anumbers(data_folder=data_folder)
    geoview = get_geoview_data(data_folder=data_folder, all=False)
    return report_anumbers.merge(geoview, on='anumber')
