import pandas as pd

from pipeline.preprocessing.processing import load_spacy_model, match_triggers
from pipeline.data.metadata import get_geoview_data, get_report_anumbers

default_event_cols = ['event_id', 'filename', 'group', 'sentence_text', 'event_text', 'Near Miss Event', 'reviewed']

def load_event_data(nlp=None, cols=None, path='events/group_all_labelled.csv'):
    nlp = nlp or load_spacy_model(output_type='text', tokenizer_only=True)
    cols = default_event_cols if cols is None else cols

    df = pd.read_csv(path, usecols=cols)
    df['labels'] = df['Near Miss Event'].astype(int)
    df['tokens'] = list(nlp.pipe(df.event_text.values))

    return df

def load_metadata(data_folder='data'):
    ''' Function to load wamex_xml file names and geoview metadata and return dataframe '''
    report_anumbers = get_report_anumbers(data_folder=data_folder)
    geoview = get_geoview_data(data_folder=data_folder, all=False)
    data = report_anumbers.merge(geoview, on='anumber')

    return data

def load_metadata_with_triggers(nlp=None, num_files=100, data_folder='data'):
    nlp = nlp or load_spacy_model(output_type='text', tokenizer_only=True)
    metadata = load_metadata()
    filenames = metadata.filename.tolist()[:num_files]
    file_triggers = match_triggers(filenames, nlp=nlp, save_json=False)
    data = file_triggers.merge(metadata, on='filename')

    return data