import pandas as pd

from pipeline.preprocessing.text import load_spacy_model, match_triggers
from pipeline.data.metadata import get_geoview_data, get_report_data

default_event_cols = ['event_id', 'filename', 'event_text', 'label']

def load_event_data(nlp=None, cols=None, confidence='high'):
    nlp = nlp or load_spacy_model(output_type='text', lemmatizer=True, geological_matcher=False,
    stopword_removal=False, punctuation_removal=True, verbose=False)

    cols = default_event_cols if cols is None else cols
    df = pd.read_csv(f'data/events/events_{confidence}-conf.csv', index_col=0, usecols=cols)

    df['tokens'] = list(nlp.pipe(df.event_text.values))

    return df

def load_metadata(data_folder='data'):
    ''' Function to load wamex_xml file names and geoview metadata and return dataframe '''
    report_anumbers = get_report_data(data_folder=data_folder)
    geoview = get_geoview_data(data_folder=data_folder, all=False)
    data = report_anumbers.merge(geoview, on='anumber')

    return data

def load_metadata_with_triggers(nlp=None, num_files=None, data_folder='data'):
    nlp = nlp or load_spacy_model(output_type='text', tokenizer_only=True)
    metadata = load_metadata()
    filenames = metadata.filename.tolist()
    if num_files != None:
        filenames = filenames[:num_files]
    file_triggers = match_triggers(filenames, nlp=nlp, save_json=False)
    data = file_triggers.merge(metadata, on='filename')

    return data