# capstone repo pipeline for labelled training data --> built event text chunks
from pipeline.preprocessing.text import match_triggers, triggers_json_to_df, load_triggers
from pipeline.preprocessing.text import load_files, load_json, load_spacy_model
from pipeline.utils.helpers import to_list
from pipeline.data.metadata import get_report_data, get_geoview_data

import numpy as np
import pandas as pd
import geopandas as gpd

from sklearn.preprocessing import MultiLabelBinarizer

import spacy

from tqdm import tqdm

def extract_text_chunks(filenames, pad=2, skip_on_trigger=False, tokenize=False, nlp=None, n_process=-1, batch_size=100):
    # if a dictionary is loaded where keys are filenames and values are pre-loaded files, we dont load from disk
    if type(filenames) == dict:
        files = filenames
    else:
        # load report files from disk to extract events on triggers
        files = load_files(filenames, data_path='data/wamex_xml', output='dict')
    
    if skip_on_trigger:
        # have not implemented skipping over overlapping text chunks
        pass
    
    if tokenize:
        nlp = nlp or load_spacy_model(output_type='text', lemmatizer=True, geological_matcher=False, 
            stopword_removal=False, punctuation_removal=True, lemmatize_triggers=True, verbose=False)
        
        files = {
            file : list(nlp.pipe(sentences, n_process=n_process, batch_size=batch_size)) 
            for file, sentences in tqdm(files.items(), desc='Tokenizing file text')
         }
        
    
    return {
        file : [' '.join((pad*[''] + sentences + pad*[''])[idx : (1 + idx + (2*pad))]).strip()
                for idx in range(len(sentences))] 
        for file, sentences in tqdm(files.items(), desc='Extracting text chunks')
     }

def build_event_text(event, files=None, pad=0, labelled_ranges=True):
    if files == None:
        _, files = get_report_data(count_sentences=False, return_files=True)
        
    # pad file with extra sentences given pad size
    file = (pad*[''] + files[event.filename] + pad*[''])
    
    # return the customised lower and upper bounds on the labelled event (only near misses)
    if event.label and labelled_ranges:
        lower = event.sentence_idx + event.lower_bound + pad
        upper = event.sentence_idx + 1 + event.upper_bound + pad
    else:
        lower = event.sentence_idx
        upper = event.sentence_idx + 1 + (2*pad)
        
    return ' '.join(file[max(lower,0):min(upper,len(file))]).strip()

def merge_datasets(datasets: dict, confidence, users = None):
    # type checking confidence_threshold parameter input
    assert type(confidence) == str, 'confidence parameter must be a string'
    assert confidence.lower() in ('low','medium','high'), 'confidence parameter must be in ("low","medium","high")'
    
    users = users or ('daniel','charlie')  # specify data labellers
    # subset all datasets on reviewed, assign user column to user name, then concatenate and return
    df = pd.concat([datasets[user].loc[datasets[user].reviewed].assign(user=user) for user in users])

    if confidence.lower() == 'high':
        df.loc[df.confidence != 'High', 'label'] = False
    elif confidence.lower() == 'medium':
        df.loc[df.confidence == 'Low', 'label'] = False
        
    agg_cols = {'label': 'all', 'lower_bound': 'min', 'upper_bound': 'max'}
    remove_cols = ['confidence','user','reviewed']

    # get unique data from df
    df_unique = df.drop(columns=remove_cols)
    df_unique = df_unique.loc[~df_unique[['filename','sentence_idx']].duplicated(keep=False)]

    # merge the cleaned duplicates with the extra column metadata
    df_merged = df.drop(columns=remove_cols + list(agg_cols.keys())).merge(
        df.loc[df[['filename','sentence_idx']].duplicated(keep=False)].groupby(
        ['filename','sentence_idx']).agg(agg_cols).reset_index(),
        on=['filename','sentence_idx'], how='right').drop_duplicates()

    return pd.concat([df_unique, df_merged])
    
def load_group_all_labelled(geoview = None, capstone_files=None, files=None, pad=2):
    import yaml
    if type(geoview) != pd.DataFrame:
        from pipeline.data.metadata import get_geoview_data
        geoview = get_geoview_data()
        
    if (files == None) | (type(capstone_files) != pd.DataFrame):
        from pipeline.data.metadata import get_report_data
        capstone_files, files = get_report_data(count_sentences=True, return_files=True)
        
    # load data and drop any duplicated events
    old_events = pd.read_csv('data/labels/group_all_labelled.csv')
    old_events = old_events.loc[~old_events[['filename','sentence_idx']].duplicated(keep=False)]
    
    # fix string and lists
    for group in (2,3,4,6):
        old_events.loc[old_events.group == group, 'trigger_words_in_sentence'] = old_events.loc[
            old_events.group == group, 'trigger_words_in_sentence'].map(yaml.safe_load).apply(
        lambda triggers : ', '.join(triggers).strip())
   
    # edit old columns and delete unnecessary ones
    old_events.drop(columns=['group','n_trigger_words_in_sentence','sentence_text','n_trigger_words_in_event',
        'trigger_words_in_event', 'n_trigger_words','event_label','reviewed','Key trigger phrase',
        'STRAT','ROCK','LOCATION','MINERAL','ORE_DEPOSIT','TIMESCALE'], inplace=True)
    old_events.rename(columns={'trigger_words_in_sentence':'sentence_triggers','Near Miss Event': 'label'}, inplace=True)
    
    # merge with necessary metadata
    old_events = old_events.merge(capstone_files, on='filename').merge(geoview[['anumber','report_type']], on='anumber')
    
    # add in upper and lower bound on text chunks given n_sentences_extract=2
    old_events['lower_idx'] = old_events.apply(
        lambda row : max(row.sentence_idx - pad, 0), axis=1) 
    old_events['upper_idx'] = old_events.apply(
        lambda row : min(row.sentence_idx + pad, row.sentence_count), axis=1)
    
    # previous event text appeared to be error prone?
    old_events['event_text'] = old_events.apply(lambda row : build_event_text(
        row, pad=pad, labelled_ranges=False, files=files), axis=1)
    
    old_events['event_id'] = old_events['event_id'] + '_old'
    
    return old_events[['event_id','filename','anumber','report_type','sentence_count',
        'sentence_idx','sentence_triggers','event_text', 'label','lower_idx','upper_idx']]

def build_event_data(datasets: dict, confidence, pad=0, batch_size=100, n_process=6, nlp=None,
                     labelled_ranges=True, group_all_labelled=False, named_entities=None,
                     return_entities=True, geoview=None, capstone_files=None, files=None):
    
    # load files if files are not provided
    if (type(capstone_files) != pd.DataFrame) | (files == None):
        capstone_files, files = get_report_data(count_sentences=False, return_files=True)
        
    # merge datasets provided by individual labellers
    df = merge_datasets(datasets, confidence=confidence)
    
    # apply the build event text function to build text chunk from labelled sentences
    df.insert(6,'event_text', df.apply(lambda row : build_event_text(
        row, pad=2, labelled_ranges=labelled_ranges, files=files), axis=1))

    # insert the event_id natural key which is f'{filename}_{idx}'
    df.insert(0, 'event_id', df.apply(lambda row : '_'.join(
        [row.filename.rsplit('.', 1)[0], str(row.sentence_idx)]), axis=1))
    
    # return the text chunk start and end positions, or lower_bound/upper_bound
    if labelled_ranges:
        df['lower_idx'] = df['sentence_idx'] + df['lower_bound']
        df['upper_idx'] = df['sentence_idx'] + df['upper_bound']
    else:
        df['lower_idx'] = df['sentence_idx'] - pad
        df['upper_idx'] = df['sentence_idx'] + pad 
    
    df.rename(columns = {'triggers':'sentence_triggers'}, inplace=True)
    df.drop(columns=['lower_bound','upper_bound'], inplace=True)
    
    # load old event labels from group labelling early in semester
    if group_all_labelled and confidence.lower() != 'high':  # only returns old ones if conf = medium
        old_events = load_group_all_labelled(geoview=geoview, capstone_files=capstone_files, files=files)
        final_index = df.index[-1]
        old_events.index = np.arange(df.index[-1], len(old_events)+df.index[-1])
        df = df.append(old_events)
        
    # run named entity recognition with spacy on text chunk
    if return_entities:
        nlp = nlp or load_spacy_model(output_type='doc', trigger_matcher=True, lemmatizer=False,
            geological_matcher=True, stopword_removal=False, punctuation_removal=False, lemmatize_triggers=True)

        named_entities = named_entities or ['DATE','LOCATION','TRIGGER','STRAT', 'ROCK', 
                                            'LOCATION', 'MINERAL', 'ORE_DEPOSIT', 'TIMESCALE']

        # create a list of tuples for each entity in each event id
        event_entities = [(event_id, ent.text, ent.label_) for event_id, doc in tqdm(zip(df.event_id.values,
            nlp.pipe(df.event_text.values, batch_size=batch_size, n_process=n_process)),
            desc=f'Extracting {confidence} confidence events') for ent in doc.ents if ent.label_ in named_entities]

        # join entity labels together as a string and then merge onto original dataframe
        df = df.merge(pd.DataFrame(data=event_entities, columns=['event_id','entity','label']).groupby(
            ['event_id','label']).apply(lambda x : ', '.join(x.entity)).unstack(level='label'),
                 on='event_id',how='left').fillna('')
        
    assert all(files[event.filename][event.sentence_idx] in event.event_text 
               for event in df.itertuples()), f'sentences not matched in {confidence} confidence event text'
    
    return df

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Specify user source data and confidence thresholds for building events.')
    parser.add_argument('-c', '--confidence', type=str, metavar='N', nargs='+',
                        default='high', help='Confidence required to label as near miss')
    parser.add_argument('-u', '--users', type=str, metavar='N', nargs='+',
                        default=['daniel','charlie'],help='Specify labelling user to source data from')
    parser.add_argument('--all_events', dest='events', action='store_true')
    parser.add_argument('--new_events', dest='events', action='store_false')
    parser.set_defaults(events=False)
    
    args = parser.parse_args()
    
    # specify confidence and user labelling settings 
    confs = (args.confidence,) if type(args.confidence) == str else args.confidence
    users = args.users
    group_all_labelled = args.events
    
    print(f'Building training data from labelled near miss instances by: {" ".join(user for user in users)}')
    print(f'Confidence thresholds to label text chunks as near miss: {" ".join(conf for conf in confs)}')
    if group_all_labelled:
        print('Including old labelled training data.')
    
    # load spacy model
    nlp = load_spacy_model(output_type='doc', trigger_matcher=True, lemmatizer=False, geological_matcher=True,
        stopword_removal=False, punctuation_removal=False, lemmatize_triggers=True, verbose=False)

    # load files and geoview metadata
    capstone_files, files = get_report_data(count_sentences=True, return_files=True)

    metadata = pd.read_csv('data/geoview/capstone_metadata.zip', compression='zip', parse_dates=['report_year'],
        usecols=['anumber','title','report_type','project','keywords','commodity','report_year'])

    # geoview = gpd.read_file('zip://data/geoview/capstone_shapefiles.shp.zip')

    # specify labellers
    # users = users or ('daniel','charlie')
    # confs = ('medium','high',)
    
    # read data from file
    dataset = {
        user : pd.read_csv(f'data/labels/{user}_dataset.csv', index_col=0).rename(
            columns={'idx': 'sentence_idx'}) for user in users}

    # print confirmation of data load for each user
    for user in users:
        print(f'{len(dataset[user].loc[dataset[user].reviewed])} events labelled by {user}.')
        
    # loads events by confidence - note will not load group labelled
    events = {conf: build_event_data(dataset, confidence=conf, files=files, nlp=nlp, capstone_files=capstone_files,
        geoview=metadata, return_entities=True, group_all_labelled=group_all_labelled) for conf in confs}

    # build geopandas.geodataframe.GeoDataFrame (to start with geoview to preserve data type for plotly map)
    # join geoview shape files, geoview metadata, capstone json to anumber mapping, and aggregated event statistics
    df = {conf : metadata.merge(capstone_files, on='anumber').merge(
        events[conf].groupby('filename')['label'].sum().reset_index(), on='filename') for conf in confs}

    # store one hot encodings for each of the events dataframes
    commodities = {}
    for conf in confs:
        df[conf]['commodity_list'] = df[conf]['commodity'].apply(lambda x : to_list(x, sep=';', default='NO TARGET COMMODITY'))  
        # expand string separated strings to list
        mlb = MultiLabelBinarizer()
        mlb.fit(df[conf]['commodity_list'])
        commodities[conf] = pd.DataFrame(mlb.transform(df[conf]['commodity_list']), columns=mlb.classes_, index=df[conf].index)

    for conf in confs:
        if group_all_labelled:
            events[conf].to_csv(f'data/events/events_{conf}-conf-extra.csv')
            commodities[conf].to_csv(f'data/events/commodities_{conf}-conf-extra.csv')
        else:
            events[conf].to_csv(f'data/events/events_{conf}-conf.csv')
            commodities[conf].to_csv(f'data/events/commodities_{conf}-conf.csv')
            
    print('Saved events and commodities for each confidence threshold to data/events/')
