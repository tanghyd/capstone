import warnings

warnings.filterwarnings('ignore')

from pathlib import Path

import pandas as pd
import numpy as np

import spacy
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import click

BASE_PATH = Path('.')
events_path = BASE_PATH / 'events'
dictionary_path = BASE_PATH / 'dictionary'
data_path = BASE_PATH / 'data'
subset_reports_path = data_path / 'subset'
subset_reports_path_txt = data_path / 'subset_txt'
df_path = data_path / 'dataframes'
patterns_path = dictionary_path / 'patterns'
triggers_path = dictionary_path / 'trigger phrases'

# should we save d2v.model to a model folder for better structure?

def train_doc2vec(df, tag_col='event_id', token_col='tokens', max_epochs=100,
                min_count=2, vec_size=50, alpha=0.025, min_alpha=0.00025, dm=1,
                model_path="d2v.model", num_workers=4, save=False):
    # produce a mapping tagging all text with their ids
    assert type(df) == pd.DataFrame, 'Input data must be a dataframe with tag_col and token_col specified, or a dict.'

    tagged_data = [TaggedDocument(words=getattr(row, token_col), tags=[getattr(row, tag_col)]) for row in df.itertuples()]

    # train model on tagged data
    model = Doc2Vec(tagged_data, vector_size=vec_size, alpha=alpha, min_alpha=min_alpha, min_count=min_count,
                dm=dm, epochs=max_epochs, workers=num_workers)

    if save:
        print(f'Saving model to {model_path}')
        model.save(model_path)

    return model

def load_doc2vec(model_path='d2v.model'):
    return Doc2Vec.load(model_path)

def get_vecs_from_model(data, model, tag_col='event_id', as_dataframe=True):
    # produce a mapping tagging all text with their ids
    vecs = []
    for tag in data[tag_col]:
        try:
            vec = model.docvecs[tag]
        except:
            vec = np.nan
        vecs.append(vec)

    if as_dataframe:
        return pd.DataFrame(vecs, index=data[tag_col])    
    return vecs

def get_vectors(df, tag_col='event_id', token_col='tokens', max_epochs=100,
                min_count=2, vec_size=50, alpha=0.025, min_alpha=0.00025, dm=1,
                model_path="d2v.model", num_workers=4, load=True, save=False, as_dataframe=True):
    if load:
        model = load_doc2vec(model_path=model_path)

    else:  # train model from scratch - does not save
        model = train_doc2vec(df, tag_col=tag_col, token_col=token_col, max_epochs=max_epochs,
                min_count=min_count, vec_size=vec_size, alpha=alpha, min_alpha=min_alpha, dm=dm,
                model_path=model_path, num_workers=num_workers, save=save)

    return get_vecs_from_model(df, model, tag_col=tag_col, as_dataframe=as_dataframe)

# not complete
# # add one hot encoded named entity features to main df
# def get_onehot_features(full_df, labelled_df):
#     from sklearn.preprocessing import MultiLabelBinarizer
#     mlb = MultiLabelBinarizer()

#     def to_list(x):
#         if isinstance(x, str):
#             return x.split(',')
#         else:
#             return ['unknown']
            
#     for feature in ['STRAT', 'ROCK', 'LOCATION', 'MINERAL', 'ORE_DEPOSIT', 'TIMESCALE']:
#         s = full_df[feature].apply(to_list)
#         wide_df = pd.DataFrame(mlb.fit_transform(s), columns=mlb.classes_, index=df.index)
#         wide_df = wide_df.add_prefix(f'{feature}_')
#         labelled_df = labelled_df.merge(wide_df, how='left', left_index=True, right_index=True)
#     return labelled_df


# need to combine with new vectorize.py

#@click.command()
#@click.option('--group', help='Group Number.')
# def vectorize(group, nlp=None, load=False, save=True):    
#     filename = events_path / f'group_{group}_labelled.csv'
#     df = pd.read_csv(filename)
#     df = df[df.columns]

#     labelled_df = df.loc[df.reviewed][['event_id', 'event_text', 'Near Miss Event']]

#     if nlp == None:
#         from .processing import load_spacy_model
#         nlp = load_spacy_model(output_type='text', tokenizer_only=True, verbose=False)
   
#     # Event Text Vector
#     labelled_df['tokens'] = list(nlp.pipe(labelled_df['event_text'].values))
    
#     get_vectors(labelled_df, load=load, save=save)

#     # One hot encoded Feature
#     labelled_df = add_features(df, labelled_df)

#     # Label
#     labelled_df['label'] = labelled_df['Near Miss Event'].astype(int)

#     # Drop unused labels
#     labelled_df = labelled_df.drop(columns=['tokens', 'event_text', 'Near Miss Event'])

#     output_filename = events_path / f'group_{group}_processed.csv'
#     labelled_df.to_csv(output_filename, index=False)
    
# if __name__ == '__main__':
#     vectorize(2)
