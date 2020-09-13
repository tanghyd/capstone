import warnings

warnings.filterwarnings('ignore')

import pandas as pd
from pathlib import Path

import numpy as np

import spacy

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import click

BASE_PATH = Path('..')
events_path = BASE_PATH / 'events'
dictionary_path = BASE_PATH / 'dictionary'
data_path = BASE_PATH / 'data'
subset_reports_path = data_path / 'subset'
subset_reports_path_txt = data_path / 'subset_txt'
df_path = data_path / 'dataframes'
patterns_path = dictionary_path / 'patterns'
triggers_path = dictionary_path / 'trigger phrases'

from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()

nlp = spacy.load("en_core_web_lg")


def get_tokens(doc):
    return [w.lemma_ for w in nlp(doc) if (w.is_alpha and not w.is_stop)]


def get_vectors(df, max_epochs=100, vec_size=20, alpha=0.025, model_path="d2v.model"):
    tagged_data = [TaggedDocument(words=row.tokens, tags=[row.event_id]) for row in df.itertuples()]
    model = Doc2Vec(size=vec_size, alpha=alpha, min_alpha=0.00025, min_count=1, dm=1)
    model.build_vocab(tagged_data)
    for epoch in range(max_epochs):
        model.train(tagged_data, total_examples=model.corpus_count, epochs=model.iter)
        model.alpha -= 0.0002
        model.min_alpha = model.alpha

    model.save(model_path)
    vecs = []

    for event_id in df['event_id']:
        try:
            vec = model.docvecs[event_id]
        except:
            vec = np.nan
        vecs.append(vec)

    return vecs


def to_list(x):
    if isinstance(x, str):
        return x.split(',')
    else:
        return ['unknown']


def add_features(full_df, labelled_df):
    for feature in ['STRAT', 'ROCK', 'LOCATION', 'MINERAL', 'ORE_DEPOSIT', 'TIMESCALE']:
        s = full_df[feature].apply(to_list)
        wide_df = pd.DataFrame(mlb.fit_transform(s), columns=mlb.classes_, index=df.index)
        wide_df = wide_df.add_prefix(f'{feature}_')
        labelled_df = labelled_df.merge(wide_df, how='left', left_index=True, right_index=True)
    return labelled_df


@click.command()
@click.option('--group', help='Group Number.')
def vectorize(group):
    filename = events_path / f'group_{group}_labelled.csv'
    df = pd.read_csv(filename)
    df = df[df.columns[2:]]

    labelled_df = df.loc[df.reviewed][['event_id', 'event_text', 'Near Miss Event']]

    # Event Text Vector
    labelled_df['tokens'] = labelled_df['event_text'].apply(get_tokens)
    labelled_df['event_text_vector'] = get_vectors(labelled_df)

    # One hot encoded Feature
    labelled_df = add_features(df, labelled_df)

    # Label
    labelled_df['label'] = labelled_df['Near Miss Event'].astype(int)

    # Drop unused labels
    labelled_df = labelled_df.drop(columns=['tokens', 'event_text', 'Near Miss Event'])

    output_filename = events_path / f'group_{group}_processed.csv'
    labelled_df.to_csv(output_filename, index=False)
