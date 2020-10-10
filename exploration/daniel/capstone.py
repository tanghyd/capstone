from typing import Dict, Set, List
from collections import defaultdict
import json
import glob
from tqdm import tqdm

import pandas as pd
import spacy

# timer.py
from dataclasses import dataclass, field
import time
from typing import Callable, ClassVar, Dict, Optional
class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

@dataclass
class Timer:
    '''Custom class to implement a manual timer. Used with "with" statements.'''
    timers: ClassVar[Dict[str, float]] = dict()
    name: Optional[str] = None
    text: str = "Elapsed time: {:0.4f} seconds"
    logger: Optional[Callable[[str], None]] = print
    _start_time: Optional[float] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Add timer to dict of timers after initialization"""
        if self.name is not None:
            self.timers.setdefault(self.name, 0)

    def start(self) -> None:
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        # Calculate elapsed time
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        # Report elapsed time
        if self.logger:
            self.logger(self.text.format(elapsed_time))
        if self.name:
            self.timers[self.name] += elapsed_time

        return elapsed_time
    
    def __enter__(self):
        """Start a new timer as a context manager"""
        self.start()
        return self

    def __exit__(self, *exc_info):
        """Stop the context manager timer"""
        self.stop()

# import user-specified path directory from path.py
import os

# count number of '..' between current dir and base path
# dir_walk_count = 0
# current_dir = os.getcwd()
# while current_dir.rsplit('/')[-1] != 'capstone':  # won't work on windows
#     current_dir = os.path.dirname(current_dir)
#     dir_walk_count += 1

# produce base_dir based on current dir
#up_n_levels = '/'.join([str('..') for _ in range(dir_walk_count)]) 
up_n_levels = '../..'

from pathlib import Path
BASE_PATH = Path(up_n_levels)  # produce base_dir based on current dir)
events_path = BASE_PATH / 'events'
dictionary_path = BASE_PATH / 'dictionary'
subset_reports_path = BASE_PATH / 'data' / 'subset'
reports_path = BASE_PATH / 'data' / 'wamex_xml'  # should be set to wamex_xml when ready
entities_path = dictionary_path / 'patterns'
triggers_path = dictionary_path / 'trigger_phrases'
#data_path = BASE_PATH / 'data'

# utility function
flatten = lambda l: [item for sublist in l for item in sublist]  # returns a list

### WAMEX XML ETL FOR REPORT METADATA ###

# ETL from reports zip file to dimension table for geoview
def load_geoview_data(data_file='Exploration_Reports_GDA2020_shp.zip', data_folder=os.path.join(up_n_levels,'data'), zip=True, headers=None, all=False):
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
def get_report_anumbers(data_folder=os.path.join(up_n_levels,'data')): 
    import os
    #filenames = [file.split('/',4)[-1] for file in glob.glob(f'{data_folder}/wamex_xml/*.json')]  # does not work on windows due to file path
    wamex_folder_path = wamex_folder_path = os.path.join(*(str(data_folder).split('/')), 'wamex_xml')
    filenames = [file.split('/',4)[-1] for file in os.listdir(wamex_folder_path) if file.split('.',1)[-1] == 'json']
    anumbers = [int(file.split("_",1)[0].replace("a","")) for file in filenames]
    files = pd.DataFrame.from_dict({'filename': filenames, 'anumber': anumbers})
    return files

def load_metadata(data_folder=os.path.join(up_n_levels,'data')):
    ''' Function to load wamex_xml file names and geoview metadata and return dataframe '''
    report_anumbers = get_report_anumbers(data_folder=data_folder)
    geoview = load_geoview_data(data_folder=data_folder, all=False)
    return report_anumbers.merge(geoview, on='anumber')


### TEXT PREPROCESSING AND FEATURE EXTRACTION ###

def clean(text):
    '''Function to pre-process text using regular expressions.'''
    
    import re
    text = text.strip('[(),- :\'\"\n]\s*').lower()
    #text = re.sub('([A-Za-z0-9\)]{2,}\.)([A-Z]+[a-z]*)', r"\g<1> \g<2>", text, flags=re.UNICODE)
    text = re.sub('\s+', ' ', text, flags=re.UNICODE).strip()
    text = re.sub('","', ' ', text, flags=re.UNICODE).strip()
    text = re.sub('-', ' ', text, flags=re.UNICODE).strip()
    # text = re.sub('\(', ' ', text, flags=re.UNICODE).strip()
    # text = re.sub('\)', ' ', text, flags=re.UNICODE).strip()
    text = re.sub('\/', ' ', text, flags=re.UNICODE).strip()
    text = text.replace("\\", ' ')

    text = ' '.join(text.split())

    if (text[len(text) - 1] != '.'):
        text += '.'

    return text

def load_patterns(path: Path):
    ''' Function to load geological entities from a folder of .json files given a pathlib Path object.'''
    # store patterns in list
    patterns = []

    # load all "patterns" in the entities_path directory
    for p in path.glob('*.json'):
        with open(p, encoding="utf8") as f:
            patterns += json.load(f)

    return patterns

def load_triggers(path: Path = triggers_path, triggers_from_labelling: bool = True):
    ''' Function to load trigger words from a folder of .txt files given a pathlib Path object.'''
    # store triggers in list
    triggers = []

    # load all trigger word files in triggers_path directory
    for p in path.glob('*.txt'): ##
        with open(p, 'r') as f:
            for line in f:
                if len(line) > 1:
                    triggers.append(line[:-2]) #.split()

    # manual labelling process provided opportunity to list new trigger words to search
    if triggers_from_labelling:  # then add these new trigger words to trigger lislt
        new_phrases = []
        for group in (0,1,2,3,4,6):  # previouslly labelled report groups
            events = pd.read_csv(events_path / f'group_{group}_labelled.csv')
            events = events.loc[events['Key trigger phrase'].notna(), ]
            events_triggers = set(events['Key trigger phrase'].tolist())
            new_phrases += events_triggers
            new_phrases = list(set(new_phrases))
        for phrase in new_phrases:
            triggers.append(phrase) #.split()

    return triggers

def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def load_spacy_model(base_model="en_core_web_lg", output_type='doc', entity_ruler=True, entities_path=entities_path,
    lemmatizer=True, stopword_removal=True, tokenizer_only=True, verbose=True):
    '''Utility function to load standardised NLP English pipeline from spaCy
    
    output_type : str --- Specifies the output type for the pipeline. Default: 'doc'.
        'doc' --> spacy.tokens.doc.Doc (Returns full sentence as spaCy Doc object)
        'sentence' --> str (Returns full sentence as plain text string)
        'text' --> list of str (Returns a list of all tokens converted to plain text string) 
    '''    
    
    # output type check
    assert output_type in ('doc', 'sentence', 'text'), "output_type not valid. Select one of ('doc', 'token', 'sentence', 'text')."

    # initialise language pipeline with base model
    nlp = spacy.load(base_model)
    
    if entity_ruler and not tokenizer_only:
        # Instantiate a spaCy EntityRuler and load patterns for manual pattern matching
        from spacy.pipeline import EntityRuler
        
        # import patterns from file
        patterns = load_patterns(entities_path)
                
        # instantiate pattern matcher and add to pipeline
        ruler = EntityRuler(nlp, overwrite_ents=True)
        ruler.add_patterns(patterns)
        
        nlp.add_pipe(ruler, name='entityruler')
        if verbose:
            print('Added entity ruler pipe')
    
    # spaCy pipeline's final output type logic
    if output_type == 'doc':
        if verbose:
            print(f'Loading spaCy model with spaCy {output_type} output.')
    
    elif output_type in ('text', 'sentence'):
        # function to take doc object to text
        def to_text(doc, lower_case=False):
            if lower_case:  # Use token.text to return strings, which we'll need for Gensim.
                return [token.text.lower() for token in doc]
            else:
                return [token.text for token in doc]

        if lemmatizer:
            def lemmatizer(doc):
                # This takes in a doc of tokens from the NER and lemmatizes them. 
                doc = [token.lemma_.lower().strip() for token in doc if token.lemma_ != '-PRON-']
                doc = u' '.join(doc)
                return nlp.make_doc(doc)
            
            nlp.add_pipe(lemmatizer, name='lemmatizer')
            if verbose:
                print('Added lemmatizer pipe')
        
        if stopword_removal:
            def remove_stopwords(doc):
                # This will remove stopwords and punctuation.
                doc = [token.text.lower().strip() for token in doc if token.is_stop != True and token.is_punct != True]
                doc = u' '.join(doc)
                return nlp.make_doc(doc)
            
            nlp.add_pipe(remove_stopwords, name='stopwords')
            if verbose:
                print('Added stopwords and punctuation pipe')

        if output_type == 'text': 
            nlp.add_pipe(to_text, name='text_output', last=True)
            if verbose:
                print(f'Loading spaCy model with list of token {output_type} strings as output.')

        elif output_type == 'sentence':
            def to_sentence(doc, sep=' '):
                return sep.join(to_text(doc))

            nlp.add_pipe(to_sentence, name='sentence_output', last=True)
            if verbose:
                print('Loading spaCy model with string output.')
    
    if tokenizer_only:
        pipes = ['tagger','parser','ner']
        if verbose:
            print(f'Loading tokenizer only - disabling: {pipes}.')
        for pipe in pipes:
            nlp.remove_pipe(pipe)

    return nlp

def create_phrase_matcher(phrases, nlp=None, label="Trigger"):
    if nlp == None:
        print('Initialising spaCy model')
        nlp = load_spacy_model(output_type='doc')

    # Create phrase matcher
    from spacy.matcher import PhraseMatcher

    # creates a phrase matcher using nlp model's vocabulary, matching on the LOWER attribute
    matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
    patterns = list(nlp.tokenizer.pipe(phrases))
    matcher.add(label, None, *patterns)  # If pattern matches, return match label as label

    return matcher

def get_matches(doc, matcher) -> Set:
    '''Gets tokens as text given a spaCy doc and a spaCy PhraseMatcher'''
    return {doc[start:end].text for _, start, end in matcher(doc)}  # set comprehension

### Function that produces sentiment scores for text chunck
def get_sentiment_scores(text):
    ## Sentiement Analyser Imports
    from textblob import TextBlob
    import nltk
    #nltk.download('vader_lexicon')
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    ##Intialise sentiment analyser from ntlk
    sentiment_analyser = SentimentIntensityAnalyzer()
    score = sentiment_analyser.polarity_scores(text)
    sentiment_dict = {
        'polarity': TextBlob(text).sentiment.polarity,
        'subjectivity': TextBlob(text).sentiment.subjectivity,
        'negative': score['neg'],
        'neutral': score['neu'],
        'positive': score['pos'],
        'compound': score['compound']}
    
    return sentiment_dict

def get_extracted_geology_ents(event_text) -> Dict:
    extracted_ents_dict = defaultdict(set)

    for ent in event_text.ents:
        label = ent.label_
        if label in ['ORE_DEPOSIT', 'ROCK', 'MINERAL', 'STRAT', 'LOCATION', 'TIMESCALE']:
            extracted_ents_dict[label].add(ent.text)

    result = {}
    for k, v in extracted_ents_dict.items():
        values = ', '.join(v) if len(v) > 0 else ''
        result[k] = values

    return result

def load_files(filenames, data_path : Path, output : str = 'dict'):
    if output == 'dict':
        return {filename: load_json(data_path / filename) for filename in filenames}
    elif output == 'list':
        return [load_json(data_path / filename) for filename in filenames]
    
def match_triggers(filenames, triggers_from_labelling=True, nlp=None, save_json=False, 
                json_file='sample_triggers.json', return_json=False, triggers_path=triggers_path, 
                data_path : Path = reports_path, save_path : Path = BASE_PATH / 'data'):  # not ideal path naming here
    if nlp == None:
        # Load standard spacy model, output tokens (not text strings) 
        nlp = load_spacy_model(output_type='doc', entity_ruler=False, verbose=False)

    # load trigger phrases from file
    triggers = load_triggers(triggers_path, triggers_from_labelling=triggers_from_labelling)

    # create phrase matcher that matches on our trigger words 
    trigger_matcher = create_phrase_matcher(triggers, nlp=nlp)

    # load report files to extract events on triggers from
    files = load_files(filenames, data_path=data_path, output='dict')

    # get sentence with trigger word match for each file in files
    # note: sentence idx is saved as a string for compatibility between json / df
    file_triggers = {
        file : dict(filter(lambda match : len(match[1]) > 0,  # remove any sentences that do not have matches
            ((str(idx), list(get_matches(doc, trigger_matcher))) for idx, doc in enumerate(nlp.tokenizer.pipe(sentences)))))
        for file, sentences in tqdm(files.items(), desc='Matching sentences in files with trigger phrases')
    }
    
    if save_json:  # save json object if specified
        # tqdm.write(f'Saving {json_file} to disk at {save_path.resolve()}.', end="")
        with open(save_path / json_file, 'w+') as f:
            json.dump(file_triggers, f)
            
    if return_json:  # returns json without repeated file names
        return file_triggers
    
    else:  # convert to dataframe with repeated filename per row
        return triggers_json_to_df(file_triggers)

def triggers_json_to_df(file_triggers):
    return pd.DataFrame([{"filename": file, "idx": idx, "triggers": ', '.join(triggers)} 
            for file, matches in file_triggers.items() for idx, triggers in matches.items()])

def get_events(entities_path, triggers_path, reports_path, n_sentences_extract=2, triggers_from_labelling=True, nlp=None):
    if nlp == None:
        # Load standard spacy model, output tokens (not text strings) 
        nlp = load_spacy_model(output_type='doc', entities_path=entities_path, verbose=False)

    # load trigger phrases from file
    triggers = load_triggers(triggers_path, triggers_from_labelling=triggers_from_labelling)

    # create phrase matcher that matches on our trigger words 
    trigger_matcher = create_phrase_matcher(triggers, nlp)

    # load report files to extract events on triggers from
    filenames = list(reports_path.glob('*.json'))

    all_event_data = []
    total_sentences = 0
        
    for filename in tqdm(filenames, desc='Extracting events from files'):
        # load json from file and count number of sentences in text chunk
        text = load_json(filename)
        n_sentences = len(text)  # counter number of sentences in text chunk
        if n_sentences == 0:  # then text is empty
            continue  # skip to next file
        total_sentences += n_sentences

        # loop through each sentence in the text chunk, with sentence_idx as loop counter
        sentence_idx = 0
        while sentence_idx < len(text):
            
            # get trigger words in sentence
            sentence = clean(text[sentence_idx])  # get sentence at specified idx from text chunk
            sentence_doc = nlp(sentence)  # process sentence text
            sentence_triggers = get_matches(sentence_doc, trigger_matcher)  # get triggers in sentence
            
            if len(sentence_triggers) > 0:  # Sentence contains at least 1 trigger word or phrase
                lower_chunk_idx = max(0, sentence_idx - n_sentences_extract)
                upper_chunk_idx = min(n_sentences, sentence_idx + n_sentences_extract)

                # process event text by joining sentences together and clean
                event_text = clean(" ".join(text[lower_chunk_idx: upper_chunk_idx]))
                event_doc = nlp(event_text) # pass event text with spacy nlp model
                event_triggers = get_matches(event_doc, trigger_matcher)  # extract other triggers in text chunk

                all_event_data.append({
                #  'event_id':                    f"{filename.with_suffix('').name}_{sentence_idx}",
                    'filename':                    filename.name,
                    'sentence_idx':                sentence_idx,
                    'sentence_text':               sentence_doc.text,
                #  'n_trigger_words_in_sentence': len(sentence_triggers),
                    'trigger_words_in_sentence':   ', '.join(sentence_triggers),
                #  'n_trigger_words_in_event':    len(event_triggers),
                    'trigger_words_in_event':      ', '.join(event_triggers),
                    'event_text':                  event_doc.text,
                # **get_sentiment_scores(event_text),
                # **get_extracted_geology_ents(event_doc),
                #    'trigger_list_version':        trigger_version
                })

                sentence_idx = upper_chunk_idx

            sentence_idx += 1

    print(f'found {len(all_event_data)} events from a total of {total_sentences} sentences')

    return pd.DataFrame(all_event_data)

### CLASSIFICATION ###

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted
import numpy as np

class CosineMeanClassifier(BaseEstimator, ClassifierMixin):
    """ 
    https://scikit-learn.org/stable/developers/develop.html
    sci-kit Learn compatible classifier that implements classification based on cosine similarity.

    Classifier is fit to a n x m numpy vector (n = number of observations, m = number of column vectors)

    Parameters
    ----------
    cut_off : float, default=0.5
        A parameter used to determine cut-off threshold for prediction.
    Attributes
    ----------
    n_features_ : int
        The number of features of the data passed to :meth:`fit`.
    mean_vector : np.array
        The mean vector of the input data fitted to the classifier.

    Example usage:
    >> from capstone import CosineMeanClassifier     
    >> from sklearn.feature_extraction.text import TfidfVectorizer
    >> from sklearn.decomposition import TruncatedSVD
    >> from sklearn.pipeline import Pipeline

    >> cosine_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(min_df=2, tokenizer = nlp, sublinear_tf = True)),  # outputs (n x v) array
        ('svd', TruncatedSVD(n_components=50)),  # outputs n x 50 array
        ('cosine', CosineMeanClassifier(cut_off=0.5))
    ])

    >> cosine_pipeline.fit(event_text)  # event text is a pd.Series or np.array of strings
    Pipeline(steps=[('tfidf',
                 TfidfVectorizer(min_df=2, sublinear_tf=True,
                                 tokenizer=<spacy.lang.en.English object at 0x7f7de6d47f70>)),
                ('svd', TruncatedSVD(n_components=50)),
                ('cosine', CosineMeanClassifier())])

    >> cosine_pipeline.predict(df.event_text.values)
    array([0, 0, 0, ..., 0, 0, 1])
    """

    def __init__(self, cutoff=0.5):
        self.cutoff = cutoff  # threshold cut-off for classification : cosine distance is bounded [0,1]

    def fit(self, X, y=None):  # Does not involve labels - we are just using mean vector like a nearest neighbor
        """A reference implementation of a fitting function for a transformer.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.
        Returns
        -------
        self : object
            Returns self.
        """
        self.mean_vector = X.mean(axis=0)
        self.n_features_ = X.shape[1]
        return self # Return the classifier
    
    def transform(self, X, y=None):
        """ A reference implementation of a transform function.
        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        X_transformed : array, shape (n_samples, 1)
            The array containing the cosine similarities of the input samples.
            in ``X``.
        """
        # Check is fit had been called
        check_is_fitted(self, 'n_features_')
        X = check_array(X)  # input validation
        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
            
        # compute cosine similarity
        magnitude = lambda x : np.sqrt(np.sum(np.power(x,2)))  # utility function to calculate magnitude
        similarity = lambda a, b : (np.dot(a, b)) / (magnitude(a)*magnitude(b)) 
        
        return np.array([similarity(x, self.mean_vector) for x in X])
    
    def predict_proba(self, X):
        """ A reference implementation of a transform function.
        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        X_transformed : array, shape (n_samples, 1)
            The array containing the cosine similarities of the input samples.
            in ``X``.
        """
        return self.transform(X)
    
    def predict(self, X):
        """ A reference implementation of a transform function.
        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        X_transformed : array, shape (n_samples, 1)
            The array containing a predictions based on taking a boolean condition cosine similarities of the input samples
            in ``X``. If the cosine similiarity is greater than self.cutoff, then return 1, else 0.
        """
        # Check is fit had been called
        check_is_fitted(self, 'n_features_')
        X = check_array(X)  # input validation
        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')

        # compute cosine similarity
        magnitude = lambda x : np.sqrt(np.sum(np.power(x,2)))  # utility function to calculate magnitude
        similarity = lambda a, b : (np.dot(a, b)) / (magnitude(a)*magnitude(b)) 
        
        # apply mean vertically over col and return a (1,n) vector,  return binary integer array
        return np.array([(similarity(x, self.mean_vector) > self.cutoff) for x in X], dtype=np.int64)
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)