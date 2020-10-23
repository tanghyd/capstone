from pathlib import Path

up_n_levels = '.'
BASE_PATH = Path(up_n_levels)
events_path = BASE_PATH / 'events'
dictionary_path = BASE_PATH / 'dictionary'
subset_reports_path = BASE_PATH / 'data' / 'subset'
reports_path = BASE_PATH / 'data' / 'wamex_xml'  # should be set to wamex_xml when ready
entities_path = dictionary_path / 'patterns'
triggers_path = dictionary_path / 'trigger_phrases'
data_path = BASE_PATH / 'data'

from typing import Dict, Set
from collections import defaultdict
import json
from tqdm import tqdm

import pandas as pd
import spacy


### TEXT PREPROCESSING AND FEATURE EXTRACTION ###
def clean(text):
    '''Function to pre-process text using regular expressions.'''

    import re
    text = text.strip('[(),- :\'\"\n]\s*').lower()
    # text = re.sub('([A-Za-z0-9\)]{2,}\.)([A-Z]+[a-z]*)', r"\g<1> \g<2>", text, flags=re.UNICODE)
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
    for p in path.glob('*.txt'):  ##
        with open(p, 'r') as f:
            for line in f:
                if len(line) > 1:
                    triggers.append(line[:-2])  # .split()

    # manual labelling process provided opportunity to list new trigger words to search
    if triggers_from_labelling:  # then add these new trigger words to trigger lislt
        new_phrases = []
        for group in (0, 1, 2, 3, 4, 6):  # previouslly labelled report groups
            events = pd.read_csv(events_path / f'group_{group}_labelled.csv')
            events = events.loc[events['Key trigger phrase'].notna(),]
            events_triggers = set(events['Key trigger phrase'].tolist())
            new_phrases += events_triggers
            new_phrases = list(set(new_phrases))
        for phrase in new_phrases:
            triggers.append(phrase)  # .split()

    return triggers


def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def load_spacy_model(base_model="en_core_web_lg", output_type='doc', tokenizer_only=False,
                    geological_matcher=True, entities_path=entities_path, lemmatize_triggers=True,
                    trigger_matcher=False, triggers_from_labelling=False, triggers_path=triggers_path,
                    lemmatizer=False, stopword_removal=False, punctuation_removal=True, verbose=True):
    '''Utility function to load standardised NLP English pipeline from spaCy
    
    output_type : str --- Specifies the output type for the pipeline. Default: 'doc'.
        'doc' --> spacy.tokens.doc.Doc (Returns full sentence as spaCy Doc object)
        'sentence' --> str (Returns full sentence as plain text string)
        'text' --> list of str (Returns a list of all tokens converted to plain text string) 
    '''

    # output type check
    assert output_type in ('doc', 'sentence', 'text'), "output_type not valid. Select one of ('doc', 'token', 'sentence', 'text')."
    from spacy.pipeline import EntityRuler
    # initialise language pipeline with base model
    nlp = spacy.load(base_model)

    def lemmatize(doc):
        # This takes in a doc of tokens from the NER and lemmatizes them. 
        doc = [token.lemma_.lower().strip() for token in doc if token.lemma_ != '-PRON-']
        doc = u' '.join(doc)
        return nlp.make_doc(doc)

    if lemmatizer:
        nlp.add_pipe(lemmatize, name='lemmatizer', before='ner')
        if verbose:
            print('Added lemmatizer pipe')

    if stopword_removal:
        def remove_stopwords(doc):
            # This will remove stopwords and punctuation.
            doc = [token.text.lower().strip() for token in doc if token.is_stop != True]
            doc = u' '.join(doc)
            return nlp.make_doc(doc)

        nlp.add_pipe(remove_stopwords, name='stopwords', before='ner')
        if verbose:
            print('Added stopword removal pipe')

    if punctuation_removal:
        def remove_punctuation(doc):
            # Temp func to remove punctuation
            doc = [token.text.lower().strip() for token in doc if token.is_punct != True]
            doc = u' '.join(doc)
            return nlp.make_doc(doc)

        nlp.add_pipe(remove_punctuation, name='punctuation', before='tagger')
        if verbose:
            print('Added punctuation removal pipe')

    if geological_matcher and not tokenizer_only:
        # Instantiate a spaCy EntityRuler and load patterns for manual pattern matching
        patterns = load_patterns(entities_path)
        geo_ruler = EntityRuler(nlp, overwrite_ents=True, validate=True)
        geo_ruler.add_patterns(patterns)

        nlp.add_pipe(geo_ruler, name='georuler', after='ner')
        if verbose:
            print('Added geological entity matcher pipe')

    # if tenement_matcher and not tokenizer_only:
    #     # specify pattern schema for a "tenement" and create entity ruler for it
    #     tenement_patterns = {}
    #     tenement_ruler = EntityRuler(nlp, overwrite_ents=True)
    #     tenement_ruler.add_patterns(tenement_patterns)

    #     nlp.add_pipe(tenement_ruler, name='tenementruler', after='ner')
    #     if verbose:
    #         print('Added tenement matcher pipe')

    if trigger_matcher and not tokenizer_only:
        trigger_ruler = create_trigger_ruler(triggers_path=triggers_path, triggers_from_labelling=triggers_from_labelling, nlp=nlp)
        nlp.add_pipe(trigger_ruler, name='triggermatcher', after='ner')

        if verbose:
            print('Added trigger phrase matcher pipe')

    if tokenizer_only:
        pipes = ['tagger', 'parser', 'ner']
        if verbose:
            print(f'Loading tokenizer only - disabling: {pipes}.')
        for pipe in pipes:
            nlp.remove_pipe(pipe)

    # spaCy pipeline's final output type logic
    if output_type == 'doc':
        if verbose:
            print(f'Loading spaCy model with spaCy {output_type} output.')

    elif output_type in ('text', 'sentence'):
        # # function to take doc object to text
        # def to_text(doc, lower_case=False):
        #     if lower_case:  # Use token.text to return strings, which we'll need for Gensim.
        #         return [token.text.lower() for token in doc]
        #     else:
        #         return [token.text for token in doc]

        if output_type == 'text':
            nlp.add_pipe(to_text, name='text_output', last=True)
            if verbose:
                print(f'Loading spaCy model with list of token {output_type} strings as output.')

        elif output_type == 'sentence':
            # def to_sentence(doc, sep=' '):
            #     return sep.join(to_text(doc))

            nlp.add_pipe(to_sentence, name='sentence_output', last=True)
            if verbose:
                print('Loading spaCy model with string output.')

    return nlp

def create_trigger_ruler(triggers_path=triggers_path, triggers_from_labelling=False, nlp=None, lemmatize_triggers=True):
    # note: if we load a default nlp class with lemmatize_triggers, we may end up with recursion
    nlp = nlp or load_spacy_model(output_type='doc', trigger_matcher=True, lemmatizer=False,
                       stopword_removal=False, punctuation_removal=True, lemmatize_triggers=False)  

    from spacy.pipeline import EntityRuler
    # load triggers and create phrase matcher that matches on our trigger words 
    triggers = load_triggers(triggers_path, triggers_from_labelling=triggers_from_labelling)
    trigger_ruler = EntityRuler(nlp, overwrite_ents=True, validate=True)
    
    # load triggers to match directly on text
    trigger_patterns = [{
        'label': 'TRIGGER',
        'pattern': [{'LOWER': word} for word in trigger]} 
        for trigger in triggers
    ]

    trigger_ruler.add_patterns(trigger_patterns)

    if lemmatize_triggers:
        # lemmatize triggers with loaded pipeline.
        lemmatized_triggers = [
            [token.lemma_.lower().strip() for token in doc if token.lemma_ != '-PRON-']
            for doc in nlp.pipe(triggers)
        ]
        # load lemmatized triggers into a rule based matcher, matching Token Lemmas
        lemma_patterns = [{
            "label": "TRIGGER",
            "pattern": [{"LEMMA": word} for word in trigger]
        } for trigger in lemmatized_triggers]

        trigger_ruler.add_patterns(lemma_patterns)

    return trigger_ruler
# function to take doc object to text
def to_text(doc, lower_case=False):
    if lower_case:  # Use token.text to return strings, which we'll need for Gensim.
        return [token.text.lower() for token in doc]
    else:
        return [token.text for token in doc]

def to_sentence(doc, sep=' '):
                return sep.join(to_text(doc))
                
def create_phrase_matcher(phrases, nlp=None, label="TRIGGER"):
    if nlp == None:
        nlp = load_spacy_model(output_type='doc', tokenizer_only=True, verbose=False)

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
    # nltk.download('vader_lexicon')
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    ## Sentiement Analyser Imports
    from textblob import TextBlob

    ##Intialise sentiment analyser from ntlk
    sentiment_analyser = SentimentIntensityAnalyzer()
    score = sentiment_analyser.polarity_scores(text)
    sentiment_dict = {
        'polarity':     TextBlob(text).sentiment.polarity,
        'subjectivity': TextBlob(text).sentiment.subjectivity,
        'negative':     score['neg'],
        'neutral':      score['neu'],
        'positive':     score['pos'],
        'compound':     score['compound']
    }

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


def load_files(filenames, data_path, output: str = 'dict'):
    if type(data_path) == str:
        data_path = Path(data_path)
    if output == 'dict':
        return {filename: load_json(data_path / filename) for filename in tqdm(filenames, desc='Loading files as dict')}
    elif output == 'list':
        return [load_json(data_path / filename) for filename in filenames in tqdm(filenames, desc='Loading files as list')]


def match_triggers(filenames, triggers_from_labelling=True, nlp=None, save_json=False,
                   json_file='sample_triggers.json', return_json=False, triggers_path=triggers_path,
                   data_path: Path = reports_path, save_path: Path = BASE_PATH / 'data'):  # not ideal path naming here):

    nlp = nlp or load_spacy_model(output_type='doc', geological_matcher=False, tokenizer_only=True, verbose=False)

    # load trigger phrases from file
    triggers = load_triggers(triggers_path, triggers_from_labelling=triggers_from_labelling)

    # create phrase matcher that matches on our trigger words 
    trigger_matcher = create_phrase_matcher(triggers, nlp=nlp)

    # if a dictionary is loaded where keys are filenames and values are pre-loaded files, we dont load from disk
    if type(filenames) == dict:
        files = filenames
    else:
        # load report files from disk to extract events on triggers
        files = load_files(filenames, data_path=data_path, output='dict')

    # get sentence with trigger word match for each file in files
    # note: sentence idx is saved as a string for compatibility between json / df
    file_triggers = {
        file: dict(filter(lambda match: len(match[1]) > 0,  # remove any sentences that do not have matches
                          ((str(idx), list(get_matches(doc, trigger_matcher))) 
                          for idx, doc in enumerate(nlp.tokenizer.pipe(sentences)))))
        for file, sentences in tqdm(files.items(), desc='Matching loaded texts with trigger phrases')
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
    return pd.DataFrame([{"filename": file, "idx": int(idx), "triggers": ', '.join(triggers)}
                         for file, matches in file_triggers.items() for idx, triggers in matches.items()])

def to_list(x, sep=',', default='unknown'):
    if isinstance(x, str):
        return [item.strip() for item in x.split(sep)]
    else:
        return [default]

def get_events(entities_path, triggers_path, reports_path, n_sentences_extract=2, triggers_from_labelling=True,
               nlp=None):
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
                event_doc = nlp(event_text)  # pass event text with spacy nlp model
                event_triggers = get_matches(event_doc, trigger_matcher)  # extract other triggers in text chunk

                all_event_data.append({
                    #  'event_id':                    f"{filename.with_suffix('').name}_{sentence_idx}",
                    'filename':                  filename.name,
                    'sentence_idx':              sentence_idx,
                    'sentence_text':             sentence_doc.text,
                    #  'n_trigger_words_in_sentence': len(sentence_triggers),
                    'trigger_words_in_sentence': ', '.join(sentence_triggers),
                    #  'n_trigger_words_in_event':    len(event_triggers),
                    'trigger_words_in_event':    ', '.join(event_triggers),
                    'event_text':                event_doc.text,
                    # **get_sentiment_scores(event_text),
                    # **get_extracted_geology_ents(event_doc),
                    #    'trigger_list_version':        trigger_version
                })

                sentence_idx = upper_chunk_idx

            sentence_idx += 1

    print(f'found {len(all_event_data)} events from a total of {total_sentences} sentences')

    return pd.DataFrame(all_event_data)
