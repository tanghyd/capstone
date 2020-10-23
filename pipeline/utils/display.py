from spacy import displacy
from tqdm import tqdm
import os

def display_ent(doc, style="ent", colors=None, options=None, compact=True, jupyter=False):
    colors = colors or {"TRIGGER": "linear-gradient(90deg, #aa9cfc, #fc9ce7)"}
    options = options or {"ents": None, "colors": colors, "compact": compact}
    if jupyter:
        displacy.render(doc, style=style, jupyter=jupyter, options=options)
    return displacy.render(doc, style=style, jupyter=jupyter, options=options)

def generate_html(event_texts, batch_size, n_process=-1, nlp=None, save=False):
    '''Multi-threaded to pipe list-like of strings into spaCy documents and return HTML in a list.
    
    arguments:
    -- event_texts : list of str (e.g. event_text column from events dataframe).
    -- batch_size : number of strings to process per batch on each process (afaik).
    -- n_process : number of workers/threads/cores on CPU (default = -1 : all available threads).
    -- nlp : pre-loaded spaCy language model. If None, load expected default configuration for this task.
    
    returns:
        list of HTML strings
    '''
    
    if nlp == None:
        from pipeline.preprocessing.text import load_spacy_model
        nlp = load_spacy_model(output_type='doc', trigger_matcher=True, lemmatizer=False, geological_matcher=True,
                       stopword_removal=False, punctuation_removal=False, lemmatize_triggers=True, verbose=False)
        
    # pipe text to list of docs
    #docs = list(nlp.pipe(event_texts, batch_size=batch_size, n_process=n_process))
    
    html = [display_ent(doc, jupyter=False) for doc in tqdm(
        nlp.pipe(event_texts, batch_size=batch_size, n_process=n_process),
    desc='Rendering spaCy entities as HTML')]
    
    return html

def write_html(df, batch_size=100, n_process=-1, nlp=None):
    '''Multi-threaded to pipe events dataframe into spaCy documents and save pre-rendered .html files in data/html/spacy/'''
    
    import pandas as pd
    
    # load test data
    if type(df) == str:  # if filename is passed, load file
        df = pd.read_csv(df, index_col=0)
    assert type(df) == pd.DataFrame
   
    if nlp == None:
        # load language model
        from pipeline.preprocessing.text import load_spacy_model
        nlp = load_spacy_model(output_type='doc', trigger_matcher=True, lemmatizer=False, geological_matcher=True,
            stopword_removal=False, punctuation_removal=False, lemmatize_triggers=True, verbose=False)
        
    event_html = {event_id : html for event_id, html in zip(df.event_id.values, generate_html(
        df.event_text.values, batch_size=batch_size, n_process=n_process))}

    for event_id, html in tqdm(event_html.items(), desc='Saving pre-rendered .html to data/html/spacy/'):
        with open(os.path.join('data','html','spacy',f'{event_id}.html'), 'w+') as f:
            f.write(html)