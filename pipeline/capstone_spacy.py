def load_spacy_model(base_model="en_core_web_lg", pattern_matcher=False, lemmatizer=True, stopword_removal=True,
                   output_text=True, pattern_path='../dictionary/patterns'):
    
    import spacy
    # initialise language pipeline with base model
    nlp = spacy.load(base_model)
    
    if pattern_matcher:
        from spacy.pipeline import EntityRuler
        import glob
        
        # import patterns from file
        patterns = []
        for filename in pattern_path.glob('*.json'):
            with open(filename, encoding="utf8") as f:
                patterns += json.load(f)
                
        # instantiate pattern matcher and add to pipeline
        ruler = EntityRuler(nlp, overwrite_ents=True)
        ruler.add_patterns(patterns)
        
        nlp.add_pipe(ruler, name='entityruler')
        print('Load entity ruler pipe')
        
    if lemmatizer:
        def lemmatizer(doc):
        # This takes in a doc of tokens from the NER and lemmatizes them. 
            # Pronouns (like "I" and "you" get lemmatized to '-PRON-', so I'm removing those.
            doc = [token.lemma_ for token in doc if token.lemma_ != '-PRON-']
            doc = u' '.join(doc)
            return nlp.make_doc(doc)
        
        nlp.add_pipe(lemmatizer, name='lemmatizer')
        print('Load lemmatizer pipe')
    
    if stopword_removal:
        def remove_stopwords(doc):
            # This will remove stopwords and punctuation.
            doc = [token for token in doc if token.is_stop != True and token.is_punct != True]
            return doc
        
        nlp.add_pipe(remove_stopwords, name='stopwords')
        print('Load stopwords and punctuation pipe')
    
    if output_text: # Use token.text to return strings, which we'll need for Gensim.
        def to_text(doc):
            return [token.text for token in doc]
        
        nlp.add_pipe(to_text, name='totext', last=True)
        print('Load text pipe')
        
    return nlp

def clean(text):
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