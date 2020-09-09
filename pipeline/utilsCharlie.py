import itertools
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict, Set

import pandas as pd
from spacy.matcher import PhraseMatcher
from tqdm import tqdm


def clean(text):
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



def create_event_id(filename: Path, sentence_idx: int) -> str:
    return f"{filename.with_suffix('').name}_{sentence_idx}"


def load_text_from_filename(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def get_found_trigger_words_and_phrases(doc, near_phrase_matcher) -> Set:
    matched_phrases = near_phrase_matcher(doc)

    found = set()
    for match_id, start, end in matched_phrases:
        span = doc[start:end]
        found.add(span.text)

    return found


def create_spacy_phrase_matcher(nlp, phrases: List, type_of_event: str) -> PhraseMatcher:
    phrase_matcher = PhraseMatcher(nlp.vocab)
    string_phrases = [' '.join(phrase) for phrase in phrases]
    patterns = [nlp(text) for text in string_phrases]
    phrase_matcher.add(str(type_of_event), None, *patterns)

    return phrase_matcher


def get_event_chunk(all_sentences, sentence_idx, n_sentences_extract, nlp=None):
    lower_idx = max(0, sentence_idx - n_sentences_extract)
    upper_idx = min(len(all_sentences), sentence_idx + n_sentences_extract)

    event_text = " ".join(all_sentences[lower_idx: upper_idx])

    if nlp is not None:
        event_text = nlp(clean(event_text))

    return event_text, upper_idx


def get_extracted_geology_ents(event_text, geology_ents) -> Dict:
    extracted_ents_dict = defaultdict(set)

    for ent in event_text.ents:
        label = ent.label_
        if label in geology_ents:
            extracted_ents_dict[label].add(ent.text)

    result = {}
    for k, v in extracted_ents_dict.items():
        values = ', '.join(v) if len(v) > 0 else ''
        result[k] = values

    return result


def create_event_df(nlp,
                    directory: Path,
                    trigger_phrases,
                    geology_ents,
                    n_sentences_extract=3):
    '''

    :param nlp:
    :param directory:
    :param trigger_phrases:
    :param geology_ents:
    :param n_sentences_extract:
    :return:
    '''

    if trigger_phrases is None or len(trigger_phrases) == 0:
        raise ValueError("Error: No trigger words provided")

    filenames = list(directory.glob('*.json'))

    near_miss_phrase_matcher = create_spacy_phrase_matcher(nlp, trigger_phrases, "NearMissEvent")

    all_event_data = []
    total_sentences = 0

    for i in tqdm(range(len(filenames))):
        filename = filenames[i]
        text = load_text_from_filename(filename)

        n_sentences = len(text)
        if n_sentences == 0:  # Empty Report
            continue

        total_sentences += n_sentences

        sentence_idx = 0

        while sentence_idx < len(text):
            sentence = text[sentence_idx]
            sentence_doc = nlp(clean(sentence))

            sentence_triggers = get_found_trigger_words_and_phrases(sentence_doc, near_miss_phrase_matcher)

            if len(sentence_triggers) > 0:  # Sentence contains at least 1 trigger word or phrase
                event_text, upper_chunk_idx = get_event_chunk(text, sentence_idx, n_sentences_extract, nlp)

                event_triggers = get_found_trigger_words_and_phrases(event_text, near_miss_phrase_matcher)

                extracted_geology_ents = get_extracted_geology_ents(event_text, geology_ents)

                all_event_data.append({
                    'event_id':                    create_event_id(filename, sentence_idx),
                    'filename':                    filename.name,
                    'sentence_idx':                sentence_idx,
                    'sentence_text':               sentence_doc.text,
                    'n_trigger_words_in_sentence': len(sentence_triggers),
                    'trigger_words_in_sentence':   ', '.join(sentence_triggers),
                    'n_trigger_words_in_event':    len(event_triggers),
                    'trigger_words_in_event':      ', '.join(event_triggers),
                    'event_text':                  event_text.text,
                    **extracted_geology_ents,
                    'event_label':                 0
                })

                sentence_idx = upper_chunk_idx

            sentence_idx += 1

    print(f'found {len(all_event_data)} events from a total of {total_sentences} sentences')

    return pd.DataFrame(all_event_data)


def get_feature_counts_df(df, feature):
    counts = Counter(list(itertools.chain(*df[feature].tolist())))
    return (pd.DataFrame.from_dict(counts, orient='index')
            .reset_index()
            .rename(columns={'index': feature, 0: 'count'})
            .sort_values('count', ascending=False)
            .set_index(feature)
            )
