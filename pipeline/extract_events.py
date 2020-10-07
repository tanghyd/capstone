import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, List

import click
import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher
from spacy.pipeline import EntityRuler
from tqdm import tqdm

BASE_PATH = Path('..')
events_path = BASE_PATH / 'events'
dictionary_path = BASE_PATH / 'dictionary'
subset_reports_path = BASE_PATH / 'data' / 'subset'
patterns_path = dictionary_path / 'patterns'
triggers_path = dictionary_path / 'trigger phrases'


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


def load_patterns(path: Path):
    patterns = []

    for p in path.glob('*.txt'):
        with open(p, encoding="utf8") as f:
            patterns += json.load(f)

    return patterns


def load_triggers(path: Path):
    triggers = []

    for p in path.glob('*.txt'): ##
        with open(p, 'r') as f:
            for line in f:
                if len(line) > 1:
                    triggers.append(line[:-2].split())

    return triggers


def load_report_filenames(path: Path) -> List:
    return list(path.glob('*.json'))


def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def extract_triggers(doc, matcher) -> Set:
    matched_phrases = matcher(doc)

    found = set()
    for match_id, start, end in matched_phrases:
        span = doc[start:end]
        found.add(span.text)

    return found


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


def get_events(patterns_path,
               triggers_path,
               reports_path,
               n_sentences_extract=2):
    nlp = spacy.load("en_core_web_lg")

    patterns = load_patterns(patterns_path)

    # Add patterns to nlp
    ruler = EntityRuler(nlp, overwrite_ents=True)
    ruler.add_patterns(patterns)
    nlp.add_pipe(ruler)

    triggers = load_triggers(triggers_path)

    # Create phrase matcher
    patterns = [nlp(' '.join(phrase)) for phrase in triggers]
    phrase_matcher = PhraseMatcher(nlp.vocab)
    phrase_matcher.add("NearMissEvent", None, *patterns)

    all_event_data = []
    total_sentences = 0

    filenames = load_report_filenames(reports_path)

    for i in tqdm(range(len(filenames))):
        filename = filenames[i]
        text = load_json(filename)

        n_sentences = len(text)
        if n_sentences == 0:  # Empty Report
            continue
        total_sentences += n_sentences

        sentence_idx = 0
        while sentence_idx < len(text):
            sentence = text[sentence_idx]
            sentence_doc = nlp(clean(sentence))

            sentence_triggers = extract_triggers(sentence_doc, phrase_matcher)

            if len(sentence_triggers) > 0:  # Sentence contains at least 1 trigger word or phrase
                lower_chunk_idx = max(0, sentence_idx - n_sentences_extract)
                upper_chunk_idx = min(n_sentences, sentence_idx + n_sentences_extract)

                event_doc = nlp(clean(" ".join(text[lower_chunk_idx: upper_chunk_idx])))

                event_triggers = extract_triggers(event_doc, phrase_matcher)

                extracted_geology_ents = get_extracted_geology_ents(event_doc)

                all_event_data.append({
                    'event_id':                    f"{filename.with_suffix('').name}_{sentence_idx}",
                    'filename':                    filename.name,
                    'sentence_idx':                sentence_idx,
                    'sentence_text':               sentence_doc.text,
                    'n_trigger_words_in_sentence': len(sentence_triggers),
                    'trigger_words_in_sentence':   ', '.join(sentence_triggers),
                    'n_trigger_words_in_event':    len(event_triggers),
                    'trigger_words_in_event':      ', '.join(event_triggers),
                    'event_text':                  event_doc.text,
                    **extracted_geology_ents,
                    'event_label':                 0
                })

                sentence_idx = upper_chunk_idx

            sentence_idx += 1

    print(f'found {len(all_event_data)} events from a total of {total_sentences} sentences')

    return pd.DataFrame(all_event_data)


@click.command()
@click.option('--group', help='Group Number.')
def extract_events(group):
    event_df = get_events(patterns_path=patterns_path,
                          triggers_path=triggers_path,
                          reports_path=subset_reports_path,
                          n_sentences_extract=2)

    event_path = events_path / f'group_{group}_events.csv'
    event_df.to_csv(event_path, index=False)


if __name__ == '__main__':
    extract_events()
