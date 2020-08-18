import re
import pandas as pd
import json
from pathlib import Path
import itertools
from collections import Counter


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


def load_text_from_filename(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def process_trigger_words(nlp, trigger_words):
    words = []
    phrases = []

    for trigger in trigger_words:
        trigger_doc = nlp(trigger)

        if len(trigger_doc) > 1:
            phrases.append([w.lemma_ for w in trigger_doc])
        else:
            words.append(trigger_doc[0].lemma_)

    return words, phrases


def get_found_trigger_words(doc, trigger_words):
    found = []

    for token in doc:
        lemma = token.lemma_
        if lemma in trigger_words:
            found.append(lemma)

    return found


def get_found_trigger_phrases(doc, trigger_phrases):
    found = []

    sentence = [token.lemma_ for token in doc]
    for word_list in trigger_phrases:
        if all([token in sentence for token in word_list]):
            found.append(' '.join(word_list))

    return found


def get_event_chuck(all_sentences, sentence_idx, n_sentences_extact, nlp=None):
    lower_idx = max(0, sentence_idx - n_sentences_extact)
    upper_idx = min(len(all_sentences), sentence_idx + n_sentences_extact)
    select_sentences = all_sentences[lower_idx: upper_idx]
    event_text = " ".join(select_sentences)

    if nlp is not None:
        event_text = clean(event_text)
        event_text = nlp(event_text)

    return event_text


def create_event_df(
        nlp,
        directory: Path,
        trigger_words,
        geology_ents,
        n_sentences_extact=2,
):
    if trigger_words is None or len(trigger_words) == 0:
        raise ValueError("Error: No trigger words provided")

    filenames = list(directory.glob('*.json'))
    print(f'extracting events on {len(filenames)} files')

    trigger_words, trigger_phrases = process_trigger_words(nlp, trigger_words)

    all_event_data = []
    total_sentences = 0

    for filename in filenames:

        text = load_text_from_filename(filename)

        n_sentences = len(text)
        if n_sentences == 0:  # Empty Report
            continue

        total_sentences += n_sentences

        for sentence_idx, sentence in enumerate(text):

            # Clean text
            sentence_doc = nlp(clean(sentence))

            # Collect all trigger words in the sentence
            found_trigger_words = get_found_trigger_words(sentence_doc, trigger_words)
            found_trigger_phrases = get_found_trigger_phrases(sentence_doc, trigger_phrases)
            all_found = found_trigger_words + found_trigger_phrases

            if len(all_found) > 0:  # Sentence contains at least 1 trigger word or phrase
                event_text = get_event_chuck(text, sentence_idx, n_sentences_extact, nlp)

                # create event_id from filename and sentence index
                event_id = f"{filename.with_suffix('').name}_{sentence_idx}"

                # set as constant for now
                label = 0

                event_data = [
                    event_id,
                    filename.name,
                    sentence_idx,
                    sentence_doc.text,
                    len(all_found),
                    all_found,
                    event_text.text,
                ]

                for ent_label in geology_ents:
                    event_data.append(
                            [ent.text for ent in event_text.ents if ent.label_ == ent_label]
                    )

                event_data.append(label)

                all_event_data.append(event_data)

    feature_columns = [
        'event_id',
        'filename',
        'sentence_idx',
        'sentence_text',
        'n_trigger_words',
        'trigger_words',
        'event_text'
    ]
    label_columns = ['event_label']
    columns = feature_columns + geology_ents + label_columns

    eventdf = pd.DataFrame(all_event_data, columns=columns)

    print(f'found {eventdf.shape[0]} events from a total of {total_sentences} sentences')

    return eventdf


def get_feature_counts_df(df, feature):
    counts = Counter(list(itertools.chain(*df[feature].tolist())))
    return (pd.DataFrame.from_dict(counts, orient='index')
            .reset_index()
            .rename(columns={'index': feature, 0: 'count'})
            .sort_values('count', ascending=False)
            .set_index(feature)
            )
