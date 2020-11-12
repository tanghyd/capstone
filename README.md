# CITS5553 Capstone Project
## Group 5 - Mineral Exploration Report Event Extraction

### Project proposal: 
https://docs.google.com/document/d/18tzEOiPqDGRY44DTDCC5J_Ck4rpK9Eya3hu5x5tLzg0/edit

Note: All final code was run on Ubuntu 18.04.

#### Pipeline

Sentences matched on triggers have been grouped and stored in: `data/labels/extract_triggers_grouped.csv`

#### Label Events

```
sentence_labeller.ipynb
```

Here we take the above `extract_triggers_grouped.csv` file and load it into the `sentence_labeller` notebook to label our sentences for each group member. Labellers can custom specify the range of the particular text chunk to include necessary context.

####  Build Event Data for Vectorisation

`build_event_data.ipynb` combines all the labelled event data from multiple labellers into a unified set of labelled events.

We additionally implement logic to create a 'commodities' dataframe - this is a one-hot encoding of the presence of any targeted commodity from the GeoView Reports (by ANUMBER). This data-frame prepared to be used in the dashboard to subset our data based on a user-specified input of what target commodity they would like to see.

This file does not return text - it only stores the sentence index for each JSON file for efficiency - the text is extracted in the tokenisation pipeline.

Uses the function:

```
python build_event_data.py --confidence high medium low --users daniel charlie --new_events
```

parameters:
--confidence (or -c): high medium low
--users (or -u): daniel charlie (or anyone else who has labelled data in data/labels)
--new_events (include only new events) OR
--all_events (also include old events - note these are labelled FALSE in confidence == High)

This takes labelled sentences (and their lower and upper bounds - i.e. may be a 3 sentence text chunk) as well as stored metadata to produce labelled events for vectorisation and classification.

Events can be extracted either sentence-by-sentence, or with a sliding window of specified (i.e. a `pad` is set). Old text chunks are extracted with `pad=2` to be equivalent to `n_sentences_extract=2`.

This code is dependent on the following files:
- `data/geoview/capstone_metadata.zip`
- all reports stored in `data/wamex_xml/`


#### Tokenisation Pipeline

In the `tokenisation_pipeline.ipynb` file we take all extracted sentences with trigger words from all reports (approximately 580,746 sentences (rows)) marked by their index positions. We specify a default padding of `pad=1` to construct chunks of length 3 (one sentence each side of the trigger sentence).

Text chunks are merged together if they are adjacent to one another, but duplicates are handled by cutting the size of text chunks until no more overlapping occurs. A default text chunk limit size has been set of 5 (so no text chunks should have length greater than 5).

This is only one way of chunking the sentence text - there are other methods that could be tried (allowing duplicates, different sentence limit, fixed text chunks size, etc).

We then tokenise the data by lower-casing the text, lemmatizing words and additionally removing punctuation via a spaCy-based pipeline. Again, this is only one method of tokenisation.

The data is saved and the coverage of the text reports is shown in plots at the end of the notebook.

Note: This code has been multi-threaded but it is not batch-processed for memory - if run as is it will likely require up to 16GB of RAM to run without issue (when results are saved together there is a brief memory spike).

#### Classification Pipeline

In `classification_pipeline.ipynb` we load the tokenised text chunks and extract them into a list of strings (`events.tokens`) and a concatenate string with a whitespace separator (`events.tokenized_text`). Depending on the vectorisation method (i.e. our trained `Doc2Vec` model, `sklearn` TF-IDF+Truncated SVD methods, or `LIME` explainers all need different text input representations).

We load our pretrained doc2vec model (fit to the entire collection of extracted text chunks) and fit a number of classifiers to the labelled data set (transformed by the pretrained doc2vec model). This model and its corresponding word vectors are stored in `capstone/models`.

We analyse the distributions of probabilities predicted by the two estimators in our soft voting classifier (XGBoost and SVM).

Addiitionally, we explore some of the results with LIME.

Finally, the model is used to predict probabilities of being a near miss for every single text chunk with some brief visual analysis of the different implications of our report aggregated scoring function (i.e. how do we aggregate the probability of an event being a near miss at the report level).

#### Data Cleaning

Some post-processing steps were implemented in `clean_event_dataframe.ipynb` like sorting rows and columns.

These final data files are output to `data/predictions`.

### Miscellaneous
Notebooks used to show example pipeline given a labelled set of data.

```
example.ipynb
classification_testing.ipynb
```

Notebook to pre-render spaCy HTML for named entity recognition - HTML data is transferred manually to the capstone-dashboard repository.

```
generate_html.iypnb
```

### Other Folders
```
/data/wamex_xml
```
contains full list of reports

```
/data/lemmatization/
```

shows some visual examples of the output of different lemmatization methods.

```
/dictionary/patterns
```

contains patterns for spaCy ruler

```
/dictionary/trigger phrases
```

contains trigger words

```
/labels/
```

contains events separately labelled for `sentence_labeller.ipynb`

```
/events/
```

contains build event data - i.e. named entities and metadata.


NOTE: we no longer require the files in `events/` but we may want to store them for reproducibility.
