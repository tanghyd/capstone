# CITS5553 Capstone Project
## Group 5 - Mineral Exploration Report Event Extraction

### Project proposal: 
https://docs.google.com/document/d/18tzEOiPqDGRY44DTDCC5J_Ck4rpK9Eya3hu5x5tLzg0/edit

#### Pipeline

Sentences matched on triggers have been grouped and stored in: `data/labels/extract_triggers_grouped.csv'

#### Label Events

```
sentence_labeller.ipynb
```


Here we take the above `extract_triggers_grouped.csv` file and load it into the `sentence_labeller` notebook to label our sentences for each group member. Labellers can custom specify the range of the particular text chunk to include necessary context.

####  Build Event Data for Vectorisation


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


#### 5. Vectorisation and Classification on Labelled Events

Notebooks used to show example pipeline given a labelled set of data.

```
pipeline/example.ipynb
pipeline/classification_testing.ipynb
```


### Project Structure
```
/data/wamex_xml
```

contains full list of reports

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


```
/classification/group_<group number>_events.csv
```

subset of data saved to `events/` for classification testing


NOTE: we no longer require the files in `events/` but we may want to store them for reproducibility.
