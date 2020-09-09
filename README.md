# CITS5553 Capstone Project
## Group 5 - Mineral Exploration Report Event Extraction

### Project proposal: 
https://docs.google.com/document/d/18tzEOiPqDGRY44DTDCC5J_Ck4rpK9Eya3hu5x5tLzg0/edit

### Project Structure
```
/data/wamex_xml
```

contains full list of reports

```
/data/subset
```

contains individual subset of reports 

```
/dictionary/patterns
```

contains pattenrs for spacy ruler

```
/dictionary/trigger phrases
```

contains trigger words

```
/events/group_<group number>_events.csv
```

contains extracted events


### Pipeline

#### 1. Extract individual subset of reports
```
python pipeline/extract_subset_reports.py --group=<group number>
```

#### 2. Extract individual subset of Events
```
python pipeline/extract_events.py --group=<group number>
```

#### 3. Label extracted events as Near-Miss or not
```
pipeline/sample_event_labeller.ipynb
```

#### 4. Train Classifier on labelled events
```
pipeline/sample_event_classification.ipynb
```