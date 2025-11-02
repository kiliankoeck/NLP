# Named Entity Recognition in the stenographic records of the Austrian Parliament

Team Sentimental Analysis

## Overview

In this project, we attempt to implement NER of several types of named entities in the stenographic records of the Austrian Parliament. 

## Repository structure

```
/preprocessing/
  fetch_records.py
  tokenize_sentences.py
  connlu.py

/data/
  /raw_html/                # downloaded HTML protocols
  /plain_text/              # cleaned plain-text speeches
  /sentence_tokenized/      # tokenized sentences for each speaker 
  /conllu/                  # final CoNLL-U output

README.md
requirements.txt
run_preprocessing.sh
setup_env.sh
```

## How to run the project:

To start up the virtual environment, run ```setup_env.sh```. Since activating the venv is different for linux and windows, please first comment and uncomment the appropriate lines

```
Linux:
source .venv/bin/activate

Windows:
source .venv/Scripts/activate
```

### Preprocessing: 
run the run_preprocessing.sh script. It performs three steps 
1. Fetching data (```/preprocessing/fetch_records.py```)
2. Cleaning raw txt files and tokenize sentences (```/preprocessing/tokenize_sentences.py```)
3. Formatting the data in CoNLL-U Format (```/preprocessing/connlu.py```)

The final results are stored under ```/data/connlu```

## Project Description 

### Preprocessing

For this milestone, we first fetched the raw plain text protocols of the Austrian Parliament's API with the help of chatGPT. 
It loads all stenographic records form the parliaments open data platform. The documentation of the API can be found under 
https://www.parlament.gv.at/recherchieren/open-data/daten-und-lizenz/stenographische-protkolle/index.html

The script downloads all records in their original html form, while also using beautifulsoup (https://www.crummy.com/software/BeautifulSoup/bs4/doc/) 
to transform the structured html into simple txt files (stored under ```/data/plain_text```)

We then inspected the data and thought about which data is relevant for the next steps. We decided that we want to keep the speakers and the speech text. 
We then had to think about how to extract that data only, and came up with the regex patterns used in the python file called "```tokenize_sentences.py```". 
This step results in txt files of the form

```speaker: [sentence1, sentence2, ...]```

Once we had that cleaned data, we then used the spaCy library to create the final CoNLL-U files, which consists of word lines for each word of a sentence. 
Each word line in turn gives information about this particular word, most notably the lemma or stem, and part of speech tags.
