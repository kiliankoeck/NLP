# Named Entity Recognition in the stenographic records of the Austrian Parliament

Team Sentimental Analysis

## Overview

In this project, we attempt to implement NER of several types of named entities in the stenographic records of the Austrian Parliament. 

## Repository structure

```
/preprocessing/
  xmi_parser.py
  xmi_to_plain_text.py
  clean_plain_text.py
  conllu_formatter.py

/split_data/
  split_data.py

/data/
  /raw_xmi/                 # raw xmi files from gerparcor
  /raw_json/                # xmi parsed to JSON
  /plain_text/              # cleaned plain-text speeches
  /conllu/                  # final CoNLL-U output

  /test_set/                # test set xmi data
  /test_set_conllu/         # test set conllu data
  /test_set_json/           # test set JSON data
  /test_set_txt/            # test set plain text

  /splits/                  # train, test, and validation sets of the CoNLL-U files
    /test/
    /train/
    /val/

/downloads/                 #contains the raw .tar file from gerparcor

README.md
download_corpus.sh
sample_corpus.sh
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
(not working yet, need to run individually) run the run_preprocessing.sh script. It performs four steps:
1. Parsing the XMI files from the downloads folder (from gerparcor) (```/preprocessing/xmi_parser.py```)
2. Extracting the raw text files  (```/preprocessing/xmi_to_plain_text.py```)
3. Cleaning the raw text files  (```/preprocessing/clean_plain_text.py```)
4. Formatting the data in CoNLL-U Format (```/preprocessing/conllu_formatter.py```)

The final CoNLL-U files are stored under ```/data/connlu```

### Data Splitting: 
Splitting the CoNLL-U files into training, testing, and validation sets (```/split_data/split_data.py```)
The final train/test/validation sets are store under ```/data/splits```

## MILESTONE 1:
### Preprocessing

For this milestone, we first fetched the raw plain text protocols of the Austrian Parliament's API using the ```/fetch_records.py``` script which was created with the help of chatGPT. 
It loads all stenographic records form the parliaments open data platform. The documentation of the API can be found under 
https://www.parlament.gv.at/recherchieren/open-data/daten-und-lizenz/stenographische-protkolle/index.html

The script downloads all records in their original html form, while also using beautifulsoup (https://www.crummy.com/software/BeautifulSoup/bs4/doc/) 
to transform the structured html into simple txt files (stored under ```/data/plain_text```)

We then inspected the data and thought about which data is relevant for the next steps. We decided that we want to keep the speakers and the speech text. 
We then had to think about how to extract that data only, and came up with the regex patterns used in the python file called ```tokenize_sentences.py```. 
This step results in txt files of the form

```speaker: [sentence1, sentence2, ...]```

Once we had that cleaned data, we then used the spaCy library in the script ```conllu.py``` to create the final CoNLL-U files, which consists of word lines for each word of a sentence. 
Each word line in turn gives information about this particular word, most notably the lemma or stem, and part of speech tags.

## MILESTONE 2:
### Preprocessing
For this milestone, we changed the data gathering a preprocessing. We first fetched the raw xmi protocols from GerParCor using the ```/download_corpus.sh``` script which was created with the help of chatGPT.  It loads all stenographic records form the parliaments open data platform. The documentation of the API can be found under http://lrec2022.gerparcor.texttechnologylab.org/. The .tar files downaloaded are stored under ```/downloads```. The raw xmi files downloaded (are also renamed for standardization) and stored under The original .tar files downaloaded are stored under ```/data/raw_xmi```.

The ```xmi_parser.py``` script then parses the raw xmi files using xml.etree.elementtree (https://docs.python.org/3/library/xml.etree.elementtree.html) to transform the xmi into simple json files (stored under ```/data/plain_text```). This extracts the original NER tags in the documents. Then the ```conllu_formatter.py``` script converts json into the CoNLL-U format with teh NER tags stored under ```/data/conllu```. We have the NER tags in BIO format, and we will be inspecting persons (PER), Organizations (ORG), and locations (LOC).

To get the plain text, ```xmi_to_plain_text.py``` is run and stores the plain text under ```/data/plain_text```. We found that there were weird characters extracted here and used a cleaning helper script to ensure the text was formatted properly, ```clean_plain_text.py``` and stored under ```/data/plain_text``` (overwrites the weird characters).

### Splitting the Data

We created a script that can possibly be used for future model creation/evaluation. It splits the data into training (70%), testing (15%), and validation (15%) sets. The ```split_data.py``` script used the ```train_test_split``` function from the scikit-learn library to preform the randomized splits into the data subsets. The script stores the resulting sets under the ```/data/splits/``` directory ready to be used in the training and evaluation of our models. 


### Models 

For the baseline models, we chose one rule-based model and two machine learning method. For the rule based model, we collect name lists from GeoNames and the Austrian parliament API, turn them into spaCy EntityRuler patterns, and then match them in text. It returns non-overlapping entity spans as `Entity` objects for persons, locations, and organizations. One of the machine learning models is spacy, specifically the german spaCy model which reads each plain-text file from the test folder, runs spaCy NER on the entire text, filters entities to PER/ORG/LOC, and writes the extracted entities to an output file. The other machine learning method we used was flair - which is a hugging face tokem classifier for the german language. 
All of those methods are being called in our ner pipeline which first loads our XMI files and then runs the NER models. For all the models, we calculate the f1 score, the precision and the recal so that we can then effectiviely compare their performance. 
