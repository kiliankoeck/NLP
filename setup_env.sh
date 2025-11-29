#!/bin/bash

set -e

python -m venv .venv

#Linux:
#source .venv/bin/activate

#Windows:
source .venv/Scripts/activate

pip install -r requirements.txt

python -m spacy download de_core_news_sm
python -m spacy download de_core_news_md

echo "Environment ready"