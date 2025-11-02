#!/bin/bash

cd ./preprocessing || exit

echo "Starting Preprocessing"

# Fetch Records - comment out if unwanted
#printf  "Fetching Records\n"
#python fetch_records.py

# Clean plain txt files into sentence tokenized speeches
printf "\n==================================\n"
printf  "Cleaning plain txt files\n"
python tokenize_sentences.py

# Format Records into CoNLL-U Format
printf "\n==================================\n"
printf  "Format records in CoNLL-U Format\n"
python conllu.py
