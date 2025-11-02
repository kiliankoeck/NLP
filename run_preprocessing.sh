#!/bin/bash

cd ./preprocessing || exit

echo "Starting Preprocessing"

# Fetch Records - comment out if unwanted
#printf  "[1/3] Fetching Records\n"
#python fetch_records.py
printf "Skipping [1/3] Fetch\n"

# Clean plain txt files into sentence tokenized speeches
printf "\n==================================\n"
printf  "[2/3] Cleaning plain txt files\n"
python tokenize_sentences.py

# Format Records into CoNLL-U Format
printf "\n==================================\n"
printf  "[3/3] Format records in CoNLL-U Format\n"
python conllu.py
