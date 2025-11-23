#!/bin/bash

cd ./preprocessing || exit

echo "Starting Preprocessing"

# Fetch Records - comment out if unwanted
#printf  "[1/4] Fetching Records\n"
#python fetch_records.py
printf "Skipping [1/4] Fetch\n"

# Clean plain txt files into sentence tokenized speeches
printf "\n==================================\n"
printf  "[2/4] Cleaning plain txt files\n"
python tokenize_sentences.py

# Format Records into CoNLL-U Format
printf "\n==================================\n"
printf  "[3/4] Format records in CoNLL-U Format\n"
python conllu.py

# Split CoNLL-U file into train/test/val sets
printf "\n==================================\n"
printf  "[4/4] Splitting CoNLL-U files into train, test, and validation sets\n"
python split_data.py