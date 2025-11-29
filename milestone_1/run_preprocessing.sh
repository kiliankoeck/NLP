#!/bin/bash

cd ./preprocessing || exit

echo "Starting Preprocessing"

#XMI parsing 
printf "\n==================================\n"
printf  "[1/4] Parsing the xmi files to JSON\n"
python xmi_parser.py

#Convert xmi to plain txt files
printf "\n==================================\n"
printf  "[2/4] Converting to plain txt files\n"
python xmi_to_plain_text.py

# Clean plain text files
printf "\n==================================\n"
printf  "[3/4] Cleaning plain txt files\n"
python clean_plain_text.py

# Format Records into CoNLL-U Format
printf "\n==================================\n"
printf  "[4/4] Format records in CoNLL-U Format\n"
python conllu_formatter.py