#!/usr/bin/env bash

set -e

DOWNLOAD_DIR="downloads"
RAW_DIR="data/raw_xmi"
BASE_URL="http://lrec2022.gerparcor.texttechnologylab.org/data"

FILE_LIST=("Bundesrat.tar" "Nationalrat.tar")

mkdir -p "$DOWNLOAD_DIR"
mkdir -p "$RAW_DIR"

for FILE_NAME in "${FILE_LIST[@]}"; do
    echo "Processing $FILE_NAME ..."
    FULL_URL="$BASE_URL/$FILE_NAME"
    SAVE_PATH="$DOWNLOAD_DIR/$FILE_NAME"

    echo "Downloading..."
    curl -L "$FULL_URL" -o "$SAVE_PATH"

    echo "Extracting..."
    #temp folder to help remove nested files
    TMP_DIR=$(mktemp -d)
    tar -xf "$SAVE_PATH" -C "$TMP_DIR"

    echo "Collecting .gz files..."
    find "$TMP_DIR" -type f -name "*.gz" -exec cp {} "$RAW_DIR" \;

    rm -rf "$TMP_DIR"
done

echo "--- Decompressing .gz ---"
find "$RAW_DIR" -type f -name "*.gz" -exec gunzip -f {} \;


echo "--- renaming ---"

for f in "$RAW_DIR"/*; do
    base=$(basename "$f")

    #remove duplicate extensions
    clean="${base%.xmi}"
    clean="${clean%.gz}"
    clean="${clean%.xmi}"
    clean="$clean.xmi"

    #nationalrat: 01.02.1938_29._Sitzung.xmi.xmi
    if [[ "$clean" =~ ^([0-9]{2}\.[0-9]{2}\.[0-9]{4})_([0-9]+)\._Sitzung ]]; then
        date="${BASH_REMATCH[1]}"
        session="${BASH_REMATCH[2]}"
        new="NR_${session}.S_${date}.xmi"
        mv "$f" "$RAW_DIR/$new"
        continue
    fi

    #bundesrat: Plenarprotokoll_2._Sitzung,_12.09.1949.xmi.gz.xmi
    if [[ "$clean" =~ ^Plenarprotokoll_([0-9]+)\._Sitzung,_([0-9]{2}\.[0-9]{2}\.[0-9]{4}) ]]; then
        session="${BASH_REMATCH[1]}"
        date="${BASH_REMATCH[2]}"
        new="BR_${session}.S_${date}.xmi"
        mv "$f" "$RAW_DIR/$new"
        continue
    fi

    #default cleanup rename
    mv "$f" "$RAW_DIR/$clean"
done

echo "---"
echo "Download Corpus is complete. Final xmi files saved in '$RAW_DIR'"