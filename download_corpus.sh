#!/usr/bin/env bash

set -e

DOWNLOAD_DIR="downloads"
RAW_DIR="data/raw_xmi"
BASE_URL="http://lrec2022.gerparcor.texttechnologylab.org/data"

FILE_LIST=("Bundesrat.tar" "Nationalrat.tar")

mkdir -p "$DOWNLOAD_DIR" "$RAW_DIR"

for FILE_NAME in "${FILE_LIST[@]}"; do
    echo "Processing $FILE_NAME ..."
    FULL_URL="$BASE_URL/$FILE_NAME"
    SAVE_PATH="$DOWNLOAD_DIR/$FILE_NAME"

    if [ -f "$SAVE_PATH" ]; then
        echo "File already exists, skipping download: $SAVE_PATH"
    else
        echo "Downloading..."
        curl -L "$FULL_URL" -o "$SAVE_PATH"
    fi

    echo "Extracting..."
    TMP_DIR=$(mktemp -d)
    tar -xf "$SAVE_PATH" -C "$TMP_DIR"

    echo "Collecting files..."
    find "$TMP_DIR" -type f -exec cp {} "$RAW_DIR" \;

    rm -rf "$TMP_DIR"
done

echo "--- renaming ---"

for f in "$RAW_DIR"/*; do
    [ -f "$f" ] || continue
    base=$(basename "$f")

    # remember if it was gzipped
    gz_suffix=""
    stem="$base"
    if [[ "$stem" == *.gz ]]; then
        gz_suffix=".gz"
        stem="${stem%.gz}"
    fi

    # remove duplicate extensions, normalize to single .xmi
    clean="${stem%.xmi}"
    clean="${clean%.xmi}"
    clean="$clean.xmi"

    # nationalrat: 01.02.1938_29._Sitzung.xmi(.gz)
    if [[ "$clean" =~ ^([0-9]{2}\.[0-9]{2}\.[0-9]{4})_([0-9]+)\._Sitzung ]]; then
        date="${BASH_REMATCH[1]}"
        session="${BASH_REMATCH[2]}"
        new_base="NR_${session}.S_${date}.xmi"
        mv "$f" "$RAW_DIR/${new_base}${gz_suffix}"
        continue
    fi

    # bundesrat: Plenarprotokoll_2._Sitzung,_12.09.1949.xmi(.gz)
    if [[ "$clean" =~ ^Plenarprotokoll_([0-9]+)\._Sitzung,_([0-9]{2}\.[0-9]{2}\.[0-9]{4}) ]]; then
        session="${BASH_REMATCH[1]}"
        date="${BASH_REMATCH[2]}"
        new_base="BR_${session}.S_${date}.xmi"
        mv "$f" "$RAW_DIR/${new_base}${gz_suffix}"
        continue
    fi

    # default cleanup rename
    mv "$f" "$RAW_DIR/${clean}${gz_suffix}"
done

echo "---"
echo "Download complete. Raw (possibly compressed) XMI files in '$RAW_DIR'"