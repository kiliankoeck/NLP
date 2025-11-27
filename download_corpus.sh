#!/usr/bin/env bash

set -e

DOWNLOAD_DIR="downloads"

RAW_DIR="data/raw_xmi"
BASE_URL="http://lrec2022.gerparcor.texttechnologylab.org/data"

FILE_LIST=("Bundesrat.tar" "Nationalrat.tar")

mkdir -p "$DOWNLOAD_DIR"
mkdir -p "$RAW_DIR"

for FILE_NAME in "${FILE_LIST[@]}"; do
    echo "---"
    echo "Starting process for: $FILE_NAME"

    FULL_URL="$BASE_URL/$FILE_NAME"
    SAVE_PATH="$DOWNLOAD_DIR/$FILE_NAME"

    echo "Downloading..."
    curl -L "$FULL_URL" -o "$SAVE_PATH"

    echo "Extracting..."
    tar -xf "$SAVE_PATH" -C "$RAW_DIR"
done

echo "unzipping .gz files..."

find "$RAW_DIR" -type f -name "*.gz" -exec gunzip -f {} \;

echo "---"
echo "Download is complete - into '$RAW_DIR' folder"