#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Step 1: Downloading corpus ==="
./download_corpus.sh

echo "=== Step 2: Running NER pipeline ==="
python -m milestone_2.ner_pipeline

echo "=== All done ==="
