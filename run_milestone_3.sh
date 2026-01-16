#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Step 1: Running NER pipeline ==="
python -m milestone_3.ner_pipeline

echo "=== Step 2: Evaluate Results ==="
python -m milestone_3.evaluate_results

echo "=== All done ==="
