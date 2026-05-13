#!/bin/bash
# Generate the full dataset (300 images per class). Expect ~30-60 min on CPU.
set -e

N_PER_CLASS=${1:-300}
OUTPUT_DIR=${2:-data/raw}

echo "=== Generating full dataset ==="
echo "  N per class : $N_PER_CLASS"
echo "  Output dir  : $OUTPUT_DIR"
echo ""

python data/generate_dataset.py \
    --categories all \
    --n_per_class "$N_PER_CLASS" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "=== Done. metadata.csv updated. ==="
