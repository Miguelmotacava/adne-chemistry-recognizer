#!/bin/bash
# Headlessly execute notebooks 00..05 with papermill, writing *_executed.ipynb.
set -e

NB_DIR="notebooks"
for nb in 00_dataset_generation 01_EDA 02_classical_ML \
          03_CNN_scratch 04_transfer_learning 05_market_comparison; do
    echo ""
    echo "=== Running: ${NB_DIR}/${nb}.ipynb ==="
    papermill "${NB_DIR}/${nb}.ipynb" "${NB_DIR}/${nb}_executed.ipynb"
done

echo ""
echo "=== All notebooks executed ==="
