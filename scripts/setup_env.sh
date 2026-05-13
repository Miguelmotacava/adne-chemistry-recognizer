#!/bin/bash
set -e

echo "=== Chemistry Recognizer: Environment Setup ==="
echo ""
echo "Step 1: Creating conda environment from environment.yml..."
conda env create -f environment.yml || echo "(env may already exist)"

echo ""
echo "Step 2: Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chem-adne

echo ""
echo "Step 3: Installing package in development mode..."
pip install -e .

echo ""
echo "Step 4: Enabling Jupyter widgets..."
jupyter nbextension enable --py widgetsnbextension --sys-prefix || true
jupyter nbextension enable --py ipycanvas --sys-prefix || true

echo ""
echo "Step 5: Generating test dataset (5 images per class) to verify setup..."
python data/generate_dataset.py \
    --categories inorganica \
    --subcategories oxidos_metalicos,hidroxidos \
    --n_per_class 5 \
    --output_dir data/raw

echo ""
echo "Step 6: Running tests..."
python -m pytest tests/ -v

echo ""
echo "=== Setup complete! ==="
echo "To start: conda activate chem-adne && jupyter notebook"
echo "Open notebooks/ and start with 00_dataset_generation.ipynb"
