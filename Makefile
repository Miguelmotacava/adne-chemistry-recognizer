.PHONY: setup data-small data-full train demo test clean

# Create environment and install everything
setup:
	conda env create -f environment.yml
	conda run -n chem-adne pip install -e .
	conda run -n chem-adne jupyter nbextension enable --py widgetsnbextension --sys-prefix
	conda run -n chem-adne jupyter nbextension enable --py ipycanvas --sys-prefix
	@echo "Run: conda activate chem-adne && jupyter notebook"

# Generate small dataset for testing (5 images per class)
data-small:
	conda run -n chem-adne python data/generate_dataset.py \
		--categories all --n_per_class 5 --output_dir data/raw

# Generate full dataset (300 images per class - takes 20-40 min)
data-full:
	conda run -n chem-adne python data/generate_dataset.py \
		--categories all --n_per_class 300 --output_dir data/raw

# Run all training notebooks headlessly (requires papermill)
train:
	conda run -n chem-adne papermill notebooks/01_EDA.ipynb notebooks/01_EDA_executed.ipynb
	conda run -n chem-adne papermill notebooks/02_classical_ML.ipynb notebooks/02_classical_ML_executed.ipynb
	conda run -n chem-adne papermill notebooks/03_CNN_scratch.ipynb notebooks/03_CNN_scratch_executed.ipynb
	conda run -n chem-adne papermill notebooks/04_transfer_learning.ipynb notebooks/04_transfer_learning_executed.ipynb

# Open only the interactive demo
demo:
	conda run -n chem-adne jupyter notebook notebooks/06_interactive_demo.ipynb

# Run tests
test:
	conda run -n chem-adne python -m pytest tests/ -v

# Remove generated data and models (keeps code)
clean:
	rm -rf data/raw/*
	rm -rf saved_models/*.pt
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete
