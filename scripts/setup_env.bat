@echo off
setlocal

echo === Chemistry Recognizer: Environment Setup ===
echo.

echo Step 1: Creating conda environment...
call conda env create -f environment.yml
if errorlevel 1 echo (env may already exist)

echo.
echo Step 2: Activating environment...
call conda activate chem-adne

echo.
echo Step 3: Installing package in development mode...
pip install -e .

echo.
echo Step 4: Enabling Jupyter widgets...
jupyter nbextension enable --py widgetsnbextension --sys-prefix
jupyter nbextension enable --py ipycanvas --sys-prefix

echo.
echo Step 5: Generating test dataset (5 images per class)...
python data\generate_dataset.py --categories inorganica --subcategories oxidos_metalicos,hidroxidos --n_per_class 5 --output_dir data\raw

echo.
echo Step 6: Running tests...
python -m pytest tests\ -v

echo.
echo === Setup complete! ===
echo To start: conda activate chem-adne
echo Then:    jupyter notebook
echo Open notebooks\ and start with 00_dataset_generation.ipynb

endlocal
