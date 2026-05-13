"""Tests for src/dataset.py — only runs if data/metadata.csv exists."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_dataset_loads():
    metadata_path = ROOT / "data" / "metadata.csv"
    if not metadata_path.exists() or metadata_path.stat().st_size == 0:
        pytest.skip("Dataset not generated yet — "
                    "run `python data/generate_dataset.py` first")

    import pandas as pd
    df = pd.read_csv(metadata_path)
    if len(df) == 0:
        pytest.skip("metadata.csv is empty — run generate_dataset.py first")

    from src.dataset import get_dataloaders
    train_loader, val_loader, test_loader, class_names, class_to_idx = (
        get_dataloaders(metadata_path=str(metadata_path),
                        batch_size=4, num_workers=0, root_dir=ROOT)
    )
    images, labels = next(iter(train_loader))
    assert images.shape[1:] == (3, 224, 224), (
        f"Expected (B, 3, 224, 224), got {images.shape}"
    )
    assert labels.ndim == 1
    assert len(class_names) > 0
    assert len(class_to_idx) == len(class_names)


def test_chemdataset_import():
    """Smoke test: ChemDataset and get_dataloaders are importable."""
    from src.dataset import ChemDataset, get_dataloaders
    assert ChemDataset is not None
    assert get_dataloaders is not None
