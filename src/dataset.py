"""
PyTorch Dataset + DataLoader factory for the Chemistry Recognizer project.

ChemDataset       : wraps a metadata DataFrame; returns (tensor, label) pairs.
get_dataloaders() : returns train/val/test loaders + class names + class_to_idx.
                    Uses WeightedRandomSampler on the training loader so that
                    minority classes are oversampled (handles class imbalance).
"""

from pathlib import Path
from typing import Iterable, Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from src.augmentation import TRAIN_TRANSFORM, VAL_TRANSFORM


class ChemDataset(Dataset):
    """A torch Dataset that reads images from disk based on a metadata DataFrame."""

    def __init__(self,
                 df: pd.DataFrame,
                 transform=None,
                 root_dir: Path = Path("."),
                 class_to_idx: Optional[Dict[str, int]] = None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.root_dir = Path(root_dir)
        if class_to_idx is None:
            self.classes = sorted(self.df["compound_id"].unique().tolist())
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        else:
            self.class_to_idx = dict(class_to_idx)
            self.classes = sorted(self.class_to_idx.keys(),
                                  key=lambda k: self.class_to_idx[k])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_path = self.root_dir / row["filepath"]
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        label = self.class_to_idx[row["compound_id"]]
        return image, label


def _filter(df: pd.DataFrame,
            category: Optional[str],
            subcategories: Optional[Iterable[str]],
            difficulty) -> pd.DataFrame:
    if category:
        df = df[df["category"] == category]
    if subcategories:
        df = df[df["subcategory"].isin(list(subcategories))]
    if difficulty is not None:
        if isinstance(difficulty, str):
            difficulty = [difficulty]
        df = df[df["difficulty"].isin(list(difficulty))]
    return df


def get_dataloaders(metadata_path: str = "data/metadata.csv",
                    category: Optional[str] = None,
                    subcategories: Optional[Iterable[str]] = None,
                    difficulty=None,
                    batch_size: int = 64,
                    num_workers: int = 4,
                    root_dir: Path = Path("."),
                    train_transform=TRAIN_TRANSFORM,
                    eval_transform=VAL_TRANSFORM,
                    use_weighted_sampler: bool = True,
                    persistent_workers: Optional[bool] = None
                    ) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], Dict[str, int]]:
    """
    Build train/val/test DataLoaders from a metadata CSV.

    Returns:
        train_loader, val_loader, test_loader, class_names, class_to_idx
    """
    metadata_path = Path(metadata_path)
    df = pd.read_csv(metadata_path)

    df = _filter(df, category, subcategories, difficulty)

    if len(df) == 0:
        raise ValueError(
            "No data found with the given filters. "
            "Run `python data/generate_dataset.py` first or relax the filters."
        )

    # Build class_to_idx from the FULL filtered set so all splits share labels
    all_classes = sorted(df["compound_id"].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(all_classes)}

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    train_ds = ChemDataset(train_df, transform=train_transform,
                           root_dir=root_dir, class_to_idx=class_to_idx)
    val_ds = ChemDataset(val_df, transform=eval_transform,
                         root_dir=root_dir, class_to_idx=class_to_idx)
    test_ds = ChemDataset(test_df, transform=eval_transform,
                          root_dir=root_dir, class_to_idx=class_to_idx)

    if persistent_workers is None:
        persistent_workers = num_workers > 0

    common = dict(num_workers=num_workers, pin_memory=True,
                  persistent_workers=persistent_workers)

    if use_weighted_sampler and len(train_df) > 0:
        class_counts = train_df["compound_id"].value_counts().to_dict()
        weights = [1.0 / max(class_counts.get(cid, 1), 1)
                   for cid in train_df["compound_id"].tolist()]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights),
                                        replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  sampler=sampler, **common)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True, **common)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)

    return train_loader, val_loader, test_loader, all_classes, class_to_idx
