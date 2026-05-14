"""
src/ package — Chemistry Recognizer.

Re-exports the most-used symbols so notebooks can do:

    from src import (ChemCNN, PretrainedModel, ChemDataset, get_dataloaders,
                     train_model, full_evaluation_report, plot_training_curves,
                     TRAIN_TRANSFORM, VAL_TRANSFORM)
"""

from src.augmentation import TRAIN_TRANSFORM, VAL_TRANSFORM, AUGMENT_ONLY, HANDWRITTEN_TRAIN_TRANSFORM
from src.dataset import ChemDataset, get_dataloaders
from src.models import ChemCNN, PretrainedModel
from src.train import train_model, train_epoch, validate_epoch
from src.evaluate import (
    plot_training_curves,
    full_evaluation_report,
    plot_confusion_matrix,
    plot_per_class_f1,
    plot_roc_curves,
    find_top_confused_pairs,
)
from src.vae import ConditionalVAE, vae_loss

__all__ = [
    "TRAIN_TRANSFORM", "VAL_TRANSFORM", "AUGMENT_ONLY", "HANDWRITTEN_TRAIN_TRANSFORM",
    "ChemDataset", "get_dataloaders",
    "ChemCNN", "PretrainedModel",
    "train_model", "train_epoch", "validate_epoch",
    "plot_training_curves", "full_evaluation_report",
    "plot_confusion_matrix", "plot_per_class_f1",
    "plot_roc_curves", "find_top_confused_pairs",
    "ConditionalVAE", "vae_loss",
]
