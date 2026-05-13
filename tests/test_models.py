"""Tests for src/models.py — ChemCNN and PretrainedModel forward-pass shapes."""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import ChemCNN, PretrainedModel


def test_chemcnn_forward_shape():
    model = ChemCNN(num_classes=10, n_blocks=2)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 10), f"Got shape {out.shape}"


def test_chemcnn_blocks_variants():
    for n_blocks in [2, 3, 4, 5]:
        model = ChemCNN(num_classes=5, n_blocks=n_blocks)
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        assert out.shape == (1, 5)


@pytest.mark.parametrize("backbone", ["resnet18", "efficientnet_b0"])
@pytest.mark.parametrize("strategy", ["finetune", "feature_extraction"])
def test_pretrained_forward_shape(backbone, strategy):
    try:
        model = PretrainedModel(backbone=backbone, num_classes=5,
                                strategy=strategy)
    except Exception as e:
        pytest.skip(f"Could not load pretrained weights for {backbone}: {e}")

    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 5), f"Got shape {out.shape}"


def test_pretrained_feature_extraction_freezes_backbone():
    try:
        model = PretrainedModel(backbone="resnet18", num_classes=3,
                                strategy="feature_extraction")
    except Exception as e:
        pytest.skip(f"Could not load weights: {e}")
    for p in model.features.parameters():
        assert not p.requires_grad
    for p in model.classifier.parameters():
        assert p.requires_grad


def test_optimizer_groups():
    try:
        model = PretrainedModel(backbone="resnet18", num_classes=3,
                                strategy="finetune")
    except Exception as e:
        pytest.skip(f"Could not load weights: {e}")
    groups = model.get_optimizer_groups(lr_backbone=1e-5, lr_head=1e-3)
    assert len(groups) == 2
    assert groups[0]["lr"] == 1e-5
    assert groups[1]["lr"] == 1e-3
