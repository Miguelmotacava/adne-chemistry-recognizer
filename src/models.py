"""
Model architectures for the Chemistry Recognizer project.

ChemCNN         : Configurable CNN trained from scratch (n_blocks, base_filters, dropout).
PretrainedModel : Wraps torchvision pretrained backbones (ResNet18, EfficientNet-B0)
                  with two strategies: 'finetune' or 'feature_extraction'.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ChemCNN(nn.Module):
    """Configurable CNN from scratch for chemical structure classification."""

    def __init__(self,
                 num_classes: int,
                 n_blocks: int = 4,
                 base_filters: int = 32,
                 dropout: float = 0.4):
        super().__init__()
        self.n_blocks = n_blocks
        self.base_filters = base_filters
        self.dropout_p = dropout

        layers = []
        in_ch = 3
        out_ch = base_filters
        for i in range(n_blocks):
            out_ch = min(base_filters * (2 ** i), 256)
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            ]
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_ch, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


class PretrainedModel(nn.Module):
    """Wrapper for pretrained backbones with configurable strategy."""

    BACKBONES = {
        "resnet18": (models.resnet18, "ResNet18_Weights", 512),
        "efficientnet_b0": (models.efficientnet_b0, "EfficientNet_B0_Weights", 1280),
    }

    def __init__(self,
                 backbone: str = "resnet18",
                 num_classes: int = 100,
                 strategy: str = "finetune",
                 dropout: float = 0.3):
        super().__init__()
        assert backbone in self.BACKBONES, (
            f"backbone must be one of {list(self.BACKBONES.keys())}"
        )
        assert strategy in ("finetune", "feature_extraction")

        model_fn, weights_attr, in_features = self.BACKBONES[backbone]
        weights = getattr(models, weights_attr).DEFAULT
        base = model_fn(weights=weights)

        if backbone == "resnet18":
            base.fc = nn.Identity()
        elif backbone == "efficientnet_b0":
            base.classifier = nn.Identity()

        self.backbone_name = backbone
        self.strategy = strategy
        self.in_features = in_features
        self.features = base

        if strategy == "feature_extraction":
            for param in self.features.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        # EfficientNet may return a 4D tensor before its (removed) classifier
        if feats.dim() > 2:
            feats = feats.mean(dim=[2, 3])
        return self.classifier(feats)

    def get_optimizer_groups(self,
                             lr_backbone: float = 1e-5,
                             lr_head: float = 1e-3):
        """Parameter groups for differential learning rates."""
        return [
            {"params": self.features.parameters(), "lr": lr_backbone},
            {"params": self.classifier.parameters(), "lr": lr_head},
        ]
