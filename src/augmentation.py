"""
Albumentations pipelines for training, validation, and dataset augmentation.

TRAIN_TRANSFORM : full training pipeline with Normalize + ToTensorV2
VAL_TRANSFORM   : validation/test pipeline (Resize + Normalize + ToTensorV2)
AUGMENT_ONLY    : geometric/photometric distortions ONLY, no Normalize/ToTensor.
                  Used by data/generate_dataset.py to create on-disk PNG variants.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# Albumentations 2.x uses Affine (not ShiftScaleRotate), GaussNoise(std_range=...),
# ElasticTransform without alpha_affine. We probe the API once.
_A_V2 = hasattr(A, "Affine") and "std_range" in A.GaussNoise.__init__.__doc__ if A.GaussNoise.__init__.__doc__ else False
try:
    A.GaussNoise(std_range=(0.05, 0.2))
    _A_V2 = True
except Exception:
    _A_V2 = False


def _shift_scale_rotate(p):
    if _A_V2:
        return A.Affine(translate_percent=(-0.1, 0.1),
                        scale=(0.8, 1.2),
                        rotate=(-15, 15), p=p)
    return A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2,
                              rotate_limit=15, p=p)


def _gauss_noise(p, strong=False):
    if _A_V2:
        rng = (0.05, 0.20) if strong else (0.02, 0.10)
        return A.GaussNoise(std_range=rng, p=p)
    return A.GaussNoise(var_limit=(10.0, 80.0 if strong else 50.0), p=p)


def _elastic(p, strong=False):
    a = 50 if strong else 30
    s = 7 if strong else 5
    return A.ElasticTransform(alpha=a, sigma=s, p=p)


TRAIN_TRANSFORM = A.Compose([
    A.Resize(224, 224),
    _shift_scale_rotate(p=0.7),
    _gauss_noise(p=0.5, strong=False),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    _elastic(p=0.3, strong=False),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])


VAL_TRANSFORM = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])


AUGMENT_ONLY = A.Compose([
    A.Resize(224, 224),
    _shift_scale_rotate(p=0.9),
    _gauss_noise(p=0.8, strong=True),
    _elastic(p=0.7, strong=True),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.6),
])
