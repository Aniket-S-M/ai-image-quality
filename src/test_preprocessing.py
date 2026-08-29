from pathlib import Path

import torch
from torchvision import transforms

from image_quality_dataset import ImageQualityDataset


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path(
    "data/generated/dataset_expanded"
)

IMAGE_SIZE = 224


# ============================================================
# IMAGENET NORMALIZATION
# ============================================================

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406,
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225,
]


# ============================================================
# TRAINING TRANSFORM
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    ),
])


# ============================================================
# VALIDATION / TEST TRANSFORM
# ============================================================

eval_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    ),
])


# ============================================================
# DATASETS
# ============================================================

train_dataset = ImageQualityDataset(
    DATASET_ROOT / "train",
    transform=train_transform,
)

val_dataset = ImageQualityDataset(
    DATASET_ROOT / "val",
    transform=eval_transform,
)

test_dataset = ImageQualityDataset(
    DATASET_ROOT / "test",
    transform=eval_transform,
)


# ============================================================
# INSPECT SAMPLES
# ============================================================

print("=" * 60)
print("IMAGE PREPROCESSING TEST")
print("=" * 60)

print(f"Train images : {len(train_dataset)}")
print(f"Val images   : {len(val_dataset)}")
print(f"Test images  : {len(test_dataset)}")


# ============================================================
# TRAIN SAMPLE
# ============================================================

train_image, train_label = train_dataset[0]

print()
print("=" * 60)
print("TRAIN SAMPLE")
print("=" * 60)

print(f"Shape : {train_image.shape}")
print(f"Dtype : {train_image.dtype}")
print(f"Label : {train_label}")

print(
    f"Range : "
    f"{train_image.min().item():.4f} → "
    f"{train_image.max().item():.4f}"
)


# ============================================================
# VALIDATION SAMPLE
# ============================================================

val_image, val_label = val_dataset[0]

print()
print("=" * 60)
print("VALIDATION SAMPLE")
print("=" * 60)

print(f"Shape : {val_image.shape}")
print(f"Dtype : {val_image.dtype}")
print(f"Label : {val_label}")

print(
    f"Range : "
    f"{val_image.min().item():.4f} → "
    f"{val_image.max().item():.4f}"
)


# ============================================================
# VALIDATION
# ============================================================

assert train_image.shape == (
    3,
    IMAGE_SIZE,
    IMAGE_SIZE,
)

assert val_image.shape == (
    3,
    IMAGE_SIZE,
    IMAGE_SIZE,
)

assert train_image.dtype == torch.float32
assert val_image.dtype == torch.float32

assert 0 <= train_label < 6
assert 0 <= val_label < 6


print()
print("=" * 60)
print("PREPROCESSING TEST PASSED")
print("=" * 60)