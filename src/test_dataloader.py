from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from image_quality_dataset import ImageQualityDataset


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path(
    "data/generated/dataset_expanded"
)

IMAGE_SIZE = 224
BATCH_SIZE = 32


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


# ============================================================
# DATASETS
# ============================================================

train_dataset = ImageQualityDataset(
    DATASET_ROOT / "train",
    transform=transform,
)

val_dataset = ImageQualityDataset(
    DATASET_ROOT / "val",
    transform=transform,
)

test_dataset = ImageQualityDataset(
    DATASET_ROOT / "test",
    transform=transform,
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# ============================================================
# DATASET INFORMATION
# ============================================================

print("=" * 60)
print("PYTORCH DATALOADER TEST")
print("=" * 60)

print(f"Train dataset : {len(train_dataset)}")
print(f"Val dataset   : {len(val_dataset)}")
print(f"Test dataset  : {len(test_dataset)}")

print()
print(f"Batch size    : {BATCH_SIZE}")


# ============================================================
# GET ONE TRAINING BATCH
# ============================================================

images, labels = next(iter(train_loader))


print()
print("=" * 60)
print("TRAINING BATCH")
print("=" * 60)

print(f"Images shape : {images.shape}")
print(f"Labels shape : {labels.shape}")

print(f"Images dtype : {images.dtype}")
print(f"Labels dtype : {labels.dtype}")

print()

print(
    f"Pixel range : "
    f"{images.min().item():.4f} → "
    f"{images.max().item():.4f}"
)

print()

print("Labels:")
print(labels.tolist())


# ============================================================
# VALIDATION
# ============================================================

assert images.shape == (
    BATCH_SIZE,
    3,
    IMAGE_SIZE,
    IMAGE_SIZE,
)

assert labels.shape == (BATCH_SIZE,)

assert images.dtype == torch.float32

assert labels.dtype == torch.int64


print()
print("=" * 60)
print("DATALOADER TEST PASSED")
print("=" * 60)