from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path("data/generated/dataset_expanded")

BATCH_SIZE = 32
IMAGE_SIZE = 224


# ============================================================
# TRANSFORMS
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


# ============================================================
# DATASETS
# ============================================================

train_dataset = datasets.ImageFolder(
    DATASET_DIR / "train",
    transform=transform,
)

val_dataset = datasets.ImageFolder(
    DATASET_DIR / "val",
    transform=transform,
)

test_dataset = datasets.ImageFolder(
    DATASET_DIR / "test",
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
print("PYTORCH IMAGE QUALITY DATA PIPELINE")
print("=" * 60)

print(f"Dataset directory : {DATASET_DIR}")
print()

print(f"Training images   : {len(train_dataset)}")
print(f"Validation images : {len(val_dataset)}")
print(f"Test images       : {len(test_dataset)}")

print()

print("Classes:")
print(train_dataset.classes)

print()

print("Class → label mapping:")
print(train_dataset.class_to_idx)


# ============================================================
# INSPECT ONE BATCH
# ============================================================

images, labels = next(iter(train_loader))

print()
print("=" * 60)
print("FIRST TRAINING BATCH")
print("=" * 60)

print(f"Image tensor shape : {images.shape}")
print(f"Label tensor shape : {labels.shape}")
print(f"Image tensor type  : {images.dtype}")
print(f"Label tensor type  : {labels.dtype}")

print()

print(f"Image value range  : {images.min().item():.4f} "
      f"to {images.max().item():.4f}")

print(f"Labels in batch    : {labels.tolist()}")

print()

print("=" * 60)
print("PIPELINE TEST PASSED")
print("=" * 60)