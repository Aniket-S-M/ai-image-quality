from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

from image_quality_dataset import ImageQualityDataset


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path(
    "data/generated/dataset_expanded"
)

IMAGE_SIZE = 224
BATCH_SIZE = 32

NUM_CLASSES = 6

NUM_WORKERS = 0


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("RESNET-18 TRAINING SETUP")
print("=" * 60)

print(f"Device: {device}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# ============================================================
# IMAGE TRANSFORMS
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


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)


# ============================================================
# CLASS COUNTS
# ============================================================

class_counts = torch.zeros(
    NUM_CLASSES,
    dtype=torch.long,
)

for _, label in train_dataset:
    class_counts[label] += 1


print()
print("=" * 60)
print("CLASS DISTRIBUTION")
print("=" * 60)

for idx, count in enumerate(class_counts):

    class_name = train_dataset.CLASS_NAMES[idx]

    print(
        f"{idx}: {class_name:15s} "
        f"{count.item():4d}"
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

total_samples = class_counts.sum().item()

class_weights = total_samples / (
    NUM_CLASSES * class_counts.float()
)


print()
print("=" * 60)
print("CLASS WEIGHTS")
print("=" * 60)

for idx, weight in enumerate(class_weights):

    class_name = train_dataset.CLASS_NAMES[idx]

    print(
        f"{class_name:15s}: "
        f"{weight.item():.4f}"
    )


# ============================================================
# LOAD PRETRAINED RESNET-18
# ============================================================

weights = ResNet18_Weights.DEFAULT

model = resnet18(
    weights=weights
)


# ============================================================
# REPLACE CLASSIFIER
# ============================================================

input_features = model.fc.in_features

model.fc = nn.Linear(
    input_features,
    NUM_CLASSES,
)


model = model.to(device)


# ============================================================
# LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights.to(device)
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4,
)


# ============================================================
# SETUP SUMMARY
# ============================================================

print()
print("=" * 60)
print("MODEL SETUP")
print("=" * 60)

print(f"Model          : ResNet-18")
print(f"Input size     : {IMAGE_SIZE} × {IMAGE_SIZE}")
print(f"Classes        : {NUM_CLASSES}")
print(f"Batch size     : {BATCH_SIZE}")
print(f"Learning rate  : 0.0001")
print(f"Weight decay   : 0.0001")

print()

print("Classifier:")
print(model.fc)


# ============================================================
# TEST ONE BATCH
# ============================================================

images, labels = next(iter(train_loader))

images = images.to(device)
labels = labels.to(device)

outputs = model(images)

loss = criterion(
    outputs,
    labels,
)


print()
print("=" * 60)
print("TRAINING BATCH TEST")
print("=" * 60)

print(f"Input shape    : {images.shape}")
print(f"Output shape   : {outputs.shape}")
print(f"Loss           : {loss.item():.4f}")


# ============================================================
# VALIDATION
# ============================================================

assert outputs.shape == (
    BATCH_SIZE,
    NUM_CLASSES,
)

assert labels.shape == (
    BATCH_SIZE,
)

assert torch.isfinite(loss)


print()
print("=" * 60)
print("TRAINING SETUP TEST PASSED")
print("=" * 60)