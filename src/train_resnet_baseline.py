from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

from sklearn.metrics import accuracy_score, f1_score

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

EPOCHS = 5
LEARNING_RATE = 1e-3

NUM_WORKERS = 0


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("RESNET-18 BASELINE TRAINING")
print("=" * 60)

print(f"Device : {device}")


# ============================================================
# NORMALIZATION
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
# TRANSFORMS
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
# CLASS WEIGHTS
# ============================================================

class_counts = torch.zeros(
    NUM_CLASSES,
    dtype=torch.long,
)

for _, label in train_dataset:
    class_counts[label] += 1


total_samples = class_counts.sum().item()

class_weights = total_samples / (
    NUM_CLASSES * class_counts.float()
)

class_weights = class_weights.to(device)


# ============================================================
# MODEL
# ============================================================

weights = ResNet18_Weights.DEFAULT

model = resnet18(
    weights=weights
)


# Freeze pretrained layers
for parameter in model.parameters():
    parameter.requires_grad = False


# Replace classifier
input_features = model.fc.in_features

model.fc = nn.Linear(
    input_features,
    NUM_CLASSES,
)


model = model.to(device)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.fc.parameters(),
    lr=LEARNING_RATE,
)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0

    all_predictions = []
    all_labels = []

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels,
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().tolist()
        )

        all_labels.extend(
            labels.cpu().tolist()
        )

    epoch_loss = (
        running_loss /
        len(train_dataset)
    )

    epoch_accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    epoch_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
    )

    return (
        epoch_loss,
        epoch_accuracy,
        epoch_f1,
    )


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def evaluate():

    model.eval()

    running_loss = 0.0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

            running_loss += (
                loss.item() * images.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                labels.cpu().tolist()
            )

    epoch_loss = (
        running_loss /
        len(val_dataset)
    )

    epoch_accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    epoch_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
    )

    return (
        epoch_loss,
        epoch_accuracy,
        epoch_f1,
    )


# ============================================================
# TRAINING LOOP
# ============================================================

best_val_f1 = 0.0

print()
print("=" * 60)
print("TRAINING")
print("=" * 60)

for epoch in range(1, EPOCHS + 1):

    train_loss, train_acc, train_f1 = (
        train_one_epoch()
    )

    val_loss, val_acc, val_f1 = evaluate()

    print()
    print(
        f"Epoch {epoch}/{EPOCHS}"
    )

    print(
        f"Train Loss    : {train_loss:.4f}"
    )

    print(
        f"Train Accuracy: {train_acc:.4f}"
    )

    print(
        f"Train Macro-F1: {train_f1:.4f}"
    )

    print(
        f"Val Loss      : {val_loss:.4f}"
    )

    print(
        f"Val Accuracy  : {val_acc:.4f}"
    )

    print(
        f"Val Macro-F1  : {val_f1:.4f}"
    )

    # Save best model based on validation Macro-F1
    if val_f1 > best_val_f1:

        best_val_f1 = val_f1

        torch.save(
            model.state_dict(),
            "best_resnet18_baseline.pth",
        )

        print(
            "✓ Best model saved"
        )


print()
print("=" * 60)
print("BASELINE TRAINING COMPLETE")
print("=" * 60)

print(
    f"Best validation Macro-F1: "
    f"{best_val_f1:.4f}"
)