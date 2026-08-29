from pathlib import Path

import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
)

from image_quality_dataset import ImageQualityDataset


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path(
    "data/generated/dataset_expanded"
)

MODEL_PATH = Path(
    "best_resnet18_finetuned.pth"
)

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 6


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("RESNET-18 VALIDATION DIAGNOSTICS")
print("=" * 60)

print(f"Device : {device}")


# ============================================================
# TRANSFORM
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
# VALIDATION DATASET
# ============================================================

val_dataset = ImageQualityDataset(
    DATASET_ROOT / "val",
    transform=eval_transform,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# ============================================================
# LOAD MODEL
# ============================================================

model = resnet18(
    weights=None
)

input_features = model.fc.in_features

model.fc = torch.nn.Linear(
    input_features,
    NUM_CLASSES,
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
    )
)

model = model.to(device)
model.eval()


print()
print(f"Validation images : {len(val_dataset)}")
print(f"Model loaded      : {MODEL_PATH}")


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)

        outputs = model(images)

        predictions = outputs.argmax(
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().tolist()
        )

        all_labels.extend(
            labels.tolist()
        )


# ============================================================
# OVERALL METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions,
)

macro_f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro",
)


print()
print("=" * 60)
print("OVERALL VALIDATION PERFORMANCE")
print("=" * 60)

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Macro-F1 : {macro_f1:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

class_names = (
    val_dataset.CLASS_NAMES
)

print()
print("=" * 60)
print("PER-CLASS PERFORMANCE")
print("=" * 60)

print(
    classification_report(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES)),
)


print()
print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print("Rows = Actual")
print("Columns = Predicted")
print()

print(cm)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

output_dir = Path(
    "data/evaluation"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


plt.figure(
    figsize=(9, 7)
)

plt.imshow(cm)

plt.title(
    "ResNet-18 Validation Confusion Matrix"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)

plt.xticks(
    range(NUM_CLASSES),
    class_names,
    rotation=45,
    ha="right",
)

plt.yticks(
    range(NUM_CLASSES),
    class_names,
)

# Write numbers inside cells
for i in range(NUM_CLASSES):

    for j in range(NUM_CLASSES):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
        )


plt.tight_layout()

output_path = (
    output_dir
    / "resnet18_validation_confusion_matrix.png"
)

plt.savefig(
    output_path,
    dpi=200,
)

plt.close()


print()
print(
    f"Confusion matrix saved to:"
)

print(output_path)


print()
print("=" * 60)
print("CNN VALIDATION DIAGNOSTICS COMPLETE")
print("=" * 60)