from pathlib import Path
from collections import defaultdict

import torch
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18

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

OUTPUT_DIR = Path(
    "data/evaluation/resnet_misclassified"
)

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 6

# Maximum number of images per confusion pair
MAX_IMAGES_PER_PAIR = 20


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("RESNET-18 VISUAL ERROR ANALYSIS")
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

transform = transforms.Compose([
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
# DATASET
# ============================================================

val_dataset = ImageQualityDataset(
    DATASET_ROOT / "val",
    transform=transform,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# ============================================================
# MODEL
# ============================================================

model = resnet18(
    weights=None
)

model.fc = torch.nn.Linear(
    model.fc.in_features,
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


# ============================================================
# FIND MISCLASSIFICATIONS
# ============================================================

errors = defaultdict(list)

total = 0
wrong = 0

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)

        outputs = model(images)

        predictions = outputs.argmax(
            dim=1
        )

        for i in range(len(labels)):

            actual = labels[i].item()
            predicted = predictions[i].item()

            total += 1

            if actual != predicted:

                wrong += 1

                image_path = (
                    val_dataset.image_paths[
                        total - 1
                    ]
                )

                pair = (
                    val_dataset.CLASS_NAMES[actual],
                    val_dataset.CLASS_NAMES[predicted],
                )

                if len(errors[pair]) < MAX_IMAGES_PER_PAIR:

                    errors[pair].append(
                        image_path
                    )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("ERROR SUMMARY")
print("=" * 60)

print(f"Validation images : {total}")
print(f"Misclassified     : {wrong}")
print(
    f"Error rate        : "
    f"{wrong / total:.4f}"
)

print()
print("Confusion pairs:")

for pair, paths in sorted(
    errors.items(),
    key=lambda item: len(item[1]),
    reverse=True,
):

    print(
        f"{pair[0]:15s} → "
        f"{pair[1]:15s} : "
        f"{len(paths)} images"
    )


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CREATE CONTACT SHEETS
# ============================================================

for (actual, predicted), paths in errors.items():

    if not paths:
        continue

    # --------------------------------------------------------
    # Sheet dimensions
    # --------------------------------------------------------

    columns = 5

    rows = (
        len(paths) + columns - 1
    ) // columns

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(15, 3 * rows),
    )

    # Make axes iterable even for one row
    if rows == 1:
        axes = [axes]

    axes_flat = []

    for row in axes:
        if hasattr(row, "__iter__"):
            axes_flat.extend(row)
        else:
            axes_flat.append(row)


    # --------------------------------------------------------
    # Plot images
    # --------------------------------------------------------

    for idx, image_path in enumerate(paths):

        image = Image.open(
            image_path
        ).convert("RGB")

        ax = axes_flat[idx]

        ax.imshow(image)

        ax.set_title(
            image_path.name,
            fontsize=7,
        )

        ax.axis("off")


    # Hide unused axes
    for idx in range(
        len(paths),
        len(axes_flat),
    ):

        axes_flat[idx].axis("off")


    fig.suptitle(
        f"Actual: {actual}  →  Predicted: {predicted}",
        fontsize=14,
    )

    fig.tight_layout()


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    filename = (
        f"{actual}_to_{predicted}.png"
    )

    output_path = (
        OUTPUT_DIR / filename
    )

    fig.savefig(
        output_path,
        dpi=150,
    )

    plt.close(fig)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 60)
print("VISUAL ERROR ANALYSIS COMPLETE")
print("=" * 60)

print(
    f"Contact sheets saved to:"
)

print(OUTPUT_DIR)