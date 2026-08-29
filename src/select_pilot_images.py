from pathlib import Path
import random
import shutil
import csv

# Project root
ROOT = Path(__file__).resolve().parents[1]

# COCO images
SOURCE_DIR = ROOT / "downloads" / "val2017"

# Our working clean-image dataset
DEST_DIR = ROOT / "data" / "raw" / "clean"

# Metadata
METADATA_DIR = ROOT / "data" / "metadata"
METADATA_FILE = METADATA_DIR / "pilot_sources.csv"

# Reproducibility
SEED = 42
NUM_IMAGES = 50


def main():
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    # Find all COCO JPG images
    images = sorted(SOURCE_DIR.glob("*.jpg"))

    if len(images) < NUM_IMAGES:
        raise RuntimeError(
            f"Only {len(images)} images found, "
            f"but {NUM_IMAGES} are required."
        )

    # Reproducible random selection
    rng = random.Random(SEED)
    selected_images = rng.sample(images, NUM_IMAGES)

    # Copy selected images
    rows = []

    for image_path in selected_images:
        destination = DEST_DIR / image_path.name

        shutil.copy2(image_path, destination)

        rows.append({
            "source_image_id": image_path.stem,
            "filename": image_path.name,
            "source_dataset": "COCO 2017 val2017"
        })

    # Save metadata
    with METADATA_FILE.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_image_id",
                "filename",
                "source_dataset"
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"Selected {len(selected_images)} images.")
    print(f"Copied to: {DEST_DIR}")
    print(f"Metadata:   {METADATA_FILE}")


if __name__ == "__main__":
    main()