from pathlib import Path
import csv

import cv2
import numpy as np

from degradation import (
    apply_blur,
    apply_underexposure,
    apply_overexposure,
    apply_gaussian_noise,
    apply_salt_pepper_noise,
    apply_jpeg_corruption,
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = ROOT / "data" / "raw" / "clean"
SPLIT_DIR = ROOT / "data" / "splits"

OUTPUT_DIR = ROOT / "data" / "generated" / "dataset"
METADATA_DIR = ROOT / "data" / "metadata"

OUTPUT_METADATA = (
    METADATA_DIR / "degradation_dataset.csv"
)


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

SEED = 42


# --------------------------------------------------
# Degradation configuration
# --------------------------------------------------

SEVERITIES = [
    "mild",
    "moderate",
    "severe",
]


# --------------------------------------------------
# Generate one source image
# --------------------------------------------------

def generate_samples(
    image_path,
    split_name,
):
    """
    Generate clean and degraded variants
    for one source image.
    """

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise RuntimeError(
            f"Could not read image: {image_path}"
        )

    source_id = image_path.stem

    split_output_dir = (
        OUTPUT_DIR / split_name
    )

    split_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    records = []

    # ----------------------------------------------
    # 1. CLEAN IMAGE
    # ----------------------------------------------

    clean_filename = (
        f"{source_id}_clean.jpg"
    )

    clean_output = (
        split_output_dir / clean_filename
    )

    cv2.imwrite(
        str(clean_output),
        image
    )

    records.append({
        "filename": clean_filename,
        "source_image_id": source_id,
        "split": split_name,
        "issue": "none",
        "severity": "none",
        "degradation": "clean"
    })

    # ----------------------------------------------
    # 2. BLUR
    # ----------------------------------------------

    for severity in SEVERITIES:

        degraded = apply_blur(
            image,
            severity
        )

        filename = (
            f"{source_id}_blur_{severity}.jpg"
        )

        output_path = (
            split_output_dir / filename
        )

        cv2.imwrite(
            str(output_path),
            degraded
        )

        records.append({
            "filename": filename,
            "source_image_id": source_id,
            "split": split_name,
            "issue": "blur",
            "severity": severity,
            "degradation": "gaussian_blur"
        })

    # ----------------------------------------------
    # 3. UNDEREXPOSURE
    # ----------------------------------------------

    for severity in SEVERITIES:

        degraded = apply_underexposure(
            image,
            severity
        )

        filename = (
            f"{source_id}_underexposure_{severity}.jpg"
        )

        output_path = (
            split_output_dir / filename
        )

        cv2.imwrite(
            str(output_path),
            degraded
        )

        records.append({
            "filename": filename,
            "source_image_id": source_id,
            "split": split_name,
            "issue": "underexposure",
            "severity": severity,
            "degradation": "brightness_reduction"
        })

    # ----------------------------------------------
    # 4. OVEREXPOSURE
    # ----------------------------------------------

    for severity in SEVERITIES:

        degraded = apply_overexposure(
            image,
            severity
        )

        filename = (
            f"{source_id}_overexposure_{severity}.jpg"
        )

        output_path = (
            split_output_dir / filename
        )

        cv2.imwrite(
            str(output_path),
            degraded
        )

        records.append({
            "filename": filename,
            "source_image_id": source_id,
            "split": split_name,
            "issue": "overexposure",
            "severity": severity,
            "degradation": "brightness_increase"
        })

    # ----------------------------------------------
    # 5. GAUSSIAN NOISE
    # ----------------------------------------------

    for severity in SEVERITIES:

        degraded = apply_gaussian_noise(
            image,
            severity
        )

        filename = (
            f"{source_id}_gaussian_noise_{severity}.jpg"
        )

        output_path = (
            split_output_dir / filename
        )

        cv2.imwrite(
            str(output_path),
            degraded
        )

        records.append({
            "filename": filename,
            "source_image_id": source_id,
            "split": split_name,
            "issue": "noise",
            "severity": severity,
            "degradation": "gaussian_noise"
        })

    # ----------------------------------------------
    # 6. SALT & PEPPER NOISE
    # ----------------------------------------------

    for severity in SEVERITIES:

        degraded = apply_salt_pepper_noise(
            image,
            severity
        )

        filename = (
            f"{source_id}_saltpepper_{severity}.jpg"
        )

        output_path = (
            split_output_dir / filename
        )

        cv2.imwrite(
            str(output_path),
            degraded
        )

        records.append({
            "filename": filename,
            "source_image_id": source_id,
            "split": split_name,
            "issue": "noise",
            "severity": severity,
            "degradation": "salt_pepper_noise"
        })

    # ----------------------------------------------
    # 7. JPEG CORRUPTION
    # ----------------------------------------------

    for severity in SEVERITIES:

        degraded = apply_jpeg_corruption(
            image,
            severity
        )

        filename = (
            f"{source_id}_corruption_{severity}.jpg"
        )

        output_path = (
            split_output_dir / filename
        )

        cv2.imwrite(
            str(output_path),
            degraded
        )

        records.append({
            "filename": filename,
            "source_image_id": source_id,
            "split": split_name,
            "issue": "corruption",
            "severity": severity,
            "degradation": "jpeg_compression"
        })

    return records


# --------------------------------------------------
# Read source split
# --------------------------------------------------

def read_split(split_name):

    split_file = (
        SPLIT_DIR /
        f"{split_name}.csv"
    )

    if not split_file.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_file}"
        )

    with split_file.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        return list(reader)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    # ----------------------------------------------
    # Reproducible random state
    # ----------------------------------------------

    np.random.seed(SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    all_records = []

    # ----------------------------------------------
    # Generate each split separately
    # ----------------------------------------------

    for split_name in [
        "train",
        "val",
        "test"
    ]:

        source_images = read_split(
            split_name
        )

        print(
            f"\nGenerating {split_name} dataset..."
        )

        print(
            f"Source images: "
            f"{len(source_images)}"
        )

        for index, row in enumerate(
            source_images,
            start=1
        ):

            image_path = (
                IMAGE_DIR /
                row["filename"]
            )

            records = generate_samples(
                image_path,
                split_name
            )

            all_records.extend(
                records
            )

            print(
                f"[{index}/{len(source_images)}] "
                f"{row['filename']}"
            )

    # ----------------------------------------------
    # Save metadata
    # ----------------------------------------------

    fieldnames = [
        "filename",
        "source_image_id",
        "split",
        "issue",
        "severity",
        "degradation"
    ]

    with OUTPUT_METADATA.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(all_records)

    # ----------------------------------------------
    # Summary
    # ----------------------------------------------

    print("\n===================================")
    print("DATASET GENERATION COMPLETE")
    print("===================================")

    print(
        f"Total generated samples: "
        f"{len(all_records)}"
    )

    for split_name in [
        "train",
        "val",
        "test"
    ]:

        count = sum(
            row["split"] == split_name
            for row in all_records
        )

        print(
            f"{split_name:5s}: {count}"
        )

    print(
        f"\nMetadata: {OUTPUT_METADATA}"
    )


if __name__ == "__main__":
    main()