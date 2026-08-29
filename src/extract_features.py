from pathlib import Path
import csv

import cv2

from features import (
    calculate_sharpness,
    calculate_brightness,
    calculate_highlight_clipping,
    calculate_contrast,
    calculate_saturation,
    calculate_edge_density,
    calculate_noise_estimate,
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    ROOT
    / "data"
    / "generated"
    / "dataset"
)

METADATA_FILE = (
    ROOT
    / "data"
    / "metadata"
    / "degradation_dataset.csv"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "features"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "features.csv"
)


# --------------------------------------------------
# Feature extraction
# --------------------------------------------------

def extract_features(image):
    """
    Extract all image-quality features
    from one image.
    """

    return {
        "sharpness": calculate_sharpness(image),
        "brightness": calculate_brightness(image),
        "highlight_clipping": calculate_highlight_clipping(image),
        "contrast": calculate_contrast(image),
        "saturation": calculate_saturation(image),
        "edge_density": calculate_edge_density(image),
        "noise_estimate": calculate_noise_estimate(image),
    }


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ----------------------------------------------
    # Read degradation metadata
    # ----------------------------------------------

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata not found: {METADATA_FILE}"
        )

    with METADATA_FILE.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        records = list(
            csv.DictReader(file)
        )

    print(
        f"Metadata records: {len(records)}"
    )

    # ----------------------------------------------
    # Feature extraction
    # ----------------------------------------------

    feature_records = []

    for index, record in enumerate(
        records,
        start=1
    ):

        split = record["split"]
        filename = record["filename"]

        image_path = (
            DATASET_DIR
            / split
            / filename
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            print(
                f"[SKIP] Could not read: "
                f"{image_path}"
            )
            continue

        features = extract_features(
            image
        )

        # Combine metadata + features
        feature_record = {
            **record,
            **features
        }

        feature_records.append(
            feature_record
        )

        if index % 50 == 0:
            print(
                f"Processed "
                f"{index}/{len(records)}"
            )

    # ----------------------------------------------
    # Save feature dataset
    # ----------------------------------------------

    fieldnames = [
        "filename",
        "source_image_id",
        "split",
        "issue",
        "severity",
        "degradation",
        "sharpness",
        "brightness",
        "highlight_clipping",
        "contrast",
        "saturation",
        "edge_density",
        "noise_estimate",
    ]

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            feature_records
        )

    # ----------------------------------------------
    # Summary
    # ----------------------------------------------

    print("\n===================================")
    print("FEATURE EXTRACTION COMPLETE")
    print("===================================")

    print(
        f"Input records : {len(records)}"
    )

    print(
        f"Output records: "
        f"{len(feature_records)}"
    )

    print(
        f"Features      : 7"
    )

    print(
        f"Output file   : {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()