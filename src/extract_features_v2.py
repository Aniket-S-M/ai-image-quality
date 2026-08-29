from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ==================================================
# PATHS
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FEATURES = (
    ROOT
    / "data"
    / "features"
    / "features.csv"
)

DATASET_DIR = (
    ROOT
    / "data"
    / "generated"
    / "dataset"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "features"
    / "features_v2.csv"
)


# ==================================================
# ORIGINAL FEATURES
# ==================================================

def calculate_sharpness(gray):
    return float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()
    )


def calculate_brightness(image):
    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    return float(
        hsv[:, :, 2].mean()
    )


def calculate_highlight_clipping(image):
    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    value = hsv[:, :, 2]

    return float(
        np.mean(value >= 250) * 100
    )


def calculate_contrast(gray):
    return float(
        gray.std()
    )


def calculate_saturation(image):
    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    return float(
        hsv[:, :, 1].mean()
    )


def calculate_edge_density(gray):

    edges = cv2.Canny(
        gray,
        100,
        200
    )

    return float(
        np.mean(edges > 0)
    )


def calculate_noise_estimate(gray):

    blurred = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    residual = (
        gray.astype(np.float32)
        - blurred.astype(np.float32)
    )

    return float(
        residual.std()
    )


# ==================================================
# NEW FEATURES
# ==================================================

def calculate_shadow_clipping(image):

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    value = hsv[:, :, 2]

    return float(
        np.mean(value <= 5) * 100
    )


def calculate_dark_pixel_ratio(gray):

    return float(
        np.mean(gray < 40) * 100
    )


def calculate_bright_pixel_ratio(gray):

    return float(
        np.mean(gray > 215) * 100
    )


def calculate_blockiness(gray):

    gray = gray.astype(
        np.float32
    )

    height, width = gray.shape

    vertical_scores = []
    horizontal_scores = []

    # ----------------------------------------------
    # Vertical 8x8 boundaries
    # ----------------------------------------------

    for x in range(
        8,
        width,
        8
    ):

        difference = np.abs(
            gray[:, x]
            - gray[:, x - 1]
        )

        vertical_scores.append(
            difference.mean()
        )

    # ----------------------------------------------
    # Horizontal 8x8 boundaries
    # ----------------------------------------------

    for y in range(
        8,
        height,
        8
    ):

        difference = np.abs(
            gray[y, :]
            - gray[y - 1, :]
        )

        horizontal_scores.append(
            difference.mean()
        )

    if vertical_scores:

        vertical_score = np.mean(
            vertical_scores
        )

    else:

        vertical_score = 0.0

    if horizontal_scores:

        horizontal_score = np.mean(
            horizontal_scores
        )

    else:

        horizontal_score = 0.0

    return float(
        (
            vertical_score
            + horizontal_score
        ) / 2
    )


# ==================================================
# EXTRACT ALL FEATURES
# ==================================================

def extract_features(image_path):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        raise ValueError(
            f"Could not read image: "
            f"{image_path}"
        )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return {

        # Existing features

        "sharpness":
            calculate_sharpness(
                gray
            ),

        "brightness":
            calculate_brightness(
                image
            ),

        "highlight_clipping":
            calculate_highlight_clipping(
                image
            ),

        "contrast":
            calculate_contrast(
                gray
            ),

        "saturation":
            calculate_saturation(
                image
            ),

        "edge_density":
            calculate_edge_density(
                gray
            ),

        "noise_estimate":
            calculate_noise_estimate(
                gray
            ),

        # New features

        "shadow_clipping":
            calculate_shadow_clipping(
                image
            ),

        "dark_pixel_ratio":
            calculate_dark_pixel_ratio(
                gray
            ),

        "bright_pixel_ratio":
            calculate_bright_pixel_ratio(
                gray
            ),

        "blockiness":
            calculate_blockiness(
                gray
            ),
    }


# ==================================================
# MAIN
# ==================================================

def main():

    print(
        "==================================="
    )

    print(
        "FEATURE EXTRACTION V2"
    )

    print(
        "==================================="
    )

    # ----------------------------------------------
    # Load existing metadata
    # ----------------------------------------------

    if not INPUT_FEATURES.exists():

        raise FileNotFoundError(
            f"Missing:\n{INPUT_FEATURES}"
        )

    metadata = pd.read_csv(
        INPUT_FEATURES
    )

    print(
        f"Input records: "
        f"{len(metadata)}"
    )

    records = []

    # ----------------------------------------------
    # Process every image
    # ----------------------------------------------

    for index, row in metadata.iterrows():

        filename = row[
            "filename"
        ]

        split = row[
            "split"
        ]

        image_path = (
            DATASET_DIR
            / split
            / filename
        )

        try:

            features = extract_features(
                image_path
            )

            record = row.to_dict()

            # Remove old feature values
            # so they can be replaced cleanly.

            for feature in [
                "sharpness",
                "brightness",
                "highlight_clipping",
                "contrast",
                "saturation",
                "edge_density",
                "noise_estimate",
            ]:

                record.pop(
                    feature,
                    None
                )

            record.update(
                features
            )

            records.append(
                record
            )

        except Exception as e:

            print(
                f"[skip] {filename}: {e}"
            )

        if (
            (index + 1) % 50 == 0
        ):

            print(
                f"Processed "
                f"{index + 1}/"
                f"{len(metadata)}"
            )

    # ----------------------------------------------
    # Create dataframe
    # ----------------------------------------------

    output_df = pd.DataFrame(
        records
    )

    # ----------------------------------------------
    # Save
    # ----------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ----------------------------------------------
    # Summary
    # ----------------------------------------------

    print(
        "\n==================================="
    )

    print(
        "FEATURE EXTRACTION V2 COMPLETE"
    )

    print(
        "==================================="
    )

    print(
        f"Input records : "
        f"{len(metadata)}"
    )

    print(
        f"Output records: "
        f"{len(output_df)}"
    )

    print(
        "Features      : 11"
    )

    print(
        f"Output file   : "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()