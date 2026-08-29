from pathlib import Path
import random
import shutil

import cv2
import numpy as np
import pandas as pd


# ==================================================
# PATHS
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

SOURCE_DIR = (
    ROOT
    / "data"
    / "raw"
    / "clean_expanded"
)

SOURCE_METADATA = (
    ROOT
    / "data"
    / "metadata"
    / "expanded_sources.csv"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "generated"
    / "dataset_expanded"
)

METADATA_DIR = (
    ROOT
    / "data"
    / "metadata"
)

OUTPUT_METADATA = (
    METADATA_DIR
    / "expanded_dataset_metadata.csv"
)


# ==================================================
# SETTINGS
# ==================================================

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ==================================================
# DEGRADATION SETTINGS
# ==================================================

BLUR_SIGMAS = {
    "mild": 1.0,
    "moderate": 2.5,
    "severe": 5.0,
}

NOISE_SIGMAS = {
    "mild": 10,
    "moderate": 25,
    "severe": 50,
}

JPEG_QUALITIES = {
    "mild": 50,
    "moderate": 25,
    "severe": 10,
}

UNDEREXPOSURE_FACTORS = {
    "mild": 0.70,
    "moderate": 0.45,
    "severe": 0.25,
}

OVEREXPOSURE_FACTORS = {
    "mild": 1.25,
    "moderate": 1.60,
    "severe": 2.00,
}


# ==================================================
# IMAGE FUNCTIONS
# ==================================================

def gaussian_blur(image, sigma):

    return cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
    )


def gaussian_noise(image, sigma):

    noise = np.random.normal(
        0,
        sigma,
        image.shape,
    )

    noisy = (
        image.astype(np.float32)
        + noise
    )

    return np.clip(
        noisy,
        0,
        255,
    ).astype(np.uint8)


def salt_pepper_noise(
    image,
    amount,
):

    result = image.copy()

    total_pixels = (
        image.shape[0]
        * image.shape[1]
    )

    num_pixels = int(
        total_pixels * amount
    )

    # Salt
    coords = (
        np.random.randint(
            0,
            image.shape[0],
            num_pixels,
        ),
        np.random.randint(
            0,
            image.shape[1],
            num_pixels,
        ),
    )

    result[coords] = 255

    # Pepper
    coords = (
        np.random.randint(
            0,
            image.shape[0],
            num_pixels,
        ),
        np.random.randint(
            0,
            image.shape[1],
            num_pixels,
        ),
    )

    result[coords] = 0

    return result


def jpeg_corruption(
    image,
    quality,
):

    encode_params = [
        cv2.IMWRITE_JPEG_QUALITY,
        quality,
    ]

    success, encoded = cv2.imencode(
        ".jpg",
        image,
        encode_params,
    )

    if not success:

        raise RuntimeError(
            "JPEG encoding failed."
        )

    decoded = cv2.imdecode(
        encoded,
        cv2.IMREAD_COLOR,
    )

    if decoded is None:

        raise RuntimeError(
            "JPEG decoding failed."
        )

    return decoded


def underexposure(
    image,
    factor,
):

    result = (
        image.astype(np.float32)
        * factor
    )

    return np.clip(
        result,
        0,
        255,
    ).astype(np.uint8)


def overexposure(
    image,
    factor,
):

    result = (
        image.astype(np.float32)
        * factor
    )

    return np.clip(
        result,
        0,
        255,
    ).astype(np.uint8)


# ==================================================
# SPLIT ASSIGNMENT
# ==================================================

def create_source_splits(
    source_ids,
):

    source_ids = list(source_ids)

    random.seed(
        RANDOM_SEED
    )

    random.shuffle(
        source_ids
    )

    total = len(source_ids)

    train_count = int(
        total * TRAIN_RATIO
    )

    val_count = int(
        total * VAL_RATIO
    )

    train_ids = source_ids[
        :train_count
    ]

    val_ids = source_ids[
        train_count:
        train_count + val_count
    ]

    test_ids = source_ids[
        train_count + val_count:
    ]

    splits = {}

    for source_id in train_ids:
        splits[source_id] = "train"

    for source_id in val_ids:
        splits[source_id] = "val"

    for source_id in test_ids:
        splits[source_id] = "test"

    return splits


# ==================================================
# MAIN
# ==================================================

def main():

    print("===================================")
    print("EXPANDED DATASET GENERATION")
    print("===================================")

    # ----------------------------------------------
    # Check inputs
    # ----------------------------------------------

    if not SOURCE_DIR.exists():

        raise FileNotFoundError(
            f"Source directory not found:\n"
            f"{SOURCE_DIR}"
        )

    if not SOURCE_METADATA.exists():

        raise FileNotFoundError(
            f"Source metadata not found:\n"
            f"{SOURCE_METADATA}"
        )

    # ----------------------------------------------
    # Load metadata
    # ----------------------------------------------

    sources = pd.read_csv(
        SOURCE_METADATA,
        dtype={
            "source_image_id": str
        },
    )

    sources[
        "source_image_id"
    ] = (
        sources[
            "source_image_id"
        ]
        .astype(str)
        .str.strip()
        .str.zfill(12)
    )

    print(
        f"Source records: {len(sources)}"
    )

    if len(sources) != 500:

        raise ValueError(
            f"Expected 500 sources, "
            f"found {len(sources)}."
        )

    if (
        sources[
            "source_image_id"
        ]
        .duplicated()
        .any()
    ):

        raise ValueError(
            "Duplicate source IDs found."
        )

    # ----------------------------------------------
    # Create source-level split
    # ----------------------------------------------

    source_ids = (
        sources[
            "source_image_id"
        ]
        .tolist()
    )

    split_map = create_source_splits(
        source_ids
    )

    print(
        "\nSource split:"
    )

    print(
        f"train: "
        f"{list(split_map.values()).count('train')}"
    )

    print(
        f"val  : "
        f"{list(split_map.values()).count('val')}"
    )

    print(
        f"test : "
        f"{list(split_map.values()).count('test')}"
    )

    # ----------------------------------------------
    # Create output directories
    # ----------------------------------------------

    for split in [
        "train",
        "val",
        "test",
    ]:

        split_dir = (
            OUTPUT_DIR
            / split
        )

        split_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ----------------------------------------------
    # Generate dataset
    # ----------------------------------------------

    records = []

    total_sources = len(
        sources
    )

    for index, row in sources.iterrows():

        source_id = (
            row[
                "source_image_id"
            ]
        )

        split = split_map[
            source_id
        ]

        source_path = (
            SOURCE_DIR
            / f"{source_id}.jpg"
        )

        if not source_path.exists():

            raise FileNotFoundError(
                f"Source image missing:\n"
                f"{source_path}"
            )

        image = cv2.imread(
            str(source_path)
        )

        if image is None:

            raise RuntimeError(
                f"Could not read image:\n"
                f"{source_path}"
            )

        output_split_dir = (
            OUTPUT_DIR
            / split
        )

        # ------------------------------------------
        # Clean
        # ------------------------------------------

        clean_filename = (
            f"{source_id}_clean.jpg"
        )

        clean_path = (
            output_split_dir
            / clean_filename
        )

        cv2.imwrite(
            str(clean_path),
            image,
        )

        records.append(
            {
                "filename":
                    clean_filename,
                "source_image_id":
                    source_id,
                "split":
                    split,
                "issue":
                    "none",
                "severity":
                    "none",
                "degradation":
                    "clean",
            }
        )

        # ------------------------------------------
        # Blur
        # ------------------------------------------

        for severity, sigma in (
            BLUR_SIGMAS.items()
        ):

            degraded = gaussian_blur(
                image,
                sigma,
            )

            filename = (
                f"{source_id}_"
                f"blur_{severity}.jpg"
            )

            path = (
                output_split_dir
                / filename
            )

            cv2.imwrite(
                str(path),
                degraded,
            )

            records.append(
                {
                    "filename":
                        filename,
                    "source_image_id":
                        source_id,
                    "split":
                        split,
                    "issue":
                        "blur",
                    "severity":
                        severity,
                    "degradation":
                        "gaussian_blur",
                }
            )

        # ------------------------------------------
        # Gaussian noise
        # ------------------------------------------

        for severity, sigma in (
            NOISE_SIGMAS.items()
        ):

            degraded = gaussian_noise(
                image,
                sigma,
            )

            filename = (
                f"{source_id}_"
                f"noise_{severity}.jpg"
            )

            path = (
                output_split_dir
                / filename
            )

            cv2.imwrite(
                str(path),
                degraded,
            )

            records.append(
                {
                    "filename":
                        filename,
                    "source_image_id":
                        source_id,
                    "split":
                        split,
                    "issue":
                        "noise",
                    "severity":
                        severity,
                    "degradation":
                        "gaussian_noise",
                }
            )

        # ------------------------------------------
        # Salt and pepper noise
        # ------------------------------------------

        sp_amounts = {
            "mild": 0.005,
            "moderate": 0.015,
            "severe": 0.030,
        }

        for severity, amount in (
            sp_amounts.items()
        ):

            degraded = salt_pepper_noise(
                image,
                amount,
            )

            filename = (
                f"{source_id}_"
                f"noise_sp_{severity}.jpg"
            )

            path = (
                output_split_dir
                / filename
            )

            cv2.imwrite(
                str(path),
                degraded,
            )

            records.append(
                {
                    "filename":
                        filename,
                    "source_image_id":
                        source_id,
                    "split":
                        split,
                    "issue":
                        "noise",
                    "severity":
                        severity,
                    "degradation":
                        "salt_pepper_noise",
                }
            )

        # ------------------------------------------
        # JPEG corruption
        # ------------------------------------------

        for severity, quality in (
            JPEG_QUALITIES.items()
        ):

            degraded = jpeg_corruption(
                image,
                quality,
            )

            filename = (
                f"{source_id}_"
                f"corruption_{severity}.jpg"
            )

            path = (
                output_split_dir
                / filename
            )

            cv2.imwrite(
                str(path),
                degraded,
            )

            records.append(
                {
                    "filename":
                        filename,
                    "source_image_id":
                        source_id,
                    "split":
                        split,
                    "issue":
                        "corruption",
                    "severity":
                        severity,
                    "degradation":
                        "jpeg_corruption",
                }
            )

        # ------------------------------------------
        # Underexposure
        # ------------------------------------------

        for severity, factor in (
            UNDEREXPOSURE_FACTORS.items()
        ):

            degraded = underexposure(
                image,
                factor,
            )

            filename = (
                f"{source_id}_"
                f"underexposure_{severity}.jpg"
            )

            path = (
                output_split_dir
                / filename
            )

            cv2.imwrite(
                str(path),
                degraded,
            )

            records.append(
                {
                    "filename":
                        filename,
                    "source_image_id":
                        source_id,
                    "split":
                        split,
                    "issue":
                        "underexposure",
                    "severity":
                        severity,
                    "degradation":
                        "underexposure",
                }
            )

        # ------------------------------------------
        # Overexposure
        # ------------------------------------------

        for severity, factor in (
            OVEREXPOSURE_FACTORS.items()
        ):

            degraded = overexposure(
                image,
                factor,
            )

            filename = (
                f"{source_id}_"
                f"overexposure_{severity}.jpg"
            )

            path = (
                output_split_dir
                / filename
            )

            cv2.imwrite(
                str(path),
                degraded,
            )

            records.append(
                {
                    "filename":
                        filename,
                    "source_image_id":
                        source_id,
                    "split":
                        split,
                    "issue":
                        "overexposure",
                    "severity":
                        severity,
                    "degradation":
                        "overexposure",
                }
            )

        # ------------------------------------------
        # Progress
        # ------------------------------------------

        completed = index + 1

        if (
            completed % 25 == 0
            or completed == total_sources
        ):

            print(
                f"Processed "
                f"{completed}/"
                f"{total_sources}"
            )

    # ----------------------------------------------
    # Save metadata
    # ----------------------------------------------

    metadata_df = pd.DataFrame(
        records
    )

    metadata_df.to_csv(
        OUTPUT_METADATA,
        index=False
    )

    # ----------------------------------------------
    # Dataset checks
    # ----------------------------------------------

    expected_samples = (
        total_sources * 19
    )

    actual_samples = len(
        metadata_df
    )

    print(
        "\n==================================="
    )

    print(
        "DATASET GENERATION RESULTS"
    )

    print(
        "==================================="
    )

    print(
        f"Source images      : "
        f"{total_sources}"
    )

    print(
        f"Expected samples   : "
        f"{expected_samples}"
    )

    print(
        f"Generated samples  : "
        f"{actual_samples}"
    )

    print(
        "\nSamples by split:"
    )

    print(
        metadata_df[
            "split"
        ]
        .value_counts()
        .sort_index()
    )

    print(
        "\nSamples by issue:"
    )

    print(
        metadata_df[
            "issue"
        ]
        .value_counts()
        .sort_index()
    )

    print(
        "\nSamples by severity:"
    )

    print(
        metadata_df[
            "severity"
        ]
        .value_counts()
        .sort_index()
    )

    # ----------------------------------------------
    # Source leakage check
    # ----------------------------------------------

    leakage = 0

    for source_id, group in (
        metadata_df
        .groupby(
            "source_image_id"
        )
    ):

        unique_splits = (
            group[
                "split"
            ]
            .nunique()
        )

        if unique_splits > 1:

            leakage += 1

    print(
        f"\nSource-image leakage: "
        f"{leakage}"
    )

    # ----------------------------------------------
    # Final validation
    # ----------------------------------------------

    if actual_samples != expected_samples:

        raise RuntimeError(
            "Generated sample count "
            "does not match expected count."
        )

    if leakage != 0:

        raise RuntimeError(
            "Source-image leakage detected."
        )

    if (
        metadata_df[
            "filename"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate filenames detected."
        )

    print(
        "\n==================================="
    )

    print(
        "EXPANDED DATASET GENERATION "
        "COMPLETE"
    )

    print(
        "==================================="
    )

    print(
        f"Total generated samples: "
        f"{actual_samples}"
    )

    print(
        f"Metadata:"
        f"\n{OUTPUT_METADATA}"
    )

    print(
        f"Dataset:"
        f"\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()