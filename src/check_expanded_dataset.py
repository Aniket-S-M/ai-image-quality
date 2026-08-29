from pathlib import Path

import pandas as pd
from PIL import Image


# ==================================================
# PATHS
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    ROOT
    / "data"
    / "generated"
    / "dataset_expanded"
)

METADATA_FILE = (
    ROOT
    / "data"
    / "metadata"
    / "expanded_dataset_metadata.csv"
)


# ==================================================
# EXPECTED VALUES
# ==================================================

EXPECTED_TOTAL = 9500
EXPECTED_SOURCES = 500

EXPECTED_SPLITS = {
    "train": 6650,
    "val": 1425,
    "test": 1425,
}

EXPECTED_ISSUES = {
    "blur": 1500,
    "corruption": 1500,
    "noise": 3000,
    "none": 500,
    "overexposure": 1500,
    "underexposure": 1500,
}

EXPECTED_SEVERITIES = {
    "mild": 3000,
    "moderate": 3000,
    "severe": 3000,
    "none": 500,
}


# ==================================================
# MAIN
# ==================================================

def main():

    print("===================================")
    print("EXPANDED DATASET SANITY CHECK")
    print("===================================")

    # ----------------------------------------------
    # Check paths
    # ----------------------------------------------

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory not found:\n"
            f"{DATASET_DIR}"
        )

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata file not found:\n"
            f"{METADATA_FILE}"
        )

    # ----------------------------------------------
    # Load metadata
    # ----------------------------------------------

    df = pd.read_csv(
        METADATA_FILE,
        dtype={
            "source_image_id": str
        }
    )

    df["source_image_id"] = (
        df["source_image_id"]
        .astype(str)
        .str.strip()
        .str.zfill(12)
    )

    print(
        f"Metadata records: {len(df)}"
    )

    # ----------------------------------------------
    # Required columns
    # ----------------------------------------------

    required_columns = {
        "filename",
        "source_image_id",
        "split",
        "issue",
        "severity",
        "degradation",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise RuntimeError(
            f"Missing metadata columns: "
            f"{missing_columns}"
        )

    # ----------------------------------------------
    # Find actual image files
    # ----------------------------------------------

    image_files = list(
        DATASET_DIR.rglob("*.jpg")
    )

    print(
        f"Image files: {len(image_files)}"
    )

    # ----------------------------------------------
    # Missing files
    # ----------------------------------------------

    metadata_filenames = set(
        df["filename"]
    )

    actual_filenames = {
        path.name
        for path in image_files
    }

    missing_files = (
        metadata_filenames
        - actual_filenames
    )

    unexpected_files = (
        actual_filenames
        - metadata_filenames
    )

    print(
        f"Missing files: "
        f"{len(missing_files)}"
    )

    print(
        f"Unexpected files: "
        f"{len(unexpected_files)}"
    )

    # ----------------------------------------------
    # Duplicate filenames
    # ----------------------------------------------

    duplicate_filenames = (
        df["filename"]
        .duplicated()
        .sum()
    )

    print(
        f"Duplicate filenames: "
        f"{duplicate_filenames}"
    )

    # ----------------------------------------------
    # Duplicate source IDs
    #
    # NOTE:
    # A source ID SHOULD occur 19 times.
    # We therefore check duplicate source-image
    # pairs differently from duplicate filenames.
    # ----------------------------------------------

    source_counts = (
        df["source_image_id"]
        .value_counts()
    )

    bad_source_counts = (
        source_counts[
            source_counts != 19
        ]
    )

    print(
        f"Unique source images: "
        f"{df['source_image_id'].nunique()}"
    )

    print(
        f"Sources with != 19 variants: "
        f"{len(bad_source_counts)}"
    )

    # ----------------------------------------------
    # Split distribution
    # ----------------------------------------------

    print(
        "\nSamples by split:"
    )

    split_counts = (
        df["split"]
        .value_counts()
        .sort_index()
    )

    print(split_counts)

    # ----------------------------------------------
    # Issue distribution
    # ----------------------------------------------

    print(
        "\nSamples by issue:"
    )

    issue_counts = (
        df["issue"]
        .value_counts()
        .sort_index()
    )

    print(issue_counts)

    # ----------------------------------------------
    # Severity distribution
    # ----------------------------------------------

    print(
        "\nSamples by severity:"
    )

    severity_counts = (
        df["severity"]
        .value_counts()
        .sort_index()
    )

    print(severity_counts)

    # ----------------------------------------------
    # Source-level leakage
    # ----------------------------------------------

    leakage = 0

    leakage_sources = []

    for source_id, group in (
        df.groupby(
            "source_image_id"
        )
    ):

        unique_splits = (
            group["split"]
            .nunique()
        )

        if unique_splits != 1:

            leakage += 1

            leakage_sources.append(
                source_id
            )

    print(
        f"\nSource-image leakage: "
        f"{leakage}"
    )

    if leakage_sources:

        print(
            "Leaking source IDs:"
        )

        for source_id in (
            leakage_sources[:20]
        ):

            print(
                f"  {source_id}"
            )

    # ----------------------------------------------
    # Image readability check
    # ----------------------------------------------

    print(
        "\nChecking image readability..."
    )

    unreadable = []
    zero_dimension = []
    image_count = 0

    for path in image_files:

        try:

            with Image.open(path) as image:

                image.verify()

            with Image.open(path) as image:

                width, height = image.size

                if width <= 0 or height <= 0:

                    zero_dimension.append(
                        path.name
                    )

            image_count += 1

        except Exception:

            unreadable.append(
                path.name
            )

        if image_count % 1000 == 0:

            print(
                f"Checked "
                f"{image_count}/"
                f"{len(image_files)}"
            )

    print(
        f"Unreadable images: "
        f"{len(unreadable)}"
    )

    print(
        f"Invalid dimensions: "
        f"{len(zero_dimension)}"
    )

    # ----------------------------------------------
    # Verify metadata/file split agreement
    # ----------------------------------------------

    split_path_errors = 0

    for _, row in df.iterrows():

        filename = row["filename"]
        expected_split = row["split"]

        expected_path = (
            DATASET_DIR
            / expected_split
            / filename
        )

        if not expected_path.exists():

            split_path_errors += 1

    print(
        f"Split path errors: "
        f"{split_path_errors}"
    )

    # ----------------------------------------------
    # Expected count checks
    # ----------------------------------------------

    print(
        "\n==================================="
    )

    print(
        "EXPECTED VS ACTUAL"
    )

    print(
        "==================================="
    )

    total_ok = (
        len(df)
        == EXPECTED_TOTAL
    )

    sources_ok = (
        df["source_image_id"]
        .nunique()
        == EXPECTED_SOURCES
    )

    splits_ok = all(
        split_counts.get(
            split,
            0
        )
        == expected
        for split, expected
        in EXPECTED_SPLITS.items()
    )

    issues_ok = all(
        issue_counts.get(
            issue,
            0
        )
        == expected
        for issue, expected
        in EXPECTED_ISSUES.items()
    )

    severities_ok = all(
        severity_counts.get(
            severity,
            0
        )
        == expected
        for severity, expected
        in EXPECTED_SEVERITIES.items()
    )

    # ----------------------------------------------
    # Final result
    # ----------------------------------------------

    passed = all(
        [
            total_ok,
            len(image_files)
            == EXPECTED_TOTAL,
            sources_ok,
            splits_ok,
            issues_ok,
            severities_ok,
            len(missing_files) == 0,
            len(unexpected_files) == 0,
            duplicate_filenames == 0,
            len(bad_source_counts) == 0,
            leakage == 0,
            len(unreadable) == 0,
            len(zero_dimension) == 0,
            split_path_errors == 0,
        ]
    )

    print(
        "\n==================================="
    )

    if passed:

        print(
            "✅ EXPANDED DATASET SANITY "
            "CHECK PASSED"
        )

    else:

        print(
            "❌ EXPANDED DATASET SANITY "
            "CHECK FAILED"
        )

        # ------------------------------------------
        # Detailed failures
        # ------------------------------------------

        if not total_ok:
            print(
                f"Expected metadata records: "
                f"{EXPECTED_TOTAL}"
            )

        if len(image_files) != EXPECTED_TOTAL:
            print(
                f"Expected image files: "
                f"{EXPECTED_TOTAL}"
            )

        if not sources_ok:
            print(
                f"Expected source images: "
                f"{EXPECTED_SOURCES}"
            )

        if not splits_ok:
            print(
                "Split distribution mismatch."
            )

        if not issues_ok:
            print(
                "Issue distribution mismatch."
            )

        if not severities_ok:
            print(
                "Severity distribution mismatch."
            )

        if missing_files:
            print(
                "Metadata references missing files."
            )

        if unexpected_files:
            print(
                "Unexpected image files found."
            )

        if duplicate_filenames:
            print(
                "Duplicate filenames detected."
            )

        if bad_source_counts.any():
            print(
                "Some sources do not have "
                "exactly 19 variants."
            )

        if leakage:
            print(
                "Source-image leakage detected."
            )

        if unreadable:
            print(
                "Unreadable images detected."
            )

        if zero_dimension:
            print(
                "Invalid image dimensions detected."
            )

        if split_path_errors:
            print(
                "Metadata split/path mismatch."
            )

    print(
        "==================================="
    )

    if not passed:
        raise RuntimeError(
            "Dataset sanity check failed."
        )


if __name__ == "__main__":
    main()