from pathlib import Path
import shutil

import pandas as pd


# ==================================================
# PATHS
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

COCO_DIR = (
    ROOT
    / "downloads"
    / "val2017"
)

EXISTING_CLEAN_DIR = (
    ROOT
    / "data"
    / "raw"
    / "clean"
)

EXPANDED_CLEAN_DIR = (
    ROOT
    / "data"
    / "raw"
    / "clean_expanded"
)

METADATA_FILE = (
    ROOT
    / "data"
    / "metadata"
    / "expanded_sources.csv"
)


# ==================================================
# MAIN
# ==================================================

def main():

    print("===================================")
    print("PREPARE EXPANDED SOURCE DATASET")
    print("===================================")

    # ----------------------------------------------
    # Check required paths
    # ----------------------------------------------

    if not COCO_DIR.exists():
        raise FileNotFoundError(
            f"COCO directory not found:\n{COCO_DIR}"
        )

    if not EXISTING_CLEAN_DIR.exists():
        raise FileNotFoundError(
            f"Existing clean directory not found:\n"
            f"{EXISTING_CLEAN_DIR}"
        )

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Expanded metadata not found:\n"
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
        .str.strip()
        .str.zfill(12)
    )

    print(
        f"Metadata sources: {len(df)}"
    )

    # ----------------------------------------------
    # Safety checks
    # ----------------------------------------------

    duplicate_ids = (
        df["source_image_id"]
        .duplicated()
        .sum()
    )

    if duplicate_ids != 0:
        raise ValueError(
            f"Duplicate source IDs found: "
            f"{duplicate_ids}"
        )

    # ----------------------------------------------
    # Existing source IDs
    # ----------------------------------------------

    existing_ids = {
        path.stem.zfill(12)
        for path in EXISTING_CLEAN_DIR.glob("*.jpg")
    }

    print(
        f"Existing clean images: "
        f"{len(existing_ids)}"
    )

    # ----------------------------------------------
    # Create expanded directory
    # ----------------------------------------------

    EXPANDED_CLEAN_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ----------------------------------------------
    # Copy all 500 sources
    # ----------------------------------------------

    copied_new = 0
    already_present = 0
    missing = []

    for _, row in df.iterrows():

        source_id = (
            str(row["source_image_id"])
            .zfill(12)
        )

        filename = (
            str(row["filename"])
        )

        source_path = (
            COCO_DIR
            / filename
        )

        destination_path = (
            EXPANDED_CLEAN_DIR
            / f"{source_id}.jpg"
        )

        # ------------------------------------------
        # Verify source exists
        # ------------------------------------------

        if not source_path.exists():

            missing.append(
                source_id
            )

            continue

        # ------------------------------------------
        # Copy if not already present
        # ------------------------------------------

        if destination_path.exists():

            already_present += 1

        else:

            shutil.copy2(
                source_path,
                destination_path
            )

            copied_new += 1

    # ----------------------------------------------
    # Final file count
    # ----------------------------------------------

    expanded_files = list(
        EXPANDED_CLEAN_DIR.glob("*.jpg")
    )

    expanded_ids = {
        path.stem.zfill(12)
        for path in expanded_files
    }

    metadata_ids = set(
        df["source_image_id"]
    )

    # ----------------------------------------------
    # Missing metadata ↔ files
    # ----------------------------------------------

    missing_files = (
        metadata_ids
        - expanded_ids
    )

    unexpected_files = (
        expanded_ids
        - metadata_ids
    )

    # ----------------------------------------------
    # Final report
    # ----------------------------------------------

    print(
        "\n==================================="
    )

    print(
        "SOURCE PREPARATION RESULTS"
    )

    print(
        "==================================="
    )

    print(
        f"Metadata sources      : {len(df)}"
    )

    print(
        f"Existing source IDs   : "
        f"{len(existing_ids)}"
    )

    print(
        f"New files copied      : {copied_new}"
    )

    print(
        f"Already present       : "
        f"{already_present}"
    )

    print(
        f"Missing COCO files    : "
        f"{len(missing)}"
    )

    print(
        f"Expanded files        : "
        f"{len(expanded_files)}"
    )

    print(
        f"Missing from output   : "
        f"{len(missing_files)}"
    )

    print(
        f"Unexpected files      : "
        f"{len(unexpected_files)}"
    )

    # ----------------------------------------------
    # Detailed problems
    # ----------------------------------------------

    if missing:

        print(
            "\nMissing COCO source IDs:"
        )

        for source_id in missing[:20]:
            print(
                f"  {source_id}"
            )

    if missing_files:

        print(
            "\nMetadata IDs without files:"
        )

        for source_id in sorted(
            missing_files
        )[:20]:

            print(
                f"  {source_id}"
            )

    if unexpected_files:

        print(
            "\nUnexpected files:"
        )

        for source_id in sorted(
            unexpected_files
        )[:20]:

            print(
                f"  {source_id}"
            )

    # ----------------------------------------------
    # Final validation
    # ----------------------------------------------

    if (
        len(df) != 500
        or len(expanded_files) != 500
        or len(missing) != 0
        or len(missing_files) != 0
        or len(unexpected_files) != 0
    ):

        raise RuntimeError(
            "\nSOURCE PREPARATION FAILED"
        )

    print(
        "\n==================================="
    )

    print(
        "SOURCE PREPARATION COMPLETE"
    )

    print(
        "==================================="
    )

    print(
        "500 source images verified."
    )

    print(
        "0 missing files."
    )

    print(
        "0 unexpected files."
    )

    print(
        f"\nOutput directory:\n"
        f"{EXPANDED_CLEAN_DIR}"
    )


if __name__ == "__main__":
    main()