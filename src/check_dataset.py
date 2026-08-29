from pathlib import Path
import csv
from collections import Counter


ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = ROOT / "data" / "generated" / "dataset"
METADATA_FILE = ROOT / "data" / "metadata" / "degradation_dataset.csv"


def main():

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata not found: {METADATA_FILE}"
        )

    with METADATA_FILE.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        records = list(csv.DictReader(file))

    print(f"Metadata records: {len(records)}")

    # -----------------------------------------
    # Check files exist
    # -----------------------------------------

    missing_files = []

    for record in records:

        split = record["split"]
        filename = record["filename"]

        path = (
            DATASET_DIR
            / split
            / filename
        )

        if not path.exists():
            missing_files.append(
                str(path)
            )

    print(
        f"Missing files: {len(missing_files)}"
    )

    # -----------------------------------------
    # Check duplicate records
    # -----------------------------------------

    filenames = [
        record["filename"]
        for record in records
    ]

    filename_counts = Counter(
        filenames
    )

    duplicates = [
        filename
        for filename, count
        in filename_counts.items()
        if count > 1
    ]

    print(
        f"Duplicate filenames: "
        f"{len(duplicates)}"
    )

    # -----------------------------------------
    # Count by split
    # -----------------------------------------

    split_counts = Counter(
        record["split"]
        for record in records
    )

    print("\nSamples by split:")

    for split in [
        "train",
        "val",
        "test"
    ]:

        print(
            f"{split:5s}: "
            f"{split_counts[split]}"
        )

    # -----------------------------------------
    # Count by issue
    # -----------------------------------------

    issue_counts = Counter(
        record["issue"]
        for record in records
    )

    print("\nSamples by issue:")

    for issue, count in sorted(
        issue_counts.items()
    ):

        print(
            f"{issue:15s}: {count}"
        )

    # -----------------------------------------
    # Count by severity
    # -----------------------------------------

    severity_counts = Counter(
        record["severity"]
        for record in records
    )

    print("\nSamples by severity:")

    for severity, count in sorted(
        severity_counts.items()
    ):

        print(
            f"{severity:10s}: {count}"
        )

    # -----------------------------------------
    # Check source-image leakage
    # -----------------------------------------

    source_split_map = {}

    leakage = []

    for record in records:

        source_id = record[
            "source_image_id"
        ]

        split = record["split"]

        if source_id not in source_split_map:

            source_split_map[source_id] = split

        elif source_split_map[source_id] != split:

            leakage.append(
                source_id
            )

    leakage = list(set(leakage))

    print(
        f"\nSource-image leakage: "
        f"{len(leakage)}"
    )

    # -----------------------------------------
    # Final result
    # -----------------------------------------

    if (
        len(records) == 950
        and len(missing_files) == 0
        and len(duplicates) == 0
        and len(leakage) == 0
    ):

        print(
            "\n✅ DATASET SANITY CHECK PASSED"
        )

    else:

        print(
            "\n⚠️ DATASET SANITY CHECK FAILED"
        )


if __name__ == "__main__":
    main()