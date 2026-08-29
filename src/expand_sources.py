from pathlib import Path
import random

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

CLEAN_DIR = (
    ROOT
    / "data"
    / "raw"
    / "clean"
)

PILOT_METADATA = (
    ROOT
    / "data"
    / "metadata"
    / "pilot_sources.csv"
)

OUTPUT_METADATA = (
    ROOT
    / "data"
    / "metadata"
    / "expanded_sources.csv"
)


# ==================================================
# SETTINGS
# ==================================================

TARGET_TOTAL = 500

RANDOM_SEED = 42


# ==================================================
# MAIN
# ==================================================

def main():

    print("===================================")
    print("SOURCE DATASET EXPANSION")
    print("===================================")

    # ----------------------------------------------
    # Check paths
    # ----------------------------------------------

    if not COCO_DIR.exists():

        raise FileNotFoundError(
            f"COCO directory not found:\n"
            f"{COCO_DIR}"
        )

    if not CLEAN_DIR.exists():

        raise FileNotFoundError(
            f"Clean directory not found:\n"
            f"{CLEAN_DIR}"
        )

    if not PILOT_METADATA.exists():

        raise FileNotFoundError(
            f"Pilot metadata not found:\n"
            f"{PILOT_METADATA}"
        )

    # ----------------------------------------------
    # Load existing sources
    # ----------------------------------------------

    pilot_df = pd.read_csv(
        PILOT_METADATA,
        dtype={
            "source_image":str
        }
    )

    existing_ids = set(
        pilot_df[
            "source_image_id"
        ]
        .astype(str)
        .str.zfill(12)
    )

    print(
        f"Existing sources: "
        f"{len(existing_ids)}"
    )

    # ----------------------------------------------
    # Read COCO images
    # ----------------------------------------------

    coco_files = sorted(
        COCO_DIR.glob("*.jpg")
    )

    print(
        f"COCO images available: "
        f"{len(coco_files)}"
    )

    # ----------------------------------------------
    # Find candidates
    # ----------------------------------------------

    candidates = []

    for image_path in coco_files:

        image_id = image_path.stem.zfill(12)

        if image_id not in existing_ids:

            candidates.append(
                image_path
            )

    print(
        f"Unused candidate images: "
        f"{len(candidates)}"
    )

    # ----------------------------------------------
    # Determine how many new images are needed
    # ----------------------------------------------

    new_required = (
        TARGET_TOTAL
        - len(existing_ids)
    )

    if new_required <= 0:

        print(
            "\nTarget already reached."
        )

        return

    if len(candidates) < new_required:

        raise ValueError(
            "Not enough unused COCO images "
            "to reach target."
        )

    print(
        f"New sources required: "
        f"{new_required}"
    )

    # ----------------------------------------------
    # Select new sources
    # ----------------------------------------------

    random.seed(
        RANDOM_SEED
    )

    selected = random.sample(
        candidates,
        new_required
    )

    selected = sorted(
        selected,
        key=lambda x: x.stem
    )

    print(
        "\nSelected new sources:"
    )

    for image_path in selected[:10]:

        print(
            f"  {image_path.name}"
        )

    if len(selected) > 10:

        print(
            f"  ... and "
            f"{len(selected) - 10} more"
        )

    # ----------------------------------------------
    # Build expanded metadata
    # ----------------------------------------------

    new_records = []

    for image_path in selected:

        new_records.append(
            {
                "source_image_id":
                    image_path.stem,

                "filename":
                    image_path.name,

                "source_dataset":
                    "COCO 2017 val2017",
            }
        )

    new_df = pd.DataFrame(
        new_records
    )

    expanded_df = pd.concat(
        [
            pilot_df,
            new_df,
        ],
        ignore_index=True
    )

    # ----------------------------------------------
    # Safety checks
    # ----------------------------------------------

    if expanded_df[
        "source_image_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate source_image_id detected!"
        )

    if len(expanded_df) != TARGET_TOTAL:

        raise ValueError(
            f"Expected {TARGET_TOTAL} sources, "
            f"got {len(expanded_df)}"
        )

    # ----------------------------------------------
    # Save
    # ----------------------------------------------

    OUTPUT_METADATA.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    expanded_df.to_csv(
        OUTPUT_METADATA,
        index=False
    )

    # ----------------------------------------------
    # Final report
    # ----------------------------------------------

    print(
        "\n==================================="
    )

    print(
        "SOURCE EXPANSION COMPLETE"
    )

    print(
        "==================================="
    )

    print(
        f"Original sources : "
        f"{len(pilot_df)}"
    )

    print(
        f"New sources      : "
        f"{len(new_df)}"
    )

    print(
        f"Total sources    : "
        f"{len(expanded_df)}"
    )

    print(
        f"Duplicate IDs    : "
        f"{expanded_df['source_image_id'].duplicated().sum()}"
    )

    print(
        f"Output file      : "
        f"{OUTPUT_METADATA}"
    )


if __name__ == "__main__":
    main()