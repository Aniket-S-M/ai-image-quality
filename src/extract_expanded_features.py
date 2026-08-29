from pathlib import Path

import pandas as pd

from extract_features_v2 import extract_features


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    ROOT
    / "data"
    / "generated"
    / "dataset_expanded"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "features"
    / "expanded_features_v2.csv"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("EXPANDED DATASET FEATURE EXTRACTION")
    print("=" * 60)

    records = []

    # --------------------------------------------------------
    # Process train / val / test
    # --------------------------------------------------------

    for split in ["train", "val", "test"]:

        split_dir = DATASET_DIR / split

        if not split_dir.exists():

            print(
                f"[WARNING] Missing: {split_dir}"
            )

            continue

        images = list(
            split_dir.glob("*.jpg")
        )

        print()
        print(
            f"{split.upper()} images: {len(images)}"
        )

        for index, image_path in enumerate(images):

            try:

                # ------------------------------------------------
                # Extract CV features
                # ------------------------------------------------

                features = extract_features(
                    image_path
                )

                # ------------------------------------------------
                # Derive ground-truth issue/severity
                # from filename
                #
                # Normal:
                # ID_blur_mild.jpg
                #
                # Noise SP:
                # ID_noise_sp_mild.jpg
                #
                # Clean:
                # ID_clean.jpg
                # ------------------------------------------------

                filename = image_path.stem

                parts = filename.split("_")


                # ------------------------------------------------
                # CLEAN
                # ------------------------------------------------

                if parts[-1] == "clean":

                    issue = "none"
                    severity = "none"


                # ------------------------------------------------
                # SALT-AND-PEPPER NOISE
                #
                # Treat noise_sp as the "noise" issue.
                # ------------------------------------------------

                elif (
                    len(parts) >= 3
                    and parts[-3] == "noise"
                    and parts[-2] == "sp"
                ):

                    issue = "noise"
                    severity = parts[-1]


                # ------------------------------------------------
                # NORMAL DEGRADATION
                # ------------------------------------------------

                else:

                    issue = parts[-2]
                    severity = parts[-1]


                # ------------------------------------------------
                # Validate issue labels
                # ------------------------------------------------

                valid_issues = {
                    "blur",
                    "corruption",
                    "noise",
                    "none",
                    "overexposure",
                    "underexposure",
                }

                if issue not in valid_issues:

                    print(
                        f"[WARNING] Unexpected issue "
                        f"'{issue}' in {filename}"
                    )

                    continue


                # ------------------------------------------------
                # Create record
                # ------------------------------------------------

                record = {

                    "filename":
                        image_path.name,

                    "split":
                        split,

                    "issue":
                        issue,

                    "severity":
                        severity,

                }

                record.update(
                    features
                )

                records.append(
                    record
                )


            except Exception as e:

                print(
                    f"[SKIP] "
                    f"{image_path.name}: {e}"
                )


            # ----------------------------------------------------
            # Progress
            # ----------------------------------------------------

            if (index + 1) % 500 == 0:

                print(
                    f"  Processed "
                    f"{index + 1}/"
                    f"{len(images)}"
                )


    # ============================================================
    # CREATE DATAFRAME
    # ============================================================

    df = pd.DataFrame(
        records
    )


    # ============================================================
    # SAVE
    # ============================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )


    # ============================================================
    # SUMMARY
    # ============================================================

    print()
    print("=" * 60)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 60)

    print(
        f"Total records : {len(df)}"
    )

    print(
        "Features      : 11"
    )

    print(
        f"Output        : {OUTPUT_FILE}"
    )


    # ============================================================
    # CLASS DISTRIBUTION
    # ============================================================

    print()
    print("CLASS DISTRIBUTION")
    print("-" * 60)

    distribution = (
        df.groupby(
            ["issue", "severity"]
        )
        .size()
    )

    print(
        distribution.to_string()
    )


    # ============================================================
    # ISSUE DISTRIBUTION
    # ============================================================

    print()
    print("ISSUE DISTRIBUTION")
    print("-" * 60)

    print(
        df["issue"]
        .value_counts()
        .sort_index()
        .to_string()
    )


    # ============================================================
    # SPLIT DISTRIBUTION
    # ============================================================

    print()
    print("SPLIT DISTRIBUTION")
    print("-" * 60)

    print(
        df["split"]
        .value_counts()
        .sort_index()
        .to_string()
    )


    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()