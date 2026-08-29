from pathlib import Path

import pandas as pd


# ==================================================
# PATHS
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    ROOT
    / "data"
    / "features"
    / "features_v2.csv"
)


FEATURE_COLUMNS = [
    "sharpness",
    "brightness",
    "highlight_clipping",
    "contrast",
    "saturation",
    "edge_density",
    "noise_estimate",
    "shadow_clipping",
    "dark_pixel_ratio",
    "bright_pixel_ratio",
    "blockiness",
]


def main():

    print("===================================")
    print("FEATURE CORRELATION ANALYSIS")
    print("===================================")

    df = pd.read_csv(
        FEATURE_FILE
    )

    features = df[
        FEATURE_COLUMNS
    ]

    correlation = features.corr()

    print("\nCorrelation matrix:")
    print(
        correlation.round(2)
    )

    # ----------------------------------------------
    # Find highly correlated pairs
    # ----------------------------------------------

    print(
        "\n==================================="
    )

    print(
        "HIGHLY CORRELATED FEATURE PAIRS"
    )

    print(
        "==================================="
    )

    threshold = 0.80

    found = False

    for i in range(
        len(FEATURE_COLUMNS)
    ):

        for j in range(
            i + 1,
            len(FEATURE_COLUMNS)
        ):

            feature_a = FEATURE_COLUMNS[i]
            feature_b = FEATURE_COLUMNS[j]

            value = correlation.loc[
                feature_a,
                feature_b
            ]

            if abs(value) >= threshold:

                print(
                    f"{feature_a:25s} "
                    f"<-> "
                    f"{feature_b:25s} "
                    f"{value:.3f}"
                )

                found = True

    if not found:

        print(
            "No feature pairs above "
            f"{threshold:.2f}"
        )

    print(
        "\nCorrelation analysis complete."
    )


if __name__ == "__main__":
    main()