from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    ROOT
    / "data"
    / "features"
    / "features_v2.csv"
)


NEW_FEATURES = [
    "shadow_clipping",
    "dark_pixel_ratio",
    "bright_pixel_ratio",
    "blockiness",
]


def main():

    print("===================================")
    print("FEATURE V2 ANALYSIS")
    print("===================================")

    df = pd.read_csv(
        FEATURE_FILE
    )

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    print("\nNew feature statistics:")

    print(
        df[NEW_FEATURES].describe()
    )

    print(
        "\nMean by issue:"
    )

    print(
        df.groupby("issue")[
            NEW_FEATURES
        ].mean().round(3)
    )

    print(
        "\nMean by issue and severity:"
    )

    print(
        df.groupby(
            ["issue", "severity"]
        )[
            NEW_FEATURES
        ].mean().round(3)
    )

    print(
        "\nFeature V2 analysis complete."
    )


if __name__ == "__main__":
    main()