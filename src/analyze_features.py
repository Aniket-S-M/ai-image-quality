from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    ROOT
    / "data"
    / "features"
    / "features.csv"
)


def main():

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURE_FILE}"
        )

    df = pd.read_csv(
        FEATURE_FILE
    )

    print("===================================")
    print("FEATURE DATASET ANALYSIS")
    print("===================================")

    print(
        f"\nRows: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    # -----------------------------------------
    # Missing values
    # -----------------------------------------

    print("\nMissing values:")

    print(
        df.isnull().sum()
    )

    # -----------------------------------------
    # Feature summary
    # -----------------------------------------

    feature_columns = [
        "sharpness",
        "brightness",
        "highlight_clipping",
        "contrast",
        "saturation",
        "edge_density",
        "noise_estimate",
    ]

    print("\nOverall feature statistics:")

    print(
        df[feature_columns]
        .describe()
        .round(3)
    )

    # -----------------------------------------
    # Mean features by issue
    # -----------------------------------------

    print("\nMean features by issue:")

    issue_means = (
        df.groupby("issue")[feature_columns]
        .mean()
        .round(3)
    )

    print(
        issue_means
    )

    # -----------------------------------------
    # Mean features by issue + severity
    # -----------------------------------------

    print(
        "\nMean features by issue and severity:"
    )

    severity_means = (
        df.groupby(
            ["issue", "severity"]
        )[feature_columns]
        .mean()
        .round(3)
    )

    print(
        severity_means
    )

    # -----------------------------------------
    # Class distribution
    # -----------------------------------------

    print("\nIssue distribution:")

    print(
        df["issue"]
        .value_counts()
    )

    print("\nSeverity distribution:")

    print(
        df["severity"]
        .value_counts()
    )


if __name__ == "__main__":
    main()