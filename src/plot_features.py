from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    ROOT
    / "data"
    / "features"
    / "features.csv"
)

PLOT_DIR = (
    ROOT
    / "data"
    / "features"
    / "plots"
)


FEATURES_TO_PLOT = [
    "sharpness",
    "brightness",
    "highlight_clipping",
    "noise_estimate",
]


def main():

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURE_FILE}"
        )

    df = pd.read_csv(FEATURE_FILE)

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"Loaded {len(df)} feature records."
    )

    # -----------------------------------------
    # Calculate mean feature value by issue
    # -----------------------------------------

    means = (
        df.groupby("issue")[FEATURES_TO_PLOT]
        .mean()
    )

    # -----------------------------------------
    # Create simple bar charts
    # -----------------------------------------

    for feature in FEATURES_TO_PLOT:

        plt.figure(
            figsize=(10, 6)
        )

        means[feature].plot(
            kind="bar"
        )

        plt.title(
            f"Average {feature.replace('_', ' ').title()} by Issue"
        )

        plt.xlabel(
            "Image Quality Issue"
        )

        plt.ylabel(
            f"Average {feature.replace('_', ' ').title()}"
        )

        plt.xticks(
            rotation=30
        )

        plt.tight_layout()

        output_path = (
            PLOT_DIR
            / f"{feature}_average_by_issue.png"
        )

        plt.savefig(
            output_path,
            dpi=150
        )

        plt.close()

        print(
            f"Created: {output_path}"
        )

    print(
        "\nSimple feature graphs created successfully."
    )


if __name__ == "__main__":
    main()