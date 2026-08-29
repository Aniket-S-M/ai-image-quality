from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)


ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    ROOT
    / "data"
    / "features"
    / "features.csv"
)

MODEL_FILE = (
    ROOT
    / "models"
    / "random_forest.pkl"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "evaluation"
)

CONFUSION_MATRIX_FILE = (
    OUTPUT_DIR
    / "confusion_matrix.png"
)

FEATURE_COLUMNS = [
    "sharpness",
    "brightness",
    "highlight_clipping",
    "contrast",
    "saturation",
    "edge_density",
    "noise_estimate",
]


def main():

    print("===================================")
    print("CLASSIFIER EVALUATION")
    print("===================================")

    # -----------------------------------------
    # Load data
    # -----------------------------------------

    df = pd.read_csv(
        FEATURE_FILE
    )

    test_df = df[
        df["split"] == "test"
    ].copy()

    print(
        f"Test samples: {len(test_df)}"
    )

    # -----------------------------------------
    # Load trained model
    # -----------------------------------------

    model = joblib.load(
        MODEL_FILE
    )

    # -----------------------------------------
    # Prepare test data
    # -----------------------------------------

    X_test = test_df[
        FEATURE_COLUMNS
    ]

    y_test = test_df[
        "issue"
    ]

    # -----------------------------------------
    # Predictions
    # -----------------------------------------

    predictions = model.predict(
        X_test
    )

    # -----------------------------------------
    # Classification report
    # -----------------------------------------

    print("\n===================================")
    print("CLASSIFICATION REPORT")
    print("===================================\n")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    # -----------------------------------------
    # Confusion matrix
    # -----------------------------------------

    labels = [
        "none",
        "blur",
        "underexposure",
        "overexposure",
        "noise",
        "corruption",
    ]

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )

    print("===================================")
    print("CONFUSION MATRIX")
    print("===================================\n")

    print(
        pd.DataFrame(
            cm,
            index=[
                f"Actual {label}"
                for label in labels
            ],
            columns=[
                f"Pred {label}"
                for label in labels
            ]
        )
    )

    # -----------------------------------------
    # Save graph
    # -----------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )

    display.plot(
        ax=ax,
        xticks_rotation=30
    )

    ax.set_title(
        "Random Forest — Test Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        CONFUSION_MATRIX_FILE,
        dpi=150
    )

    plt.close()

    print(
        f"\nSaved confusion matrix:"
    )

    print(
        CONFUSION_MATRIX_FILE
    )


if __name__ == "__main__":
    main()