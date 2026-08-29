from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


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

MODEL_FILE = (
    ROOT
    / "models"
    / "random_forest_v2.pkl"
)


# ==================================================
# FEATURES
# ==================================================

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


TARGET_COLUMN = "issue"


# ==================================================
# MAIN
# ==================================================

def main():

    print("===================================")
    print("RANDOM FOREST V2 EVALUATION")
    print("===================================")

    # ----------------------------------------------
    # Load data
    # ----------------------------------------------

    df = pd.read_csv(
        FEATURE_FILE
    )

    model = joblib.load(
        MODEL_FILE
    )

    # ----------------------------------------------
    # Test set
    # ----------------------------------------------

    test_df = df[
        df["split"] == "test"
    ].copy()

    X_test = test_df[
        FEATURE_COLUMNS
    ]

    y_test = test_df[
        TARGET_COLUMN
    ]

    # ----------------------------------------------
    # Predictions
    # ----------------------------------------------

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    # ----------------------------------------------
    # Classification report
    # ----------------------------------------------

    print("\n===================================")
    print("TEST RESULTS")
    print("===================================")

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print("\nClassification report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    # ----------------------------------------------
    # Confusion matrix
    # ----------------------------------------------

    labels = sorted(
        y_test.unique()
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )

    confusion_df = pd.DataFrame(
        matrix,
        index=[
            f"Actual: {label}"
            for label in labels
        ],
        columns=[
            f"Pred: {label}"
            for label in labels
        ]
    )

    print("\n===================================")
    print("CONFUSION MATRIX")
    print("===================================")

    print(
        confusion_df
    )

    # ----------------------------------------------
    # Error summary
    # ----------------------------------------------

    results = test_df.copy()

    results[
        "predicted_issue"
    ] = predictions

    errors = results[
        results["issue"]
        != results["predicted_issue"]
    ]

    print("\n===================================")
    print("ERROR SUMMARY")
    print("===================================")

    print(
        f"Total test samples: "
        f"{len(results)}"
    )

    print(
        f"Correct predictions: "
        f"{len(results) - len(errors)}"
    )

    print(
        f"Incorrect predictions: "
        f"{len(errors)}"
    )

    print(
        "\nActual → Predicted:"
    )

    if len(errors) > 0:

        error_summary = (
            errors
            .groupby(
                [
                    "issue",
                    "predicted_issue"
                ]
            )
            .size()
            .sort_values(
                ascending=False
            )
        )

        print(
            error_summary
        )

    else:

        print(
            "No errors."
        )

    # ----------------------------------------------
    # Save confusion matrix
    # ----------------------------------------------

    output_dir = (
        ROOT
        / "data"
        / "evaluation"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    matrix_file = (
        output_dir
        / "confusion_matrix_v2.csv"
    )

    confusion_df.to_csv(
        matrix_file
    )

    print(
        "\nConfusion matrix saved:"
    )

    print(
        matrix_file
    )

    print(
        "\n==================================="
    )

    print(
        "V2 EVALUATION COMPLETE"
    )

    print(
        "==================================="
    )


if __name__ == "__main__":
    main()