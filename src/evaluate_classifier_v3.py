from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
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
    / "random_forest_v3.pkl"
)

FEATURE_LIST_FILE = (
    ROOT
    / "models"
    / "random_forest_v3_features.txt"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "evaluation"
)


# ==================================================
# LOAD SELECTED FEATURES
# ==================================================

def load_features():

    if not FEATURE_LIST_FILE.exists():

        raise FileNotFoundError(
            f"Feature list not found:\n"
            f"{FEATURE_LIST_FILE}"
        )

    with open(
        FEATURE_LIST_FILE,
        "r"
    ) as f:

        features = [
            line.strip()
            for line in f
            if line.strip()
        ]

    return features


# ==================================================
# MAIN
# ==================================================

def main():

    print(
        "==================================="
    )

    print(
        "RANDOM FOREST V3 FINAL EVALUATION"
    )

    print(
        "==================================="
    )

    # ----------------------------------------------
    # Load model
    # ----------------------------------------------

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            f"Model not found:\n"
            f"{MODEL_FILE}"
        )

    model = joblib.load(
        MODEL_FILE
    )

    # ----------------------------------------------
    # Load features
    # ----------------------------------------------

    df = pd.read_csv(
        FEATURE_FILE
    )

    features = load_features()

    print(
        f"Total records: {len(df)}"
    )

    print(
        f"Selected features: {len(features)}"
    )

    print(
        "\nFeatures used:"
    )

    for feature in features:

        print(
            f"  - {feature}"
        )

    # ----------------------------------------------
    # Test set
    # ----------------------------------------------

    test_df = df[
        df["split"] == "test"
    ].copy()

    X_test = test_df[
        features
    ]

    y_test = test_df[
        "issue"
    ]

    print(
        f"\nTest samples: {len(test_df)}"
    )

    # ----------------------------------------------
    # Predictions
    # ----------------------------------------------

    predictions = model.predict(
        X_test
    )

    # ----------------------------------------------
    # Accuracy
    # ----------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )

    print(
        "\n==================================="
    )

    print(
        "V3 TEST RESULTS"
    )

    print(
        "==================================="
    )

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print(
        f"Macro F1: {macro_f1:.4f}"
    )

    # ----------------------------------------------
    # Classification report
    # ----------------------------------------------

    print(
        "\nClassification report:"
    )

    report = classification_report(
        y_test,
        predictions,
        zero_division=0
    )

    print(
        report
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

    print(
        "\n==================================="
    )

    print(
        "V3 CONFUSION MATRIX"
    )

    print(
        "==================================="
    )

    print(
        confusion_df.to_string()
    )

    # ----------------------------------------------
    # Error analysis
    # ----------------------------------------------

    results = test_df.copy()

    results[
        "predicted_issue"
    ] = predictions

    errors = results[
        results["issue"]
        != results["predicted_issue"]
    ]

    print(
        "\n==================================="
    )

    print(
        "V3 ERROR SUMMARY"
    )

    print(
        "==================================="
    )

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

    if len(errors) > 0:

        print(
            "\nActual → Predicted:"
        )

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

    # ----------------------------------------------
    # Save evaluation files
    # ----------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    matrix_file = (
        OUTPUT_DIR
        / "confusion_matrix_v3.csv"
    )

    confusion_df.to_csv(
        matrix_file
    )

    error_file = (
        OUTPUT_DIR
        / "v3_misclassified.csv"
    )

    errors.to_csv(
        error_file,
        index=False
    )

    # ----------------------------------------------
    # Final summary
    # ----------------------------------------------

    print(
        "\n==================================="
    )

    print(
        "V3 EVALUATION COMPLETE"
    )

    print(
        "==================================="
    )

    print(
        f"Confusion matrix:"
        f"\n{matrix_file}"
    )

    print(
        f"Misclassified samples:"
        f"\n{error_file}"
    )


if __name__ == "__main__":
    main()