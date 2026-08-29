from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    ROOT
    / "data"
    / "features"
    / "expanded_features_v2.csv"
)

MODEL_FILE = (
    ROOT
    / "models"
    / "random_forest_v3_expanded.pkl"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "evaluation"
)


# ============================================================
# V3 FEATURES
# ============================================================

FEATURES = [
    "sharpness",
    "brightness",
    "highlight_clipping",
    "contrast",
    "saturation",
    "edge_density",
    "shadow_clipping",
    "dark_pixel_ratio",
    "bright_pixel_ratio",
    "blockiness",
]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("RANDOM FOREST V3 EXPANDED - FINAL EVALUATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_FILE}"
        )

    model = joblib.load(
        MODEL_FILE
    )

    # --------------------------------------------------------
    # Load features
    # --------------------------------------------------------

    if not FEATURE_FILE.exists():

        raise FileNotFoundError(
            f"Feature file not found:\n{FEATURE_FILE}"
        )

    df = pd.read_csv(
        FEATURE_FILE
    )

    # --------------------------------------------------------
    # Test split ONLY
    # --------------------------------------------------------

    test_df = df[
        df["split"] == "test"
    ].copy()

    X_test = test_df[
        FEATURES
    ]

    y_test = test_df[
        "issue"
    ]

    print()
    print(
        f"Total dataset : {len(df)}"
    )

    print(
        f"Test samples  : {len(test_df)}"
    )

    print(
        f"Features      : {len(FEATURES)}"
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("EXPANDED V3 TEST RESULTS")
    print("=" * 60)

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Macro F1 : {macro_f1:.4f}"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report_dict = classification_report(
        y_test,
        predictions,
        zero_division=0,
        output_dict=True,
    )

    report = classification_report(
        y_test,
        predictions,
        zero_division=0,
    )

    print()
    print("=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)

    print(report)

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    labels = sorted(
        y_test.unique()
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels,
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
        ],
    )

    print()
    print("=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)

    print(
        confusion_df.to_string()
    )

    # --------------------------------------------------------
    # Error analysis
    # --------------------------------------------------------

    results_df = test_df.copy()

    results_df[
        "predicted_issue"
    ] = predictions

    errors = results_df[
        results_df["issue"]
        != results_df["predicted_issue"]
    ].copy()

    print()
    print("=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)

    print(
        f"Total test samples : {len(results_df)}"
    )

    print(
        f"Correct predictions: "
        f"{len(results_df) - len(errors)}"
    )

    print(
        f"Incorrect predictions: "
        f"{len(errors)}"
    )

    if len(errors) > 0:

        print()
        print(
            "Actual → Predicted:"
        )

        error_summary = (
            errors
            .groupby(
                [
                    "issue",
                    "predicted_issue",
                ]
            )
            .size()
            .sort_values(
                ascending=False
            )
        )

        print(
            error_summary.to_string()
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    matrix_file = (
        OUTPUT_DIR
        / "confusion_matrix_v3_expanded.csv"
    )

    confusion_df.to_csv(
        matrix_file
    )

    # --------------------------------------------------------
    # Misclassified samples
    # --------------------------------------------------------

    error_file = (
        OUTPUT_DIR
        / "v3_expanded_misclassified.csv"
    )

    errors.to_csv(
        error_file,
        index=False,
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report_file = (
        OUTPUT_DIR
        / "v3_expanded_classification_report.csv"
    )

    pd.DataFrame(
        report_dict
    ).transpose().to_csv(
        report_file
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary_file = (
        OUTPUT_DIR
        / "v3_expanded_final_results.csv"
    )

    summary = pd.DataFrame(
        [
            {
                "model":
                    "Random Forest V3 Expanded",

                "total_samples":
                    len(df),

                "test_samples":
                    len(test_df),

                "num_features":
                    len(FEATURES),

                "accuracy":
                    accuracy,

                "macro_f1":
                    macro_f1,
            }
        ]
    )

    summary.to_csv(
        summary_file,
        index=False,
    )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 60)
    print("EXPANDED V3 EVALUATION COMPLETE")
    print("=" * 60)

    print()
    print(
        f"Confusion matrix:\n{matrix_file}"
    )

    print()
    print(
        f"Misclassified samples:\n{error_file}"
    )

    print()
    print(
        f"Classification report:\n{report_file}"
    )

    print()
    print(
        f"Final results:\n{summary_file}"
    )


if __name__ == "__main__":

    main()