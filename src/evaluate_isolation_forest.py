from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
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
    / "isolation_forest.pkl"
)

SCALER_FILE = (
    ROOT
    / "models"
    / "isolation_forest_scaler.pkl"
)


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
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


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ISOLATION FOREST TEST EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = pd.read_csv(
        FEATURE_FILE
    )

    test_df = df[
        df["split"] == "test"
    ].copy()

    print(
        f"Test images : {len(test_df)}"
    )

    # --------------------------------------------------------
    # Load artifacts
    # --------------------------------------------------------

    model = joblib.load(
        MODEL_FILE
    )

    scaler = joblib.load(
        SCALER_FILE
    )

    # --------------------------------------------------------
    # Prepare features
    # --------------------------------------------------------

    X = test_df[
        FEATURES
    ].copy()

    X = X.fillna(
        X.median()
    )

    X_scaled = scaler.transform(
        X
    )

    # --------------------------------------------------------
    # Isolation Forest prediction
    #
    #  1  = normal
    # -1  = anomaly
    # --------------------------------------------------------

    predictions = model.predict(
        X_scaled
    )

    test_df["anomaly_prediction"] = predictions

    test_df["is_anomaly"] = (
        predictions == -1
    )

    # --------------------------------------------------------
    # Ground truth
    #
    # none       = normal
    # everything else = degraded/anomalous
    # --------------------------------------------------------

    test_df["ground_truth_anomaly"] = (
        test_df["issue"] != "none"
    )

    # --------------------------------------------------------
    # Overall anomaly detection
    # --------------------------------------------------------

    y_true = (
        test_df["ground_truth_anomaly"]
        .astype(int)
    )

    y_pred = (
        test_df["is_anomaly"]
        .astype(int)
    )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    # --------------------------------------------------------
    # Overall results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("OVERALL ANOMALY DETECTION")
    print("=" * 70)

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1        : {f1:.4f}"
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    print()
    print(
        "CONFUSION MATRIX"
    )
    print(
        "Rows = actual"
    )
    print(
        "Columns = predicted"
    )

    print()
    print(
        "                 Normal  Anomaly"
    )

    print(
        f"Actual Normal    "
        f"{cm[0][0]:6d}  "
        f"{cm[0][1]:7d}"
    )

    print(
        f"Actual Degraded  "
        f"{cm[1][0]:6d}  "
        f"{cm[1][1]:7d}"
    )

    # --------------------------------------------------------
    # Clean image behaviour
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CLEAN IMAGE ANALYSIS")
    print("=" * 70)

    clean_df = test_df[
        test_df["issue"] == "none"
    ]

    clean_anomaly_rate = (
        clean_df["is_anomaly"].mean()
    )

    print(
        f"Clean images : {len(clean_df)}"
    )

    print(
        f"Flagged as anomaly : "
        f"{clean_df['is_anomaly'].sum()}"
    )

    print(
        f"False anomaly rate : "
        f"{clean_anomaly_rate:.4f}"
    )

    # --------------------------------------------------------
    # Issue-level anomaly rates
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ANOMALY RATE BY ISSUE")
    print("=" * 70)

    issue_results = []

    for issue in sorted(
        test_df["issue"].unique()
    ):

        subset = test_df[
            test_df["issue"] == issue
        ]

        anomaly_count = (
            subset["is_anomaly"].sum()
        )

        anomaly_rate = (
            anomaly_count
            / len(subset)
        )

        issue_results.append(
            {
                "issue": issue,
                "images": len(subset),
                "anomalies": anomaly_count,
                "anomaly_rate": anomaly_rate,
            }
        )

        print(
            f"{issue:<16}"
            f"{len(subset):>5} images   "
            f"{anomaly_count:>5} anomalies   "
            f"{anomaly_rate:.4f}"
        )

    # --------------------------------------------------------
    # Severity-level anomaly rates
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ANOMALY RATE BY SEVERITY")
    print("=" * 70)

    severity_results = []

    for severity in [
        "none",
        "mild",
        "moderate",
        "severe",
    ]:

        subset = test_df[
            test_df["severity"] == severity
        ]

        if len(subset) == 0:
            continue

        anomaly_count = (
            subset["is_anomaly"].sum()
        )

        anomaly_rate = (
            anomaly_count
            / len(subset)
        )

        severity_results.append(
            {
                "severity": severity,
                "images": len(subset),
                "anomalies": anomaly_count,
                "anomaly_rate": anomaly_rate,
            }
        )

        print(
            f"{severity:<10}"
            f"{len(subset):>5} images   "
            f"{anomaly_count:>5} anomalies   "
            f"{anomaly_rate:.4f}"
        )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[
                "normal",
                "degraded",
            ],
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Save evaluation results
    # --------------------------------------------------------

    evaluation_dir = (
        ROOT
        / "data"
        / "evaluation"
    )

    evaluation_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        evaluation_dir
        / "isolation_forest_test_results.csv"
    )

    test_df.to_csv(
        output_file,
        index=False
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print("=" * 70)
    print("ISOLATION FOREST EVALUATION COMPLETE")
    print("=" * 70)

    print(
        f"Results saved to:\n{output_file}"
    )


if __name__ == "__main__":
    main()