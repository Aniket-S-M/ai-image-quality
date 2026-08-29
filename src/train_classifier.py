from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    ROOT
    / "data"
    / "features"
    / "features.csv"
)

MODEL_DIR = (
    ROOT
    / "models"
)

MODEL_FILE = (
    MODEL_DIR
    / "random_forest.pkl"
)


# --------------------------------------------------
# Features used by the model
# --------------------------------------------------

FEATURE_COLUMNS = [
    "sharpness",
    "brightness",
    "highlight_clipping",
    "contrast",
    "saturation",
    "edge_density",
    "noise_estimate",
]


TARGET_COLUMN = "issue"


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("===================================")
    print("RANDOM FOREST TRAINING")
    print("===================================")

    # ----------------------------------------------
    # Load feature dataset
    # ----------------------------------------------

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURE_FILE}"
        )

    df = pd.read_csv(
        FEATURE_FILE
    )

    print(
        f"Total records: {len(df)}"
    )

    # ----------------------------------------------
    # Split using existing source-level split
    # ----------------------------------------------

    train_df = df[
        df["split"] == "train"
    ].copy()

    val_df = df[
        df["split"] == "val"
    ].copy()

    test_df = df[
        df["split"] == "test"
    ].copy()

    print(
        f"Train: {len(train_df)}"
    )

    print(
        f"Validation: {len(val_df)}"
    )

    print(
        f"Test: {len(test_df)}"
    )

    # ----------------------------------------------
    # Prepare X and y
    # ----------------------------------------------

    X_train = train_df[
        FEATURE_COLUMNS
    ]

    y_train = train_df[
        TARGET_COLUMN
    ]

    X_val = val_df[
        FEATURE_COLUMNS
    ]

    y_val = val_df[
        TARGET_COLUMN
    ]

    X_test = test_df[
        FEATURE_COLUMNS
    ]

    y_test = test_df[
        TARGET_COLUMN
    ]

    # ----------------------------------------------
    # Train Random Forest
    # ----------------------------------------------

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    print(
        "\nTraining Random Forest..."
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "Training complete."
    )

    # ----------------------------------------------
    # Validation evaluation
    # ----------------------------------------------

    val_predictions = model.predict(
        X_val
    )

    val_accuracy = accuracy_score(
        y_val,
        val_predictions
    )

    print(
        "\n==================================="
    )

    print(
        "VALIDATION RESULTS"
    )

    print(
        "==================================="
    )

    print(
        f"Accuracy: "
        f"{val_accuracy:.4f}"
    )

    print(
        "\nClassification report:"
    )

    print(
        classification_report(
            y_val,
            val_predictions,
            zero_division=0
        )
    )

    # ----------------------------------------------
    # Test evaluation
    # ----------------------------------------------

    test_predictions = model.predict(
        X_test
    )

    test_accuracy = accuracy_score(
        y_test,
        test_predictions
    )

    print(
        "\n==================================="
    )

    print(
        "TEST RESULTS"
    )

    print(
        "==================================="
    )

    print(
        f"Accuracy: "
        f"{test_accuracy:.4f}"
    )

    print(
        "\nClassification report:"
    )

    print(
        classification_report(
            y_test,
            test_predictions,
            zero_division=0
        )
    )

    # ----------------------------------------------
    # Feature importance
    # ----------------------------------------------

    importance = pd.Series(
        model.feature_importances_,
        index=FEATURE_COLUMNS
    ).sort_values(
        ascending=False
    )

    print(
        "\n==================================="
    )

    print(
        "FEATURE IMPORTANCE"
    )

    print(
        "==================================="
    )

    print(
        importance
    )

    # ----------------------------------------------
    # Save model
    # ----------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_FILE
    )

    print(
        "\n==================================="
    )

    print(
        "MODEL SAVED"
    )

    print(
        "==================================="
    )

    print(
        MODEL_FILE
    )


if __name__ == "__main__":
    main()