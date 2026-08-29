from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


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

MODEL_DIR = (
    ROOT
    / "models"
)

MODEL_FILE = (
    MODEL_DIR
    / "random_forest_v2.pkl"
)


# ==================================================
# V2 FEATURES
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
    print("RANDOM FOREST V2 TRAINING")
    print("===================================")

    # ----------------------------------------------
    # Load feature dataset
    # ----------------------------------------------

    if not FEATURE_FILE.exists():

        raise FileNotFoundError(
            f"Feature file not found:\n"
            f"{FEATURE_FILE}"
        )

    df = pd.read_csv(
        FEATURE_FILE
    )

    print(
        f"Total records: {len(df)}"
    )

    # ----------------------------------------------
    # Existing source-level split
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
    # X and y
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
    # Random Forest
    # ----------------------------------------------

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    print(
        "\nTraining Random Forest V2..."
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "Training complete."
    )

    # ----------------------------------------------
    # Validation
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
        "V2 VALIDATION RESULTS"
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
    # Test
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
        "V2 TEST RESULTS"
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
        "V2 FEATURE IMPORTANCE"
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
        "V2 MODEL SAVED"
    )

    print(
        "==================================="
    )

    print(
        MODEL_FILE
    )


if __name__ == "__main__":
    main()