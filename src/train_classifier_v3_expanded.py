from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


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

MODEL_DIR = (
    ROOT
    / "models"
)

EVALUATION_DIR = (
    ROOT
    / "data"
    / "evaluation"
)


# ============================================================
# RANDOM FOREST V3 FEATURE SET
#
# This is the SAME 10-feature configuration selected
# for the original RF V3 model.
#
# noise_estimate is intentionally excluded.
# ============================================================

V3_FEATURES = [
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
# TRAIN MODEL
# ============================================================

def train_model(
    train_df,
):

    X_train = train_df[
        V3_FEATURES
    ]

    y_train = train_df[
        "issue"
    ]

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


# ============================================================
# EVALUATE
# ============================================================

def evaluate_model(
    model,
    dataframe,
):

    X = dataframe[
        V3_FEATURES
    ]

    y = dataframe[
        "issue"
    ]

    predictions = model.predict(
        X
    )

    accuracy = accuracy_score(
        y,
        predictions,
    )

    macro_f1 = f1_score(
        y,
        predictions,
        average="macro",
    )

    return (
        accuracy,
        macro_f1,
        predictions,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print(
        "RANDOM FOREST V3 - EXPANDED DATASET"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Check feature file
    # --------------------------------------------------------

    if not FEATURE_FILE.exists():

        raise FileNotFoundError(
            f"Feature file not found:\n"
            f"{FEATURE_FILE}"
        )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(
        FEATURE_FILE
    )

    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    required_columns = (
        V3_FEATURES
        + [
            "split",
            "issue",
        ]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(
                missing_columns
            )
        )

    # --------------------------------------------------------
    # Create splits
    # --------------------------------------------------------

    train_df = df[
        df["split"] == "train"
    ].copy()

    val_df = df[
        df["split"] == "val"
    ].copy()

    test_df = df[
        df["split"] == "test"
    ].copy()

    # --------------------------------------------------------
    # Dataset summary
    # --------------------------------------------------------

    print()

    print(
        f"Total samples      : {len(df)}"
    )

    print(
        f"Training samples   : {len(train_df)}"
    )

    print(
        f"Validation samples : {len(val_df)}"
    )

    print(
        f"Test samples       : {len(test_df)}"
    )

    print()

    print(
        f"Features used      : {len(V3_FEATURES)}"
    )

    print()

    for feature in V3_FEATURES:

        print(
            f"  - {feature}"
        )

    # ========================================================
    # CLASS DISTRIBUTION
    # ========================================================

    print()

    print("=" * 60)

    print(
        "TRAIN CLASS DISTRIBUTION"
    )

    print("=" * 60)

    print(
        train_df[
            "issue"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()

    print("=" * 60)

    print(
        "VALIDATION CLASS DISTRIBUTION"
    )

    print("=" * 60)

    print(
        val_df[
            "issue"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()

    print("=" * 60)

    print(
        "TEST CLASS DISTRIBUTION"
    )

    print("=" * 60)

    print(
        test_df[
            "issue"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # ========================================================
    # TRAIN
    # ========================================================

    print()

    print("=" * 60)

    print(
        "TRAINING"
    )

    print("=" * 60)

    model = train_model(
        train_df
    )

    print(
        "Training complete."
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    validation_accuracy, validation_macro_f1, _ = (
        evaluate_model(
            model,
            val_df,
        )
    )

    print()

    print("=" * 60)

    print(
        "VALIDATION RESULTS"
    )

    print("=" * 60)

    print(
        f"Accuracy : {validation_accuracy:.4f}"
    )

    print(
        f"Macro F1 : {validation_macro_f1:.4f}"
    )

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    importance = pd.Series(
        model.feature_importances_,
        index=V3_FEATURES,
    ).sort_values(
        ascending=False
    )

    print()

    print("=" * 60)

    print(
        "FEATURE IMPORTANCE"
    )

    print("=" * 60)

    print(
        importance.to_string()
    )

    # ========================================================
    # SAVE EXPANDED MODEL
    # ========================================================

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_file = (
        MODEL_DIR
        / "random_forest_v3_expanded.pkl"
    )

    joblib.dump(
        model,
        model_file,
    )

    # --------------------------------------------------------
    # Save feature list
    # --------------------------------------------------------

    feature_file = (
        MODEL_DIR
        / "random_forest_v3_expanded_features.txt"
    )

    with open(
        feature_file,
        "w",
    ) as f:

        for feature in V3_FEATURES:

            f.write(
                feature + "\n"
            )

    # ========================================================
    # SAVE TRAINING RESULTS
    # ========================================================

    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = pd.DataFrame(
        [
            {
                "experiment":
                    "V3_expanded_9500",

                "total_samples":
                    len(df),

                "train_samples":
                    len(train_df),

                "validation_samples":
                    len(val_df),

                "test_samples":
                    len(test_df),

                "num_features":
                    len(V3_FEATURES),

                "validation_accuracy":
                    validation_accuracy,

                "validation_macro_f1":
                    validation_macro_f1,
            }
        ]
    )

    results_file = (
        EVALUATION_DIR
        / "v3_expanded_training_results.csv"
    )

    results.to_csv(
        results_file,
        index=False,
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()

    print("=" * 60)

    print(
        "EXPANDED V3 TRAINING COMPLETE"
    )

    print("=" * 60)

    print(
        f"Model saved:"
        f"\n{model_file}"
    )

    print()

    print(
        f"Feature list:"
        f"\n{feature_file}"
    )

    print()

    print(
        f"Results:"
        f"\n{results_file}"
    )

    print()

    print(
        "Existing production model was NOT modified:"
    )

    print(
        ROOT
        / "models"
        / "random_forest_v3.pkl"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()