from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score


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


# ==================================================
# FEATURE SETS
# ==================================================

ALL_FEATURES = [
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


# Candidate V3 feature sets

V3_A_FEATURES = [
    feature
    for feature in ALL_FEATURES
    if feature != "edge_density"
]


V3_B_FEATURES = [
    feature
    for feature in ALL_FEATURES
    if feature != "noise_estimate"
]


EXPERIMENTS = {
    "V3_A_remove_edge_density":
        V3_A_FEATURES,

    "V3_B_remove_noise_estimate":
        V3_B_FEATURES,
}


# ==================================================
# TRAIN MODEL
# ==================================================

def train_model(
    features,
    train_df
):

    X_train = train_df[
        features
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
        y_train
    )

    return model


# ==================================================
# EVALUATE
# ==================================================

def evaluate_model(
    model,
    features,
    dataframe
):

    X = dataframe[
        features
    ]

    y = dataframe[
        "issue"
    ]

    predictions = model.predict(
        X
    )

    accuracy = accuracy_score(
        y,
        predictions
    )

    macro_f1 = f1_score(
        y,
        predictions,
        average="macro"
    )

    return (
        accuracy,
        macro_f1,
        predictions
    )


# ==================================================
# MAIN
# ==================================================

def main():

    print(
        "==================================="
    )

    print(
        "RANDOM FOREST V3 FEATURE SELECTION"
    )

    print(
        "==================================="
    )

    # ----------------------------------------------
    # Load data
    # ----------------------------------------------

    df = pd.read_csv(
        FEATURE_FILE
    )

    train_df = df[
        df["split"] == "train"
    ].copy()

    val_df = df[
        df["split"] == "val"
    ].copy()

    print(
        f"Total records: {len(df)}"
    )

    print(
        f"Train: {len(train_df)}"
    )

    print(
        f"Validation: {len(val_df)}"
    )

    # ----------------------------------------------
    # Validation experiments
    # ----------------------------------------------

    results = []

    trained_models = {}

    for name, features in EXPERIMENTS.items():

        print(
            "\n-----------------------------------"
        )

        print(
            f"Experiment: {name}"
        )

        print(
            f"Features: {len(features)}"
        )

        model = train_model(
            features,
            train_df
        )

        accuracy, macro_f1, predictions = (
            evaluate_model(
                model,
                features,
                val_df
            )
        )

        print(
            f"Validation Accuracy: "
            f"{accuracy:.4f}"
        )

        print(
            f"Validation Macro F1: "
            f"{macro_f1:.4f}"
        )

        results.append(
            {
                "experiment": name,
                "num_features": len(features),
                "validation_accuracy": accuracy,
                "validation_macro_f1": macro_f1,
            }
        )

        trained_models[name] = (
            model,
            features
        )

    # ----------------------------------------------
    # Results
    # ----------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        "validation_macro_f1",
        ascending=False
    )

    print(
        "\n==================================="
    )

    print(
        "V3 VALIDATION COMPARISON"
    )

    print(
        "==================================="
    )

    print(
        results_df.to_string(
            index=False,
            formatters={
                "validation_accuracy":
                    "{:.4f}".format,

                "validation_macro_f1":
                    "{:.4f}".format,
            }
        )
    )

    # ----------------------------------------------
    # Select best feature set
    # ----------------------------------------------

    best_name = results_df.iloc[
        0
    ]["experiment"]

    best_features = trained_models[
        best_name
    ][1]

    best_model = trained_models[
        best_name
    ][0]

    print(
        "\n==================================="
    )

    print(
        "SELECTED V3 MODEL"
    )

    print(
        "==================================="
    )

    print(
        f"Selected experiment: "
        f"{best_name}"
    )

    print(
        f"Selected features: "
        f"{len(best_features)}"
    )

    print(
        "\nFeatures:"
    )

    for feature in best_features:

        print(
            f"  - {feature}"
        )

    # ----------------------------------------------
    # Feature importance
    # ----------------------------------------------

    importance = pd.Series(
        best_model.feature_importances_,
        index=best_features
    ).sort_values(
        ascending=False
    )

    print(
        "\n==================================="
    )

    print(
        "SELECTED MODEL FEATURE IMPORTANCE"
    )

    print(
        "==================================="
    )

    print(
        importance
    )

    # ----------------------------------------------
    # Save selected model
    # ----------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model_file = (
        MODEL_DIR
        / "random_forest_v3.pkl"
    )

    joblib.dump(
        best_model,
        model_file
    )

    # ----------------------------------------------
    # Save selected feature list
    # ----------------------------------------------

    feature_file = (
        MODEL_DIR
        / "random_forest_v3_features.txt"
    )

    with open(
        feature_file,
        "w"
    ) as f:

        for feature in best_features:

            f.write(
                feature + "\n"
            )

    # ----------------------------------------------
    # Save validation results
    # ----------------------------------------------

    evaluation_dir = (
        ROOT
        / "data"
        / "evaluation"
    )

    evaluation_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    results_file = (
        evaluation_dir
        / "v3_validation_results.csv"
    )

    results_df.to_csv(
        results_file,
        index=False
    )

    print(
        "\n==================================="
    )

    print(
        "V3 MODEL SAVED"
    )

    print(
        "==================================="
    )

    print(
        f"Model: {model_file}"
    )

    print(
        f"Features: {feature_file}"
    )

    print(
        f"Results: {results_file}"
    )


if __name__ == "__main__":
    main()