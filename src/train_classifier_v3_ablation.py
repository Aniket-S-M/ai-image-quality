from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


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


# ==================================================
# ALL V2 FEATURES
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


# ==================================================
# EXPERIMENTS
# ==================================================

EXPERIMENTS = {

    # V2 baseline
    "V2_all_11": ALL_FEATURES,

    # Remove one correlated feature
    "remove_blockiness": [
        f for f in ALL_FEATURES
        if f != "blockiness"
    ],

    "remove_edge_density": [
        f for f in ALL_FEATURES
        if f != "edge_density"
    ],

    "remove_noise_estimate": [
        f for f in ALL_FEATURES
        if f != "noise_estimate"
    ],

    # Remove highly redundant structural features
    "remove_edge_and_blockiness": [
        f for f in ALL_FEATURES
        if f not in [
            "edge_density",
            "blockiness",
        ]
    ],

    # Exposure redundancy experiments
    "remove_dark_pixel_ratio": [
        f for f in ALL_FEATURES
        if f != "dark_pixel_ratio"
    ],

    "remove_bright_pixel_ratio": [
        f for f in ALL_FEATURES
        if f != "bright_pixel_ratio"
    ],
}


# ==================================================
# TRAIN + EVALUATE
# ==================================================

def evaluate_feature_set(
    name,
    features,
    train_df,
    test_df
):

    X_train = train_df[
        features
    ]

    y_train = train_df[
        "issue"
    ]

    X_test = test_df[
        features
    ]

    y_test = test_df[
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

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )

    return {
        "experiment": name,
        "num_features": len(features),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }


# ==================================================
# MAIN
# ==================================================

def main():

    print("===================================")
    print("FEATURE ABLATION EXPERIMENT")
    print("===================================")

    df = pd.read_csv(
        FEATURE_FILE
    )

    train_df = df[
        df["split"] == "train"
    ].copy()

    test_df = df[
        df["split"] == "test"
    ].copy()

    print(
        f"Train samples: {len(train_df)}"
    )

    print(
        f"Test samples : {len(test_df)}"
    )

    results = []

    # ----------------------------------------------
    # Run experiments
    # ----------------------------------------------

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

        result = evaluate_feature_set(
            name,
            features,
            train_df,
            test_df
        )

        results.append(
            result
        )

        print(
            f"Accuracy: "
            f"{result['accuracy']:.4f}"
        )

        print(
            f"Macro F1: "
            f"{result['macro_f1']:.4f}"
        )

    # ----------------------------------------------
    # Results table
    # ----------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        "macro_f1",
        ascending=False
    )

    print(
        "\n==================================="
    )

    print(
        "ABLATION RESULTS"
    )

    print(
        "==================================="
    )

    print(
        results_df.to_string(
            index=False,
            formatters={
                "accuracy":
                    "{:.4f}".format,

                "macro_f1":
                    "{:.4f}".format,
            }
        )
    )

    # ----------------------------------------------
    # Save results
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

    output_file = (
        output_dir
        / "feature_ablation_results.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print(
        "\nResults saved:"
    )

    print(
        output_file
    )

    print(
        "\n==================================="
    )

    print(
        "ABLATION COMPLETE"
    )

    print(
        "==================================="
    )


if __name__ == "__main__":
    main()