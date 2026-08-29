from pathlib import Path
import shutil

import joblib
import pandas as pd


# ==================================================
# PROJECT PATHS
# ==================================================

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

EVALUATION_DIR = (
    ROOT
    / "data"
    / "evaluation"
)

ERROR_CSV = (
    EVALUATION_DIR
    / "misclassified.csv"
)

ERROR_IMAGE_DIR = (
    EVALUATION_DIR
    / "misclassified"
)


# ==================================================
# FEATURES USED BY RANDOM FOREST
# ==================================================

FEATURE_COLUMNS = [
    "sharpness",
    "brightness",
    "highlight_clipping",
    "contrast",
    "saturation",
    "edge_density",
    "noise_estimate",
]


# ==================================================
# FIND IMAGE FILE
# ==================================================

def find_image(filename):

    # Search generated images first
    generated_dir = (
        ROOT
        / "data"
        / "generated"
    )

    matches = list(
        generated_dir.rglob(filename)
    )

    if matches:
        return matches[0]

    # Search clean images as fallback
    clean_dir = (
        ROOT
        / "data"
        / "raw"
        / "clean"
    )

    clean_path = clean_dir / filename

    if clean_path.exists():
        return clean_path

    return None


# ==================================================
# MAIN
# ==================================================

def main():

    print("===================================")
    print("RANDOM FOREST ERROR ANALYSIS")
    print("===================================")

    # ------------------------------------------------
    # Check required files
    # ------------------------------------------------

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Feature file not found:\n{FEATURE_FILE}"
        )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model file not found:\n{MODEL_FILE}"
        )

    # ------------------------------------------------
    # Load data and model
    # ------------------------------------------------

    df = pd.read_csv(
        FEATURE_FILE
    )

    model = joblib.load(
        MODEL_FILE
    )

    # ------------------------------------------------
    # Select TEST SET ONLY
    # ------------------------------------------------

    test_df = df[
        df["split"] == "test"
    ].copy()

    print(
        f"Test samples: {len(test_df)}"
    )

    # ------------------------------------------------
    # Prepare features
    # ------------------------------------------------

    X_test = test_df[
        FEATURE_COLUMNS
    ]

    y_test = test_df[
        "issue"
    ]

    # ------------------------------------------------
    # Predictions
    # ------------------------------------------------

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )

    # ------------------------------------------------
    # Find confidence of predicted class
    # ------------------------------------------------

    class_names = model.classes_

    predicted_confidence = []

    for prediction, probability in zip(
        predictions,
        probabilities
    ):

        predicted_index = list(
            class_names
        ).index(prediction)

        confidence = probability[
            predicted_index
        ]

        predicted_confidence.append(
            confidence
        )

    # ------------------------------------------------
    # Add predictions to dataframe
    # ------------------------------------------------

    test_df["predicted_issue"] = predictions

    test_df["prediction_confidence"] = (
        predicted_confidence
    )

    # ------------------------------------------------
    # Keep only incorrect predictions
    # ------------------------------------------------

    errors = test_df[
        test_df["issue"]
        != test_df["predicted_issue"]
    ].copy()

    print(
        f"Misclassified samples: {len(errors)}"
    )

    # ------------------------------------------------
    # Create output directories
    # ------------------------------------------------

    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    ERROR_IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------
    # Copy misclassified images
    # ------------------------------------------------

    copied = 0
    missing = 0

    image_paths = []

    for _, row in errors.iterrows():

        filename = row["filename"]

        source_path = find_image(
            filename
        )

        if source_path is None:

            image_paths.append("NOT_FOUND")

            missing += 1

            continue

        destination_name = (
            f"{row['issue']}_TO_"
            f"{row['predicted_issue']}__"
            f"{filename}"
        )

        destination_path = (
            ERROR_IMAGE_DIR
            / destination_name
        )

        shutil.copy2(
            source_path,
            destination_path
        )

        image_paths.append(
            str(destination_path)
        )

        copied += 1

    errors["image_path"] = image_paths

    # ------------------------------------------------
    # Select useful columns
    # ------------------------------------------------

    output_columns = [
        "filename",
        "source_image_id",
        "issue",
        "predicted_issue",
        "severity",
        "degradation",
        "prediction_confidence",
        "sharpness",
        "brightness",
        "highlight_clipping",
        "contrast",
        "saturation",
        "edge_density",
        "noise_estimate",
        "image_path",
    ]

    errors = errors[
        output_columns
    ]

    # ------------------------------------------------
    # Save error CSV
    # ------------------------------------------------

    errors.to_csv(
        ERROR_CSV,
        index=False
    )

    # ------------------------------------------------
    # Print summary
    # ------------------------------------------------

    print("\n===================================")
    print("ERROR SUMMARY")
    print("===================================")

    print(
        "\nActual → Predicted:"
    )

    summary = (
        errors
        .groupby(
            ["issue", "predicted_issue"]
        )
        .size()
        .sort_values(
            ascending=False
        )
    )

    print(
        summary
    )

    print(
        "\nImages copied:"
        f" {copied}"
    )

    print(
        "Images not found:"
        f" {missing}"
    )

    print(
        "\n==================================="
    )

    print(
        "ERROR ANALYSIS COMPLETE"
    )

    print(
        "==================================="
    )

    print(
        f"CSV:\n{ERROR_CSV}"
    )

    print(
        f"Images:\n{ERROR_IMAGE_DIR}"
    )


if __name__ == "__main__":
    main()