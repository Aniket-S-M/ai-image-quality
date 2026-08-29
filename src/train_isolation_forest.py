from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


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

MODEL_DIR = ROOT / "models"

MODEL_FILE = (
    MODEL_DIR
    / "isolation_forest.pkl"
)

SCALER_FILE = (
    MODEL_DIR
    / "isolation_forest_scaler.pkl"
)

FEATURE_LIST_FILE = (
    MODEL_DIR
    / "isolation_forest_features.txt"
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
    print("ISOLATION FOREST TRAINING")
    print("=" * 70)

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

    print(
        f"Total records : {len(df)}"
    )

    # --------------------------------------------------------
    # TRAINING DATA
    #
    # IMPORTANT:
    # Isolation Forest learns NORMAL image-quality
    # behaviour from clean images only.
    # --------------------------------------------------------

    train_df = df[
        (df["split"] == "train")
        & (df["issue"] == "none")
    ].copy()

    print(
        f"Clean training images : {len(train_df)}"
    )

    if len(train_df) == 0:

        raise ValueError(
            "No clean training images found."
        )

    # --------------------------------------------------------
    # Extract features
    # --------------------------------------------------------

    X_train = train_df[
        FEATURES
    ].copy()

    # --------------------------------------------------------
    # Handle missing values
    # --------------------------------------------------------

    if X_train.isnull().any().any():

        print(
            "[WARNING] Missing feature values detected."
        )

        X_train = X_train.fillna(
            X_train.median()
        )

    # --------------------------------------------------------
    # Standardization
    #
    # Important because feature scales are very different:
    #
    # sharpness       -> thousands
    # brightness      -> hundreds
    # clipping        -> percentage
    # blockiness      -> small values
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X_train
    )

    # --------------------------------------------------------
    # Train Isolation Forest
    # --------------------------------------------------------

    model = IsolationForest(
        n_estimators=300,
        contamination=0.05,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_scaled
    )

    # --------------------------------------------------------
    # Training sanity check
    # --------------------------------------------------------

    predictions = model.predict(
        X_scaled
    )

    anomalies = (
        predictions == -1
    ).sum()

    normal = (
        predictions == 1
    ).sum()

    print()
    print(
        "TRAINING SANITY CHECK"
    )
    print("-" * 70)

    print(
        f"Normal samples    : {normal}"
    )

    print(
        f"Anomalous samples : {anomalies}"
    )

    print(
        f"Anomaly rate      : "
        f"{anomalies / len(predictions):.4f}"
    )

    # --------------------------------------------------------
    # Save model artifacts
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_FILE
    )

    joblib.dump(
        scaler,
        SCALER_FILE
    )

    with open(
        FEATURE_LIST_FILE,
        "w"
    ) as f:

        for feature in FEATURES:

            f.write(
                feature + "\n"
            )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ISOLATION FOREST SAVED")
    print("=" * 70)

    print(
        f"Model  : {MODEL_FILE}"
    )

    print(
        f"Scaler : {SCALER_FILE}"
    )

    print(
        f"Features : {FEATURE_LIST_FILE}"
    )


if __name__ == "__main__":
    main()