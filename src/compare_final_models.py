from pathlib import Path

import joblib
import pandas as pd
import torch
import torch.nn as nn

from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from torchvision import transforms
from torchvision.models import resnet18


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

RF_MODEL_FILE = (
    ROOT
    / "models"
    / "random_forest_v3.pkl"
)

RF_FEATURE_FILE = (
    ROOT
    / "models"
    / "random_forest_v3_features.txt"
)

RESNET_MODEL_FILE = (
    ROOT
    / "best_resnet18_finetuned.pth"
)

DATASET_DIR = (
    ROOT
    / "data"
    / "generated"
    / "dataset_expanded"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "evaluation"
    / "final_model_comparison.csv"
)


# ============================================================
# CLASS ORDER
# ============================================================

CLASS_NAMES = [
    "blur",
    "corruption",
    "noise",
    "none",
    "overexposure",
    "underexposure",
]


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# RESNET PREPROCESSING
# ============================================================

TRANSFORM = transforms.Compose([
    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406,
        ],
        std=[
            0.229,
            0.224,
            0.225,
        ],
    ),
])


# ============================================================
# LOAD RESNET
# ============================================================

def load_resnet():

    model = resnet18(
        weights=None
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        len(CLASS_NAMES),
    )

    state_dict = torch.load(
        RESNET_MODEL_FILE,
        map_location=DEVICE,
    )

    model.load_state_dict(
        state_dict
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    return model


# ============================================================
# RESNET PREDICTION
# ============================================================

def predict_resnet(
    model,
    image_path,
):

    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )

    tensor = TRANSFORM(
        image
    )

    tensor = tensor.unsqueeze(
        0
    )

    tensor = tensor.to(
        DEVICE
    )

    with torch.no_grad():

        outputs = model(
            tensor
        )

        predictions = torch.argmax(
            outputs,
            dim=1,
        )

    return CLASS_NAMES[
        predictions.item()
    ]


# ============================================================
# LOAD RF FEATURES
# ============================================================

def load_rf_features():

    with open(
        RF_FEATURE_FILE,
        "r"
    ) as f:

        features = [
            line.strip()
            for line in f
            if line.strip()
        ]

    return features


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("FINAL RANDOM FOREST vs RESNET-18 COMPARISON")
    print("=" * 72)

    # --------------------------------------------------------
    # Load feature dataframe
    # --------------------------------------------------------

    df = pd.read_csv(
        FEATURE_FILE
    )

    test_df = df[
        df["split"] == "test"
    ].copy()

    print()
    print(
        f"Test images : {len(test_df)}"
    )

    # --------------------------------------------------------
    # Load RF
    # --------------------------------------------------------

    print()
    print(
        "Loading Random Forest..."
    )

    rf_model = joblib.load(
        RF_MODEL_FILE
    )

    rf_features = load_rf_features()

    print(
        f"RF features : {len(rf_features)}"
    )

    # --------------------------------------------------------
    # RF prediction
    # --------------------------------------------------------

    X_test = test_df[
        rf_features
    ].copy()

    X_test = X_test.fillna(
        X_test.median()
    )

    rf_predictions = rf_model.predict(
        X_test
    )

    # --------------------------------------------------------
    # ResNet
    # --------------------------------------------------------

    print(
        "Loading ResNet-18..."
    )

    resnet_model = load_resnet()

    print(
        "Running ResNet on test set..."
    )

    resnet_predictions = []

    for index, row in test_df.iterrows():

        image_path = (
            DATASET_DIR
            / "test"
            / row["filename"]
        )

        prediction = predict_resnet(
            resnet_model,
            image_path,
        )

        resnet_predictions.append(
            prediction
        )

        if (
            len(resnet_predictions)
            % 250
            == 0
        ):

            print(
                f"  Processed "
                f"{len(resnet_predictions)}"
                f"/{len(test_df)}"
            )

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    y_true = test_df[
        "issue"
    ].tolist()

    # --------------------------------------------------------
    # Metrics helper
    # --------------------------------------------------------

    def calculate_metrics(
        name,
        predictions,
    ):

        accuracy = accuracy_score(
            y_true,
            predictions
        )

        precision = precision_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0,
        )

        macro_f1 = f1_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0,
        )

        weighted_f1 = f1_score(
            y_true,
            predictions,
            average="weighted",
            zero_division=0,
        )

        return {
            "model": name,
            "accuracy": accuracy,
            "macro_precision": precision,
            "macro_recall": recall,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        }

    # --------------------------------------------------------
    # Calculate results
    # --------------------------------------------------------

    results = []

    results.append(
        calculate_metrics(
            "Random Forest V3",
            rf_predictions,
        )
    )

    results.append(
        calculate_metrics(
            "ResNet-18 Fine-tuned",
            resnet_predictions,
        )
    )

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Print comparison
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("MODEL COMPARISON")
    print("=" * 72)

    print(
        results_df.to_string(
            index=False,
            formatters={
                "accuracy":
                    "{:.4f}".format,

                "macro_precision":
                    "{:.4f}".format,

                "macro_recall":
                    "{:.4f}".format,

                "macro_f1":
                    "{:.4f}".format,

                "weighted_f1":
                    "{:.4f}".format,
            }
        )
    )

    # --------------------------------------------------------
    # ResNet detailed report
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("RESNET-18 TEST CLASSIFICATION REPORT")
    print("=" * 72)

    print(
        classification_report(
            y_true,
            resnet_predictions,
            labels=CLASS_NAMES,
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # RF detailed report
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("RANDOM FOREST TEST CLASSIFICATION REPORT")
    print("=" * 72)

    print(
        classification_report(
            y_true,
            rf_predictions,
            labels=CLASS_NAMES,
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Confusion matrices
    # --------------------------------------------------------

    rf_cm = confusion_matrix(
        y_true,
        rf_predictions,
        labels=CLASS_NAMES,
    )

    resnet_cm = confusion_matrix(
        y_true,
        resnet_predictions,
        labels=CLASS_NAMES,
    )

    print()
    print("=" * 72)
    print("RANDOM FOREST CONFUSION MATRIX")
    print("=" * 72)

    print(
        pd.DataFrame(
            rf_cm,
            index=CLASS_NAMES,
            columns=CLASS_NAMES,
        )
    )

    print()
    print("=" * 72)
    print("RESNET-18 CONFUSION MATRIX")
    print("=" * 72)

    print(
        pd.DataFrame(
            resnet_cm,
            index=CLASS_NAMES,
            columns=CLASS_NAMES,
        )
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    comparison_df = test_df[
        [
            "filename",
            "issue",
            "severity",
        ]
    ].copy()

    comparison_df[
        "random_forest_prediction"
    ] = rf_predictions

    comparison_df[
        "resnet_prediction"
    ] = resnet_predictions

    comparison_df[
        "models_agree"
    ] = (
        comparison_df[
            "random_forest_prediction"
        ]
        ==
        comparison_df[
            "resnet_prediction"
        ]
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    predictions_file = (
        ROOT
        / "data"
        / "evaluation"
        / "final_model_predictions.csv"
    )

    comparison_df.to_csv(
        predictions_file,
        index=False
    )

    # --------------------------------------------------------
    # Agreement
    # --------------------------------------------------------

    agreement = (
        comparison_df[
            "models_agree"
        ].mean()
    )

    print()
    print("=" * 72)
    print("MODEL AGREEMENT")
    print("=" * 72)

    print(
        f"RF / ResNet agreement : "
        f"{agreement:.4f}"
    )

    # --------------------------------------------------------
    # Winner
    # --------------------------------------------------------

    best_row = results_df.sort_values(
        "macro_f1",
        ascending=False
    ).iloc[0]

    print()
    print("=" * 72)
    print("BEST MODEL")
    print("=" * 72)

    print(
        f"Selected by Macro-F1 : "
        f"{best_row['model']}"
    )

    print(
        f"Macro-F1             : "
        f"{best_row['macro_f1']:.4f}"
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("FINAL MODEL COMPARISON COMPLETE")
    print("=" * 72)

    print(
        f"Metrics saved to:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        f"Predictions saved to:\n"
        f"{predictions_file}"
    )


if __name__ == "__main__":
    main()