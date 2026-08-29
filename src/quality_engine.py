from pathlib import Path

import joblib
import pandas as pd

from src.extract_features_v2 import extract_features


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "models"

RF_MODEL_FILE = MODEL_DIR / "random_forest_v3_expanded.pkl"
RF_FEATURE_FILE = MODEL_DIR / "random_forest_v3_expanded_features.txt"

IF_MODEL_FILE = MODEL_DIR / "isolation_forest.pkl"
IF_SCALER_FILE = MODEL_DIR / "isolation_forest_scaler.pkl"
IF_FEATURE_FILE = MODEL_DIR / "isolation_forest_features.txt"


# ============================================================
# LOAD MODELS
# ============================================================

RF_MODEL = joblib.load(
    RF_MODEL_FILE
)

ISOLATION_FOREST = joblib.load(
    IF_MODEL_FILE
)

IF_SCALER = joblib.load(
    IF_SCALER_FILE
)


# ============================================================
# LOAD FEATURE LISTS
# ============================================================

with open(
    RF_FEATURE_FILE,
    "r"
) as f:

    RF_FEATURES = [
        line.strip()
        for line in f
        if line.strip()
    ]


with open(
    IF_FEATURE_FILE,
    "r"
) as f:

    IF_FEATURES = [
        line.strip()
        for line in f
        if line.strip()
    ]


# ============================================================
# CONSTANTS
# ============================================================

ISSUES = {
    "blur",
    "corruption",
    "noise",
    "overexposure",
    "underexposure",
}


# ============================================================
# RANDOM FOREST PREDICTION
# ============================================================

def predict_issue(features):

    values = {
        feature: features[feature]
        for feature in RF_FEATURES
    }

    X = pd.DataFrame(
        [values],
        columns=RF_FEATURES,
    )

    X = X.fillna(0)

    prediction = RF_MODEL.predict(
        X
    )[0]

    probabilities = (
        RF_MODEL.predict_proba(
            X
        )[0]
    )

    classes = RF_MODEL.classes_

    probability_map = {
        str(classes[i]):
        float(probabilities[i])
        for i in range(
            len(classes)
        )
    }

    confidence = probability_map.get(
        str(prediction),
        0.0,
    )

    return (
        str(prediction),
        confidence,
        probability_map,
    )


# ============================================================
# SEVERITY ESTIMATION
# ============================================================

def estimate_severity(
    issue,
    features,
):

    # --------------------------------------------------------
    # BLUR
    # --------------------------------------------------------

    if issue == "blur":

        sharpness = features[
            "sharpness"
        ]

        if sharpness < 3:

            return "high", 1.0

        if sharpness <= 8:

            return "medium", 0.85

        if sharpness <= 25:

            return "medium", 0.70

        return "low", 0.70


    # --------------------------------------------------------
    # NOISE
    # --------------------------------------------------------

    if issue == "noise":

        noise = features[
            "noise_estimate"
        ]

        if noise >= 23:

            return "high", 1.0

        if noise >= 16:

            return "medium", 0.85

        if noise >= 11:

            return "low", 0.70

        return "low", 0.50


    # --------------------------------------------------------
    # OVEREXPOSURE
    # --------------------------------------------------------

    if issue == "overexposure":

        clipping = features[
            "highlight_clipping"
        ]

        brightness = features[
            "brightness"
        ]

        if (
            clipping >= 42
            or brightness >= 195
        ):

            return "high", 1.0

        if (
            clipping >= 25
            or brightness >= 170
        ):

            return "medium", 0.85

        if (
            clipping >= 8
            or brightness >= 145
        ):

            return "low", 0.70

        return "low", 0.50


    # --------------------------------------------------------
    # UNDEREXPOSURE
    # --------------------------------------------------------

    if issue == "underexposure":

        brightness = features[
            "brightness"
        ]

        dark_ratio = features[
            "dark_pixel_ratio"
        ]

        if (
            brightness < 45
            or dark_ratio > 65
        ):

            return "high", 1.0

        if (
            brightness <= 75
            or dark_ratio >= 40
        ):

            return "medium", 0.85

        if (
            brightness <= 105
            or dark_ratio >= 25
        ):

            return "low", 0.70

        return "low", 0.50


    # --------------------------------------------------------
    # CORRUPTION
    # --------------------------------------------------------

    if issue == "corruption":

        blockiness = features[
            "blockiness"
        ]

        if blockiness >= 15:

            return "high", 0.85

        if blockiness >= 10:

            return "medium", 0.70

        if blockiness >= 7:

            return "low", 0.55

        return "low", 0.40


    return "none", 0.0


# ============================================================
# ISOLATION FOREST
# ============================================================

def detect_anomaly(features):

    values = {
        feature: features[feature]
        for feature in IF_FEATURES
    }

    X = pd.DataFrame(
        [values],
        columns=IF_FEATURES,
    )

    X = X.fillna(0)

    X_scaled = IF_SCALER.transform(
        X
    )

    prediction = (
        ISOLATION_FOREST.predict(
            X_scaled
        )[0]
    )

    anomaly_score = (
        ISOLATION_FOREST.decision_function(
            X_scaled
        )[0]
    )

    # IMPORTANT:
    # Convert numpy.bool_ -> Python bool
    # so FastAPI can serialize it as JSON.

    is_anomaly = bool(
        prediction == -1
    )

    return (
        is_anomaly,
        float(anomaly_score),
    )


# ============================================================
# QUALITY SCORE
# ============================================================

def calculate_quality_score(
    issue,
    severity,
    classifier_confidence,
    anomaly=False,
):

    if issue == "none":

        if anomaly:

            return 70

        return 100


    penalties = {
        "low": 15,
        "medium": 40,
        "high": 70,
    }

    base_penalty = penalties.get(
        severity,
        30,
    )

    confidence = max(
        0.0,
        min(
            1.0,
            float(
                classifier_confidence
            ),
        ),
    )

    confidence_factor = (
        0.60
        + 0.40 * confidence
    )

    penalty = (
        base_penalty
        * confidence_factor
    )

    if anomaly:

        penalty += 5

    score = 100 - penalty

    return int(
        round(
            max(
                0,
                min(
                    100,
                    score,
                ),
            )
        )
    )


# ============================================================
# QUALITY LABEL
# ============================================================

def get_quality_label(score):

    if score >= 75:

        return "ACCEPTABLE"

    if score >= 45:

        return "DEGRADED"

    return "POTENTIALLY_DEFECTIVE"


# ============================================================
# ACCEPT / REJECT
# ============================================================

def get_decision(score):

    if score >= 75:

        return "ACCEPTED"

    return "REJECTED"


# ============================================================
# IMAGE STATISTICS
# ============================================================

def get_image_statistics(features):

    """
    Convert extracted CV features into
    frontend-friendly JSON-compatible values.
    """

    statistics = {}

    feature_names = [

        "sharpness",

        "brightness",

        "contrast",

        "saturation",

        "edge_density",

        "noise_estimate",

        "highlight_clipping",

        "shadow_clipping",

        "dark_pixel_ratio",

        "bright_pixel_ratio",

        "blockiness",

    ]

    for feature in feature_names:

        if feature not in features:

            continue

        value = features[
            feature
        ]

        try:

            value = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):

            continue

        statistics[feature] = round(
            value,
            3,
        )

    return statistics


# ============================================================
# FINAL ASSESSMENT
# ============================================================

def assess_image(
    image_path
):

    # --------------------------------------------------------
    # Extract CV features
    # --------------------------------------------------------

    features = extract_features(
        image_path
    )


    # --------------------------------------------------------
    # Random Forest V3 prediction
    # --------------------------------------------------------

    (
        prediction,
        confidence,
        probabilities,
    ) = predict_issue(
        features
    )


    # --------------------------------------------------------
    # Isolation Forest
    # --------------------------------------------------------

    (
        is_anomaly,
        anomaly_score,
    ) = detect_anomaly(
        features
    )


    # --------------------------------------------------------
    # Image statistics
    # --------------------------------------------------------

    image_statistics = (
        get_image_statistics(
            features
        )
    )


    # ========================================================
    # CLEAN IMAGE
    # ========================================================

    if prediction == "none":

        if not is_anomaly:

            score = 100

            return {

                "quality_score": score,

                "quality_label":
                    "ACCEPTABLE",

                "decision":
                    "ACCEPTED",

                "issues": [],

                "image_statistics":
                    image_statistics,

                "anomaly": {

                    "detected":
                        bool(False),

                    "score":
                        float(
                            round(
                                anomaly_score,
                                4,
                            )
                        ),
                },
            }


        # ----------------------------------------------------
        # RF says clean but Isolation Forest
        # detects unusual feature pattern.
        # ----------------------------------------------------

        anomaly_confidence = max(
            0.0,
            min(
                1.0,
                1.0 - (
                    anomaly_score
                    + 0.5
                ),
            ),
        )

        score = 70

        return {

            "quality_score": score,

            "quality_label":
                "DEGRADED",

            "decision":
                "REJECTED",

            "issues": [

                {

                    "type":
                        "potential_anomaly",

                    "severity":
                        "low",

                    "confidence":
                        round(
                            float(
                                anomaly_confidence
                            ),
                            4,
                        ),
                }

            ],

            "image_statistics":
                image_statistics,

            "anomaly": {

                "detected":
                    bool(True),

                "score":
                    float(
                        round(
                            anomaly_score,
                            4,
                        )
                    ),
            },
        }


    # ========================================================
    # UNKNOWN CLASS
    # ========================================================

    if prediction not in ISSUES:

        score = 50

        return {

            "quality_score": score,

            "quality_label":
                "DEGRADED",

            "decision":
                "REJECTED",

            "issues": [],

            "image_statistics":
                image_statistics,

            "anomaly": {

                "detected":
                    bool(is_anomaly),

                "score":
                    float(
                        round(
                            anomaly_score,
                            4,
                        )
                    ),
            },
        }


    # ========================================================
    # SEVERITY
    # ========================================================

    severity, severity_confidence = (
        estimate_severity(
            prediction,
            features,
        )
    )


    # ========================================================
    # QUALITY SCORE
    # ========================================================

    score = calculate_quality_score(
        issue=prediction,
        severity=severity,
        classifier_confidence=confidence,
        anomaly=is_anomaly,
    )


    # ========================================================
    # QUALITY LABEL
    # ========================================================

    label = get_quality_label(
        score
    )


    # ========================================================
    # ACCEPT / REJECT
    # ========================================================

    decision = get_decision(
        score
    )


    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    return {

        "quality_score":
            int(score),

        "quality_label":
            str(label),

        "decision":
            str(decision),

        "issues": [

            {

                "type":
                    str(prediction),

                "severity":
                    str(severity),

                "confidence":
                    float(
                        round(
                            confidence,
                            4,
                        )
                    ),
            }

        ],

        "image_statistics":
            image_statistics,

        "anomaly": {

            "detected":
                bool(is_anomaly),

            "score":
                float(
                    round(
                        anomaly_score,
                        4,
                    )
                ),
        },
    }