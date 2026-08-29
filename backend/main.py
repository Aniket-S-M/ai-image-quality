from io import BytesIO
import os
import tempfile

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
)

from fastapi.middleware.cors import CORSMiddleware

from PIL import Image

from src.quality_engine import assess_image

from backend.database import (
    save_analysis,
    get_all_analyses,
    get_analysis,
)


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="AI Image Quality API",
    description=(
        "AI-powered image quality assessment API using "
        "computer vision features, Random Forest V3 Expanded, "
        "and Isolation Forest anomaly detection."
    ),
    version="1.1.0",
)


# ============================================================
# MODEL INFORMATION
# ============================================================

MODEL_NAME = "Random Forest V3 Expanded"
MODEL_VERSION = "v3-expanded"

TRAINING_SAMPLES = 6650
VALIDATION_SAMPLES = 1425
TEST_SAMPLES = 1425

TEST_ACCURACY = 0.8821
TEST_MACRO_F1 = 0.8364

RF_FEATURE_COUNT = 10

ANOMALY_DETECTOR = "Isolation Forest"

CLASS_NAMES = [
    "blur",
    "corruption",
    "noise",
    "none",
    "overexposure",
    "underexposure",
]


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "model": "random_forest_v3_expanded",
        "model_version": MODEL_VERSION,
        "anomaly_detector": "isolation_forest",
    }


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.get("/model")
def model_info():

    return {
        "model": MODEL_NAME,
        "version": MODEL_VERSION,

        "training_samples":
            TRAINING_SAMPLES,

        "validation_samples":
            VALIDATION_SAMPLES,

        "test_samples":
            TEST_SAMPLES,

        "test_accuracy":
            TEST_ACCURACY,

        "test_macro_f1":
            TEST_MACRO_F1,

        "anomaly_detector":
            ANOMALY_DETECTOR,

        "classes":
            CLASS_NAMES,

        "features":
            RF_FEATURE_COUNT,

        "feature_selection":
            "V3 selected features; noise_estimate excluded from RF classifier",

        "quality_threshold":
            75,
    }


# ============================================================
# IMAGE VALIDATION
# ============================================================

def read_uploaded_image(contents):

    try:

        image = Image.open(
            BytesIO(contents)
        )

        # ----------------------------------------------------
        # Verify image integrity
        # ----------------------------------------------------

        image.verify()

        # verify() invalidates the image object,
        # therefore reopen it.
        image = Image.open(
            BytesIO(contents)
        )

        return image.convert("RGB")

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Invalid image file.",
        )


# ============================================================
# FINAL QUALITY ANALYSIS
# ============================================================

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...)
):

    # ========================================================
    # VALIDATE FILE TYPE
    # ========================================================

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp",
    }

    if file.content_type not in allowed_types:

        raise HTTPException(
            status_code=400,
            detail=(
                "Please upload a JPG, PNG, "
                "or WEBP image."
            ),
        )

    # ========================================================
    # READ UPLOADED FILE
    # ========================================================

    contents = await file.read()

    if not contents:

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    # ========================================================
    # VALIDATE IMAGE CONTENT
    # ========================================================

    image = read_uploaded_image(
        contents
    )

    # ========================================================
    # TEMPORARY IMAGE
    #
    # The quality engine expects an image path.
    # ========================================================

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            suffix=".jpg",
            delete=False,
        ) as temp:

            temp_path = temp.name

            image.save(
                temp_path,
                format="JPEG",
            )

        # ====================================================
        # COMPLETE QUALITY PIPELINE
        # ====================================================

        assessment = assess_image(
            temp_path
        )

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Quality analysis failed: {e}"
            ),
        )

    finally:

        # ====================================================
        # DELETE TEMPORARY FILE
        # ====================================================

        if temp_path is not None:

            try:

                if os.path.exists(
                    temp_path
                ):

                    os.remove(
                        temp_path
                    )

            except Exception:

                pass

    # ========================================================
    # EXTRACT ASSESSMENT
    # ========================================================

    quality_score = assessment.get(
        "quality_score",
        0,
    )

    quality_label = assessment.get(
        "quality_label",
        "UNKNOWN",
    )

    issues = assessment.get(
        "issues",
        [],
    )

    image_statistics = assessment.get(
        "image_statistics",
        {},
    )

    anomaly = assessment.get(
        "anomaly",
        {
            "detected": False,
            "score": 0.0,
        },
    )

    # ========================================================
    # FINAL DECISION
    # ========================================================

    decision = assessment.get(
        "decision"
    )

    if decision is None:

        if quality_score >= 75:

            decision = "ACCEPTED"

        else:

            decision = "REJECTED"

    # ========================================================
    # SAVE TO SQLITE
    # ========================================================

    try:

        analysis_id = save_analysis(
            filename=file.filename,
            quality_score=quality_score,
            quality_label=quality_label,
            issues=issues,
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Could not save analysis: {e}"
            ),
        )

    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    return {
        "id":
            analysis_id,

        "filename":
            file.filename,

        "quality_score":
            quality_score,

        "quality_label":
            quality_label,

        "decision":
            decision,

        "issues":
            issues,

        "image_statistics":
            image_statistics,

        "anomaly":
            anomaly,
    }


# ============================================================
# ANALYSIS HISTORY
# ============================================================

@app.get("/analyses")
def analyses():

    records = get_all_analyses()

    return {
        "count":
            len(records),

        "analyses":
            records,
    }


# ============================================================
# SINGLE ANALYSIS
# ============================================================

@app.get("/analyses/{analysis_id}")
def analysis(
    analysis_id: int
):

    result = get_analysis(
        analysis_id
    )

    if result is None:

        raise HTTPException(
            status_code=404,
            detail="Analysis not found.",
        )

    return result


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "name":
            "AI Image Quality API",

        "version":
            "1.1.0",

        "status":
            "running",

        "model":
            MODEL_NAME,

        "main_endpoint":
            "/analyze",

        "history_endpoint":
            "/analyses",

        "health_endpoint":
            "/health",

        "model_endpoint":
            "/model",
    }