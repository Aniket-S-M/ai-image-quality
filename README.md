# AI Image Quality Assessment

An end-to-end computer-vision and machine-learning system for automated image quality assessment.

The system accepts an image, validates it, extracts measurable visual features, classifies known degradation using **Random Forest V3 Expanded**, detects unusual feature patterns using **Isolation Forest**, estimates severity, calculates a quality score, and returns an acceptance/rejection decision through a FastAPI backend and React frontend.

## Key Results

| Metric | Result |
|---|---:|
| Expanded dataset | 9,500 samples |
| Source images | 500 |
| Training samples | 6,650 |
| Validation samples | 1,425 |
| Test samples | 1,425 |
| Production CV features | 10 |
| Test accuracy | **88.21%** |
| Test Macro-F1 | **83.64%** |

### Per-Class F1

| Class | F1 |
|---|---:|
| Blur | 0.95 |
| Corruption | 0.78 |
| Noise | 0.91 |
| None | 0.54 |
| Overexposure | 0.85 |
| Underexposure | 0.98 |

## Architecture

```text
React Frontend
      |
      | HTTP
      v
FastAPI Backend
      |
      v
Image Validation
      |
      v
CV Feature Extraction
      |
      +----------------------+
      |                      |
      v                      v
Random Forest V3       Isolation Forest
Expanded              Anomaly Detection
      |                      |
      +----------+-----------+
                 |
                 v
          Severity Estimation
                 |
                 v
           Quality Score
                 |
                 v
        ACCEPTED / REJECTED
                 |
                 v
             SQLite
```

## Production Pipeline

### 1. Image validation

The API accepts JPEG, PNG, and WEBP images. Uploaded content is validated before processing.

### 2. Computer-vision feature extraction

The production classifier uses 10 measurable features:

- sharpness
- brightness
- highlight clipping
- contrast
- saturation
- edge density
- shadow clipping
- dark pixel ratio
- bright pixel ratio
- blockiness

### 3. Random Forest V3 Expanded

The classifier predicts one of six classes:

- blur
- corruption
- noise
- none
- overexposure
- underexposure

The model was trained on 6,650 training samples and evaluated on a held-out test set of 1,425 samples.

**Test accuracy: 88.21%**

**Test Macro-F1: 83.64%**

### 4. Severity estimation

The predicted degradation and image statistics are used to estimate severity as low, medium, or high.

### 5. Isolation Forest

Isolation Forest is a complementary anomaly-detection layer.

Random Forest answers:

> Which known degradation class does this image resemble?

Isolation Forest answers:

> Is this feature pattern unusual compared with the learned distribution?

An anomaly is therefore an additional signal; it does not automatically mean that an image is defective.

### 6. Quality scoring

The quality engine combines classification, confidence, severity, and anomaly information into a score from 0 to 100.

Current decision threshold:

```text
Score >= 75  -> ACCEPTED
Score < 75   -> REJECTED
```

A clean image classified as `none` without an anomaly can legitimately receive **100/100**.

## Global Feature Importance

The expanded Random Forest reported:

| Feature | Importance |
|---|---:|
| Sharpness | 24.14% |
| Highlight clipping | 16.42% |
| Bright pixel ratio | 15.57% |
| Blockiness | 13.53% |
| Edge density | 9.33% |
| Brightness | 6.07% |
| Contrast | 5.20% |
| Shadow clipping | 3.49% |
| Dark pixel ratio | 3.16% |
| Saturation | 3.10% |

These are **global model-level importances**, not explanations of an individual prediction.

## Backend API

The FastAPI application is in `backend/main.py`.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API information |
| GET | `/health` | Health/model status |
| GET | `/model` | Model metadata |
| POST | `/analyze` | Main production quality analysis |
| GET | `/analyses` | Analysis history |
| GET | `/analyses/{id}` | Retrieve one analysis |
| POST | `/predict` | Legacy/experimental ResNet prediction |

Interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

### Main `/analyze` flow

```text
Upload
  -> file validation
  -> image-content validation
  -> temporary image
  -> CV feature extraction
  -> Random Forest V3 Expanded
  -> severity estimation
  -> Isolation Forest
  -> quality score
  -> accept/reject
  -> SQLite
  -> JSON response
```

### Example

```bash
curl -X POST "http://127.0.0.1:8000/analyze" -F "file=@image.jpg"
```

## ResNet-18 Experimental Endpoint

The repository also contains a ResNet-18 inference pipeline exposed through `/predict`.

This is intentionally separate from the production `/analyze` workflow.

```text
/analyze
    -> CV features
    -> Random Forest V3 Expanded
    -> Isolation Forest
    -> quality assessment

/predict
    -> ResNet-18
    -> degradation prediction
    -> confidence/probabilities
```

The production application therefore uses Random Forest V3 Expanded + Isolation Forest, while ResNet-18 is retained as an experimental/alternative model.

## Frontend

The frontend uses:

- React 19
- Vite 6
- Lucide React

The interface provides image upload, quality results, issue/severity information, image statistics, anomaly information, model/evaluation information, and analysis history.

## Database

SQLite stores analysis history in:

```text
data/quality_analysis.db
```

The Docker Compose configuration mounts a named volume for `/app/data` so history persists when the backend container is recreated.

## Local Setup

### Backend

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

API:

```text
http://127.0.0.1:8000
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

### Frontend

```powershell
cd frontend-react
npm install
npm run dev
```

The Vite development server normally runs at:

```text
http://localhost:5173
```

## Docker

Build the backend image:

```powershell
docker build -t ai-image-quality .
```

Run it:

```powershell
docker run --rm -p 8000:8000 ai-image-quality
```

Or use Compose:

```powershell
docker compose up --build
```

## Testing

The repository contains scripts covering:

- degradation generation
- feature extraction
- dataset loading
- preprocessing
- quality engine
- blur/noise/exposure/corruption degradation
- Random Forest evaluation
- Isolation Forest
- ResNet experiments
- API behavior

The API has been tested for valid analysis, health/model endpoints, analysis history, invalid file types, invalid image content, and nonexistent analysis IDs.

## Evaluation and Failure Cases

The strongest classes are blur and underexposure. The main weaknesses are the `none` class and confusion involving corruption/noise.

The held-out test results show:

- Blur F1: 0.95
- Corruption F1: 0.78
- Noise F1: 0.91
- None F1: 0.54
- Overexposure F1: 0.85
- Underexposure F1: 0.98

This makes the model useful for automated screening while leaving clear opportunities for further improvement.

## Limitations

1. The `none` class is the weakest class at F1 0.54.
2. Corruption is more difficult to distinguish from noise and other classes.
3. Quality scoring is a rule-based decision layer on top of model predictions and severity.
4. Isolation Forest detects unusual feature patterns; anomaly does not automatically mean defect.
5. The expanded dataset contains generated degradation variants, so performance on unseen real-world artifacts may differ.

## Project Structure

```text
ai-image-quality/
├── backend/
│   ├── main.py
│   └── database.py
├── src/
│   ├── quality_engine.py
│   ├── inference.py
│   ├── extract_features_v2.py
│   └── ...
├── models/
├── frontend-react/
├── data/
├── requirements.txt
├── requirements-dev.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
└── README.md
```

## Future Improvements

- improve the `none` class
- add more real-world degradation examples
- probability calibration
- automated threshold optimization
- richer API response schemas
- authentication and rate limiting
- model monitoring
- cloud object storage
- CI/CD
