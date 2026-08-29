# AI Image Quality Assessment

An end-to-end computer-vision and machine-learning system that automatically evaluates image quality, identifies known degradations, detects unusual image patterns, estimates severity, and returns an acceptance/rejection decision.

The application uses a **React frontend**, **FastAPI backend**, **Random Forest V3 Expanded**, **Isolation Forest**, and **SQLite**.
## Production Model Naming

The final production classifier is referred to in the repository as **Random Forest V3 Expanded**.

The name describes how the final model was produced:

```text
V3
│
├── Final feature-selection version
│
│   └── 10 selected computer-vision features
│
└── Expanded
    │
    └── Trained using the expanded dataset
        generated from 500 source images

## 🚀 Overview

The production pipeline is:

```text
Image Upload
     ↓
File & Image Validation
     ↓
CV Feature Extraction
     ↓
Random Forest V3 Expanded
     ↓
Severity Estimation
     ↓
Isolation Forest Anomaly Detection
     ↓
Quality Score (0–100)
     ↓
ACCEPTED / REJECTED
     ↓
SQLite Analysis History

```
## Sample Images Of Production 
<img width="1367" height="728" alt="Screenshot 2026-08-29 223246" src="https://github.com/user-attachments/assets/8fdf44a5-d88b-4135-b890-841da1a2b188" />
<img width="1429" height="836" alt="Screenshot 2026-08-29 223310" src="https://github.com/user-attachments/assets/dabfe964-610b-4cbd-81d4-33f33bb2a465" />
<img width="1356" height="669" alt="Screenshot 2026-08-29 223344" src="https://github.com/user-attachments/assets/62cd898f-c0a3-42c7-96c2-a2b1f4c5fd18" />
<img width="1430" height="886" alt="Screenshot 2026-08-29 223330" src="https://github.com/user-attachments/assets/b6300f87-6130-4155-86ec-9bd532008ae1" />






## 🧠 Production ML Pipeline

### 1. Image Validation

The API accepts:

- JPEG
- PNG
- WEBP

Uploaded content is validated before processing.

### 2. Computer-Vision Features

The production classifier uses 10 measurable image features:

1. Sharpness
2. Brightness
3. Highlight clipping
4. Contrast
5. Saturation
6. Edge density
7. Shadow clipping
8. Dark pixel ratio
9. Bright pixel ratio
10. Blockiness

### 3. Random Forest V3 Expanded

The Random Forest predicts six degradation classes:

- `blur`
- `corruption`
- `noise`
- `none`
- `overexposure`
- `underexposure`

It was trained using 6,650 training samples and evaluated on a held-out test set of 1,425 samples.

**Test accuracy: 88.21%**

**Test Macro-F1: 83.64%**

### 4. Severity Estimation

The predicted degradation and image statistics are used to estimate severity:

- Low
- Medium
- High

### 5. Isolation Forest

Isolation Forest provides a complementary anomaly-detection signal.

The two models answer different questions:

```text
Random Forest
→ Which known degradation does this image resemble?

Isolation Forest
→ Is this feature pattern unusual compared with the learned distribution?
```

An anomaly is therefore an additional signal and does **not automatically mean the image is defective**.

### 6. Quality Scoring

The quality engine combines:

- classification
- model confidence
- severity
- anomaly information

to produce a score from **0 to 100**.

Current decision threshold:

```text
Score >= 75  → ACCEPTED
Score < 75   → REJECTED
```

A clean image classified as `none` without an anomaly can receive **100/100**.

## 📈 Feature Importance

Global Random Forest feature importance:

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

These are **global model-level importances**, not explanations for an individual prediction.

## 🏗️ Architecture

```text
┌──────────────────────┐
│   React Frontend     │
│      React 19        │
│       Vite 6         │
└──────────┬───────────┘
           │ HTTP
           ▼
┌──────────────────────┐
│   FastAPI Backend    │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Image Validation    │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ CV Feature Extraction│
└──────────┬───────────┘
           │
      ┌────┴─────┐
      ▼          ▼
 Random Forest  Isolation
 V3 Expanded    Forest
      │          │
      └────┬─────┘
           ▼
┌──────────────────────┐
│ Severity Estimation  │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│   Quality Score      │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ ACCEPTED / REJECTED  │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│       SQLite         │
└──────────────────────┘
```

## 🌐 Deployment

The application has been successfully deployed to **Render**.

The deployed system provides the same production analysis pipeline used locally.

> Add the live Render URL here:
>
> `https://ai-image-quality-frontend.onrender.com/`

## 🔌 Backend API

The FastAPI application is located at:

```text
backend/main.py
```

### Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API information |
| GET | `/health` | Health/model status |
| GET | `/model` | Model metadata |
| POST | `/analyze` | Main production quality analysis |
| GET | `/analyses` | Analysis history |
| GET | `/analyses/{id}` | Retrieve one analysis |
| POST | `/predict` | Legacy/experimental ResNet prediction |

### Main `/analyze` Flow

```text
Upload
  ↓
File validation
  ↓
Image-content validation
  ↓
Temporary image
  ↓
CV feature extraction
  ↓
Random Forest V3 Expanded
  ↓
Severity estimation
  ↓
Isolation Forest
  ↓
Quality score
  ↓
Accept / Reject
  ↓
SQLite
  ↓
JSON response
```

### Example Request

```bash
curl -X POST "http://127.0.0.1:8000/analyze" -F "file=@image.jpg"
```

### Interactive API Documentation

When running locally:

```text
http://127.0.0.1:8000/docs
```

## 🧪 ResNet-18 Experimental Pipeline

The repository also contains a separate ResNet-18 inference pipeline exposed through `/predict`.

This is **not the production analysis workflow**.

```text
/analyze
    ↓
CV Features
    ↓
Random Forest V3 Expanded
    ↓
Isolation Forest
    ↓
Quality Assessment
```

```text
/predict
    ↓
ResNet-18
    ↓
Degradation Prediction
    ↓
Confidence / Probabilities
```

The production application therefore uses **Random Forest V3 Expanded + Isolation Forest**, while ResNet-18 is retained as an experimental/alternative model.

## 💾 Database

SQLite stores analysis history in:

```text
data/quality_analysis.db
```

The Docker Compose configuration mounts a named volume for `/app/data` so analysis history can persist when the backend container is recreated.

## 💻 Local Setup

### Backend

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start FastAPI:

```bash
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

```bash
cd frontend-react
npm install
npm run dev
```

The Vite development server normally runs at:

```text
http://localhost:5173
```

## 🐳 Docker

Build:

```bash
docker build -t ai-image-quality .
```

Run:

```bash
docker run --rm -p 8000:8000 ai-image-quality
```

Or use Docker Compose:

```bash
docker compose up --build
```

## 🧪 Testing

The repository contains tests and scripts covering:

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

API behavior has been tested for:

- valid image analysis
- health endpoint
- model endpoint
- analysis history
- invalid file types
- invalid image content
- nonexistent analysis IDs

## 📁 Project Structure

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

## ⚠️ Limitations

1. The `none` class is the weakest class at F1 0.54.
2. Corruption is harder to distinguish from noise and other classes.
3. Quality scoring is a rule-based decision layer on top of model predictions and severity.
4. Isolation Forest detects unusual feature patterns; anomaly does not automatically mean defect.
5. The expanded dataset contains generated degradation variants, so performance on unseen real-world artifacts may differ.

## 🔮 Future Improvements

- Improve the `none` class
- Add more real-world degradation examples
- Probability calibration
- Automated threshold optimization
- Richer API response schemas
- Authentication and rate limiting
- Model monitoring
- Cloud object storage
- CI/CD

## 🛠️ Tech Stack

**Machine Learning**
- Python
- Scikit-learn
- Random Forest
- Isolation Forest
- ResNet-18

**Computer Vision**
- Image feature extraction
- Image statistics
- Sharpness and edge analysis
- Exposure and clipping analysis
- Blockiness analysis

**Backend**
- FastAPI
- Uvicorn
- SQLite

**Frontend**
- React 19
- Vite 6
- Lucide React

**Deployment**
- Docker
- Docker Compose
- Render

## 📌 Key Takeaway

This project demonstrates a complete ML application lifecycle:

```text
Data
 ↓
Feature Engineering
 ↓
Model Training
 ↓
Model Evaluation
 ↓
Inference Pipeline
 ↓
API
 ↓
Frontend
 ↓
Docker
 ↓
Cloud Deployment
```

The system is designed as an automated image-quality screening pipeline rather than a single standalone classifier.
