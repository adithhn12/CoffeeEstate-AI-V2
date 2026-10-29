# Coffee Estate AI — Cherry Ripeness Analysis & AI Model Status

**Coffee Estate AI** is a specialized computer vision application engineered for coffee plantations. The system provides real-time detection of coffee cherries, maturity classification across 5 developmental stages, harvesting recommendations, and live model diagnostics.

---

## 1. System Architecture

```text
CoffeeEstateAI/
├── backend/                        # FastAPI Backend
│   ├── app/
│   │   ├── config.py               # Paths, upload directories & configuration
│   │   ├── database.py             # SQLite engine & SQLAlchemy ORM
│   │   ├── database_migrations.py  # Safe column & schema migrations
│   │   ├── main.py                 # FastAPI application, CORS, static image mounts
│   │   ├── models/                 # SQLAlchemy database models
│   │   │   ├── estate.py           # Estate & Section models
│   │   │   └── analysis.py         # ImageAnalysis model
│   │   ├── schemas/                # Pydantic validation schemas
│   │   │   └── analysis.py         # ImageAnalysis, BatchAnalysis & ModelStatus schemas
│   │   ├── services/
│   │   │   ├── vision_service.py   # Ultralytics YOLOv8 inference & maturity interpreter
│   │   │   └── sample_data.py      # Initial demonstration seed data
│   │   └── routes/                 # REST API Endpoints
│   │       ├── health.py           # Health & GPU status (/api/health)
│   │       └── analysis.py         # Image upload, history & model status (/api/analysis)
│   ├── requirements.txt            # Python dependencies
│   └── tests/
│       ├── run_tests.py            # Automated Unit Test Suite
│       └── test_backend.py         # Backend test cases
├── frontend/                       # React 18 + Vite + TypeScript + Tailwind CSS
│   ├── src/
│   │   ├── api/client.ts           # Clean typed API Client
│   │   ├── components/Layout.tsx   # Responsive Sidebar & Mobile Navigation
│   │   ├── pages/
│   │   │   ├── RipenessAnalysis.tsx# Coffee Cherry Ripeness Analysis & History
│   │   │   └── SettingsModel.tsx   # Model Information & Class Definitions
│   │   └── types/index.ts          # TypeScript Model Interfaces
├── ml/
│   ├── weights/
│   │   └── best.pt                 # Fine-tuned YOLOv8 coffee cherry weights (6.49 MB)
│   ├── train_yolo.py               # YOLOv8 Training Script
│   └── evaluate_model.py           # Model Validation & Evaluation Script
├── scripts/
│   └── verify_regression.py        # Prediction regression verification against baseline
├── coffee_cherry.yaml              # 5-class YOLO dataset descriptor
├── baseline_regression_sample.json # Canonical baseline prediction sample
└── README.md                       # Documentation
```

---

## 2. Core Functional Modules

### A. Coffee Cherry Ripeness Analysis
* **Modular YOLOv8 Vision Pipeline**: Detects coffee cherries from branch photographs with color-coded bounding boxes.
* **5 Biological Cherry Classes**:
  * `0`: Dark Brown Cherry (Overripe / dried fruit)
  * `1`: Green Cherry (Unripe fruit)
  * `2`: Low Visibility / Unsure (Occluded fruit; excluded from denominator)
  * `3`: Red Cherry (Prime ripe fruit)
  * `4`: Yellow Cherry (Semi-ripe / turning fruit)
* **Comprehensive Metrics**: Exact cherry counts, evaluable ripeness percentages, confidence scores, and maturity stage classification.
* **Actionable Harvesting Recommendations**: Guidance on selective picking, fly picking, processing route (washed vs natural), and pulping timelines.
* **Analysis History**: View recent checks with zooming inspection and safe individual record deletion.

### B. Model Diagnostics & Information
* **Verified Model Details**: Architecture, model tag, training dataset source, and dataset split.
* **Biological Class Definitions**: Detailed explanations for each cherry category and its processing significance.
* **Hardware & Runtime Status**: Live detection of CUDA GPU acceleration or CPU execution.

---

## 3. Quick Start Guide

### Step 1: Backend Setup (FastAPI)

```bash
# In the backend directory:
cd backend

# Install Python dependencies:
pip install -r requirements.txt

# Start the FastAPI backend server:
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
* Interactive Swagger API documentation: `http://127.0.0.1:8000/docs`
* Health check: `http://127.0.0.1:8000/api/health`

### Step 2: Frontend Setup (React + Vite)

```bash
# In a separate terminal, from the frontend directory:
cd frontend

# Install Node dependencies:
npm install

# Start the Vite development server:
npm run dev
```
* Open `http://localhost:5173` in your browser.

---

## 4. Verification & Testing

### Backend Unit Tests
Run the automated test suite covering health check, weights loading, 5-class parity, invalid image handling, maturity calculations, and history deletion:
```bash
python backend/tests/run_tests.py
```

### AI Prediction Regression Test
Verify that model predictions, detection counts, bounding boxes, and confidence scores match the canonical baseline with zero drift:
```bash
python scripts/verify_regression.py
```

### Frontend Type Check & Build
Validate TypeScript definitions and compile the production build:
```bash
cd frontend
npm run build
```
