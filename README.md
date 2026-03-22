# Diabetes Prediction API (k-NN + FastAPI)

A lightweight ML service that predicts diabetes risk using a `KNeighborsClassifier` trained on the Pima Indians Diabetes dataset.

## Project Overview

This repository contains:
- Model training script: `main.py`
- API endpoint: `api/routes/predict.py`
- Serialized artifacts: `models/knn_diabetes_model.pkl`, `models/knn_diabetes_scaler.pkl`
- Dependency lock-style requirements: `requirements.txt`

## Tech Stack

- Python 3.13 (based on current local artifacts)
- FastAPI + Uvicorn
- scikit-learn (k-NN)
- pandas / numpy
- joblib for model persistence

## Repository Structure

```text
.
|-- api/
|   `-- routes/
|       `-- predict.py
|-- models/
|   |-- knn_diabetes_model.pkl
|   `-- knn_diabetes_scaler.pkl
|-- main.py
|-- requirements.txt
`-- README.md
```

## Prerequisites

- Python 3.10+ (3.13 recommended to match current environment)
- A virtual environment
- Dataset file available at the path expected by training script:
  - `../../Datasets/diabetes.csv` (relative to repository root when running `main.py`)

If your dataset lives elsewhere, update this line in `main.py`:

```python
df = pd.read_csv("../../Datasets/diabetes.csv")
```

## Setup

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### macOS/Linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train / Refresh the Model

Run training to regenerate model artifacts:

```bash
python main.py
```

Outputs written by `main.py`:
- `models/knn_diabetes_model.pkl`
- `models/knn_diabetes_scaler.pkl`

## Run the API

From the repository root:

```bash
uvicorn api.routes.predict:app --reload
```

API docs:
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## Prediction Contract

### Endpoint

- `POST /predict`

### Request Body

```json
{
  "pregnancies": 2,
  "glucose": 120,
  "blood_pressure": 70,
  "skin_thickness": 20,
  "insulin": 85,
  "bmi": 28.5,
  "diabetes_pedigree_function": 0.45,
  "age": 34
}
```

### Response

Returns a plain string:
- `"Diabetic"`
- `"Non-Diabetic"`

## Quick Test with cURL

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "pregnancies": 2,
    "glucose": 120,
    "blood_pressure": 70,
    "skin_thickness": 20,
    "insulin": 85,
    "bmi": 28.5,
    "diabetes_pedigree_function": 0.45,
    "age": 34
  }'
```

## Known Gaps / Recommended Next Steps

- Return structured JSON (e.g., prediction + confidence) instead of plain string.
- Add input validation rules (range checks for each feature).
- Version the model artifact and store metrics alongside it.
- Add tests for API contract and model-loading behavior.
- Externalize paths/config via environment variables or settings module.

## License

No license file is currently present in this repository.
Add one if this project will be shared or published.
