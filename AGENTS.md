# AGENTS.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## What this repo is
A small Python + Flask project that trains an XGBoost-based credit score classifier and serves it via a REST API consumed by a static HTML/JS UI.

## Common commands
All commands are intended to be run from the repo root (paths in the code assume that).

### Install dependencies
```bash
pip install -r requirements.txt
```

### Train / retrain the model
This produces the artifacts used by the API:
- `models/credit_model.pkl`
- `models/preprocessor.pkl`

```bash
# Train using the German Credit dataset (downloads to data/german.data on first run)
python src/train_model_german.py --dataset german

# Train using synthetic data (often preferred for the web UI’s “typical” inputs)
python src/train_model_german.py --dataset synthetic
```

Older training entrypoint (synthetic-only):
```bash
python src/train_model.py
```

### Run the API server
```bash
python src/api.py
```
Endpoints:
- `GET  /health`
- `POST /predict`
- `GET  /feature_importance`

### Run the web UI
The UI is static and calls the API at `http://localhost:5000`.

- Open `web/index.html` directly in a browser, or
- Serve it locally:

```bash
python -m http.server 8080 --directory web
```
Then open `http://localhost:8080`.

### Tests / smoke checks
There is no test runner (e.g. pytest) configured in this repo; tests are Python scripts.

Model-level smoke test (does not require the API server):
```bash
python test_model.py
```

API-level smoke tests (require `python src/api.py` running separately):
```bash
# Quick valid request
python test_api_valid.py

# Exercises Poor/Fair/Good/Excellent profiles
python test_api_all_categories.py

# Sends nulls to validate error handling
python test_api_crash.py
```

### Debugging a single prediction locally
Runs preprocessing + `predict_single()` directly with a hard-coded sample input:
```bash
python debug_prediction.py
```

## High-level architecture (big picture)
### Data + feature pipeline
- `src/data_preprocessing.py` defines `DataPreprocessor`:
  - `generate_synthetic_data()` creates labeled synthetic profiles for the 4 classes.
  - `engineer_features()` adds derived features (e.g. `loan_to_income`, flags like `high_debt`).
  - `prepare_data()` applies feature engineering + scaling via an internal `StandardScaler`.
  - The fitted `DataPreprocessor` instance is persisted as `models/preprocessor.pkl` and is used at inference time.

- `src/load_german_credit.py` defines `GermanCreditDataLoader`:
  - Downloads `data/german.data` from the UCI repository if missing.
  - Encodes categorical columns.
  - Maps the original binary target into the repo’s 4-class `credit_score` via a computed risk score.
  - Converts German dataset fields into the model’s expected input schema (the same 10 “core” fields as the API expects).

### Model
- `src/model.py` defines `CreditScoreModel`, a thin wrapper around `xgboost.XGBClassifier`.
- The trained object (including the fitted `LabelEncoder`) is saved via `CreditScoreModel.save()` to `models/credit_model.pkl`.
- Inference:
  - The API first uses `preprocessor.prepare_data(..., fit_scaler=False)`.
  - Then calls `model.predict_single()` on the scaled + engineered feature dict.

### Training entrypoints
- `src/train_model_german.py` is the main orchestrator:
  - Chooses dataset (`--dataset german|synthetic`).
  - Trains the model and writes artifacts under `models/`.
  - For the German dataset path, it applies SMOTE before training.
- `src/train_model.py` is an older synthetic-only training script.

### Serving (API + UI)
- `src/api.py`:
  - Loads `models/credit_model.pkl` and `models/preprocessor.pkl` at startup.
  - Implements `/predict` by validating inputs, preprocessing into the scaled feature space, and returning `CreditScoreModel.predict_single()`.
  - Also serves static files from `web/` (route `/` serves `web/index.html`).

- `web/script.js`:
  - POSTs form values to `${API_URL}/predict` and renders the returned probabilities.
  - `API_URL` is hard-coded to `http://localhost:5000`.

## Gotchas that affect day-to-day work
- Most scripts and `src/api.py` use relative paths like `models/credit_model.pkl`; run commands from the repo root so these resolve correctly.
- If the API fails early on requests, check `src/api.py` logging around `/predict` input validation first (that’s where most runtime issues surface).