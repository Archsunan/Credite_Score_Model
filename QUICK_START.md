# 🚀 Quick Start Guide - Credit Score Web App

## Running Your Project

### Step 1: Start the API Server

Open a terminal/PowerShell in the project directory and run:

```bash
python src/api.py
```

**You should see:**
```
Loading model artifacts...
Model and preprocessor loaded successfully!

Starting Flask API server...
API will be available at: http://localhost:5000

Endpoints:
  - GET  /health              : Health check
  - POST /predict             : Predict credit score
  - GET  /feature_importance  : Get feature importance

Press Ctrl+C to stop the server
```

### Step 2: Open the Web Application

**Option A: Double-click** `web/index.html` in File Explorer

**Option B: In browser, open:**
```
file:///C:/Users/Admin/OneDrive/Desktop/credit_score/web/index.html
```

**Option C: Use a local server (recommended):**
```bash
cd web
python -m http.server 8080
```
Then open: `http://localhost:8080`

### Step 3: Use the Application!

1. The form will be pre-filled with default values
2. Modify any values as needed
3. Click **"Analyze Profile"**
4. View the credit score prediction!

---

## Testing Different Profiles

### Excellent Credit Profile (Default)
Just click "Analyze Profile" with default values!
- Expected: **Excellent** score

### Good Credit Profile  
Modify these values:
- Delinquencies: 1
- Debt-to-income: 0.40
- Inquiries: 2
- Expected: **Good/Excellent** score

### Fair Credit Profile
Modify these values:
- Age: 33
- Income: 52000
- Employment: 5
- Delinquencies: 1
- Debt-to-income: 0.45
- Inquiries: 3
- Expected: **Fair** score

### Poor Credit Profile
Modify these values:
- Age: 22
- Income: 28000
- Employment: 0.5
- Loan amount: 35000
- Loan term: 60
- Credit history: 1
- Credit lines: 10
- Delinquencies: 5
- Debt-to-income: 0.85
- Inquiries: 8
- Expected: **Poor** score

---

## Optional: Testing Commands

### Test Model Predictions
```bash
python test_model.py
```

### Test All Categories via API
```bash
python test_api_all_categories.py
```

### Check API Health
Open in browser: `http://localhost:5000/health`

Or in PowerShell:
```powershell
Invoke-WebRequest -Uri http://localhost:5000/health
```

---

## Troubleshooting

### Issue: "Model not found"
**Solution:** Train the model first:
```bash
python src/train_model_german.py --dataset synthetic
```

### Issue: "Cannot connect to API"
**Solution:** Make sure the API server is running (Step 1)

### Issue: Port 5000 already in use
**Solution:** Stop other applications using port 5000 or modify the port in `src/api.py`

### Issue: Missing dependencies
**Solution:** Install requirements:
```bash
pip install -r requirements.txt
```

---

## Stopping the Server

Press **Ctrl+C** in the terminal where the API is running

---

## Project Commands Summary

| Command | Purpose |
|---------|---------|
| `python src/api.py` | Start API server |
| `python test_model.py` | Test model with samples |
| `python test_api_all_categories.py` | Test all categories |
| `python src/train_model_german.py --dataset synthetic` | Retrain model |

---

## Access Points

- **Web App**: `web/index.html` or `http://localhost:8080`
- **API**: `http://localhost:5000`
- **Health Check**: `http://localhost:5000/health`
- **Predict Endpoint**: `http://localhost:5000/predict` (POST)

---

**That's it! Your credit score web application is ready to use!** 🎉
