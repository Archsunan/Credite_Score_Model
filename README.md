# 🏦 Credit Score Model

An AI-powered credit scoring system that predicts creditworthiness using machine learning. Built with XGBoost, Flask, and a modern web interface.

## ✨ Features

- **🤖 Machine Learning Model**: XGBoost-based classification with 78.6% accuracy and 95.1% ROC AUC
- **📊 Automated Data Processing**: Feature engineering and preprocessing pipeline
- **🚀 REST API**: Flask-based API with CORS support for predictions
- **💻 Modern Web Interface**: Responsive UI with real-time predictions
- **📈 Model Evaluation**: Confusion matrix, feature importance, and comprehensive metrics
- **🎯 Four Credit Categories**: Excellent, Good, Fair, Poor with probability distributions

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python src/train_model.py
```
This generates 10,000 synthetic credit records, trains the model, and saves:
- `models/credit_model.pkl` - Trained model
- `models/preprocessor.pkl` - Data preprocessor
- `models/feature_importance.png` - Feature importance visualization
- `models/confusion_matrix.png` - Model performance matrix

### 3. Start the API Server
```bash
python src/api.py
```
API will be available at: **http://localhost:5000**

### 4. Open the Web Interface
Open `web/index.html` in your browser or visit the local file path.

## 📊 Model Performance

- **Accuracy**: 78.6%
- **Weighted ROC AUC**: 95.1%
- **Classes**: Excellent (88% precision), Good (54% precision), Fair (49% precision), Poor (81% precision)

## Project Structure

```
credit_score/
├── src/
│   ├── data_preprocessing.py  # Data cleaning and feature engineering
│   ├── train_model.py         # Model training pipeline
│   ├── model.py               # Model class and prediction logic
│   └── api.py                 # Flask API server
├── web/
│   ├── index.html            # Frontend interface
│   ├── style.css             # Styling
│   └── script.js             # Frontend logic
└── models/                    # Saved model files (created after training)
```

## 🔌 API Endpoints

### Health Check
```
GET http://localhost:5000/health
```
Returns server health and model status.

### Predict Credit Score
```
POST http://localhost:5000/predict
```

**Request Body:**
```json
{
  "age": 35,
  "income": 65000,
  "employment_length": 5,
  "loan_amount": 25000,
  "loan_term": 36,
  "credit_history_length": 10,
  "num_credit_lines": 4,
  "debt_to_income": 0.35,
  "num_delinquencies": 0,
  "num_inquiries": 1
}
```

**Response:**
```json
{
  "credit_score": "Good",
  "probability": 0.82,
  "risk_level": "Low",
  "all_probabilities": {
    "Excellent": 0.15,
    "Good": 0.82,
    "Fair": 0.02,
    "Poor": 0.01
  }
}
```

### Feature Importance
```
GET http://localhost:5000/feature_importance
```
Returns the importance scores of all features used in the model.


