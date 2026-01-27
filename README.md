
#  Credit Score Model

An AI-powered credit scoring system that predicts creditworthiness using machine learning. Built with XGBoost, Flask, and a modern web interface.

##  Features

- ** Machine Learning Model**: XGBoost-based classification with 78.6% accuracy and 95.1% ROC AUC
- **Automated Data Processing**: Feature engineering and preprocessing pipeline
- **REST API**: Flask-based API with CORS support for predictions
- **Modern Web Interface**: Responsive UI with real-time predictions
- **Model Evaluation**: Confusion matrix, feature importance, and comprehensive metrics
- **Four Credit Categories**: Excellent, Good, Fair, Poor with probability distributions

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
# Train on German Credit Dataset (Default)
python src/train_model_german.py

# Optional: Train on Synthetic Data
python src/train_model_german.py --dataset synthetic
```
This trains the model and saves:
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

##  Model Performance

- **Accuracy**: 78.6%
- **Weighted ROC AUC**: 95.1%
- **Classes**: Excellent (88% precision), Good (54% precision), Fair (49% precision), Poor (81% precision)

## Project Structure

```
credit_score/
├── src/
│   ├── data_preprocessing.py  # Data cleaning and feature engineering
│   ├── train_model_german.py  # Main training pipeline (German & Synthetic)
│   ├── load_german_credit.py  # German Credit dataset loader
│   ├── model.py               # Model class and prediction logic
│   └── api.py                 # Flask API server
├── web/
│   ├── index.html            # Frontend interface
│   ├── style.css             # Styling
│   └── script.js             # Frontend logic
└── models/                    # Saved model files (created after training)
```

##  API Endpoints

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


The model uses **10 core features** with automatic feature engineering:

### Core Features
| Feature | Description | Example |
|---------|-------------|----------|
| Age | Applicant's age | 35 |
| Annual Income | Yearly income in USD | 65000 |
| Employment Length | Years employed | 5.0 |
| Loan Amount | Requested loan amount | 25000 |
| Loan Term | Loan duration in months | 36 |
| Credit History Length | Years of credit history | 10.0 |
| Number of Credit Lines | Active credit accounts | 4 |
| Debt-to-Income Ratio | Debt as proportion of income | 0.35 |
| Number of Delinquencies | Past payment failures | 0 |
| Number of Inquiries | Recent credit checks | 1 |

### Engineered Features
The system automatically generates additional features:
- Loan-to-Income Ratio
- Credit Lines per Year
- Age-Income Interaction
- High Debt Indicator
- Has Delinquencies Flag
- Many Inquiries Flag

## Credit Score Categories

| Category | Risk Level | Description |
|----------|-----------|-------------|
| **Excellent** | Very Low | High creditworthiness, minimal risk |
| **Good** | Low | Moderate-low risk, reliable borrower |
| **Fair** | Moderate | Some concerns, careful evaluation needed |
| **Poor** | High | High risk, significant credit concerns |

## Technologies Used

- **Python 3.13+** - Core programming language
- **XGBoost 3.1** - Gradient boosting model
- **scikit-learn** - Data preprocessing and metrics
- **pandas & numpy** - Data manipulation
- **Flask 3.1** - REST API framework
- **matplotlib & seaborn** - Visualizations
- **HTML/CSS/JavaScript** - Web interface




### Test Model Predictions
```bash
python test_model.py
```

### Test All Categories via API
```bash
python test_api_all_categories.py
```

