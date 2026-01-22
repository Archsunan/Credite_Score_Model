from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Load model and preprocessor
model = None
preprocessor = None

def load_artifacts():
    """Load trained model and preprocessor."""
    global model, preprocessor
    
    model_path = 'models/credit_model.pkl'
    preprocessor_path = 'models/preprocessor.pkl'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Please train the model first by running: python src/train_model.py"
        )
    
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(
            f"Preprocessor not found at {preprocessor_path}. "
            "Please train the model first by running: python src/train_model.py"
        )
    
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    print("Model and preprocessor loaded successfully!")

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict credit score for a given applicant.
    
    Expected JSON payload:
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
    """
    try:
        # Get request data
        data = request.get_json()
        
        # Validate required fields
        required_fields = [
            'age', 'income', 'employment_length', 'loan_amount',
            'loan_term', 'credit_history_length', 'num_credit_lines',
            'debt_to_income', 'num_delinquencies', 'num_inquiries'
        ]
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'error': f"Missing required fields: {', '.join(missing_fields)}"
            }), 400
        
        # Create DataFrame
        input_df = pd.DataFrame([{
            'age': float(data['age']),
            'income': float(data['income']),
            'employment_length': float(data['employment_length']),
            'loan_amount': float(data['loan_amount']),
            'loan_term': float(data['loan_term']),
            'credit_history_length': float(data['credit_history_length']),
            'num_credit_lines': float(data['num_credit_lines']),
            'debt_to_income': float(data['debt_to_income']),
            'num_delinquencies': float(data['num_delinquencies']),
            'num_inquiries': float(data['num_inquiries'])
        }])
        
        # Preprocess
        X, _ = preprocessor.prepare_data(input_df, fit_scaler=False)
        
        # Make prediction
        result = model.predict_single(X.iloc[0].to_dict())
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/feature_importance', methods=['GET'])
def feature_importance():
    """Get feature importance scores."""
    try:
        importance_df = model.get_feature_importance()
        return jsonify({
            'features': importance_df.to_dict('records')
        })
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("Loading model artifacts...")
    load_artifacts()
    
    print("\nStarting Flask API server...")
    print("API will be available at: http://localhost:5000")
    print("\nEndpoints:")
    print("  - GET  /health              : Health check")
    print("  - POST /predict             : Predict credit score")
    print("  - GET  /feature_importance  : Get feature importance")
    print("\nPress Ctrl+C to stop the server\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
