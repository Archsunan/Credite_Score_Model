from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import os
import pandas as pd

# Initialize Flask app serving static files from the 'web' directory
WEB_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '../web'))
app = Flask(__name__, static_folder=WEB_FOLDER, static_url_path='')
CORS(app)

# Load model and preprocessor
model = None
preprocessor = None

REQUIRED_FIELDS = [
    'age', 'income', 'employment_length', 'loan_amount',
    'loan_term', 'credit_history_length', 'num_credit_lines',
    'debt_to_income', 'num_delinquencies', 'num_inquiries'
]


def load_artifacts():
    """Load trained model and preprocessor."""
    global model, preprocessor

    model_path = 'models/credit_model.pkl'
    preprocessor_path = 'models/preprocessor.pkl'

    train_hint = "python src/train_model_german.py --dataset synthetic"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            f"Please train the model first by running: {train_hint}"
        )

    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(
            f"Preprocessor not found at {preprocessor_path}. "
            f"Please train the model first by running: {train_hint}"
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
    """Predict credit score for a given applicant."""
    try:
        if model is None or preprocessor is None:
            return jsonify({
                'error': 'Model not loaded. Train the model and restart the server.',
                'hint': 'python src/train_model_german.py --dataset synthetic'
            }), 500

        # Get request data (avoid exceptions on invalid/missing JSON)
        data = request.get_json(silent=True)

        if data is None:
            return jsonify({'error': 'Request body must be JSON'}), 400
        if not isinstance(data, dict):
            return jsonify({'error': 'JSON payload must be an object'}), 400

        # Validate required fields
        missing_fields = [field for field in REQUIRED_FIELDS if field not in data]
        if missing_fields:
            return jsonify({
                'error': f"Missing required fields: {', '.join(missing_fields)}"
            }), 400

        # Validate data types
        for field in REQUIRED_FIELDS:
            value = data.get(field)
            if value is None:
                return jsonify({'error': f"Field '{field}' cannot be empty/null"}), 400
            try:
                float(value)
            except (ValueError, TypeError):
                return jsonify({
                    'error': f"Field '{field}' must be a number, got {type(value).__name__}"
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
        return jsonify({'error': str(e)}), 500

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

# Static file serving routes (must be last to not override API routes)
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    # Avoid serving API routes as static files
    if path in ['health', 'predict', 'feature_importance']:
        return jsonify({'error': 'Not found'}), 404
    return send_from_directory(app.static_folder, path)

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
