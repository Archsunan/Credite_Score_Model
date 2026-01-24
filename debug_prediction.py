import joblib
import pandas as pd
import sys
import os

# Add src to path so we can import classes
sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    from data_preprocessing import DataPreprocessor
    from model import CreditScoreModel
except ImportError:
    # If running from root
    sys.path.append('src')
    from data_preprocessing import DataPreprocessor
    from model import CreditScoreModel

def debug_prediction():
    print("Loading artifacts...")
    try:
        model = joblib.load('models/credit_model.pkl')
        preprocessor = joblib.load('models/preprocessor.pkl')
        print("Artifacts loaded successfully.")
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        return

    # Sample input from API usage
    sample_input = {
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
    
    print("\nInput data:")
    print(sample_input)

    try:
        # Create DataFrame
        input_df = pd.DataFrame([sample_input])
        
        print("\nStep 1: Preprocessing...")
        X, _ = preprocessor.prepare_data(input_df, fit_scaler=False)
        print("Preprocessing successful.")
        print(f"Processed feature shape: {X.shape}")
        print("Processed features columns:", X.columns.tolist())
        print("Processed features values:\n", X.head())
        
        print("\nStep 2: Prediction...")
        # Simulating api.py calling model.predict_single
        # In api.py: result = model.predict_single(X.iloc[0].to_dict())
        
        feature_dict = X.iloc[0].to_dict()
        print("Feature dict passed to predict_single:", feature_dict)
        
        result = model.predict_single(feature_dict)
        print("\nPrediction Result:")
        print(result)
        
    except Exception as e:
        print(f"\nCaught exception during prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_prediction()
