"""
Test script to verify the trained German Credit model works correctly.
Uses the API interface to make predictions.
"""
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import joblib
import pandas as pd
from data_preprocessing import DataPreprocessor
from model import CreditScoreModel

def test_model():
    """Test the trained model with sample predictions."""
    print("="*60)
    print("Testing German Credit Model")
    print("="*60)
    
    # Load model and preprocessor
    print("\nLoading model and preprocessor...")
    try:
        model = CreditScoreModel.load('models/credit_model.pkl')
        preprocessor = joblib.load('models/preprocessor.pkl')
        print("[OK] Model and preprocessor loaded successfully!")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("Please train the model first: python src/train_model_german.py")
        return
    
    # Test cases
    test_cases = [
        {
            'name': 'Young Professional (Good Risk)',
            'age': 28,
            'income': 75000,
            'employment_length': 4,
            'loan_amount': 15000,
            'loan_term': 24,
            'credit_history_length': 8,
            'num_credit_lines': 3,
            'debt_to_income': 0.25,
            'num_delinquencies': 0,
            'num_inquiries': 1
        },
        {
            'name': 'Mid-Career Professional (Excellent Risk)',
            'age': 42,
            'income': 95000,
            'employment_length': 15,
            'loan_amount': 30000,
            'loan_term': 36,
            'credit_history_length': 20,
            'num_credit_lines': 5,
            'debt_to_income': 0.30,
            'num_delinquencies': 0,
            'num_inquiries': 0
        },
        {
            'name': 'High Risk Applicant',
            'age': 25,
            'income': 35000,
            'employment_length': 1,
            'loan_amount': 25000,
            'loan_term': 60,
            'credit_history_length': 3,
            'num_credit_lines': 8,
            'debt_to_income': 0.70,
            'num_delinquencies': 3,
            'num_inquiries': 5
        }
    ]
    
    print("\n" + "="*60)
    print("Making Predictions")
    print("="*60)
    
    for i, test_case in enumerate(test_cases, 1):
        name = test_case.pop('name')
        print(f"\n{i}. {name}")
        print("-" * 60)
        
        # Display input
        print("   Input Features:")
        for key, value in test_case.items():
            print(f"      {key:25s}: {value}")
        
        # Create DataFrame and preprocess
        input_df = pd.DataFrame([test_case])
        X_preprocessed, _ = preprocessor.prepare_data(input_df, fit_scaler=False)
        
        # Make prediction using the preprocessed features
        result = model.predict_single(X_preprocessed.iloc[0].to_dict())
        
        # Display results
        print(f"\n   Prediction Results:")
        print(f"      Credit Score: {result['credit_score']}")
        print(f"      Confidence: {result['probability']:.1%}")
        print(f"      Risk Level: {result['risk_level']}")
        print(f"\n      Probability Distribution:")
        for cls, prob in sorted(result['all_probabilities'].items(), 
                               key=lambda x: x[1], reverse=True):
            bar = '#' * int(prob * 50)
            print(f"         {cls:12s}: {prob:6.1%} {bar}")
    
    print("\n" + "="*60)
    print("Model Test Completed Successfully!")
    print("="*60)
    print("\nThe model is ready to use!")
    print("Start the API with: python src/api.py")

if __name__ == "__main__":
    test_model()
