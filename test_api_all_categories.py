"""
Test script to verify Poor credit score predictions work correctly.
"""
import sys
import os
import requests
import json

# Test data for Poor credit score
poor_credit_data = {
    "age": 22,
    "income": 28000,
    "employment_length": 0.5,
    "loan_amount": 35000,
    "loan_term": 60,
    "credit_history_length": 1,
    "num_credit_lines": 10,
    "debt_to_income": 0.85,
    "num_delinquencies": 5,
    "num_inquiries": 8
}

fair_credit_data = {
    "age": 33,
    "income": 52000,
    "employment_length": 5,
    "loan_amount": 28000,
    "loan_term": 48,
    "credit_history_length": 7,
    "num_credit_lines": 6,
    "debt_to_income": 0.45,
    "num_delinquencies": 1,
    "num_inquiries": 3
}

good_credit_data = {
    "age": 36,
    "income": 68000,
    "employment_length": 7,
    "loan_amount": 32000,
    "loan_term": 36,
    "credit_history_length": 10,
    "num_credit_lines": 5,
    "debt_to_income": 0.40,
    "num_delinquencies": 1,
    "num_inquiries": 2
}

excellent_credit_data = {
    "age": 45,
    "income": 95000,
    "employment_length": 15,
    "loan_amount": 30000,
    "loan_term": 24,
    "credit_history_length": 20,
    "num_credit_lines": 3,
    "debt_to_income": 0.20,
    "num_delinquencies": 0,
    "num_inquiries": 0
}

def test_api_endpoint(data, expected_category):
    """Test the API with given data."""
    url = "http://localhost:5000/predict"
    
    try:
        response = requests.post(url, json=data, timeout=5)
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n{'='*60}")
            print(f"Expected: {expected_category}")
            print(f"{'='*60}")
            print(f"Credit Score: {result['credit_score']}")
            print(f"Confidence: {result['probability']*100:.1f}%")
            print(f"Risk Level: {result['risk_level']}")
            print(f"\nProbability Distribution:")
            for category, prob in sorted(result['all_probabilities'].items(), 
                                        key=lambda x: x[1], reverse=True):
                bar = '#' * int(prob * 50)
                print(f"  {category:12s}: {prob*100:6.1f}% {bar}")
            
            # Check if prediction matches expected
            if result['credit_score'] == expected_category:
                print(f"\n[PASS] Got expected '{expected_category}' score")
            else:
                print(f"\n✗ FAIL: Expected '{expected_category}' but got '{result['credit_score']}'")
            
            return result
        else:
            print(f"Error: API returned status code {response.status_code}")
            print(response.text)
            return None
            
    except requests.exceptions.ConnectionError:
        print("\n" + "="*60)
        print("ERROR: Cannot connect to API server!")
        print("="*60)
        print("\nPlease start the API server first:")
        print("  python src/api.py")
        print("\nThen run this test again.")
        print("="*60)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    print("="*60)
    print("Testing All Credit Score Categories")
    print("="*60)
    
    tests = [
        (poor_credit_data, "Poor"),
        (fair_credit_data, "Fair"),
        (good_credit_data, "Good"),
        (excellent_credit_data, "Excellent")
    ]
    
    results = []
    for data, expected in tests:
        result = test_api_endpoint(data, expected)
        if result:
            results.append((expected, result['credit_score']))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for expected, actual in results:
        status = "[PASS]" if expected == actual else "[FAIL]"
        print(f"{status}: Expected '{expected}', Got '{actual}'")
    
    print("\n" + "="*60)
    print("All category tests completed!")
    print("="*60)

if __name__ == "__main__":
    main()
