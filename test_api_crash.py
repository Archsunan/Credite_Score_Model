import requests
import json

def test_api_null_values():
    url = 'http://localhost:5000/predict'
    
    # Payload with nulls (simulating NaN from JS)
    payload = {
        "age": None,
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
    
    try:
        response = requests.post(url, json=payload)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_api_null_values()
