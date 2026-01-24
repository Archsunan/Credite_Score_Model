import requests
import json

def test_api_valid():
    url = 'http://localhost:5000/predict'
    
    # Valid Payload
    payload = {
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
    
    try:
        print(f"Sending request to {url}...")
        response = requests.post(url, json=payload)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("Response:", json.dumps(response.json(), indent=2))
        else:
            print("Error Response:", response.text)
            
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_api_valid()
