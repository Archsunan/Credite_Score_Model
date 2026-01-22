const API_URL = 'http://localhost:5000';

// Form submission handler
document.getElementById('creditForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Collect form data
    const formData = {
        age: parseFloat(document.getElementById('age').value),
        income: parseFloat(document.getElementById('income').value),
        employment_length: parseFloat(document.getElementById('employment_length').value),
        loan_amount: parseFloat(document.getElementById('loan_amount').value),
        loan_term: parseFloat(document.getElementById('loan_term').value),
        credit_history_length: parseFloat(document.getElementById('credit_history_length').value),
        num_credit_lines: parseFloat(document.getElementById('num_credit_lines').value),
        debt_to_income: parseFloat(document.getElementById('debt_to_income').value),
        num_delinquencies: parseFloat(document.getElementById('num_delinquencies').value),
        num_inquiries: parseFloat(document.getElementById('num_inquiries').value)
    };
    
    // Show loading overlay
    document.getElementById('loadingOverlay').classList.add('active');
    
    try {
        // Make API request
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });
        
        if (!response.ok) {
            throw new Error(`API error: ${response.statusText}`);
        }
        
        const result = await response.json();
        
        // Display results
        displayResults(result);
        
    } catch (error) {
        console.error('Error:', error);
        alert(`Error calculating credit score: ${error.message}\n\nMake sure the API server is running:\npython src/api.py`);
    } finally {
        // Hide loading overlay
        document.getElementById('loadingOverlay').classList.remove('active');
    }
});

function displayResults(result) {
    const resultSection = document.getElementById('resultSection');
    const scoreBadge = document.getElementById('scoreBadge');
    const scoreValue = document.getElementById('scoreValue');
    const riskLevel = document.getElementById('riskLevel');
    const confidence = document.getElementById('confidence');
    const probabilityBars = document.getElementById('probabilityBars');
    
    // Update credit score
    scoreValue.textContent = result.credit_score;
    
    // Update badge color based on score
    scoreBadge.className = 'credit-score-badge ' + result.credit_score.toLowerCase();
    
    // Update risk level and confidence
    riskLevel.textContent = result.risk_level;
    confidence.textContent = `${(result.probability * 100).toFixed(1)}%`;
    
    // Create probability bars
    probabilityBars.innerHTML = '';
    const categories = ['Excellent', 'Good', 'Fair', 'Poor'];
    const colors = {
        'Excellent': 'linear-gradient(90deg, #11998e 0%, #38ef7d 100%)',
        'Good': 'linear-gradient(90deg, #667eea 0%, #764ba2 100%)',
        'Fair': 'linear-gradient(90deg, #f093fb 0%, #f5576c 100%)',
        'Poor': 'linear-gradient(90deg, #eb3349 0%, #f45c43 100%)'
    };
    
    categories.forEach(category => {
        const probability = result.all_probabilities[category] || 0;
        const percentage = (probability * 100).toFixed(1);
        
        const barHtml = `
            <div class="probability-bar">
                <div class="probability-label">
                    <span>${category}</span>
                    <span>${percentage}%</span>
                </div>
                <div class="bar-container">
                    <div class="bar-fill" style="width: ${percentage}%; background: ${colors[category]};"></div>
                </div>
            </div>
        `;
        
        probabilityBars.innerHTML += barHtml;
    });
    
    // Show result section
    resultSection.style.display = 'block';
    
    // Scroll to results
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Check API health on page load
window.addEventListener('load', async () => {
    try {
        const response = await fetch(`${API_URL}/health`);
        const health = await response.json();
        
        if (!health.model_loaded) {
            console.warn('Model not loaded on the server');
        }
    } catch (error) {
        console.warn('Could not connect to API server. Make sure to run: python src/api.py');
    }
});
