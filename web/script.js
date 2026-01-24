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
    const scoreValue = document.getElementById('scoreValue');
    const riskLevel = document.getElementById('riskLevel');
    const confidence = document.getElementById('confidence');
    const probabilityBars = document.getElementById('probabilityBars');
    const circle = document.getElementById('progressRing');
    const scoreCircle = document.getElementById('scoreCircle');

    // Define Colors and Offsets
    // Radius = 90, Circumference = 2 * PI * 90 ≈ 565.48
    const circumference = 565.48;
    const colors = {
        'Excellent': '#10b981', // green
        'Good': '#5383d0ff',      // blue
        'Fair': '#f59e0b',      // amber
        'Poor': '#ef4444'       // red
    };

    const color = colors[result.credit_score] || '#ffffff';

    // Update Text Data
    scoreValue.textContent = result.credit_score;

    // Update score circle background color
    if (scoreCircle) {
        scoreCircle.style.background = `radial-gradient(circle at center, ${color}20, transparent)`;
        scoreCircle.style.borderColor = color;
    }

    riskLevel.textContent = result.risk_level;

    confidence.textContent = `${(result.probability * 100).toFixed(1)}%`;

    // Update Circle Animation
    const offset = circumference - (result.probability * circumference);
    circle.style.strokeDashoffset = offset;
    circle.style.stroke = color;

    // Create probability bars
    probabilityBars.innerHTML = '';
    const categories = ['Excellent', 'Good', 'Fair', 'Poor'];

    // Find max probability to scale bars relative to each other if wanted, 
    // but standard percentage width is usually better for clarity.

    categories.forEach(category => {
        const probability = result.all_probabilities[category] || 0;
        const percentage = (probability * 100).toFixed(1);
        const barColor = colors[category];

        const barHtml = `
            <div class="probability-bar">
                <div class="probability-label">
                    <span>${category}</span>
                    <span>${percentage}%</span>
                </div>
                <div class="bar-container">
                    <div class="bar-fill" style="width: ${percentage}%; background-color: ${barColor}; box-shadow: 0 0 10px ${barColor}80;"></div>
                </div>
            </div>
        `;

        probabilityBars.innerHTML += barHtml;
    });


    // Show result section with flex (removed display:none)
    resultSection.classList.remove('hidden');

    // Load feature importance
    loadFeatureImportance();

    // Scroll to results
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

async function loadFeatureImportance() {
    const wrapper = document.getElementById('featureImportanceWrapper');
    const barsContainer = document.getElementById('featureImportanceBars');

    try {
        const response = await fetch(`${API_URL}/feature_importance`);
        if (!response.ok) return;

        const data = await response.json();
        const features = data.features; // Array of {feature: 'name', importance: 0.123}

        // Sort by importance desc and take top 5
        features.sort((a, b) => b.importance - a.importance);
        const topFeatures = features.slice(0, 5);
        const maxImportance = topFeatures[0].importance;

        barsContainer.innerHTML = '';

        topFeatures.forEach(item => {
            // Normalize width relative to the top feature
            const percentage = (item.importance / maxImportance) * 100;
            // Clean up feature name (e.g. "debt_to_income" -> "Debt to Income")
            const displayName = item.feature
                .split('_')
                .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                .join(' ');

            const barHtml = `
                <div class="probability-bar">
                    <div class="probability-label">
                        <span>${displayName}</span>
                        <span>${(item.importance * 100).toFixed(1)}%</span>
                    </div>
                    <div class="bar-container">
                        <div class="bar-fill" style="width: ${percentage}%; background-color: #a855f7; box-shadow: 0 0 10px #a855f780;"></div>
                    </div>
                </div>
            `;
            barsContainer.innerHTML += barHtml;
        });

        wrapper.classList.remove('hidden');

    } catch (e) {
        console.error("Failed to load feature importance", e);
    }
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


