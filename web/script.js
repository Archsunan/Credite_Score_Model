const API_URL = 'http://localhost:5000';
// Form submission handler
// Form submission handler
const creditForm = document.getElementById('creditForm');
if (creditForm) {
    creditForm.addEventListener('submit', async (e) => {
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
}
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
        'Excellent': '#16a34a', // Green
        'Good': '#2563eb',      // Blue
        'Fair': '#d97706',      // Brown/Amber (as requested)
        'Poor': '#dc2626'       // Red
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

    // Save to history
    saveToHistory(result);


}

function setupActionButtons(result) {
    const printBtn = document.getElementById('printBtn');
    const downloadBtn = document.getElementById('downloadBtn');

    if (printBtn) {
        printBtn.onclick = () => window.print();
    }

    if (downloadBtn) {
        // Change text to PDF
        downloadBtn.innerHTML = '<i class="fa-solid fa-file-pdf"></i> Download PDF';
        // Pass the result object to the function
        downloadBtn.onclick = () => downloadResultAsPDF(result);
    }
}

function downloadResultAsPDF(result) {
    // 1. Collect Input Data
    const inputs = {
        'Age': document.getElementById('age').value,
        'Annual Income': '$' + parseInt(document.getElementById('income').value).toLocaleString(),
        'Employment Length': document.getElementById('employment_length').value + ' years',
        'Loan Amount': '$' + parseInt(document.getElementById('loan_amount').value).toLocaleString(),
        'Loan Term': document.getElementById('loan_term').value + ' months',
        'Credit History': document.getElementById('credit_history_length').value + ' years',
        'Active Credit Lines': document.getElementById('num_credit_lines').value,
        'Debt-to-Income': document.getElementById('debt_to_income').value,
        'Delinquencies': document.getElementById('num_delinquencies').value,
        'Inquiries': document.getElementById('num_inquiries').value
    };

    // 2. Create the Report HTML Structure
    const reportContainer = document.createElement('div');
    reportContainer.className = 'pdf-report';
    reportContainer.style.padding = '40px';
    reportContainer.style.background = 'white';
    reportContainer.style.color = 'black';
    reportContainer.style.fontFamily = "'Inter', sans-serif";
    reportContainer.style.width = '800px';

    // Header
    const header = `
        <div style="border-bottom: 2px solid #4f46e5; padding-bottom: 20px; margin-bottom: 30px; display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 style="color: #4f46e5; margin: 0; font-size: 24px;">CrediSure</h1>
                <p style="margin: 5px 0 0; color: #64748b; font-size: 14px;">Advanced AI Credit Assessment</p>
            </div>
            <div style="text-align: right;">
                <p style="margin: 0; font-weight: bold;">Analysis Report</p>
                <p style="margin: 5px 0 0; color: #64748b; font-size: 12px;">${new Date().toLocaleString()}</p>
            </div>
        </div>
    `;

    // Applicant Details Section
    let detailsHtml = `
        <div style="margin-bottom: 30px;">
            <h3 style="color: #1e293b; border-left: 4px solid #4f46e5; padding-left: 10px; margin-bottom: 15px;">Applicant Profile</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
    `;

    for (const [key, value] of Object.entries(inputs)) {
        detailsHtml += `
            <div style="background: #f8fafc; padding: 10px; border-radius: 6px; font-size: 14px;">
                <span style="color: #64748b; display: block; font-size: 12px;">${key}</span>
                <span style="font-weight: 600; color: #0f172a;">${value}</span>
            </div>
        `;
    }
    detailsHtml += `</div></div>`;

    // Assessment Results Section
    const scoreColor = getScoreColor(result.credit_score);
    const resultsHtml = `
        <div style="margin-bottom: 30px;">
            <h3 style="color: #1e293b; border-left: 4px solid ${scoreColor}; padding-left: 10px; margin-bottom: 15px;">Assessment Results</h3>
            
            <div style="display: flex; gap: 20px; margin-bottom: 20px;">
                <div style="flex: 1; background: ${scoreColor}15; padding: 20px; border-radius: 12px; text-align: center; border: 1px solid ${scoreColor}30;">
                    <span style="display: block; color: #64748b; font-size: 14px; margin-bottom: 5px;">Credit Score</span>
                    <span style="display: block; color: ${scoreColor}; font-size: 32px; font-weight: bold;">${result.credit_score}</span>
                </div>
                <div style="flex: 1; background: #f8fafc; padding: 20px; border-radius: 12px; text-align: center; border: 1px solid #e2e8f0;">
                    <span style="display: block; color: #64748b; font-size: 14px; margin-bottom: 5px;">Risk Level</span>
                    <span style="display: block; color: #0f172a; font-size: 24px; font-weight: 600;">${result.risk_level}</span>
                </div>
                <div style="flex: 1; background: #f8fafc; padding: 20px; border-radius: 12px; text-align: center; border: 1px solid #e2e8f0;">
                    <span style="display: block; color: #64748b; font-size: 14px; margin-bottom: 5px;">Model Confidence</span>
                    <span style="display: block; color: #0f172a; font-size: 24px; font-weight: 600;">${(result.probability * 100).toFixed(1)}%</span>
                </div>
            </div>

            <div style="background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px;">
                <h4 style="margin: 0 0 15px 0; font-size: 14px; color: #64748b;">Probability Distribution</h4>
                ${generateProbabilityBarsHtml(result.all_probabilities)}
            </div>
        </div>
    `;

    // Footer
    const footer = `
        <div style="margin-top: 50px; border-top: 1px solid #e2e8f0; padding-top: 20px; text-align: center; font-size: 12px; color: #94a3b8;">
            <p>Generated by CrediSure AI • For educational purposes only</p>
        </div>
    `;

    reportContainer.innerHTML = header + detailsHtml + resultsHtml + footer;

    // IMPORTANT: Append to body so html2canvas can render it. 
    // We use fixed positioning at 0,0 with a negative z-index to ensure it is "on screen"
    // for the renderer but invalid for the user.
    reportContainer.style.position = 'fixed';
    reportContainer.style.left = '0';
    reportContainer.style.top = '0';
    reportContainer.style.zIndex = '-10000';
    document.body.appendChild(reportContainer);

    // 3. Render PDF
    const opt = {
        margin: 0.5,
        filename: `CrediSure_Report_${Date.now()}.pdf`,
        image: { type: 'jpeg', quality: 0.98 },
        html2canvas: { scale: 2, useCORS: true, logging: false },
        jsPDF: { unit: 'in', format: 'letter', orientation: 'portrait' }
    };

    // Use html2pdf
    html2pdf().set(opt).from(reportContainer).save().then(() => {
        // Cleanup
        document.body.removeChild(reportContainer);
    });
}

// Helper to generates bars for the PDF
function generateProbabilityBarsHtml(probabilities) {
    const colors = { 'Excellent': '#16a34a', 'Good': '#2563eb', 'Fair': '#d97706', 'Poor': '#dc2626' };
    const categories = ['Excellent', 'Good', 'Fair', 'Poor'];

    let html = '';
    categories.forEach(cat => {
        const pct = (probabilities[cat] * 100).toFixed(1);
        const color = colors[cat];
        html += `
            <div style="margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 4px; font-weight: 500;">
                    <span>${cat}</span>
                    <span>${pct}%</span>
                </div>
                <div style="height: 6px; background: #f1f5f9; border-radius: 3px; overflow: hidden;">
                    <div style="width: ${pct}%; height: 100%; background: ${color};"></div>
                </div>
            </div>
        `;
    });
    return html;
}

function getScoreColor(score) {
    const colors = { 'Excellent': '#16a34a', 'Good': '#2563eb', 'Fair': '#d97706', 'Poor': '#dc2626' };
    return colors[score] || '#000';
}

// History Management
function saveToHistory(result) {
    const historyItem = {
        id: Date.now(), // Unique ID
        date: new Date().toLocaleString(),
        score: result.credit_score,
        risk: result.risk_level,
        probability: (result.probability * 100).toFixed(1),
        // Save full inputs
        age: document.getElementById('age').value,
        income: document.getElementById('income').value,
        employment_length: document.getElementById('employment_length').value,
        loan_amount: document.getElementById('loan_amount').value,
        loan_term: document.getElementById('loan_term').value,
        credit_history_length: document.getElementById('credit_history_length').value,
        num_credit_lines: document.getElementById('num_credit_lines').value,
        debt_to_income: document.getElementById('debt_to_income').value,
        num_delinquencies: document.getElementById('num_delinquencies').value,
        num_inquiries: document.getElementById('num_inquiries').value
    };

    let history = JSON.parse(localStorage.getItem('creditHistory') || '[]');
    history.unshift(historyItem); // Add to beginning
    if (history.length > 20) history.pop(); // Keep last 20 records

    localStorage.setItem('creditHistory', JSON.stringify(history));

    // Only load if on history page (though we are navigating away typically, but good for SPA feel if we had it)
    if (document.getElementById('historyPageList')) {
        loadHistoryPage();
    }
}

// Logic for the History Page
function loadHistoryPage() {
    const listContainer = document.getElementById('historyPageList');
    if (!listContainer) return;

    const history = JSON.parse(localStorage.getItem('creditHistory') || '[]');

    if (history.length === 0) {
        listContainer.innerHTML = '<div class="empty-history">No analysis history found. Run calculations in the App to see them here.</div>';
        return;
    }

    const colors = {
        'Excellent': '#10b981',
        'Good': '#5383d0ff',
        'Fair': '#f59e0b',
        'Poor': '#ef4444'
    };

    listContainer.innerHTML = history.map(item => `
        <div class="history-card-detailed" style="border-left: 5px solid ${colors[item.score]}">
            <div class="history-summary">
                <div class="summary-left">
                    <span class="history-date">${item.date}</span>
                    <div class="score-badge" style="background: ${colors[item.score]}20; color: ${colors[item.score]}; border: 1px solid ${colors[item.score]}">
                        ${item.score} (${item.probability}%)
                    </div>
                </div>
                <div class="history-actions">
                    <button class="btn-toggle" onclick="toggleDetails(${item.id})">
                        View Details <i class="fa-solid fa-chevron-down"></i>
                    </button>
                    <button class="btn-delete-item" onclick="deleteHistoryItem(${item.id})" title="Delete this record">
                        <i class="fa-solid fa-trash"></i>
                    </button>
                </div>
            </div>
            
            <div id="details-${item.id}" class="history-details-grid hidden">
                <div class="detail-item"><strong>Risk Level:</strong> ${item.risk}</div>
                <div class="detail-item"><strong>Age:</strong> ${item.age}</div>
                <div class="detail-item"><strong>Income:</strong> $${parseInt(item.income).toLocaleString()}</div>
                <div class="detail-item"><strong>Emp. Length:</strong> ${item.employment_length} yrs</div>
                <div class="detail-item"><strong>Loan Amount:</strong> $${parseInt(item.loan_amount).toLocaleString()}</div>
                <div class="detail-item"><strong>Loan Term:</strong> ${item.loan_term} mos</div>
                <div class="detail-item"><strong>History Length:</strong> ${item.credit_history_length} yrs</div>
                <div class="detail-item"><strong>Credit Lines:</strong> ${item.num_credit_lines}</div>
                <div class="detail-item"><strong>DTI Ratio:</strong> ${item.debt_to_income}</div>
                <div class="detail-item"><strong>Delinquencies:</strong> ${item.num_delinquencies}</div>
                <div class="detail-item"><strong>Inquiries:</strong> ${item.num_inquiries}</div>
            </div>
        </div>
    `).join('');
}

function toggleDetails(id) {
    const details = document.getElementById(`details-${id}`);
    if (details) {
        details.classList.toggle('hidden');
    }
}

function deleteHistoryItem(id) {
    if (confirm('Are you sure you want to delete this record?')) {
        let history = JSON.parse(localStorage.getItem('creditHistory') || '[]');
        history = history.filter(item => item.id !== id);
        localStorage.setItem('creditHistory', JSON.stringify(history));
        loadHistoryPage();
    }
}

function clearHistory() {
    if (confirm('Are you sure you want to clear all analysis history?')) {
        localStorage.removeItem('creditHistory');
        loadHistoryPage();
    }
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
    if (document.getElementById('historyPageList')) {
        loadHistoryPage();
    }

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




