/**
 * Product Bundle Recommender Web Interface
 * Fetches product descriptions from TSV and displays bundle recommendations
 */

const API_BASE_URL = 'http://localhost:5000';
const PRODUCTS_URL = '/assets/products.tsv';

/**
 * Load products from TSV file and populate dropdown
 */
async function loadProducts() {
    try {
        const response = await fetch(PRODUCTS_URL);
        const text = await response.text();
        
        const lines = text.trim().split('\n');
        const products = lines.slice(1).filter(line => line.trim()); // Skip header
        
        const select = document.getElementById('productSelect');
        
        products.forEach(product => {
            const option = document.createElement('option');
            option.value = product.trim();
            option.textContent = product.trim();
            select.appendChild(option);
        });
        
        // Enable button when dropdown is not empty
        const recommendBtn = document.getElementById('recommendBtn');
        select.addEventListener('change', () => {
            recommendBtn.disabled = !select.value;
        });
    } catch (error) {
        console.error('Error loading products:', error);
        showError('Failed to load product list');
    }
}

/**
 * Aggregate bundle suggestions from all recommenders
 * Count how often each product appears in bundles
 */
function extractTopSuggestions(recommendations, selectedProduct, topN = 3) {
    const productScores = {};
    
    // Aggregate all bundles from all recommenders
    recommendations.forEach(rec => {
        if (rec.error || !rec.bundles) return;
        
        rec.bundles.forEach(bundle => {
            bundle.forEach(product => {
                // Skip the selected product
                if (product.toLowerCase() === selectedProduct.toLowerCase()) {
                    return;
                }
                
                // Track product and average confidence across appearances
                if (!productScores[product]) {
                    productScores[product] = { count: 0, totalConfidence: 0 };
                }
                productScores[product].count += 1;
                productScores[product].totalConfidence += rec.confidence;
            });
        });
    });
    
    // Calculate average confidence for each product
    const scored = Object.entries(productScores).map(([product, data]) => ({
        product,
        confidence: data.totalConfidence / data.count,
        frequency: data.count
    }));
    
    // Sort by confidence (descending) and return top N
    return scored
        .sort((a, b) => b.confidence - a.confidence)
        .slice(0, topN);
}

/**
 * Fetch recommendations from API
 */
async function fetchRecommendations(productDescription) {
    try {
        const url = `${API_BASE_URL}/api/v1/bundles?product_description=${encodeURIComponent(productDescription)}&threshold=0.1&top_n=10`;
        const response = await fetch(url);
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.status !== 'success') {
            throw new Error(data.message || 'API returned error status');
        }
        
        return data.recommendations || [];
    } catch (error) {
        console.error('Error fetching recommendations:', error);
        throw error;
    }
}

/**
 * Display results in the UI
 */
function displayResults(suggestions) {
    const resultsList = document.getElementById('resultsList');
    resultsList.innerHTML = '';
    
    if (suggestions.length === 0) {
        resultsList.innerHTML = '<p style="color: #999;">No suggestions available</p>';
        return;
    }
    
    suggestions.forEach((item, index) => {
        const div = document.createElement('div');
        div.className = 'result-item';
        
        const confidencePercent = (item.confidence * 100).toFixed(1);
        
        div.innerHTML = `
            <div class="result-item-name">${index + 1}. ${item.product}</div>
            <div class="result-item-score">Confidence: ${confidencePercent}% | Frequency: ${item.frequency}</div>
        `;
        
        resultsList.appendChild(div);
    });
}

/**
 * Show error message
 */
function showError(message) {
    const errorDiv = document.getElementById('error');
    errorDiv.textContent = message;
    errorDiv.classList.add('show');
    setTimeout(() => {
        errorDiv.classList.remove('show');
    }, 5000);
}

/**
 * Handle recommendation button click
 */
async function handleRecommend() {
    const productSelect = document.getElementById('productSelect');
    const selectedProduct = productSelect.value;
    
    if (!selectedProduct) {
        showError('Please select a product');
        return;
    }
    
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    const error = document.getElementById('error');
    
    // Reset UI
    error.classList.remove('show');
    results.classList.remove('show');
    loading.classList.add('show');
    
    try {
        const recommendations = await fetchRecommendations(selectedProduct);
        const suggestions = extractTopSuggestions(recommendations, selectedProduct, 3);
        
        displayResults(suggestions);
        results.classList.add('show');
    } catch (err) {
        showError('Failed to get recommendations. Please try again.');
    } finally {
        loading.classList.remove('show');
    }
}

/**
 * Initialize on page load
 */
document.addEventListener('DOMContentLoaded', () => {
    loadProducts();
    
    document.getElementById('recommendBtn').addEventListener('click', handleRecommend);
    document.getElementById('productSelect').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            handleRecommend();
        }
    });
    
    // Disable button initially
    document.getElementById('recommendBtn').disabled = true;
});
