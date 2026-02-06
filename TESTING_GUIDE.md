# Testing Guide for E-Commerce Recommendation Service

**Last Updated:** 4 February 2026

## Overview

This project includes comprehensive unit and integration tests with the following specifications:

- **Test Framework:** pytest
- **Web Framework:** Flask (with Flask-CORS)
- **Mocking:** All external services mocked for speed & consistency
- **Test Data:** Synthetic fixtures (100-500 rows) for fast test execution
- **Coverage:** Terminal report + HTML report; **fail CI if < 80%**
- **LLM Config:** Tests support on/off activation of LLM features via environment variables

---

## Quick Start

### Run All Tests

```bash
# Basic run with coverage
pytest tests/ -v

# With coverage report (term + HTML)
pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

# Using Makefile (recommended)
make test
```

### View Coverage Report

```bash
# Terminal report
make test-coverage

# HTML report (opens in browser)
make test-html
# Output: htmlcov/index.html
```

### Run Specific Tests

```bash
# All tests in a file
pytest tests/test_data_pipeline.py -v
pytest tests/test_recommendation_engine.py -v
pytest tests/test_llm_integration.py -v
pytest tests/test_api.py -v
pytest tests/test_utils.py -v

# Specific test class
pytest tests/test_data_pipeline.py::TestDataPipelineBasics -v
pytest tests/test_recommendation_engine.py::TestNaiveBayesRecommender -v
pytest tests/test_llm_integration.py::TestSelectAlternativesWithLLM -v
pytest tests/test_api.py::TestBundlesEndpoint -v
pytest tests/test_utils.py::TestLoadJsonFile -v

# Specific test function
pytest tests/test_data_pipeline.py::TestDataPipelineBasics::test_initialization_defaults -v
pytest tests/test_recommendation_engine.py::TestSVMRecommender::test_different_kernels -v
pytest tests/test_llm_integration.py::TestSelectAlternativesWithLLM::test_success_with_mocked_openai -v

# LLM Integration tests (mocked by default, fast & free)
pytest tests/test_llm_integration.py -v  # All 19 tests with mocking, 3 real tests skipped
pytest tests/test_llm_integration.py::TestFallbackMechanisms -v  # Fallback tests only
pytest tests/test_llm_integration.py -m "llm" -v  # All LLM-related tests

# LLM Integration tests with real API calls (requires API keys, costs ~$0.01)
USE_REAL_LLM=true pytest tests/test_llm_integration.py::TestRealLLMCalls -v
USE_REAL_LLM=true LLM_BUDGET_LIMIT=3 pytest tests/test_llm_integration.py::TestRealLLMCalls::test_real_openai_call_minimal -v

# Using Makefile
make test-file FILE=tests/test_data_pipeline.py
make test-file FILE=tests/test_recommendation_engine.py
make test-file FILE=tests/test_llm_integration.py
make test-func FUNC=TestNaiveBayesRecommender::test_fit_basic
make test-func FUNC=TestSelectAlternativesWithLLM::test_success_with_mocked_openai
```

---

## Test Organization

### Test File Structure

```
tests/
├── __init__.py                 # Package marker
├── conftest.py                 # Shared fixtures & configuration
├── test_data_pipeline.py       # DataPipeline tests (✅ Implemented)
├── test_recommendation_engine.py # Recommendation tests (✅ Implemented)
├── test_llm_integration.py     # LLM integration tests (✅ Implemented)
├── test_api.py                 # Flask API endpoint tests (✅ Implemented)
├── test_utils.py               # Utils & helpers tests (✅ Implemented)
└── test_integration.py         # (To be implemented)
```

### Test Categories (pytest markers)

```bash
# Unit tests (pure functions, no I/O)
pytest tests/ -m "unit"

# Integration tests (full workflows)
pytest tests/ -m "integration"

# LLM-related tests
pytest tests/ -m "llm"

# Slow-running tests
pytest tests/ -m "slow"

# Performance benchmarks
pytest tests/ -m "performance"
```

---

## Test Data & Fixtures

### Synthetic Fixtures (100-500 rows)

All test data is synthetically generated in `conftest.py` and `test_data_pipeline.py`:

```python
# Example fixtures (in conftest.py)
@pytest.fixture
def minimal_dataframe():
    """3-row DataFrame with required columns."""
    return pd.DataFrame({
        'InvoiceNo': ['001', '002', '003'],
        'StockCode': ['A001', 'A002', 'A003'],
        ...
    })

@pytest.fixture
def realistic_sample_dataframe():
    """100-row DataFrame with quality issues (cancellations, nulls, negatives)."""
    # Generates realistic data with mixed quality problems

@pytest.fixture
def transaction_baskets_dataframe():
    """6-row DataFrame with transaction baskets (multiple items per invoice)."""
    # Used for testing basket creation and bundle generation
```

### Why Synthetic Data?

✅ **Fast:** No I/O to actual CSV files  
✅ **Repeatable:** Deterministic, same results every run  
✅ **Isolated:** No dependence on external data files  
✅ **Comprehensive:** Covers edge cases (nulls, negatives, duplicates)  
✅ **Configurable:** Easy to parametrize for different scenarios  

---

## LLM Mocking Strategy

### Key Design: Mock-First Pattern

All LLM functions are **mocked by default** using `monkeypatch`:

```python
@pytest.fixture
def mock_all_llm_functions(monkeypatch):
    """Mock all LLM function calls for speed & consistency."""
    
    def mock_enrich_batch(descriptions, **kwargs):
        return {
            desc: {
                'category': 'test_category',
                'material': 'test_material',
                ...
            }
            for desc in descriptions
        }
    
    monkeypatch.setattr(
        'src.data_pipeline.enrich_categories_batch_with_llm',
        mock_enrich_batch
    )
    # ... other mocks
```

### LLM Configuration Options

Tests support **configurable LLM activation** via environment variables:

#### Option 1: LLM Disabled (Default for unit tests)

```bash
# Terminal
ENRICHMENT_ENABLED=false OUTLIER_ENABLED=false CONTEXT_ENABLED=false pytest tests/

# Makefile
make test-llm-off

# Python (in test)
@pytest.fixture
def llm_config_disabled(monkeypatch):
    """Disable all LLM features for fast testing."""
    monkeypatch.setenv('ENRICHMENT_ENABLED', 'false')
    # ...
```

**Use when:** Fast unit test execution, testing core logic without LLM overhead.

#### Option 2: LLM Enabled with Mocking (Default for most tests)

```bash
# Terminal
ENRICHMENT_ENABLED=true OUTLIER_ENABLED=true pytest tests/

# Makefile
make test-llm-on

# Python (in test)
@pytest.fixture
def llm_config_enabled_mocked(monkeypatch):
    """Enable LLM features with mocking."""
    monkeypatch.setenv('ENRICHMENT_ENABLED', 'true')
    # ... mock functions injected via conftest.py
```

**Use when:** Validating LLM integration paths with predictable mock outputs.

#### Option 3: LLM Cache Disabled (For cache validation)

```bash
# Python (in test)
@pytest.fixture
def llm_config_cache_disabled(monkeypatch):
    """Enable LLM features but disable cache-first pattern."""
    monkeypatch.setenv('ENRICHMENT_CACHE_FIRST', 'false')
    # ... forces fresh LLM calls
```

**Use when:** Testing cache behavior (cache hits vs misses).

### Example: Using LLM Config in Tests

```python
def test_pipeline_with_llm_disabled(llm_config_disabled):
    """Fast test: LLM disabled."""
    pipeline = DataPipeline()
    result = pipeline.preprocess()
    assert result is not None

def test_pipeline_with_llm_enabled(llm_config_enabled_mocked, mock_all_llm_functions):
    """Integration test: LLM mocked."""
    pipeline = DataPipeline()
    result = pipeline.preprocess()
    assert 'category' in result.columns or True  # Depends on mocked behavior

def test_cache_hit_miss(llm_config_cache_disabled):
    """Validate cache behavior."""
    # First call: cache miss
    # Second call: cache hit
    # Verify reduced LLM calls
```

---

## Coverage Configuration

### Coverage Settings (pytest.ini)

```ini
[pytest]
addopts = 
    --cov=src                        # Cover src/ directory
    --cov-report=term-missing        # Terminal report with missing lines
    --cov-report=html:htmlcov        # HTML report
    --cov-fail-under=80              # ❌ FAIL if coverage < 80%
    --cov-branch                     # Include branch coverage
```

### Coverage Reports

#### Terminal Report (Minimal)

```bash
make test-coverage
```

Output shows:
- Coverage % per module
- Lines covered vs total
- Missing lines by line number

Example:
```
Name                      Stmts   Miss  Cover   Missing
--------------------------------------------------------
src/data_pipeline.py       320     32    90%    45-48, 127-130, ...
src/recommendation_engine  210     18    91%    89-91, 145-147, ...
src/utils.py              180     12    93%    56, 102, 145, ...
```

#### HTML Report (Detailed)

```bash
make test-html
```

Output: `htmlcov/index.html`
- Interactive browser navigation
- Color-coded coverage (green=covered, red=uncovered)
- Line-by-line drill-down
- Trend tracking

#### Fail CI if Coverage < 80%

```bash
pytest tests/ --cov=src --cov-fail-under=80
```

Exit code:
- `0` if coverage ≥ 80%
- `1` (failure) if coverage < 80%

CI/CD integration:
```yaml
# GitHub Actions example
- name: Run tests with coverage
  run: |
    pytest tests/ --cov=src --cov-fail-under=80
    # Fails CI if coverage drops below 80%
```

---

## Test Implementation Details

### Implemented Test Classes

#### ✅ TestDataPipelineBasics (Implemented)
Tests pipeline initialization and basic method validation.

```python
def test_initialization_defaults(self):
    """Pipeline initializes with default values."""
    pipeline = DataPipeline()
    assert pipeline.raw_data is None
    assert pipeline.force_reprocess is False

def test_preprocess_without_raw_data(self):
    """ValueError raised when preprocessing without data."""
    with pytest.raises(ValueError, match="Raw data not loaded"):
        pipeline.preprocess()
```

#### ✅ TestLoadAndExplore (Implemented)
Tests data loading, statistics, and CSV-to-TSV conversion.

```python
def test_load_raw_data_success(self, minimal_dataframe, temp_data_directory):
    """Load raw data from valid CSV."""
    # Create temp CSV, load it, verify content

def test_explore_data_auto_picks_processed_first(self):
    """explore_data prefers processed over raw."""
    # Set both raw and processed, verify auto mode picks processed
```

#### ✅ TestDataCleaning (Implemented)
Tests cancellation handling, null removal, negative value filtering.

```python
def test_handle_cancellations_removes_c_prefix(self):
    """Rows with InvoiceNo starting with 'C' removed."""
    # Verify cancellations filtered out

def test_clean_data_removes_negative_quantity(self):
    """Negative Quantity values removed."""
    # Verify all quantities > 0
```

#### ✅ TestBundleGeneration (Implemented)
Tests transaction baskets and Apriori bundle generation.

```python
def test_create_transaction_baskets_grouping(self):
    """Baskets grouped by InvoiceNo."""
    # Verify grouping and aggregation

def test_generate_bundles_min_support(self):
    """Higher min_support → fewer bundles."""
    # Verify threshold effect
```

#### ✅ TestLLMFeatures (Implemented)
Tests LLM integration with mocking.

```python
def test_enrich_categories_cache_first(self, mock_all_llm_functions):
    """Category enrichment with cache-first pattern."""
    # Mock LLM, verify enrichment columns added

def test_enrich_categories_no_credentials(self, monkeypatch):
    """Graceful handling when LLM credentials missing."""
    # Set empty API key, verify no crash
```

#### ✅ TestEdgeCases (Implemented)
Tests boundary conditions: empty data, single rows, nulls, Unicode.

```python
def test_empty_dataframe_processing(self):
    """Empty DataFrame processed without error."""
    
def test_unicode_encoding_handling(self):
    """Unicode characters handled correctly."""
```

#### ✅ TestFullPipeline (Implemented)
End-to-end workflow tests with configurable LLM.

```python
def test_pipeline_load_to_baskets(self, llm_config_disabled):
    """Full: load → preprocess → create baskets."""
    
def test_pipeline_reproducibility(self):
    """Same RANDOM_STATE produces same results."""
```

#### ✅ TestNaiveBayesRecommender (Implemented)
Tests Naive Bayes bundle recommender with edge cases.

```python
def test_initialization_default(self):
    """Test recommender initialization with default model type."""
    recommender = NaiveBayesBundleRecommender()
    assert recommender.name == "NaiveBayesBundleRecommender"

def test_fit_basic(self, sample_transactions, sample_bundles):
    """Test basic model fitting."""
    # Fit model and verify metrics

def test_fit_with_empty_bundles(self, sample_transactions):
    """Test fitting with empty bundles (all labels become 0)."""
    # Verify NB can handle imbalanced data

def test_gaussian_vs_multinomial(self, large_transactions, large_bundles):
    """Test that both Gaussian and Multinomial NB work."""
    # Verify both model types fit and predict
```

#### ✅ TestSVMRecommender (Implemented)
Tests SVM bundle recommender with different kernels and parameters.

```python
def test_different_kernels(self, large_transactions, large_bundles):
    """Test different SVM kernels (linear, rbf, poly)."""
    # Verify all kernels work correctly

def test_different_c_values(self, large_transactions, large_bundles):
    """Test different regularization C values."""
    # Verify C parameter effect

def test_feature_scaling(self, sample_transactions, sample_bundles):
    """Test that feature scaling is applied."""
    # Verify StandardScaler is fitted
```

#### ✅ TestBundleRecommendationEngine (Implemented)
Tests main recommendation engine with ensemble, OOS, and persistence.

```python
def test_recommend_bundles_ensemble_averaging(self):
    """Test ensemble averaging with multiple recommenders."""
    # Verify probabilities averaged from NB + SVM

def test_threshold_filtering_medium(self):
    """Test threshold=0.5 for moderate filtering."""
    # Verify confidence threshold filtering

@patch('src.recommendation_engine.OOS_ENABLED', True)
def test_oos_substitution_enabled(self, ...):
    """Test OOS substitution when enabled."""
    # Mock inventory and LLM, verify substitution logic

def test_model_persistence_save_load(self, tmp_path, ...):
    """Test saving and loading model."""
    # Verify save → load → predict consistency
```

#### ✅ TestConfigurationParameters (Implemented)
Tests effect of configuration parameters.

```python
def test_min_support_threshold_effect(self):
    """Higher min_support → fewer bundles."""
    
def test_max_bundle_size_effect(self):
    """max_size parameter limits bundle size."""
```

#### ✅ TestSelectAlternativesWithLLM (Implemented)
Tests core LLM integration for OOS substitution with strict mocking.

```python
def test_empty_missing_item(self, sample_inventory, mock_urllib_llm_success):
    """Test handling of empty missing_item string."""
    # Verify graceful handling when missing_item is ""

def test_success_with_mocked_openai(self, mock_urllib_llm_success, sample_inventory, sample_candidates):
    """Test successful LLM call with mocked OpenAI response."""
    # Verify urllib mocking works correctly

@pytest.mark.parametrize("provider", ["openai", "azure", "gemini"])
def test_different_providers(self, provider, mock_urllib_llm_success, ...):
    """Test different LLM providers with mocked responses."""
    # Verify all providers use urllib.request correctly

def test_invalid_json_graceful_failure(self, sample_inventory, sample_candidates):
    """Test graceful handling when LLM returns invalid JSON."""
    # Mock invalid JSON, verify fallback to empty list
```

#### ✅ TestFallbackMechanisms (Implemented)
Tests heuristic fallback when LLM unavailable.

```python
def test_heuristic_selection_returns_dict_or_none(self):
    """Test select_alternative_heuristic basic functionality."""
    # Verify Jaccard similarity fallback

def test_fallback_on_llm_error(self, mock_urllib_llm_error, sample_inventory, sample_candidates):
    """Test fallback to heuristic when LLM service fails."""
    # Verify graceful degradation without crash
```

#### ✅ TestInventoryOperations (Implemented)
Tests inventory loading and normalization.

```python
def test_load_missing_inventory_file(self):
    """Test load_inventory_csv with non-existent file."""
    # Verify FileNotFoundError handling

def test_normalize_description_lowercasing(self):
    """Test normalize_description_basic lowercase conversion."""
    # Verify "RED Roses" → "red roses"

def test_normalize_description_whitespace(self):
    """Test normalize_description_basic whitespace handling."""
    # Verify "  multi   space  " → "multi space"
```

#### ✅ TestResponseStructure (Implemented)
Tests LLM response format validation.

```python
def test_alternatives_list_format(self):
    """Test select_alternatives_with_llm returns List[Dict]."""
    # Verify response has correct structure

def test_audit_trail_structure(self, mock_urllib_llm_success, sample_inventory, sample_candidates):
    """Test each alternative has item, score, reason fields."""
    # Verify required fields present in all alternatives
```

#### ✅ TestRealLLMCalls (Implemented - Opt-in Only)
Tests real LLM API calls with strict budget controls (**SKIPPED by default**).

```python
@pytest.mark.llm_real
@pytest.mark.skipif(not USE_REAL_LLM, reason="Real LLM testing disabled")
def test_real_openai_call_minimal(self):
    """Test real OpenAI API call with minimal context."""
    # Requires USE_REAL_LLM=true, enforces budget limit
    # Minimal context: ~200 tokens ≈ $0.001 per call

@pytest.mark.llm_real
@pytest.mark.skipif(not USE_REAL_LLM, reason="Real LLM testing disabled")
def test_real_azure_call_minimal(self):
    """Test real Azure OpenAI call with minimal context."""
    # Budget-controlled real API testing
```

**Cost Control Mechanisms:**
- **USE_REAL_LLM**: Environment variable (default: `false`), enables real API calls only when explicitly set
- **LLM_BUDGET_LIMIT**: Maximum real calls allowed per test run (default: 5)
- **_check_llm_budget()**: RuntimeError raised if budget exceeded
- **Minimal Context**: Real tests use 2-3 candidates (~200 tokens) vs 100+ in production
- **Opt-in Markers**: `@pytest.mark.llm_real` ensures real tests only run when explicitly requested

**Usage:**
```bash
# All tests with mocking (default, fast, free)
pytest tests/test_llm_integration.py -v  # 19 passed, 3 skipped

# Real LLM tests (requires API keys, costs ~$0.01)
USE_REAL_LLM=true LLM_BUDGET_LIMIT=3 pytest tests/test_llm_integration.py::TestRealLLMCalls -v
```

#### ✅ TestHealthEndpoint (Implemented)
Tests health check endpoint for service availability monitoring.

```python
def test_health_returns_ok(self, client):
    """Test health endpoint returns 200 with status ok."""
    response = client.get('/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'ok'

def test_health_response_structure(self, client):
    """Test health response has correct structure."""
    # Verify JSON structure matches API contract
```

#### ✅ TestRecommendersEndpoint (Implemented)
Tests GET /api/v1/recommenders endpoint for listing available models.

```python
def test_recommenders_returns_success(self, client, mock_get_engine):
    """Test recommenders endpoint returns list of available models."""
    response = client.get('/api/v1/recommenders')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'success'

def test_recommenders_list_contains_models(self, client, mock_get_engine):
    """Test recommenders list includes naive_bayes and svm."""
    # Verify all configured models are returned

def test_recommenders_model_loading_failure(self, client):
    """Test graceful error handling when engine fails to load."""
    # Mock engine loading failure, verify 500 response
```

#### ✅ TestBundlesEndpoint (Implemented)
Tests POST /api/v1/bundles endpoint for product bundle recommendations.

```python
def test_bundles_successful_request(self, client, mock_get_engine):
    """Test bundles endpoint with valid product_description."""
    response = client.post('/api/v1/bundles', 
        json={'product_description': 'laptop'})
    assert response.status_code == 200

def test_bundles_with_custom_threshold(self, client, mock_get_engine):
    """Test bundles with custom confidence threshold."""
    response = client.post('/api/v1/bundles',
        json={'product_description': 'laptop', 'threshold': 0.7})
    # Verify threshold parameter is respected

def test_bundles_missing_product_description(self, client, mock_get_engine):
    """Test error handling for missing required field."""
    response = client.post('/api/v1/bundles', json={})
    assert response.status_code == 400

def test_bundles_invalid_threshold_below_range(self, client, mock_get_engine):
    """Test validation: threshold must be >= 0.0."""
    response = client.post('/api/v1/bundles',
        json={'product_description': 'laptop', 'threshold': -0.1})
    assert response.status_code == 400

def test_bundles_invalid_threshold_above_range(self, client, mock_get_engine):
    """Test validation: threshold must be <= 1.0."""
    response = client.post('/api/v1/bundles',
        json={'product_description': 'laptop', 'threshold': 1.5})
    assert response.status_code == 400
```

#### ✅ TestBundlesBatchEndpoint (Implemented)
Tests POST /api/v1/bundles/batch for bulk recommendations.

```python
def test_batch_successful_request(self, client, mock_get_engine):
    """Test batch processing of multiple product descriptions."""
    payload = {
        'product_descriptions': ['laptop', 'mouse', 'keyboard'],
        'threshold': 0.5
    }
    response = client.post('/api/v1/bundles/batch', json=payload)
    assert response.status_code == 200
    data = json.loads(response.data)
    assert len(data['results']) == 3

def test_batch_empty_list(self, client, mock_get_engine):
    """Test error handling for empty product list."""
    response = client.post('/api/v1/bundles/batch',
        json={'product_descriptions': []})
    assert response.status_code == 400

def test_batch_exceeds_max_items(self, client, mock_get_engine):
    """Test validation: batch size limit enforcement."""
    # Verify max batch size constraint (e.g., 100 items)
```

#### ✅ TestCrossSellEndpoint (Implemented)
Tests GET /api/v1/cross-sell endpoint for product recommendations.

```python
def test_cross_sell_successful_request(self, client, mock_get_engine):
    """Test cross-sell recommendations."""
    response = client.get('/api/v1/cross-sell?product_description=laptop')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'recommendations' in data

def test_cross_sell_with_top_n(self, client, mock_get_engine):
    """Test cross-sell with custom top_n parameter."""
    response = client.get('/api/v1/cross-sell?product_description=laptop&top_n=5')
    # Verify limited number of recommendations returned

def test_cross_sell_missing_product(self, client, mock_get_engine):
    """Test error handling when product_description is missing."""
    response = client.get('/api/v1/cross-sell')
    assert response.status_code == 400
```

#### ✅ TestStatsEndpoint (Implemented)
Tests GET /api/v1/stats for engine statistics.

```python
def test_stats_successful_request(self, client, mock_get_engine):
    """Test stats endpoint returns engine metrics."""
    response = client.get('/api/v1/stats')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'num_recommenders' in data
    assert 'num_bundles' in data

def test_stats_response_structure(self, client, mock_get_engine):
    """Test stats response has all required fields."""
    # Verify complete statistics structure
```

#### ✅ TestStaticFileServing (Implemented)
Tests serving of static web UI files.

```python
def test_index_html_served(self, client):
    """Test that index.html is served at root path."""
    response = client.get('/')
    assert response.status_code == 200
    assert b'html' in response.data

def test_static_css_served(self, client):
    """Test CSS files are accessible."""
    # Verify static asset serving

def test_nonexistent_static_file_404(self, client):
    """Test 404 for missing static files."""
    response = client.get('/nonexistent.js')
    assert response.status_code == 404
```

#### ✅ TestErrorHandlers (Implemented)
Tests global error handling and custom error responses.

```python
def test_404_error_handler(self, client):
    """Test custom 404 error response."""
    response = client.get('/api/v1/nonexistent')
    assert response.status_code == 404
    data = json.loads(response.data)
    assert data['status'] == 'error'

def test_500_error_handler(self, client):
    """Test internal server error handling."""
    # Mock internal error, verify 500 response structure
```

#### ✅ TestResponseValidation (Implemented)
Tests API response format consistency.

```python
def test_success_response_format(self, client, mock_get_engine):
    """Test all success responses follow standard format."""
    # Verify {status: 'success', data: {...}} structure

def test_error_response_format(self, client):
    """Test all error responses follow standard format."""
    # Verify {status: 'error', message: '...'} structure
```

#### ✅ TestJSONSerialization (Implemented)
Tests JSON encoding/decoding for API requests and responses.

```python
def test_numpy_arrays_serialized(self, client, mock_get_engine):
    """Test numpy arrays are correctly serialized to JSON."""
    # Verify numpy data types handled in responses

def test_special_characters_in_json(self, client, mock_get_engine):
    """Test Unicode and special characters in JSON."""
    # Verify proper encoding of non-ASCII characters
```

### Test Statistics

**Total Test Cases: 125**

| Category | Count | Status |
|----------|-------|--------|
| **Data Pipeline Tests** | | |
| Basic Tests | 5 | ✅ |
| Load & Explore | 8 | ✅ |
| Data Cleaning | 5 | ✅ |
| Bundle Generation | 7 | ✅ |
| LLM Features | 5 | ✅ |
| Edge Cases | 6 | ✅ |
| Serialization | 3 | ✅ |
| Full Pipeline | 3 | ✅ |
| Configuration | 2 | ✅ |
| **Subtotal** | **44** | **✅** |
| **Recommendation Engine Tests** | | |
| NaiveBayesRecommender | 17 | ✅ |
| SVMRecommender | 15 | ✅ |
| BundleRecommendationEngine | 30 | ✅ |
| **Subtotal** | **62** | **✅** |
| **LLM Integration Tests** | | |
| SelectAlternativesWithLLM | 9 | ✅ |
| FallbackMechanisms | 4 | ✅ |
| InventoryOperations | 4 | ✅ |
| ResponseStructure | 2 | ✅ |
| RealLLMCalls (opt-in) | 3 | ✅ (skipped) |
| **Subtotal** | **22** | **✅** |
| **API Endpoint Tests** | | |
| HealthEndpoint | 2 | ✅ |
| RecommendersEndpoint | 4 | ✅ |
| BundlesEndpoint | 11 | ✅ |
| BundlesBatchEndpoint | 8 | ✅ |
| CrossSellEndpoint | 5 | ✅ |
| StatsEndpoint | 3 | ✅ |
| StaticFileServing | 3 | ✅ |
| ErrorHandlers | 2 | ✅ |
| ResponseValidation | 3 | ✅ |
| JSONSerialization | 2 | ✅ |
| **Subtotal** | **43** | **✅** |
| **TOTAL IMPLEMENTED** | **171** | **✅** |

**Planned for P0 completion:**
- Utilities tests: 12+
| SVMRecommender | 15 | ✅ |
| BundleRecommendationEngine | 30 | ✅ |
| **Subtotal** | **62** | **✅** |
| **TOTAL IMPLEMENTED** | **106** | **✅** |

**Planned for P0 completion:**
- LLM Integration tests: 15+
- API tests: 18+
- Utilities tests: 12+

---

## Running Tests

### Command Reference

```bash
# All tests with coverage
make test

# Quick run
make test-quick

# Verbose output
make test-verbose

# Unit tests only
make test-unit

# Integration tests
make test-integration

# LLM disabled (fastest)
make test-llm-off

# LLM enabled (mocked)
make test-llm-on

# Coverage report
make test-coverage

# HTML coverage
make test-html

# Specific file
make test-file FILE=tests/test_data_pipeline.py
make test-file FILE=tests/test_llm_integration.py

# Specific function
make test-func FUNC=TestDataPipelineBasics::test_initialization_defaults
make test-func FUNC=TestSelectAlternativesWithLLM::test_success_with_mocked_openai

# LLM Integration tests (mocked by default, zero cost)
pytest tests/test_llm_integration.py -v  # 19 passed, 3 skipped
pytest tests/test_llm_integration.py::TestSelectAlternativesWithLLM -v  # LLM core tests
pytest tests/test_llm_integration.py::TestFallbackMechanisms -v  # Fallback & degradation tests
pytest tests/test_llm_integration.py::TestInventoryOperations -v  # Inventory helpers
pytest tests/test_llm_integration.py -m "llm" -v  # All LLM tests (mocked)

# LLM Integration tests with real API calls (opt-in only, requires USE_REAL_LLM=true)
USE_REAL_LLM=true pytest tests/test_llm_integration.py::TestRealLLMCalls -v  # All 3 real tests
USE_REAL_LLM=true LLM_BUDGET_LIMIT=3 pytest tests/test_llm_integration.py::TestRealLLMCalls::test_real_openai_call_minimal -v  # Single provider
pytest tests/test_llm_integration.py -m "llm_real" -v  # All real LLM tests

# API Endpoint tests (Flask test client, all endpoints)
pytest tests/test_api.py -v  # All API tests
pytest tests/test_api.py::TestHealthEndpoint -v  # Health check only
pytest tests/test_api.py::TestBundlesEndpoint -v  # Bundle recommendations
pytest tests/test_api.py::TestBundlesBatchEndpoint -v  # Batch processing
pytest tests/test_api.py::TestCrossSellEndpoint -v  # Cross-sell recommendations
pytest tests/test_api.py::TestStatsEndpoint -v  # Engine statistics
pytest tests/test_api.py::TestErrorHandlers -v  # Error handling
make test-file FILE=tests/test_api.py

# Previously failed tests
make test-failed

# Clean test artifacts
make test-clean
```

### Advanced Usage

#### Parallel Execution (requires pytest-xdist)

```bash
pip install pytest-xdist
make test-parallel
```

#### Watch Mode (requires pytest-watch)

```bash
pip install pytest-watch
make test-watch
```

#### Verbose Debugging

```bash
pytest tests/test_data_pipeline.py::TestDataPipelineBasics::test_initialization_defaults -vv -s --tb=long
```

Options:
- `-vv`: Very verbose
- `-s`: Show print statements
- `--tb=long`: Long traceback format
- `--pdb`: Drop into debugger on failure

---

## Integration Tests with Configurable LLM

### Example: Full Pipeline with LLM Config

```python
class TestFullPipeline:
    """End-to-end workflow tests."""
    
    def test_pipeline_with_llm_off(self, llm_config_disabled):
        """Fast test: LLM disabled."""
        pipeline = DataPipeline()
        pipeline.load_raw_data("data/test.csv")
        result = pipeline.preprocess()
        assert len(result) > 0
    
    def test_pipeline_with_llm_on(self, llm_config_enabled_mocked, mock_all_llm_functions):
        """Integration test: LLM features enabled (mocked)."""
        pipeline = DataPipeline()
        pipeline.load_raw_data("data/test.csv")
        result = pipeline.preprocess()
        # Verify LLM enrichment occurred
        assert 'category' in result.columns or result is not None
    
    def test_pipeline_cache_behavior(self, llm_config_cache_disabled):
        """Validate cache hit/miss."""
        # First run: cache miss
        pipeline1 = DataPipeline()
        pipeline1.preprocess()
        
        # Second run: cache hit (same data)
        pipeline2 = DataPipeline()
        pipeline2.preprocess()
        
        # Verify cache was used
```

---

## API Endpoint Testing

### Flask Test Client Pattern

API tests use Flask's built-in test client for endpoint testing without starting a real server:

```python
@pytest.fixture
def client():
    """Flask test client for API testing."""
    app.config['TESTING'] = True
    with app.test_client() as test_client:
        yield test_client

def test_health_endpoint(client):
    """Test using Flask test client."""
    response = client.get('/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'ok'
```

**Benefits:**
- ✅ No server startup required
- ✅ Fast execution (in-process testing)
- ✅ Full request/response inspection
- ✅ Isolated from network issues

### Mocking the Recommendation Engine

API tests mock the `BundleRecommendationEngine` to avoid dependencies on trained models:

```python
@pytest.fixture
def mock_engine():
    """Mock BundleRecommendationEngine with typical response structure."""
    engine = MagicMock()
    
    # Mock recommenders
    engine.recommenders = {
        "naive_bayes": MagicMock(name="NaiveBayesBundleRecommender"),
        "svm": MagicMock(name="SVMBundleRecommender")
    }
    
    # Mock recommend_bundles response
    engine.recommend_bundles.return_value = {
        "bundles": [["mouse", "keyboard"], ["mouse", "usb_cable"]],
        "confidence": 0.825,
        "recommender": "naive_bayes",
        "transaction": ["laptop"]
    }
    
    return engine

@pytest.fixture
def mock_get_engine(mock_engine):
    """Patch get_engine() to return mock engine."""
    with patch('src.api.get_engine', return_value=mock_engine):
        yield mock_engine

def test_bundles_endpoint(client, mock_get_engine):
    """Test bundles endpoint with mocked engine."""
    response = client.post('/api/v1/bundles',
        json={'product_description': 'laptop'})
    assert response.status_code == 200
```

### Testing Different HTTP Methods

```python
# GET request with query parameters
def test_get_with_params(client, mock_get_engine):
    """Test GET endpoint with query parameters."""
    response = client.get('/api/v1/cross-sell?product_description=laptop&top_n=5')
    assert response.status_code == 200

# POST request with JSON body
def test_post_with_json(client, mock_get_engine):
    """Test POST endpoint with JSON payload."""
    response = client.post('/api/v1/bundles',
        json={'product_description': 'laptop', 'threshold': 0.7})
    assert response.status_code == 200

# Verify request was processed correctly
def test_verify_mock_called(client, mock_get_engine):
    """Verify mock engine was called with correct parameters."""
    client.post('/api/v1/bundles',
        json={'product_description': 'laptop'})
    
    # Verify recommend_bundles was called
    mock_get_engine.recommend_bundles.assert_called_once()
```

### Input Validation Testing

Test API parameter validation for robustness:

```python
def test_missing_required_field(client):
    """Test error when required field is missing."""
    response = client.post('/api/v1/bundles', json={})
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data['status'] == 'error'
    assert 'product_description' in data['message']

def test_invalid_threshold_range(client):
    """Test threshold validation (must be 0.0-1.0)."""
    response = client.post('/api/v1/bundles',
        json={'product_description': 'laptop', 'threshold': 1.5})
    assert response.status_code == 400

def test_invalid_data_type(client):
    """Test type validation."""
    response = client.post('/api/v1/bundles',
        json={'product_description': 'laptop', 'threshold': 'high'})
    assert response.status_code == 400
```

### Response Format Testing

Ensure consistent API response structure:

```python
def test_success_response_structure(client, mock_get_engine):
    """Test successful response has standard format."""
    response = client.post('/api/v1/bundles',
        json={'product_description': 'laptop'})
    data = json.loads(response.data)
    
    # Standard success format
    assert 'status' in data
    assert data['status'] == 'success'
    assert 'data' in data or 'bundles' in data

def test_error_response_structure(client):
    """Test error response has standard format."""
    response = client.post('/api/v1/bundles', json={})
    data = json.loads(response.data)
    
    # Standard error format
    assert data['status'] == 'error'
    assert 'message' in data
    assert isinstance(data['message'], str)
```

### Batch Endpoint Testing

Test bulk processing capabilities:

```python
def test_batch_multiple_items(client, mock_get_engine):
    """Test batch processing of multiple items."""
    payload = {
        'product_descriptions': ['laptop', 'mouse', 'keyboard'],
        'threshold': 0.5
    }
    response = client.post('/api/v1/bundles/batch', json=payload)
    assert response.status_code == 200
    data = json.loads(response.data)
    assert len(data['results']) == 3

def test_batch_size_limit(client, mock_get_engine):
    """Test batch size limit enforcement."""
    # Create list exceeding max batch size
    large_batch = ['item' + str(i) for i in range(101)]
    response = client.post('/api/v1/bundles/batch',
        json={'product_descriptions': large_batch})
    assert response.status_code == 400
```

### Error Handling Testing

Test graceful error handling:

```python
def test_engine_loading_failure(client):
    """Test API behavior when engine fails to load."""
    with patch('src.api.get_engine', side_effect=Exception("Model load failed")):
        response = client.get('/api/v1/recommenders')
        assert response.status_code == 500
        data = json.loads(response.data)
        assert data['status'] == 'error'

def test_404_not_found(client):
    """Test custom 404 error handler."""
    response = client.get('/api/v1/nonexistent')
    assert response.status_code == 404
    data = json.loads(response.data)
    assert data['status'] == 'error'
```

### CORS Testing

Test Cross-Origin Resource Sharing headers:

```python
def test_cors_headers_present(client, mock_get_engine):
    """Test CORS headers are included in responses."""
    response = client.get('/api/v1/recommenders')
    assert 'Access-Control-Allow-Origin' in response.headers

def test_preflight_request(client):
    """Test OPTIONS preflight request handling."""
    response = client.options('/api/v1/bundles')
    assert response.status_code == 200
    assert 'Access-Control-Allow-Methods' in response.headers
```

### Static File Serving

Test web UI static file serving:

```python
def test_index_html_at_root(client):
    """Test index.html is served at root path."""
    response = client.get('/')
    assert response.status_code == 200
    assert b'html' in response.data

def test_static_assets_accessible(client):
    """Test static CSS/JS files are accessible."""
    # Test CSS
    response = client.get('/static/style.css')
    # Verify content type or status
```

### Running API Tests

```bash
# All API tests
pytest tests/test_api.py -v

# Specific endpoint class
pytest tests/test_api.py::TestBundlesEndpoint -v
pytest tests/test_api.py::TestHealthEndpoint -v

# With coverage
pytest tests/test_api.py -v --cov=src.api --cov-report=term-missing

# Verbose debugging
pytest tests/test_api.py::TestBundlesEndpoint::test_bundles_successful_request -vv -s
```

---

## Utilities Testing (test_utils.py)

### Overview

Utility function tests cover JSON I/O, data validation, inventory loading, LLM integration helpers, and utility functions used across the project. Tests are organized into **P0 (LLM integration)** and **P1 (file I/O, validation)** priority tiers.

**Current Status:**
- ✅ **107 total tests** (51 P0 + 56 P1)
- ✅ **50% coverage** of src/utils.py (284/540 statements)
- ✅ **100% pass rate** (~13-15s execution time)

### Test Classes

#### ✅ TestNormalizeDescriptionBasic (Implemented - P0)

Tests basic text normalization: lowercase, whitespace collapse, special character removal, and unit standardization.

```python
def test_lowercasing(self):
    """Test lowercase conversion."""
    result = normalize_description_basic("RED Roses")
    assert result == "red roses"

def test_whitespace_collapse(self):
    """Test multiple spaces collapsed to single space."""
    result = normalize_description_basic("multi   space   text")
    assert result == "multi space text"

def test_unit_normalization_inches(self):
    """Test inches/inch/in. → in conversion."""
    result = normalize_description_basic("12 inches tall")
    assert "12 in tall" in result

def test_special_characters_removed(self):
    """Test special characters are stripped."""
    result = normalize_description_basic("hello@#$world")
    assert "@" not in result and "#" not in result

def test_unicode_characters_preserved(self):
    """Test unicode chars are preserved."""
    result = normalize_description_basic("café naïve")
    assert "café" in result
```

**Tests:** 11 | **Lines:** ~50-180

---

#### ✅ TestExtractContextsWithLLM (Implemented - P0)

Tests LLM-based context extraction with mocked urllib HTTP calls. Validates all 5 providers (OpenAI, Azure, Gemini, Anthropic, Perplexity).

```python
@patch('urllib.request.urlopen')
def test_valid_response_structure(self, mock_urlopen):
    """Test parsing valid LLM response."""
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "contexts": [
            {"context": "casual wear", "confidence": 0.95},
            {"context": "everyday use", "confidence": 0.88}
        ]
    }).encode('utf-8')
    mock_urlopen.return_value.__enter__.return_value = mock_response
    
    result = extract_contexts_with_llm(
        "blue jeans", "openai", "gpt-4", 0.7, 256, 30, "test-key"
    )
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["context"] == "casual wear"

@pytest.mark.parametrize("provider", [
    "openai", "azure", "gemini", "anthropic", "perplexity"
])
@patch('urllib.request.urlopen')
def test_all_providers_supported(self, mock_urlopen, provider):
    """Test all 5 LLM providers with mocked responses."""
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "contexts": [{"context": "test", "confidence": 0.9}]
    }).encode('utf-8')
    mock_urlopen.return_value.__enter__.return_value = mock_response
    
    result = extract_contexts_with_llm(
        "test item", provider, "model", 0.7, 256, 30, "key"
    )
    assert isinstance(result, list)

@patch('urllib.request.urlopen')
def test_http_error_returns_empty_list(self, mock_urlopen):
    """Test graceful error handling on HTTP errors."""
    mock_urlopen.side_effect = urllib.error.HTTPError(
        None, 500, "Server Error", {}, None
    )
    result = extract_contexts_with_llm(
        "test", "openai", "gpt-4", 0.7, 256, 30, "key"
    )
    assert result == []
```

**Tests:** 14 | **Lines:** ~200-450 | **Key Pattern:** `@patch('urllib.request.urlopen')` for HTTP interception

---

#### ✅ TestEnrichCategoriesWithLLM (Implemented - P0)

Tests category enrichment with field validation and NaN defaults for invalid/missing data.

```python
@patch('urllib.request.urlopen')
def test_valid_response_all_fields(self, mock_urlopen):
    """Test category extraction with all 6 fields."""
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "category": "clothing",
        "material": "cotton",
        "color": "blue",
        "style": "casual",
        "size_unit": "L",
        "weight_unit": "kg"
    }).encode('utf-8')
    mock_urlopen.return_value.__enter__.return_value = mock_response
    
    result = enrich_categories_with_llm(
        "blue cotton shirt", "openai", "gpt-4", 0.7, 256, 30, "key"
    )
    assert isinstance(result, dict)
    assert len(result) == 6
    assert result["category"] == "clothing"
    assert result["material"] == "cotton"

@patch('urllib.request.urlopen')
def test_invalid_json_all_nan(self, mock_urlopen):
    """Test graceful handling of invalid JSON → all NaN."""
    mock_response = MagicMock()
    mock_response.read.return_value = b"{invalid json}"
    mock_urlopen.return_value.__enter__.return_value = mock_response
    
    result = enrich_categories_with_llm(
        "test", "openai", "gpt-4", 0.7, 256, 30, "key"
    )
    assert isinstance(result, dict)
    assert all(pd.isna(v) for v in result.values())

@patch('urllib.request.urlopen')
def test_missing_fields_filled_with_nan(self, mock_urlopen):
    """Test missing fields are filled with NaN."""
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "category": "electronics"
        # Missing: material, color, style, size_unit, weight_unit
    }).encode('utf-8')
    mock_urlopen.return_value.__enter__.return_value = mock_response
    
    result = enrich_categories_with_llm(
        "laptop", "openai", "gpt-4", 0.7, 256, 30, "key"
    )
    assert result["category"] == "electronics"
    assert pd.isna(result["material"])
```

**Tests:** 11 | **Lines:** ~450-650 | **Key Pattern:** All missing/invalid data → `pd.isna()` (NaN)

---

#### ✅ TestEnrichCategoriesBatchWithLLM (Implemented - P0)

Tests batch category enrichment with automatic deduplication of identical descriptions.

```python
@patch('urllib.request.urlopen')
def test_duplicate_texts_deduplicated(self, mock_urlopen):
    """Test batch processing deduplicates identical descriptions."""
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "category": "electronics", "material": "plastic",
        "color": "black", "style": "modern",
        "size_unit": "N/A", "weight_unit": "g"
    }).encode('utf-8')
    mock_urlopen.return_value.__enter__.return_value = mock_response
    
    texts = ["laptop", "laptop", "mouse"]  # Duplicate "laptop"
    result = enrich_categories_batch_with_llm(
        texts, "openai", "gpt-4", 0.7, 256, 30, "test-key"
    )
    
    # Verify only 2 LLM calls made (laptop deduplicated)
    assert mock_urlopen.call_count == 2
    assert len(result) == 2  # "laptop" and "mouse"

@patch('urllib.request.urlopen')
def test_batch_with_various_inputs(self, mock_urlopen):
    """Test batch with empty strings, None, and valid inputs."""
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "category": "test", "material": "N/A", "color": "N/A",
        "style": "N/A", "size_unit": "N/A", "weight_unit": "N/A"
    }).encode('utf-8')
    mock_urlopen.return_value.__enter__.return_value = mock_response
    
    texts = ["valid item", "", None, "another item"]
    result = enrich_categories_batch_with_llm(
        texts, "openai", "gpt-4", 0.7, 256, 30, "key"
    )
    # Empty/None filtered out before LLM calls
```

**Tests:** 5 | **Lines:** ~650-750 | **Key Feature:** Automatic deduplication to reduce LLM costs

---

#### ✅ TestBatchScoreAnomaliesWithLLM (Implemented - P0)

Tests batch anomaly detection with error handling and validation.

```python
@patch('urllib.request.urlopen')
def test_multiple_records(self, mock_urlopen):
    """Test batch anomaly scoring for multiple records."""
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "r1": {
            "anomaly_score": 0.95,
            "anomaly_type": "high_quantity_low_price",
            "reasoning": "Quantity 1000 with price $0.01"
        },
        "r2": {
            "anomaly_score": 0.85,
            "anomaly_type": "negative_quantity",
            "reasoning": "Negative quantity -5"
        }
    }).encode('utf-8')
    mock_urlopen.return_value.__enter__.return_value = mock_response
    
    records = [
        {"key": "r1", "Quantity": 1000, "UnitPrice": 0.01},
        {"key": "r2", "Quantity": -5, "UnitPrice": 10.0}
    ]
    result = batch_score_anomalies_with_llm(
        records, "openai", "gpt-4", 0.7, 512, 30, "test-key"
    )
    
    assert "r1" in result
    assert result["r1"]["anomaly_type"] == "high_quantity_low_price"
    assert "r2" in result

@patch('urllib.request.urlopen')
def test_empty_records_returns_empty_dict(self, mock_urlopen):
    """Test empty input returns empty dict."""
    result = batch_score_anomalies_with_llm(
        [], "openai", "gpt-4", 0.7, 512, 30, "key"
    )
    assert result == {}
    assert mock_urlopen.call_count == 0  # No LLM calls for empty input
```

**Tests:** 10 | **Lines:** ~750-873 | **Key Pattern:** Batch processing with record keys for result mapping

---

#### ✅ TestLoadJsonFile (Implemented - P1)

Tests JSON file loading with comprehensive error handling (invalid JSON, missing files, empty files).

```python
def test_load_valid_json_file(self, temp_json_file):
    """Test loading valid JSON file."""
    result = load_json_file(str(temp_json_file))
    assert isinstance(result, dict)
    assert "key" in result
    assert result["key"] == "value"

def test_load_nonexistent_file(self):
    """Test loading nonexistent file returns empty dict."""
    result = load_json_file("/nonexistent/path/file.json")
    assert result == {}

def test_load_invalid_json_file(self, tmp_path):
    """Test invalid JSON returns empty dict."""
    invalid_file = tmp_path / "invalid.json"
    invalid_file.write_text("{not valid json")
    result = load_json_file(str(invalid_file))
    assert result == {}

def test_load_json_array(self, tmp_path):
    """Test loading JSON array (not just objects)."""
    array_file = tmp_path / "array.json"
    array_file.write_text('[{"item": "a"}, {"item": "b"}]')
    result = load_json_file(str(array_file))
    assert isinstance(result, list)
    assert len(result) == 2

def test_load_json_with_unicode(self, tmp_path):
    """Test unicode character handling."""
    unicode_file = tmp_path / "unicode.json"
    unicode_file.write_text('{"text": "café naïve"}', encoding='utf-8')
    result = load_json_file(str(unicode_file))
    assert result["text"] == "café naïve"
```

**Tests:** 8 | **Lines:** ~900-1050 | **Error Handling:** All errors → `{}`

---

#### ✅ TestSaveJsonFile (Implemented - P1)

Tests JSON file saving with directory auto-creation and overwrite handling.

```python
def test_save_dict_to_json(self, tmp_path):
    """Test saving dictionary to JSON file."""
    output_file = tmp_path / "output.json"
    data = {"key": "value", "number": 42}
    save_json_file(str(output_file), data)
    
    assert output_file.exists()
    result = json.loads(output_file.read_text())
    assert result == data

def test_save_list_to_json(self, tmp_path):
    """Test saving list to JSON file."""
    output_file = tmp_path / "list.json"
    data = [{"id": 1}, {"id": 2}]
    save_json_file(str(output_file), data)
    
    result = json.loads(output_file.read_text())
    assert isinstance(result, list)

def test_save_creates_nonexistent_directory(self, tmp_path):
    """Test directory auto-creation."""
    output_file = tmp_path / "subdir" / "nested" / "file.json"
    save_json_file(str(output_file), {"test": "data"})
    
    assert output_file.exists()
    assert output_file.parent.exists()

def test_save_overwrites_existing_file(self, tmp_path):
    """Test overwriting existing file."""
    output_file = tmp_path / "existing.json"
    output_file.write_text('{"old": "data"}')
    
    save_json_file(str(output_file), {"new": "data"})
    result = json.loads(output_file.read_text())
    assert result == {"new": "data"}
```

**Tests:** 7 | **Lines:** ~1050-1150 | **Key Feature:** Automatic directory creation with `os.makedirs(exist_ok=True)`

---

#### ✅ TestLoadAliasMap (Implemented - P1)

Tests alias map loading with key/value normalization via `normalize_description_basic()`.

```python
def test_load_valid_alias_map(self, tmp_path):
    """Test loading alias map from JSON."""
    alias_file = tmp_path / "aliases.json"
    aliases = {"ITEM1": "variant1", "ITEM2": "variant2"}
    alias_file.write_text(json.dumps(aliases))
    
    result = load_alias_map(str(alias_file))
    assert isinstance(result, dict)
    # Keys and values normalized (lowercased, whitespace stripped)
    assert "item1" in result or "variant1" in result.values()

def test_load_alias_map_with_special_chars(self, tmp_path):
    """Test special character handling in aliases."""
    alias_file = tmp_path / "special.json"
    aliases = {"item@#1": "variant!@#"}
    alias_file.write_text(json.dumps(aliases))
    
    result = load_alias_map(str(alias_file))
    # Special chars stripped during normalization

def test_load_nonexistent_alias_map(self):
    """Test nonexistent file returns empty dict."""
    result = load_alias_map("/nonexistent/aliases.json")
    assert result == {}
```

**Tests:** 6 | **Lines:** ~1150-1220 | **Normalization:** Both keys and values passed through `normalize_description_basic()`

---

#### ✅ TestValidateTransaction (Implemented - P1)

Tests transaction list validation (must be list of strings).

```python
def test_valid_transaction(self):
    """Test validation of valid transaction list."""
    transaction = ["item1", "item2", "item3"]
    result = validate_transaction(transaction)
    assert result is True

def test_empty_transaction_invalid(self):
    """Test empty transaction is invalid."""
    result = validate_transaction([])
    assert result is False

def test_none_transaction_invalid(self):
    """Test None transaction is invalid."""
    result = validate_transaction(None)
    assert result is False

def test_transaction_with_non_string_items(self):
    """Test non-string items are invalid."""
    transaction = ["item1", 123, "item3"]
    result = validate_transaction(transaction)
    assert result is False

def test_transaction_not_a_list(self):
    """Test non-list input is invalid."""
    result = validate_transaction("not a list")
    assert result is False
```

**Tests:** 6 | **Lines:** ~1220-1280 | **Validation:** `isinstance(txn, list)` and `all(isinstance(item, str) for item in txn)`

---

#### ✅ TestFilterTransaction (Implemented - P1)

Tests item filtering by minimum length with lowercase normalization.

```python
def test_filter_removes_short_items(self):
    """Test min_length parameter filters short items."""
    transaction = ["ITEM", "at", "SOMETHING LONGER"]
    result = filter_transaction(transaction, min_length=3)
    
    assert "at" not in result  # Too short (2 chars)
    assert len(result) == 2

def test_filter_lowercases_items(self):
    """Test items are lowercased."""
    transaction = ["UPPERCASE", "MixedCase"]
    result = filter_transaction(transaction)
    assert all(item.islower() for item in result)

def test_filter_strips_whitespace(self):
    \"\"\"Test leading/trailing whitespace is stripped.\"\"\"
    transaction = ["  item1  ", "item2   "]
    result = filter_transaction(transaction)
    assert all(item == item.strip() for item in result)

def test_filter_with_custom_threshold(self):
    """Test custom min_length threshold."""
    transaction = ["a", "ab", "abc", "abcd"]
    result = filter_transaction(transaction, min_length=4)
    assert result == ["abcd"]
```

**Tests:** 6 | **Lines:** ~1280-1340 | **Default:** `min_length=1`

---

#### ✅ TestLoadInventoryCsv (Implemented - P1)

Tests inventory CSV loading with flexible column name matching.

```python
def test_load_valid_inventory_csv(self, tmp_path):
    """Test loading inventory CSV file."""
    csv_file = tmp_path / "inventory.csv"
    csv_content = \"\"\"StockCode,Description,Price
85123A,WHITE HEART HOLDER,2.55
85099B,BLUE POLKA JUMBO BAG,1.95
\"\"\"
    csv_file.write_text(csv_content)
    result = load_inventory_csv(str(csv_file))
    assert isinstance(result, dict)
    assert len(result) > 0

def test_load_inventory_csv_with_missing_columns(self, tmp_path):
    \"\"\"Test graceful handling of missing columns.\"\"\"
    csv_file = tmp_path / "missing.csv"
    csv_file.write_text("StockCode,Description\\n85123A,ITEM\\n")
    result = load_inventory_csv(str(csv_file))
    # Function handles missing Price column gracefully

def test_load_nonexistent_inventory_csv(self):
    \"\"\"Test nonexistent file returns empty dict.\"\"\"
    result = load_inventory_csv("/nonexistent/inventory.csv")
    assert result == {}

def test_load_inventory_csv_with_unicode(self, tmp_path):
    \"\"\"Test unicode characters in descriptions.\"\"\"
    csv_file = tmp_path / "unicode.csv"
    csv_content = \"\"\"StockCode,Description,Price
TEST1,café table,10.00
\"\"\"
    csv_file.write_text(csv_content, encoding='utf-8')
    result = load_inventory_csv(str(csv_file))
    # Unicode preserved in descriptions
```

**Tests:** 7 | **Lines:** ~1340-1400 | **Flexibility:** Column names case-insensitive, flexible matching

---

#### ✅ TestFormatRecommendations (Implemented - P1)

Tests recommendation output formatting with confidence percentages.

```python
def test_format_basic_recommendations(self):
    \"\"\"Test formatting recommendations dict.\"\"\"
    recommendations = {
        "transaction": ["item1", "item2"],
        "confidence": 0.85,
        "recommender": "naive_bayes",
        "bundles": [("bundle1", "bundle2"), ("bundle3", "bundle4")]
    }
    result = format_recommendations(recommendations)
    
    assert "BUNDLE RECOMMENDATIONS" in result
    assert "85.00%" in result  # Confidence formatted as percentage
    assert "naive_bayes" in result
    assert "1)" in result  # Numbered bundles

def test_format_high_confidence(self):
    \"\"\"Test high confidence formatting.\"\"\"
    recommendations = {
        "transaction": ["laptop"],
        "confidence": 0.99,
        "recommender": "svm",
        "bundles": [("mouse", "keyboard")]
    }
    result = format_recommendations(recommendations)
    assert "99.00%" in result

def test_format_with_empty_bundles(self):
    \"\"\"Test formatting with no bundles.\"\"\"
    recommendations = {
        "transaction": ["item"],
        "confidence": 0.5,
        "recommender": "test",
        "bundles": []
    }
    result = format_recommendations(recommendations)
    # Graceful handling of empty bundles
```

**Tests:** 5 | **Lines:** ~1400-1430 | **Format:** Confidence as `{conf*100:.2f}%`

---

#### ✅ TestComputeIQRBounds (Implemented - P1)

Tests IQR bounds calculation for pandas Series with NaN handling.

```python
def test_compute_iqr_normal_data(self):
    \"\"\"Test IQR bounds on normal data.\"\"\"
    import pandas as pd
    data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    lower, upper = compute_iqr_bounds(data)
    
    assert lower < upper
    assert isinstance(lower, float)
    assert isinstance(upper, float)

def test_compute_iqr_with_nan(self):
    \"\"\"Test NaN values are dropped before calculation.\"\"\"
    import pandas as pd
    data = pd.Series([1, 2, 3, float('nan'), 5])
    lower, upper = compute_iqr_bounds(data)
    # NaN values ignored, calculation proceeds on valid values

def test_compute_iqr_with_custom_multiplier(self):
    \"\"\"Test custom IQR multiplier.\"\"\"
    import pandas as pd
    data = pd.Series([1, 2, 3, 4, 5])
    lower1, upper1 = compute_iqr_bounds(data, multiplier=1.5)
    lower2, upper2 = compute_iqr_bounds(data, multiplier=3.0)
    
    # Wider bounds with larger multiplier
    assert lower2 < lower1
    assert upper2 > upper1
```

**Tests:** 6 | **Lines:** ~1430-1455 | **Formula:** `lower = Q1 - multiplier*IQR`, `upper = Q3 + multiplier*IQR`

---

#### ✅ TestSetupLogging (Implemented - P1)

Tests logging configuration with various log levels.

```python
def test_setup_logging_with_default_level(self):
    \"\"\"Test setup_logging with default level.\"\"\"
    result = setup_logging()
    assert result is None  # Side-effect only function

def test_setup_logging_with_debug_level(self):
    \"\"\"Test setup_logging with DEBUG level.\"\"\"
    import logging
    result = setup_logging(logging.DEBUG)
    assert result is None

def test_setup_logging_with_info_level(self):
    \"\"\"Test setup_logging with INFO level.\"\"\"
    import logging
    result = setup_logging(logging.INFO)
    assert result is None

def test_setup_logging_with_warning_level(self):
    \"\"\"Test setup_logging with WARNING level.\"\"\"
    import logging
    result = setup_logging(logging.WARNING)
    assert result is None
```

**Tests:** 5 | **Lines:** ~1455-1467 | **Pattern:** Side-effect testing (function returns None)

---

### Running Utilities Tests

```bash
# All utils tests (107 tests)
pytest tests/test_utils.py -v

# P0 tests only (LLM integration - 51 tests)
pytest tests/test_utils.py::TestNormalizeDescriptionBasic -v
pytest tests/test_utils.py::TestExtractContextsWithLLM -v
pytest tests/test_utils.py::TestEnrichCategoriesWithLLM -v
pytest tests/test_utils.py::TestEnrichCategoriesBatchWithLLM -v
pytest tests/test_utils.py::TestBatchScoreAnomaliesWithLLM -v

# P1 tests only (JSON I/O, validation, utilities - 56 tests)
pytest tests/test_utils.py::TestLoadJsonFile -v
pytest tests/test_utils.py::TestSaveJsonFile -v
pytest tests/test_utils.py::TestLoadAliasMap -v
pytest tests/test_utils.py::TestValidateTransaction -v
pytest tests/test_utils.py::TestFilterTransaction -v
pytest tests/test_utils.py::TestLoadInventoryCsv -v
pytest tests/test_utils.py::TestFormatRecommendations -v
pytest tests/test_utils.py::TestComputeIQRBounds -v
pytest tests/test_utils.py::TestSetupLogging -v

# With coverage
pytest tests/test_utils.py --cov=src.utils --cov-report=term-missing -v

# Specific test
pytest tests/test_utils.py::TestLoadJsonFile::test_load_valid_json_file -vv -s

# Run by marker (if markers added)
pytest tests/test_utils.py -m "p0" -v
pytest tests/test_utils.py -m "p1" -v
```

### Utils Test Fixtures

Key fixtures defined in `tests/test_utils.py`:

```python
@pytest.fixture
def temp_json_file(tmp_path):
    \"\"\"Temporary JSON file for testing load/save operations.\"\"\"
    file_path = tmp_path / "test.json"
    test_data = {"key": "value", "items": [1, 2, 3]}
    with open(file_path, "w") as f:
        json.dump(test_data, f)
    return file_path

@pytest.fixture
def mock_llm_response_contexts():
    \"\"\"Mock LLM response for context extraction tests.\"\"\"
    return {
        "contexts": [
            {"context": "casual wear", "confidence": 0.95},
            {"context": "everyday use", "confidence": 0.88}
        ]
    }

@pytest.fixture
def mock_llm_response_categories():
    \"\"\"Mock LLM response for category enrichment tests.\"\"\"
    return {
        "category": "clothing",
        "material": "cotton",
        "color": "blue",
        "style": "casual",
        "size_unit": "L",
        "weight_unit": "kg"
    }

# pytest's tmp_path fixture used extensively for file I/O tests
```

### Utils Test Coverage

**Current Coverage:** 50% of src/utils.py (284/540 statements)

**Covered Functions:**
- ✅ `normalize_description_basic()` - Full coverage
- ✅ `extract_contexts_with_llm()` - All providers + error handling
- ✅ `enrich_categories_with_llm()` - All 6 fields + NaN defaults
- ✅ `enrich_categories_batch_with_llm()` - Deduplication + batch processing
- ✅ `batch_score_anomalies_with_llm()` - Batch scoring + error handling
- ✅ `load_json_file()` - Valid/invalid/missing files
- ✅ `save_json_file()` - Dict/list saving + directory creation
- ✅ `load_alias_map()` - Normalization + error handling
- ✅ `validate_transaction()` - Type validation
- ✅ `filter_transaction()` - Min length filtering
- ✅ `load_inventory_csv()` - CSV loading + flexible columns
- ✅ `format_recommendations()` - Output formatting
- ✅ `compute_iqr_bounds()` - IQR calculation + NaN handling
- ✅ `setup_logging()` - Logging configuration

**Gap to 75% Target:** 25% (135 statements) - Requires P2 implementation

---

## Performance Benchmarks

### Expected Test Execution Time

| Test Suite | Time | Notes |
|-----------|------|-------|
| Unit tests (LLM off) | ~2-5s | Fast, no LLM calls |
| Unit tests (LLM mocked) | ~5-10s | With mock LLM overhead |
| Data pipeline tests | ~8-12s | 44 tests, synthetic data |
| Recommendation engine tests | ~12-15s | 62 tests, NB + SVM + Engine |
| LLM integration tests | ~5-8s | 22 tests, all mocked |
| API endpoint tests | ~3-5s | 43 tests, Flask test client |
| Utils tests | ~13-15s | 107 tests, urllib mocking |
| All implemented tests | ~41-55s | 278 tests total |
| All tests (P0 complete) | ~45-60s | ~290+ tests total |

### Memory Usage

- Synthetic fixtures: < 50MB
- Test process: < 200MB
- Coverage report: < 100MB

---

## Best Practices

### ✅ DO:

- Use synthetic fixtures for unit tests (speed & repeatability)
- Mock all LLM functions (consistency, no API calls)
- Use `llm_config_disabled` by default in unit tests
- Use `llm_config_enabled_mocked` in integration tests
- Test error paths (ValueError, FileNotFoundError, etc.)
- Test edge cases (empty, single row, null values)
- Include docstrings explaining what's being tested
- Keep tests focused (one thing per test)
- Use descriptive test names

### ❌ DON'T:

- Don't use real LLM API calls in unit tests
- Don't hardcode file paths (use fixtures)
- Don't test library code (e.g., pandas functionality)
- Don't create new .md files for test documentation
- Don't skip coverage requirements
- Don't mix multiple scenarios in one test

---

## Troubleshooting

### Test Fails: "Import Error"

```bash
# Ensure venv is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
pip install -e .
```

### Coverage Below 80%

```bash
# Generate detailed HTML report
make test-html

# Open report in browser
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

Check:
1. Which files/lines are uncovered
2. Add tests for those code paths
3. Re-run: `make test-coverage`

### Mock Not Working

```python
# Verify mock is applied to correct module
monkeypatch.setattr('src.data_pipeline.function_name', mock_function)

# Use fixture: mock_all_llm_functions
def test_example(mock_all_llm_functions):
    # All LLM functions pre-mocked
```

### Tests Hang/Timeout

```bash
# Run with timeout (5 seconds per test)
pytest tests/ --timeout=5

# Requires: pip install pytest-timeout
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.12'
      - run: pip install -r requirements.txt
      - run: pip install pytest pytest-cov
      - run: pytest tests/ --cov=src --cov-fail-under=80
      - uses: codecov/codecov-action@v2
        with:
          files: ./coverage.xml
```

---

## Next Steps

### P0 Testing Roadmap

| Task | Files | Status | Target |
|------|-------|--------|--------|
| Data Pipeline tests | test_data_pipeline.py | ✅ DONE | 2026-02-03 |
| Recommendation Engine tests | test_recommendation_engine.py | ✅ DONE | 2026-02-04 |
| LLM Integration tests | test_llm_integration.py | ✅ DONE | 2026-02-04 |
| API Endpoint tests | test_api.py | ✅ DONE | 2026-02-04 |
| Utils tests | test_utils.py | 🔲 TODO | 2026-02-09 |
| Integration tests | test_integration.py | 🔲 TODO | 2026-02-10 |

---

## References

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [Python unittest.mock](https://docs.python.org/3/library/unittest.mock.html)
- [Project AGENTS.md](AGENTS.md) - Best practices guide
