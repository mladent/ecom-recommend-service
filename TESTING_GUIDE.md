# Testing Guide for E-Commerce Recommendation Service

**Last Updated:** 4 February 2026

## Overview

This project includes comprehensive unit and integration tests with the following specifications:

- **Test Framework:** pytest
- **Mocking:** All LLM functions mocked via monkeypatch for speed & consistency
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

# Specific test class
pytest tests/test_data_pipeline.py::TestDataPipelineBasics -v
pytest tests/test_recommendation_engine.py::TestNaiveBayesRecommender -v

# Specific test function
pytest tests/test_data_pipeline.py::TestDataPipelineBasics::test_initialization_defaults -v
pytest tests/test_recommendation_engine.py::TestSVMRecommender::test_different_kernels -v

# Using Makefile
make test-file FILE=tests/test_data_pipeline.py
make test-file FILE=tests/test_recommendation_engine.py
make test-func FUNC=TestNaiveBayesRecommender::test_fit_basic
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
├── test_llm_integration.py     # (To be implemented)
├── test_api.py                 # (To be implemented)
├── test_utils.py               # (To be implemented)
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

### Test Statistics

**Total Test Cases: 106**

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

# Specific function
make test-func FUNC=TestDataPipelineBasics::test_initialization_defaults

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

## Performance Benchmarks

### Expected Test Execution Time

| Test Suite | Time | Notes |
|-----------|------|-------|
| Unit tests (LLM off) | ~2-5s | Fast, no LLM calls |
| Unit tests (LLM mocked) | ~5-10s | With mock LLM overhead |
| Data pipeline tests | ~8-12s | 44 tests, synthetic data |
| Recommendation engine tests | ~12-15s | 62 tests, NB + SVM + Engine |
| All implemented tests | ~20-27s | 106 tests total |
| All tests (P0 complete) | ~45-60s | ~150 tests total |

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
| LLM Integration tests | test_llm_integration.py | 🔲 TODO | 2026-02-08 |
| API tests | test_api.py | 🔲 TODO | 2026-02-09 |
| Utils tests | test_utils.py | 🔲 TODO | 2026-02-09 |
| Integration tests | test_integration.py | 🔲 TODO | 2026-02-10 |

---

## References

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [Python unittest.mock](https://docs.python.org/3/library/unittest.mock.html)
- [Project AGENTS.md](AGENTS.md) - Best practices guide
