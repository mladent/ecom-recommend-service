"""Pytest configuration and shared fixtures for all tests."""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

# Configure pytest environment
def pytest_configure(config):
    """Configure pytest session."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "llm: mark test as involving LLM features"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance benchmark"
    )


@pytest.fixture(scope="session")
def project_root():
    """Get project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_data_dir(project_root):
    """Get test data directory."""
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def cleanup_test_files(project_root):
    """Cleanup any test-generated files after test."""
    cleanup_paths = []
    
    yield cleanup_paths
    
    # Cleanup
    for path in cleanup_paths:
        try:
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                import shutil
                shutil.rmtree(path)
        except Exception as e:
            print(f"Warning: Could not cleanup {path}: {e}")


# ============================================================================
# LLM CONFIGURATION FIXTURES (for integration tests)
# ============================================================================

@pytest.fixture
def llm_config_disabled(monkeypatch):
    """Disable all LLM features for fast testing."""
    monkeypatch.setenv('ENRICHMENT_ENABLED', 'false')
    monkeypatch.setenv('ENRICHMENT_CACHE_FIRST', 'false')
    monkeypatch.setenv('OUTLIER_ENABLED', 'false')
    monkeypatch.setenv('OUTLIER_CACHE_FIRST', 'false')
    monkeypatch.setenv('CONTEXT_ENABLED', 'false')
    monkeypatch.setenv('CONTEXT_CACHE_FIRST', 'false')
    monkeypatch.setenv('NORMALIZATION_ENABLED', 'false')
    monkeypatch.setenv('NORMALIZATION_CACHE_FIRST', 'false')
    return {
        'enrichment': False,
        'outlier': False,
        'context': False,
        'normalization': False
    }


@pytest.fixture
def llm_config_enabled_mocked(monkeypatch):
    """Enable LLM features with mocking (for validation)."""
    monkeypatch.setenv('ENRICHMENT_ENABLED', 'true')
    monkeypatch.setenv('ENRICHMENT_CACHE_FIRST', 'true')
    monkeypatch.setenv('OUTLIER_ENABLED', 'true')
    monkeypatch.setenv('OUTLIER_CACHE_FIRST', 'true')
    monkeypatch.setenv('CONTEXT_ENABLED', 'true')
    monkeypatch.setenv('CONTEXT_CACHE_FIRST', 'true')
    monkeypatch.setenv('NORMALIZATION_ENABLED', 'true')
    monkeypatch.setenv('NORMALIZATION_CACHE_FIRST', 'true')
    return {
        'enrichment': True,
        'outlier': True,
        'context': True,
        'normalization': True,
        'cache_first': True
    }


@pytest.fixture
def llm_config_cache_disabled(monkeypatch):
    """Enable LLM features but disable cache-first pattern."""
    monkeypatch.setenv('ENRICHMENT_ENABLED', 'true')
    monkeypatch.setenv('ENRICHMENT_CACHE_FIRST', 'false')
    monkeypatch.setenv('OUTLIER_ENABLED', 'true')
    monkeypatch.setenv('OUTLIER_CACHE_FIRST', 'false')
    monkeypatch.setenv('CONTEXT_ENABLED', 'true')
    monkeypatch.setenv('CONTEXT_CACHE_FIRST', 'false')
    return {
        'enrichment': True,
        'outlier': True,
        'context': True,
        'cache_first': False
    }


# ============================================================================
# MOCK LLM FUNCTIONS (Reusable Across Tests)
# ============================================================================

def mock_normalize_description(description, **kwargs):
    """Mock normalize_description_with_llm."""
    return description.lower().strip() if description else ""


def mock_enrich_categories_batch(descriptions, **kwargs):
    """Mock enrich_categories_batch_with_llm."""
    return {
        desc: {
            'category': 'electronics',
            'material': 'plastic',
            'size': 'medium',
            'theme': 'general'
        }
        for desc in descriptions
    }


def mock_batch_score_anomalies(descriptions, **kwargs):
    """Mock batch_score_anomalies_with_llm."""
    return {
        desc: {
            'is_anomaly': False,
            'score': 0.05,
            'reason': 'within normal distribution'
        }
        for desc in descriptions
    }


def mock_extract_contexts(description, **kwargs):
    """Mock extract_contexts_with_llm."""
    return [f"context for {description}"]


@pytest.fixture
def mock_all_llm_functions(monkeypatch):
    """Mock all LLM functions globally for tests."""
    monkeypatch.setattr(
        'src.data_pipeline.normalize_description_with_llm',
        mock_normalize_description
    )
    monkeypatch.setattr(
        'src.data_pipeline.enrich_categories_batch_with_llm',
        mock_enrich_categories_batch
    )
    monkeypatch.setattr(
        'src.data_pipeline.batch_score_anomalies_with_llm',
        mock_batch_score_anomalies
    )
    monkeypatch.setattr(
        'src.data_pipeline.extract_contexts_with_llm',
        mock_extract_contexts
    )
    monkeypatch.setattr(
        'src.utils.normalize_description_with_llm',
        mock_normalize_description
    )
    monkeypatch.setattr(
        'src.utils.enrich_categories_batch_with_llm',
        mock_enrich_categories_batch
    )
    monkeypatch.setattr(
        'src.utils.batch_score_anomalies_with_llm',
        mock_batch_score_anomalies
    )
    monkeypatch.setattr(
        'src.utils.extract_contexts_with_llm',
        mock_extract_contexts
    )


# ============================================================================
# HELPER FIXTURES FOR TEST DATA
# ============================================================================

@pytest.fixture
def sample_csv_file(temp_dir):
    """Create a sample CSV file for testing."""
    import pandas as pd
    
    df = pd.DataFrame({
        'InvoiceNo': ['001', '002', '003'],
        'StockCode': ['A', 'B', 'C'],
        'Description': ['item1', 'item2', 'item3'],
        'Quantity': [1, 2, 3],
        'UnitPrice': [10.0, 20.0, 30.0],
        'CustomerID': [100.0, 101.0, 102.0],
        'InvoiceDate': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'Country': ['UK', 'US', 'FR']
    })
    
    filepath = os.path.join(temp_dir, 'test_data.csv')
    df.to_csv(filepath, index=False, encoding='ISO-8859-1')
    return filepath


@pytest.fixture
def multi_invoice_csv_file(temp_dir):
    """Create CSV with multiple items per invoice (transaction baskets)."""
    import pandas as pd
    
    df = pd.DataFrame({
        'InvoiceNo': ['001', '001', '001', '002', '002', '003'],
        'StockCode': ['A', 'B', 'C', 'B', 'D', 'E'],
        'Description': ['item1', 'item2', 'item3', 'item2', 'item4', 'item5'],
        'Quantity': [1, 1, 1, 2, 1, 1],
        'UnitPrice': [10.0, 20.0, 30.0, 20.0, 40.0, 50.0],
        'CustomerID': [100.0, 100.0, 100.0, 101.0, 101.0, 102.0],
        'InvoiceDate': ['2023-01-01', '2023-01-01', '2023-01-01', 
                        '2023-01-02', '2023-01-02', '2023-01-03'],
        'Country': ['UK', 'UK', 'UK', 'US', 'US', 'FR']
    })
    
    filepath = os.path.join(temp_dir, 'multi_invoice.csv')
    df.to_csv(filepath, index=False, encoding='ISO-8859-1')
    return filepath


# ============================================================================
# PARAMETRIZATION HELPERS
# ============================================================================

def pytest_generate_tests(metafunc):
    """Parametrize tests dynamically based on markers."""
    if "llm_enabled" in metafunc.fixturenames:
        metafunc.parametrize("llm_enabled", [False, True], ids=["llm_off", "llm_on"])
    
    if "min_support_threshold" in metafunc.fixturenames:
        metafunc.parametrize(
            "min_support_threshold",
            [0.01, 0.05, 0.1, 0.25],
            ids=["1%", "5%", "10%", "25%"]
        )
    
    if "svm_kernel" in metafunc.fixturenames:
        metafunc.parametrize(
            "svm_kernel",
            ["rbf", "linear", "poly"],
            ids=["rbf", "linear", "poly"]
        )
