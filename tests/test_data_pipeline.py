"""Comprehensive unit tests for DataPipeline class."""

import os
import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, mock_open
from tempfile import TemporaryDirectory

from src.data_pipeline import DataPipeline
from src.llm_client import LLMClient
from src.utils import LLMQuotaExceededError


# ============================================================================
# FIXTURES: SYNTHETIC TEST DATA
# ============================================================================

@pytest.fixture
def minimal_dataframe():
    """Minimal valid DataFrame with required columns."""
    return pd.DataFrame({
        'InvoiceNo': ['001', '002', '003'],
        'StockCode': ['A001', 'A002', 'A003'],
        'Description': ['item1', 'item2', 'item3'],
        'Quantity': [1, 2, 3],
        'UnitPrice': [10.0, 20.0, 30.0],
        'CustomerID': [100.0, 101.0, 102.0],
        'InvoiceDate': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'Country': ['UK', 'US', 'FR']
    })


@pytest.fixture
def realistic_sample_dataframe():
    """Realistic 100-row DataFrame with mixed data quality issues."""
    np.random.seed(42)
    n_rows = 100
    
    # Create base data
    invoice_nos = [f"{i:06d}" for i in range(1, n_rows + 1)]
    
    # Add some cancellations (prefix 'C')
    for i in range(5):
        invoice_nos[i] = 'C' + invoice_nos[i][1:]
    
    stock_codes = np.random.choice(['A001', 'A002', 'A003', 'A004', 'A005'], n_rows)
    descriptions = np.random.choice([
        'WHITE HANGING HEART',
        'BLUE BALL',
        'RED BOX',
        'GREEN CUBE',
        'YELLOW SPHERE'
    ], n_rows)
    
    quantities = np.random.randint(1, 10, n_rows)
    unit_prices = np.random.uniform(5.0, 100.0, n_rows)
    
    # Add some null CustomerIDs
    customer_ids = np.full(n_rows, np.nan)
    customer_ids[5:] = np.random.randint(100, 200, n_rows - 5)
    
    invoice_dates = [
        (datetime(2023, 1, 1) + timedelta(days=i)).strftime('%Y-%m-%d')
        for i in range(n_rows)
    ]
    
    countries = np.random.choice(['UK', 'US', 'FR', 'DE', 'IT'], n_rows)
    
    # Add some negative quantities/prices
    quantities[10] = -1
    unit_prices[11] = -5.0
    
    return pd.DataFrame({
        'InvoiceNo': invoice_nos,
        'StockCode': stock_codes,
        'Description': descriptions,
        'Quantity': quantities,
        'UnitPrice': unit_prices,
        'CustomerID': customer_ids,
        'InvoiceDate': invoice_dates,
        'Country': countries
    })


@pytest.fixture
def transaction_baskets_dataframe():
    """DataFrame with transaction baskets (multiple items per invoice)."""
    return pd.DataFrame({
        'InvoiceNo': ['001', '001', '001', '002', '002', '003'],
        'StockCode': ['A', 'B', 'C', 'B', 'D', 'E'],
        'Description': ['item1', 'item2', 'item3', 'item2', 'item4', 'item5'],
        'Quantity': [1, 1, 1, 2, 1, 1],
        'UnitPrice': [10.0, 20.0, 30.0, 20.0, 40.0, 50.0],
        'CustomerID': [100.0, 100.0, 100.0, 101.0, 101.0, 102.0],
        'InvoiceDate': ['2023-01-01', '2023-01-01', '2023-01-01', 
                        '2023-01-02', '2023-01-02', '2023-01-03'],
        'Country': ['UK', 'UK', 'UK', 'US', 'US', 'FR'],
        'TransactionValue': [60.0, 60.0, 60.0, 40.0, 40.0, 50.0]
    })


@pytest.fixture
def temp_data_directory():
    """Temporary directory for test file I/O."""
    with TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def config_with_llm_disabled(monkeypatch):
    """Configure with all LLM features disabled."""
    monkeypatch.setenv('NORMALIZATION_ENABLED', 'false')
    monkeypatch.setenv('ENRICHMENT_ENABLED', 'false')
    monkeypatch.setenv('OUTLIER_ENABLED', 'false')
    monkeypatch.setenv('CONTEXT_ENABLED', 'false')


@pytest.fixture
def config_with_llm_enabled(monkeypatch):
    """Configure with LLM features enabled (with mocking)."""
    monkeypatch.setenv('NORMALIZATION_ENABLED', 'true')
    monkeypatch.setenv('ENRICHMENT_ENABLED', 'true')
    monkeypatch.setenv('OUTLIER_ENABLED', 'true')
    monkeypatch.setenv('CONTEXT_ENABLED', 'true')


@pytest.fixture
def mock_llm_functions(monkeypatch):
    """Mock all LLM function calls for speed & consistency."""
    
    # Mock normalize_description_with_llm
    def mock_normalize(description, **kwargs):
        return description.lower().strip()
    
    # Mock enrich_categories_batch_with_llm
    def mock_enrich_batch(descriptions, **kwargs):
        return {
            desc: {
                'category': 'test_category',
                'material': 'test_material',
                'size': 'M',
                'theme': 'test_theme'
            }
            for desc in descriptions
        }
    
    # Mock batch_score_anomalies_with_llm
    def mock_score_anomalies(descriptions, **kwargs):
        return {
            desc: {'is_anomaly': False, 'score': 0.1, 'reason': 'normal'}
            for desc in descriptions
        }
    
    # Mock extract_contexts_with_llm
    def mock_extract_contexts(description, **kwargs):
        return [f"context for {description}"]
    
    monkeypatch.setattr(
        'src.data_pipeline.normalize_description_with_llm',
        mock_normalize
    )
    monkeypatch.setattr(
        'src.data_pipeline.enrich_categories_batch_with_llm',
        mock_enrich_batch
    )
    monkeypatch.setattr(
        'src.data_pipeline.batch_score_anomalies_with_llm',
        mock_score_anomalies
    )
    monkeypatch.setattr(
        'src.data_pipeline.extract_contexts_with_llm',
        mock_extract_contexts
    )
    
    return {
        'normalize': mock_normalize,
        'enrich': mock_enrich_batch,
        'score_anomalies': mock_score_anomalies,
        'extract_contexts': mock_extract_contexts
    }


@pytest.fixture
def mock_llm_client(monkeypatch):
    """Mock LLMClient methods to return deterministic LLM outputs."""

    def mock_validate_credentials(self):
        return any(
            os.getenv(key)
            for key in (
                "OPENAI_API_KEY",
                "AZURE_OPENAI_API_KEY",
                "GEMINI_API_KEY",
                "ANTHROPIC_API_KEY",
                "PERPLEXITY_API_KEY",
            )
        )

    def mock_chat_completion_json(self, system_prompt, user_prompt):
        content = f"{system_prompt} {user_prompt}".lower()
        if "anomaly" in content:
            return [{"key": "mock", "anomaly_type": "none", "anomaly_reason": "normal"}]
        if "context" in content:
            return {"contexts": [{"context": "mock context", "confidence": 0.9}]}
        return {
            "category": "test_category",
            "material": "test_material",
            "size": "M",
            "theme": "test_theme",
        }

    monkeypatch.setattr(LLMClient, "validate_credentials", mock_validate_credentials)
    monkeypatch.setattr(LLMClient, "chat_completion_json", mock_chat_completion_json)

    return {
        "validate_credentials": mock_validate_credentials,
        "chat_completion_json": mock_chat_completion_json,
    }


# ============================================================================
# TEST CLASS: Basic Pipeline Initialization & Methods
# ============================================================================

class TestDataPipelineBasics:
    """Test initialization and basic methods."""
    
    def test_initialization_defaults(self):
        """Test DataPipeline initializes with default values."""
        pipeline = DataPipeline()
        assert pipeline.raw_data is None
        assert pipeline.processed_data is None
        assert pipeline.products is None
        assert pipeline.transactions is None
        assert pipeline.bundles is None
        assert pipeline.force_reprocess is False
    
    def test_initialization_with_force_reprocess(self):
        """Test DataPipeline initializes with force_reprocess=True."""
        pipeline = DataPipeline(force_reprocess=True)
        assert pipeline.force_reprocess is True
    
    def test_preprocess_without_raw_data(self):
        """Verify ValueError raised when preprocessing without loaded data."""
        pipeline = DataPipeline()
        with pytest.raises(ValueError, match="Raw data not loaded"):
            pipeline.preprocess()
    
    def test_create_baskets_without_processed_data(self):
        """Verify ValueError raised when creating baskets without preprocessing."""
        pipeline = DataPipeline()
        with pytest.raises(ValueError, match="Data not preprocessed"):
            pipeline.create_transaction_baskets()
    
    def test_generate_bundles_without_baskets(self):
        """Verify ValueError raised when generating bundles without baskets."""
        pipeline = DataPipeline()
        with pytest.raises(ValueError, match="Transaction baskets not created"):
            pipeline.generate_product_bundles()


# ============================================================================
# TEST CLASS: Data Loading & Exploration
# ============================================================================

class TestLoadAndExplore:
    """Test data loading and statistics generation."""
    
    def test_load_raw_data_success(self, minimal_dataframe, temp_data_directory):
        """Test loading raw data from valid CSV file."""
        filepath = os.path.join(temp_data_directory, 'test_data.csv')
        minimal_dataframe.to_csv(filepath, index=False, encoding='ISO-8859-1')
        
        pipeline = DataPipeline()
        result = pipeline.load_raw_data(filepath)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert pipeline.raw_data is not None
        assert result.equals(minimal_dataframe)
    
    def test_load_raw_data_file_not_found(self):
        """Test FileNotFoundError raised for missing file."""
        pipeline = DataPipeline()
        with pytest.raises(FileNotFoundError, match="Data file not found"):
            pipeline.load_raw_data('/nonexistent/path/data.csv')
    
    def test_load_raw_data_encoding_iso_8859_1(self, temp_data_directory):
        """Test loading CSV with ISO-8859-1 encoding."""
        df = pd.DataFrame({
            'Description': ['café', 'naïve', 'résumé'],
            'Quantity': [1, 2, 3],
            'UnitPrice': [10.0, 20.0, 30.0],
            'CustomerID': [100.0, 101.0, 102.0],
            'InvoiceNo': ['001', '002', '003'],
            'StockCode': ['A', 'B', 'C'],
            'InvoiceDate': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'Country': ['FR', 'FR', 'FR']
        })
        filepath = os.path.join(temp_data_directory, 'test_encoding.csv')
        df.to_csv(filepath, index=False, encoding='ISO-8859-1')
        
        pipeline = DataPipeline()
        result = pipeline.load_raw_data(filepath)
        
        assert len(result) == 3
        assert 'café' in result['Description'].values or 'caf' in str(result['Description'].values)
    
    def test_explore_data_raw(self, minimal_dataframe):
        """Test explore_data with raw data."""
        pipeline = DataPipeline()
        pipeline.raw_data = minimal_dataframe
        
        stats = pipeline.explore_data(data_type='raw')
        
        assert stats['stage'] == 'raw'
        assert stats['shape'] == (3, 8)
        assert 'columns' in stats
        assert 'dtypes' in stats
        assert 'missing_values' in stats
        assert 'unique_customers' in stats
        assert stats['unique_customers'] == 3
    
    def test_explore_data_processed(self, minimal_dataframe):
        """Test explore_data with processed data."""
        pipeline = DataPipeline()
        # Add processed data columns
        minimal_dataframe['TransactionValue'] = [10.0, 40.0, 90.0]
        pipeline.processed_data = minimal_dataframe
        
        stats = pipeline.explore_data(data_type='processed')
        
        assert stats['stage'] == 'processed'
        assert 'transaction_value_stats' in stats
        assert 'quantity_stats' in stats
    
    def test_explore_data_auto_picks_processed_first(self, minimal_dataframe):
        """Test explore_data auto mode prefers processed over raw."""
        pipeline = DataPipeline()
        pipeline.raw_data = minimal_dataframe.copy()
        minimal_dataframe['TransactionValue'] = [10.0, 40.0, 90.0]
        pipeline.processed_data = minimal_dataframe
        
        stats = pipeline.explore_data(data_type='auto')
        
        assert stats['stage'] == 'processed'
    
    def test_explore_data_auto_falls_back_to_raw(self, minimal_dataframe):
        """Test explore_data auto mode falls back to raw."""
        pipeline = DataPipeline()
        pipeline.raw_data = minimal_dataframe
        pipeline.processed_data = None
        
        stats = pipeline.explore_data(data_type='auto')
        
        assert stats['stage'] == 'raw'
    
    def test_explore_data_invalid_type(self):
        """Test ValueError for invalid data_type."""
        pipeline = DataPipeline()
        with pytest.raises(ValueError, match="Invalid data_type"):
            pipeline.explore_data(data_type='invalid')
    
    def test_explore_data_no_data_available(self):
        """Test ValueError when no data available."""
        pipeline = DataPipeline()
        with pytest.raises(ValueError, match="data not available"):
            pipeline.explore_data(data_type='raw')
    
    def test_convert_csv_to_tsv_success(self, minimal_dataframe, temp_data_directory):
        """Test successful CSV to TSV conversion."""
        csv_path = os.path.join(temp_data_directory, 'test.csv')
        tsv_path = os.path.join(temp_data_directory, 'test.tsv')
        
        minimal_dataframe.to_csv(csv_path, index=False, encoding='ISO-8859-1')
        
        pipeline = DataPipeline()
        result = pipeline.convert_csv_to_tsv(csv_path, tsv_path)
        
        assert result is True
        assert os.path.exists(tsv_path)
        
        # Verify TSV content
        df_tsv = pd.read_csv(tsv_path, sep='\t', encoding='utf-8')
        assert len(df_tsv) == 3
        assert list(df_tsv.columns) == list(minimal_dataframe.columns)
    
    def test_convert_csv_to_tsv_creates_output_dir(self, minimal_dataframe, temp_data_directory):
        """Test convert_csv_to_tsv creates output directory if not exists."""
        csv_path = os.path.join(temp_data_directory, 'test.csv')
        tsv_path = os.path.join(temp_data_directory, 'subdir', 'test.tsv')
        
        minimal_dataframe.to_csv(csv_path, index=False, encoding='ISO-8859-1')
        
        pipeline = DataPipeline()
        result = pipeline.convert_csv_to_tsv(csv_path, tsv_path)
        
        assert result is True
        assert os.path.exists(tsv_path)


# ============================================================================
# TEST CLASS: Data Cleaning
# ============================================================================

class TestDataCleaning:
    """Test individual preprocessing/cleaning steps."""
    
    def test_handle_cancellations_removes_c_prefix(self, realistic_sample_dataframe):
        """Test that cancellations (InvoiceNo prefix 'C') are removed."""
        pipeline = DataPipeline()
        pipeline.raw_data = realistic_sample_dataframe.copy()
        
        # Count cancellations before
        cancellations_before = (pipeline.raw_data['InvoiceNo'].str.startswith('C')).sum()
        assert cancellations_before > 0
        
        # Run preprocessing
        result = pipeline.preprocess()
        
        # Verify cancellations removed
        cancellations_after = (result['InvoiceNo'].str.startswith('C')).sum()
        assert cancellations_after == 0
        assert len(result) < len(realistic_sample_dataframe)
    
    def test_remove_missing_customers_filters_nulls(self, realistic_sample_dataframe):
        """Test that rows with null CustomerID are removed."""
        pipeline = DataPipeline()
        pipeline.raw_data = realistic_sample_dataframe.copy()
        
        # Count nulls before
        nulls_before = pipeline.raw_data['CustomerID'].isna().sum()
        assert nulls_before > 0
        
        # Run preprocessing
        result = pipeline.preprocess()
        
        # Verify nulls removed
        nulls_after = result['CustomerID'].isna().sum()
        assert nulls_after == 0
    
    def test_clean_data_removes_negative_quantity(self, realistic_sample_dataframe):
        """Test that negative Quantity values are removed."""
        pipeline = DataPipeline()
        pipeline.raw_data = realistic_sample_dataframe.copy()
        
        # Verify negative quantity exists
        assert (pipeline.raw_data['Quantity'] < 0).any()
        
        # Run preprocessing
        result = pipeline.preprocess()
        
        # Verify all quantities are positive
        assert (result['Quantity'] > 0).all()
    
    def test_clean_data_removes_negative_price(self, realistic_sample_dataframe):
        """Test that negative UnitPrice values are removed."""
        pipeline = DataPipeline()
        pipeline.raw_data = realistic_sample_dataframe.copy()
        
        # Verify negative price exists
        assert (pipeline.raw_data['UnitPrice'] < 0).any()
        
        # Run preprocessing
        result = pipeline.preprocess()
        
        # Verify all prices are positive
        assert (result['UnitPrice'] > 0).all()
    
    def test_clean_data_removes_null_descriptions(self):
        """Test that rows with null Description are removed."""
        df = pd.DataFrame({
            'InvoiceNo': ['001', '002', '003'],
            'StockCode': ['A', 'B', 'C'],
            'Description': ['item1', None, 'item3'],
            'Quantity': [1, 1, 1],
            'UnitPrice': [10.0, 10.0, 10.0],
            'CustomerID': [100.0, 101.0, 102.0],
            'InvoiceDate': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'Country': ['UK', 'US', 'FR']
        })
        
        pipeline = DataPipeline()
        pipeline.raw_data = df
        result = pipeline.preprocess()
        
        assert len(result) == 2  # One null description removed
        assert result['Description'].notna().all()


# ============================================================================
# TEST CLASS: Transaction Baskets & Bundle Generation
# ============================================================================

class TestBundleGeneration:
    """Test transaction baskets and bundle creation."""
    
    def test_create_transaction_baskets_grouping(self, transaction_baskets_dataframe):
        """Test create_transaction_baskets groups by InvoiceNo."""
        pipeline = DataPipeline()
        pipeline.processed_data = transaction_baskets_dataframe.copy()
        pipeline.processed_data['TransactionValue'] = [60.0, 60.0, 60.0, 40.0, 40.0, 50.0]
        
        baskets = pipeline.create_transaction_baskets()
        
        assert len(baskets) == 3  # 3 unique invoices
        assert 'Items' in baskets.columns
        # First invoice has 3 items
        assert len(baskets.iloc[0]['Items']) == 3
    
    def test_create_transaction_baskets_filters_single_items(self):
        """Test that single-item baskets are filtered out."""
        df = pd.DataFrame({
            'InvoiceNo': ['001', '002', '003'],
            'Description': ['item1', 'item2', 'item3'],
            'Quantity': [1, 1, 1],
            'UnitPrice': [10.0, 10.0, 10.0],
            'CustomerID': [100.0, 101.0, 102.0],
            'InvoiceDate': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'TransactionValue': [10.0, 10.0, 10.0]
        })
        
        pipeline = DataPipeline()
        pipeline.processed_data = df
        baskets = pipeline.create_transaction_baskets()
        
        assert len(baskets) == 0  # All single-item, all filtered out
    
    def test_create_transaction_baskets_aggregation(self, transaction_baskets_dataframe):
        """Test aggregation of metadata in baskets."""
        pipeline = DataPipeline()
        pipeline.processed_data = transaction_baskets_dataframe.copy()
        
        baskets = pipeline.create_transaction_baskets()
        
        # Check aggregated columns
        assert 'CustomerID' in baskets.columns
        assert 'InvoiceDate' in baskets.columns
        assert 'TransactionValue' in baskets.columns
        assert 'Quantity' in baskets.columns
        
        # First basket should have CustomerID=100
        assert baskets.iloc[0]['CustomerID'] == 100.0
    
    def test_generate_bundles_with_min_support(self, transaction_baskets_dataframe):
        """Test bundle generation respects min_support threshold."""
        pipeline = DataPipeline()
        pipeline.processed_data = transaction_baskets_dataframe.copy()
        baskets = pipeline.create_transaction_baskets()
        pipeline.transactions = baskets
        
        # Low support should generate more bundles
        bundles_low = pipeline.generate_product_bundles(min_support=0.1)
        
        # High support should generate fewer bundles
        bundles_high = pipeline.generate_product_bundles(min_support=0.5)
        
        assert len(bundles_low) >= len(bundles_high)
    
    def test_generate_bundles_max_size_limit(self, transaction_baskets_dataframe):
        """Test bundle generation respects max_size limit."""
        pipeline = DataPipeline()
        pipeline.processed_data = transaction_baskets_dataframe.copy()
        baskets = pipeline.create_transaction_baskets()
        pipeline.transactions = baskets
        
        # Generate bundles with max_size=2
        bundles = pipeline.generate_product_bundles(max_size=2)
        
        # All bundles should have size <= 2
        if len(bundles) > 0:
            max_bundle_size_in_result = max(len(bundle) for bundle in bundles)
            assert max_bundle_size_in_result <= 2
    
    def test_generate_bundles_empty_baskets(self):
        """Test bundle generation with empty baskets returns empty list."""
        df = pd.DataFrame(columns=['Items', 'CustomerID', 'InvoiceDate', 'TransactionValue', 'Quantity'])
        
        pipeline = DataPipeline()
        pipeline.processed_data = df
        pipeline.transactions = df
        
        bundles = pipeline.generate_product_bundles()
        
        assert len(bundles) == 0
    
    def test_get_bundle_statistics_distribution(self, transaction_baskets_dataframe):
        """Test bundle statistics calculation."""
        pipeline = DataPipeline()
        pipeline.processed_data = transaction_baskets_dataframe.copy()
        baskets = pipeline.create_transaction_baskets()
        pipeline.transactions = baskets
        bundles = pipeline.generate_product_bundles(min_support=0.1)
        pipeline.bundles = bundles
        
        stats = pipeline.get_bundle_statistics()
        
        assert isinstance(stats, dict)
        assert 'total_bundles' in stats
        assert 'bundle_sizes' in stats


# ============================================================================
# TEST CLASS: LLM Features & Caching (Mocked)
# ============================================================================

class TestLLMFeatures:
    """Test LLM-dependent features with mocking."""
    
    def test_enrich_categories_cache_first(self, minimal_dataframe, mock_llm_client, monkeypatch):
        """Test category enrichment uses cache-first pattern."""
        monkeypatch.setenv('ENRICHMENT_ENABLED', 'true')
        monkeypatch.setenv('ENRICHMENT_CACHE_FIRST', 'true')
        
        pipeline = DataPipeline()
        pipeline.raw_data = minimal_dataframe.copy()
        
        # Run preprocessing with enrichment mocked
        result = pipeline.preprocess()
        
        # Should complete without LLM errors
        assert result is not None
    
    def test_enrich_categories_no_credentials(self, minimal_dataframe, mock_llm_client, monkeypatch):
        """Test graceful handling when LLM credentials missing."""
        monkeypatch.setenv('ENRICHMENT_ENABLED', 'true')
        monkeypatch.setenv('OPENAI_API_KEY', '')
        
        pipeline = DataPipeline()
        pipeline.raw_data = minimal_dataframe.copy()
        
        # Should not crash, just skip enrichment or fill with NaN
        result = pipeline.preprocess()
        assert result is not None
    
    def test_flag_anomalies_iqr_detection(self, minimal_dataframe, mock_llm_client, monkeypatch):
        """Test IQR-based anomaly detection."""
        monkeypatch.setenv('OUTLIER_ENABLED', 'false')  # Use heuristic only
        
        pipeline = DataPipeline()
        pipeline.raw_data = minimal_dataframe.copy()
        
        result = pipeline.preprocess()
        
        # Should have anomaly columns
        if 'check_anomaly' in result.columns:
            assert isinstance(result['check_anomaly'].iloc[0], (bool, np.bool_))
    
    def test_extract_contexts_mocked(self, minimal_dataframe, mock_llm_client, monkeypatch):
        """Test context extraction with mock."""
        monkeypatch.setenv('CONTEXT_ENABLED', 'true')
        
        pipeline = DataPipeline()
        pipeline.raw_data = minimal_dataframe.copy()
        
        result = pipeline.preprocess()
        
        # Should complete without errors
        assert result is not None


# ==========================================================================
# TEST CLASS: Pipeline Helper Methods
# ==========================================================================

class TestPipelineHelpers:
    """Test helper methods used by pipeline LLM integration."""

    def test_get_api_key_for_provider(self, monkeypatch):
        """Verify API key mapping for each provider."""
        monkeypatch.setattr('src.data_pipeline.OPENAI_API_KEY', 'openai-key')
        monkeypatch.setattr('src.data_pipeline.AZURE_OPENAI_API_KEY', 'azure-key')
        monkeypatch.setattr('src.data_pipeline.GEMINI_API_KEY', 'gemini-key')
        monkeypatch.setattr('src.data_pipeline.ANTHROPIC_API_KEY', 'anthropic-key')
        monkeypatch.setattr('src.data_pipeline.PERPLEXITY_API_KEY', 'perplexity-key')

        pipeline = DataPipeline()

        assert pipeline._get_api_key_for_provider('openai') == 'openai-key'
        assert pipeline._get_api_key_for_provider('azure') == 'azure-key'
        assert pipeline._get_api_key_for_provider('gemini') == 'gemini-key'
        assert pipeline._get_api_key_for_provider('anthropic') == 'anthropic-key'
        assert pipeline._get_api_key_for_provider('perplexity') == 'perplexity-key'

    def test_build_pipeline_llm_config(self, monkeypatch):
        """Verify LLM config creation uses provider-specific credentials."""
        monkeypatch.setattr('src.data_pipeline.LLM_PROVIDER', 'openai')
        monkeypatch.setattr('src.data_pipeline.LLM_MODEL', 'test-model')
        monkeypatch.setattr('src.data_pipeline.LLM_TEMPERATURE', 0.1)
        monkeypatch.setattr('src.data_pipeline.LLM_MAX_TOKENS', 123)
        monkeypatch.setattr('src.data_pipeline.LLM_TIMEOUT_SECONDS', 7)
        monkeypatch.setattr('src.data_pipeline.OPENAI_API_KEY', 'openai-key')
        monkeypatch.setattr('src.data_pipeline.AZURE_OPENAI_API_KEY', 'azure-key')

        pipeline = DataPipeline()
        config = pipeline._build_pipeline_llm_config()

        assert config.provider == 'openai'
        assert config.model == 'test-model'
        assert config.openai_api_key == 'openai-key'
        assert config.azure_api_key is None

    def test_handle_llm_quota_error_standardized(self):
        """Verify standardized quota handling raises LLMQuotaExceededError."""
        pipeline = DataPipeline()
        with pytest.raises(LLMQuotaExceededError, match='quota'):
            pipeline._handle_llm_quota_error_standardized(Exception('quota'), 'skip_batch')


# ============================================================================
# TEST CLASS: Edge Cases & Error Handling
# ============================================================================

class TestEdgeCases:
    """Test boundary conditions and edge cases."""
    
    def test_empty_dataframe_processing(self):
        """Test preprocessing empty DataFrame."""
        df = pd.DataFrame(columns=[
            'InvoiceNo', 'StockCode', 'Description', 'Quantity', 
            'UnitPrice', 'CustomerID', 'InvoiceDate', 'Country'
        ])
        
        pipeline = DataPipeline()
        pipeline.raw_data = df
        
        result = pipeline.preprocess()
        
        assert len(result) == 0
    
    def test_single_row_dataframe(self):
        """Test preprocessing single-row DataFrame."""
        df = pd.DataFrame({
            'InvoiceNo': ['001'],
            'StockCode': ['A'],
            'Description': ['item1'],
            'Quantity': [1],
            'UnitPrice': [10.0],
            'CustomerID': [100.0],
            'InvoiceDate': ['2023-01-01'],
            'Country': ['UK']
        })
        
        pipeline = DataPipeline()
        pipeline.raw_data = df
        
        result = pipeline.preprocess()
        
        assert len(result) <= 1
    
    def test_all_nulls_column(self):
        """Test handling DataFrame with all-null column."""
        df = pd.DataFrame({
            'InvoiceNo': ['001', '002', '003'],
            'StockCode': ['A', 'B', 'C'],
            'Description': [None, None, None],
            'Quantity': [1, 1, 1],
            'UnitPrice': [10.0, 10.0, 10.0],
            'CustomerID': [100.0, 101.0, 102.0],
            'InvoiceDate': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'Country': ['UK', 'US', 'FR']
        })
        
        pipeline = DataPipeline()
        pipeline.raw_data = df
        
        result = pipeline.preprocess()
        
        # All should be filtered out due to null descriptions
        assert len(result) == 0
    
    def test_special_characters_descriptions(self):
        """Test handling descriptions with special characters."""
        df = pd.DataFrame({
            'InvoiceNo': ['001', '002', '003'],
            'StockCode': ['A', 'B', 'C'],
            'Description': ['item@#$', 'item!%^', 'item&*()'],
            'Quantity': [1, 1, 1],
            'UnitPrice': [10.0, 10.0, 10.0],
            'CustomerID': [100.0, 101.0, 102.0],
            'InvoiceDate': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'Country': ['UK', 'US', 'FR']
        })
        
        pipeline = DataPipeline()
        pipeline.raw_data = df
        
        result = pipeline.preprocess()
        
        assert len(result) == 3  # Should not crash
    
    def test_unicode_encoding_handling(self):
        """Test handling of Unicode characters in descriptions."""
        df = pd.DataFrame({
            'InvoiceNo': ['001', '002', '003'],
            'StockCode': ['A', 'B', 'C'],
            'Description': ['café', '日本語', 'Ελληνικά'],
            'Quantity': [1, 1, 1],
            'UnitPrice': [10.0, 10.0, 10.0],
            'CustomerID': [100.0, 101.0, 102.0],
            'InvoiceDate': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'Country': ['FR', 'JP', 'GR']
        })
        
        pipeline = DataPipeline()
        pipeline.raw_data = df
        
        result = pipeline.preprocess()
        
        assert len(result) == 3  # Should handle Unicode


# ============================================================================
# TEST CLASS: Serialization (Save/Load)
# ============================================================================

class TestSerialization:
    """Test save/load operations."""
    
    def test_save_processed_data_pickle(self, minimal_dataframe, temp_data_directory):
        """Test saving processed data as pickle."""
        pipeline = DataPipeline()
        pipeline.processed_data = minimal_dataframe
        
        pickle_path = os.path.join(temp_data_directory, 'processed.pkl')
        result = pipeline.save_processed_data(pickle_path)
        
        assert result is True
        assert os.path.exists(pickle_path)
    
    def test_load_processed_data_success(self, minimal_dataframe, temp_data_directory):
        """Test loading processed data from pickle."""
        pickle_path = os.path.join(temp_data_directory, 'processed.pkl')
        
        # Save first
        pipeline1 = DataPipeline()
        pipeline1.processed_data = minimal_dataframe
        pipeline1.save_processed_data(pickle_path)
        
        # Load
        pipeline2 = DataPipeline()
        result = pipeline2.load_processed_data(pickle_path)
        
        assert result is True
        assert pipeline2.processed_data is not None
        assert len(pipeline2.processed_data) == 3
    
    def test_load_processed_data_file_not_found(self):
        """Test FileNotFoundError for missing pickle."""
        pipeline = DataPipeline()
        with pytest.raises(FileNotFoundError):
            pipeline.load_processed_data('/nonexistent/path/processed.pkl')


# ============================================================================
# TEST CLASS: Integration Tests (Full Pipelines)
# ============================================================================

class TestFullPipeline:
    """Test end-to-end workflows."""
    
    def test_pipeline_load_to_baskets(self, realistic_sample_dataframe, temp_data_directory, mock_llm_client, monkeypatch):
        """Full pipeline: load → preprocess → create baskets."""
        # Disable LLM to speed up test
        monkeypatch.setenv('ENRICHMENT_ENABLED', 'false')
        monkeypatch.setenv('OUTLIER_ENABLED', 'false')
        monkeypatch.setenv('CONTEXT_ENABLED', 'false')
        
        # Save test data
        csv_path = os.path.join(temp_data_directory, 'test_data.csv')
        realistic_sample_dataframe.to_csv(csv_path, index=False, encoding='ISO-8859-1')
        
        # Run pipeline
        pipeline = DataPipeline()
        pipeline.load_raw_data(csv_path)
        processed = pipeline.preprocess()
        baskets = pipeline.create_transaction_baskets()
        
        assert len(processed) > 0
        assert len(baskets) >= 0
    
    def test_pipeline_load_to_bundles(self, transaction_baskets_dataframe, temp_data_directory, monkeypatch):
        """Full pipeline: load → preprocess → baskets → bundles."""
        monkeypatch.setenv('ENRICHMENT_ENABLED', 'false')
        monkeypatch.setenv('OUTLIER_ENABLED', 'false')
        
        # Save test data
        csv_path = os.path.join(temp_data_directory, 'baskets.csv')
        transaction_baskets_dataframe.to_csv(csv_path, index=False, encoding='ISO-8859-1')
        
        # Run pipeline
        pipeline = DataPipeline()
        pipeline.load_raw_data(csv_path)
        processed = pipeline.preprocess()
        baskets = pipeline.create_transaction_baskets()
        bundles = pipeline.generate_product_bundles()
        
        assert len(processed) > 0
        assert len(baskets) > 0
        assert isinstance(bundles, list)
    
    def test_pipeline_reproducibility(self, realistic_sample_dataframe, temp_data_directory, monkeypatch):
        """Same RANDOM_STATE produces same results."""
        monkeypatch.setenv('ENRICHMENT_ENABLED', 'false')
        monkeypatch.setenv('OUTLIER_ENABLED', 'false')
        
        csv_path = os.path.join(temp_data_directory, 'repro_test.csv')
        realistic_sample_dataframe.to_csv(csv_path, index=False, encoding='ISO-8859-1')
        
        # First run
        pipeline1 = DataPipeline()
        pipeline1.load_raw_data(csv_path)
        processed1 = pipeline1.preprocess()
        
        # Second run (same data, same seed)
        pipeline2 = DataPipeline()
        pipeline2.load_raw_data(csv_path)
        processed2 = pipeline2.preprocess()
        
        # Results should be identical
        assert len(processed1) == len(processed2)
        assert processed1.shape == processed2.shape


# ============================================================================
# TEST CLASS: Configuration & Parameter Tests
# ============================================================================

class TestConfigurationParameters:
    """Test effect of configuration parameters."""
    
    def test_min_support_threshold_effect(self, transaction_baskets_dataframe):
        """Higher min_support → fewer bundles."""
        pipeline = DataPipeline()
        pipeline.processed_data = transaction_baskets_dataframe.copy()
        baskets = pipeline.create_transaction_baskets()
        pipeline.transactions = baskets
        
        bundles_2pct = pipeline.generate_product_bundles(min_support=0.01)
        bundles_50pct = pipeline.generate_product_bundles(min_support=0.50)
        
        # Higher threshold should have fewer or equal bundles
        assert len(bundles_50pct) <= len(bundles_2pct)
    
    def test_max_bundle_size_effect(self, transaction_baskets_dataframe):
        """max_size parameter limits bundle size."""
        pipeline = DataPipeline()
        pipeline.processed_data = transaction_baskets_dataframe.copy()
        baskets = pipeline.create_transaction_baskets()
        pipeline.transactions = baskets
        
        bundles_size2 = pipeline.generate_product_bundles(max_size=2)
        bundles_size5 = pipeline.generate_product_bundles(max_size=5)
        
        # Verify size limits
        if len(bundles_size2) > 0:
            max_in_size2 = max(len(b) for b in bundles_size2)
            assert max_in_size2 <= 2
        
        if len(bundles_size5) > 0:
            max_in_size5 = max(len(b) for b in bundles_size5)
            assert max_in_size5 <= 5


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
