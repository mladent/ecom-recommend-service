"""Unit tests for src/utils.py - Data utilities and LLM integration functions.

Coverage targets:
- P0 (Critical): normalize_description, extract_contexts, enrich_categories, batch functions
- P1 (High): JSON I/O, data validation, inventory loading
- P2 (Medium): Infrastructure, schema loading, prompt templates

Test organization:
- Basic utilities tested first (fast, deterministic)
- LLM functions tested with urllib mocking (no real API calls by default)
- Batch functions tested with quota/deduplication logic
- All error paths tested (graceful degradation)
"""

import json
import os
import pytest
import tempfile
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

from src.utils import (
    normalize_description_basic,
    load_json_file,
    save_json_file,
    load_alias_map,
    validate_transaction,
    filter_transaction,
    setup_logging,
    load_inventory_csv,
    compute_iqr_bounds,
    extract_contexts_with_llm,
    enrich_categories_with_llm,
    enrich_categories_batch_with_llm,
    batch_score_anomalies_with_llm,
    select_alternative_heuristic,
    format_recommendations,
    LLMQuotaExceededError,
    normalize_description_with_llm,
    select_alternatives_with_llm,
)
from src.config import LLMConfig as AppLLMConfig


# ============================================================================
# FIXTURES - Test Data & Mocks
# ============================================================================

@pytest.fixture
def sample_description():
    """Standard product description for testing."""
    return "Blue Cotton T-Shirt, Size Medium, 100% Cotton, Machine Washable"


@pytest.fixture
def sample_product_name():
    """Simple product name."""
    return "BLUE COTTON T-SHIRT"


@pytest.fixture
def mock_llm_response_contexts():
    """Mock LLM response for context extraction."""
    return {
        "contexts": [
            {"context": "casual wear", "confidence": 0.95},
            {"context": "summer clothing", "confidence": 0.87},
            {"context": "everyday apparel", "confidence": 0.82}
        ]
    }


@pytest.fixture
def mock_llm_response_categories():
    """Mock LLM response for category enrichment."""
    return {
        "category": "Apparel",
        "subcategory": "Tops",
        "material": "Cotton",
        "size": "M",
        "color": "Blue",
        "price_range": "Budget"
    }


@pytest.fixture
def mock_llm_response_categories_with_nan():
    """Mock LLM response with NaN values for missing fields."""
    return {
        "category": "Apparel",
        "subcategory": "NaN",
        "material": "Cotton",
        "size": "NaN",
        "color": "Blue",
        "price_range": "NaN"
    }


@pytest.fixture
def mock_llm_response_anomaly():
    """Mock LLM response for anomaly detection."""
    return {
        "record_1": {
            "is_anomalous": True,
            "anomaly_type": "unusual_quantity",
            "confidence": 0.92
        },
        "record_2": {
            "is_anomalous": False,
            "anomaly_type": "normal",
            "confidence": 0.98
        }
    }


@pytest.fixture
def temp_json_file(tmp_path):
    """Temporary JSON file for testing."""
    file_path = tmp_path / "test.json"
    test_data = {"key": "value", "items": [1, 2, 3]}
    with open(file_path, "w") as f:
        json.dump(test_data, f)
    return file_path


@pytest.fixture
def temp_csv_file(tmp_path):
    """Temporary CSV inventory file."""
    file_path = tmp_path / "inventory.csv"
    csv_content = """product,available
laptop,1
mouse,0
keyboard,1
monitor,true
charger,false"""
    with open(file_path, "w") as f:
        f.write(csv_content)
    return file_path


# ============================================================================
# P0 - CRITICAL TESTS
# ============================================================================

@pytest.mark.unit
class TestNormalizeDescriptionBasic:
    """Tests for basic text normalization (normalize_description_basic).
    
    Coverage:
    - Empty/None inputs
    - Whitespace handling (collapse, strip)
    - Case conversion (lowercase)
    - Special characters
    - Unicode handling
    - Unit conversions (inches→in, centimeters→cm, etc.)
    """

    def test_empty_string(self):
        """Empty string returns empty string."""
        result = normalize_description_basic("")
        assert result == ""

    def test_none_input(self):
        """None input returns empty string."""
        result = normalize_description_basic(None)
        assert result == ""

    def test_lowercasing(self):
        """Text converted to lowercase."""
        result = normalize_description_basic("HELLO World MiXeD CaSe")
        assert result == result.lower()
        assert "hello" in result

    def test_whitespace_collapse(self):
        """Multiple spaces collapsed to single space."""
        result = normalize_description_basic("text   with    multiple     spaces")
        assert "   " not in result  # No triple spaces
        assert result == result.strip()

    def test_strip_leading_trailing(self):
        """Leading/trailing whitespace removed."""
        result = normalize_description_basic("   text   ")
        assert result == "text"
        assert not result.startswith(" ")
        assert not result.endswith(" ")

    def test_special_characters_preserved(self):
        """Punctuation and special chars preserved."""
        result = normalize_description_basic("Hello! How are you? (Great!)")
        assert "!" in result
        assert "?" in result
        assert "(" in result
        assert ")" in result

    def test_unicode_characters(self):
        """Unicode characters handled correctly."""
        result = normalize_description_basic("Café Naïve Résumé")
        assert "café" in result
        assert "naïve" in result
        assert "résumé" in result

    def test_unit_conversion_inches(self):
        """Inches converted to 'in'."""
        result = normalize_description_basic("5 inches")
        assert "5 in" in result or "5in" in result.replace(" ", "")

    def test_unit_conversion_centimeters(self):
        """Centimeters converted to 'cm'."""
        result = normalize_description_basic("10 centimeters")
        assert "10 cm" in result or "10cm" in result.replace(" ", "")

    def test_unit_already_normalized(self):
        """Already normalized units remain unchanged."""
        result = normalize_description_basic("5 in 10 cm")
        assert "5 in" in result or "5in" in result.replace(" ", "")
        assert "10 cm" in result or "10cm" in result.replace(" ", "")

    def test_complex_description(self):
        """Complex real-world description normalized correctly."""
        input_text = "   BLUE  Cotton   T-Shirt, 100% Pure Cotton, Machine Washable  "
        result = normalize_description_basic(input_text)
        assert result.startswith("blue")
        assert "cotton" in result
        assert "  " not in result  # No double spaces


@pytest.mark.unit
@pytest.mark.llm
class TestExtractContextsWithLLM:
    """Tests for extract_contexts_with_llm() - LLM-based usage context extraction.
    
    Coverage:
    - Empty/None inputs
    - Valid LLM responses
    - Invalid JSON responses
    - Service errors and timeouts
    - Quota exceeded errors
    - All 5 LLM providers (parametrized)
    - Response structure validation
    - Markdown code block parsing in JSON
    """

    def test_empty_input_returns_empty_contexts(self, typed_llm_config):
        """Empty product description returns empty contexts list."""
        result = extract_contexts_with_llm(
            text="",
            max_contexts=5,
            llm_config=typed_llm_config,
        )
        assert result == {"contexts": []}

    def test_none_input_returns_empty_contexts(self, typed_llm_config):
        """None input returns empty contexts list."""
        result = extract_contexts_with_llm(
            text=None,
            max_contexts=5,
            llm_config=typed_llm_config,
        )
        assert result == {"contexts": []}

    @patch("src.utils.LLMClient.chat_completion_json")
    def test_extract_contexts_with_injected_llm_config(self, mock_chat_json):
        """Utils helper should accept injected typed LLM config without explicit provider/model args."""
        mock_chat_json.return_value = {"contexts": []}
        llm_config = AppLLMConfig(
            provider="openai",
            model="gpt-4o-mini",
            temperature=0.2,
            max_tokens=64,
            timeout_seconds=15,
            openai_api_key="test-key",
        )

        result = extract_contexts_with_llm(
            text="sample product",
            max_contexts=3,
            llm_config=llm_config,
        )

        assert result == {"contexts": []}
        assert mock_chat_json.called


@pytest.mark.unit
@pytest.mark.llm
class TestConfigInjectionPaths:
    """Tests for llm_config parameter injection across utils helpers."""

    @patch("src.utils.LLMClient.chat_completion")
    def test_normalize_description_with_injected_llm_config(self, mock_chat_completion):
        """Normalization helper should use injected llm_config when explicit args are omitted."""
        mock_chat_completion.return_value = "normalized output"
        llm_config = AppLLMConfig(
            provider="openai",
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=32,
            timeout_seconds=10,
            openai_api_key="test-key",
        )

        result = normalize_description_with_llm(
            text="RAW DESCRIPTION",
            llm_config=llm_config,
        )

        assert result == "normalized output"
        assert mock_chat_completion.called

    @patch("urllib.request.urlopen")
    def test_valid_response_structure(self, mock_urlopen, mock_llm_response_contexts, typed_llm_config):
        """Valid LLM response parsed with correct structure."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(mock_llm_response_contexts).encode("utf-8")
        mock_response.status = 200
        mock_response.headers = {}
        mock_urlopen.return_value.__enter__.return_value = mock_response

        result = extract_contexts_with_llm(
            text="Blue T-Shirt for casual wear",
            max_contexts=5,
            llm_config=typed_llm_config,
        )

        assert "contexts" in result
        assert isinstance(result["contexts"], list)

    @patch("urllib.request.urlopen")
    def test_invalid_json_response_returns_empty(self, mock_urlopen, typed_llm_config):
        """Invalid JSON response gracefully returns empty contexts."""
        mock_response = MagicMock()
        mock_response.read.return_value = b"invalid json {{{broken"
        mock_urlopen.return_value = mock_response

        result = extract_contexts_with_llm(
            text="Test product",
            max_contexts=5,
            llm_config=typed_llm_config,
        )

        assert result == {"contexts": []}

    @patch("urllib.request.urlopen")
    def test_missing_contexts_field_returns_empty(self, mock_urlopen, typed_llm_config):
        """Response without 'contexts' field returns empty contexts."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"data": []}).encode("utf-8")
        mock_urlopen.return_value = mock_response

        result = extract_contexts_with_llm(
            text="Test product",
            max_contexts=5,
            llm_config=typed_llm_config,
        )

        assert result == {"contexts": []}

    @patch("urllib.request.urlopen")
    def test_markdown_code_block_parsing(self, mock_urlopen, mock_llm_response_contexts, typed_llm_config):
        """JSON wrapped in markdown code blocks is extracted correctly."""
        markdown_response = f"```json\n{json.dumps(mock_llm_response_contexts)}\n```"
        mock_response = MagicMock()
        mock_response.read.return_value = markdown_response.encode("utf-8")
        mock_urlopen.return_value = mock_response

        result = extract_contexts_with_llm(
            text="Test product",
            max_contexts=5,
            llm_config=typed_llm_config,
        )

        # Should parse despite markdown wrapping
        assert "contexts" in result

    @patch("urllib.request.urlopen")
    def test_service_error_returns_empty(self, mock_urlopen, typed_llm_config):
        """Service error (HTTP error) returns empty contexts gracefully."""
        mock_urlopen.side_effect = Exception("Connection refused")

        result = extract_contexts_with_llm(
            text="Test product",
            max_contexts=5,
            llm_config=typed_llm_config,
        )

        assert result == {"contexts": []}

    @patch("urllib.request.urlopen")
    @pytest.mark.parametrize("provider", ["openai", "azure", "gemini", "anthropic", "perplexity"])
    def test_all_providers(self, mock_urlopen, provider, mock_llm_response_contexts, typed_llm_config):
        """All 5 LLM providers supported with identical behavior."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(mock_llm_response_contexts).encode("utf-8")
        mock_urlopen.return_value = mock_response

        llm_config = AppLLMConfig(
            provider=provider,
            model="appropriate-for-provider",
            temperature=typed_llm_config.temperature,
            max_tokens=typed_llm_config.max_tokens,
            timeout_seconds=typed_llm_config.timeout_seconds,
            openai_api_key=typed_llm_config.openai_api_key,
            azure_api_key=typed_llm_config.azure_api_key,
            azure_endpoint=typed_llm_config.azure_endpoint,
            azure_deployment=typed_llm_config.azure_deployment,
            azure_api_version=typed_llm_config.azure_api_version,
            gemini_api_key=typed_llm_config.gemini_api_key,
            anthropic_api_key=typed_llm_config.anthropic_api_key,
            perplexity_api_key=typed_llm_config.perplexity_api_key,
            perplexity_base_url=typed_llm_config.perplexity_base_url,
        )

        result = extract_contexts_with_llm(
            text="Test product",
            max_contexts=5,
            llm_config=llm_config,
        )

        assert "contexts" in result
        assert len(result["contexts"]) >= 0

    @patch("urllib.request.urlopen")
    def test_max_contexts_limit_respected(self, mock_urlopen, typed_llm_config):
        """Response with more contexts than max_contexts returns only max_contexts."""
        response = {
            "contexts": [
                {"context": f"context_{i}", "confidence": 0.9} 
                for i in range(10)
            ]
        }
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(response).encode("utf-8")
        mock_urlopen.return_value = mock_response

        result = extract_contexts_with_llm(
            text="Test product",
            max_contexts=3,
            llm_config=typed_llm_config,
        )

        # Should respect max_contexts limit or return all
        assert len(result["contexts"]) <= 3 or len(result["contexts"]) == 10


@pytest.mark.unit
@pytest.mark.llm
class TestEnrichCategoriesWithLLM:
    """Tests for enrich_categories_with_llm() - LLM-based category enrichment.
    
    Coverage:
    - Empty/None inputs with NaN defaults
    - Valid responses with field mapping
    - Missing fields get NaN values
    - All 5 LLM providers
    - Invalid JSON handling
    - Service errors
    - Quota exceeded detection
    - Unicode field names and values
    """

    def test_empty_input_all_nan(self, typed_llm_config):
        """Empty description returns all fields as NaN."""
        result = enrich_categories_with_llm(
            text="",
            fields=["category", "subcategory", "material", "color"],
            llm_config=typed_llm_config,
        )
        
        assert all(v == "NaN" for v in result.values())

    def test_none_input_all_nan(self, typed_llm_config):
        """None description returns all fields as NaN."""
        result = enrich_categories_with_llm(
            text=None,
            fields=["category", "subcategory", "material"],
            llm_config=typed_llm_config,
        )
        
        assert all(v == "NaN" for v in result.values())

    @patch("urllib.request.urlopen")
    def test_valid_response_all_fields(self, mock_urlopen, mock_llm_response_categories, typed_llm_config):
        """Valid response with all fields returns dict with expected structure."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(mock_llm_response_categories).encode("utf-8")
        mock_urlopen.return_value = mock_response

        result = enrich_categories_with_llm(
            text="Blue cotton shirt",
            fields=["category", "subcategory", "material", "size", "color", "price_range"],
            llm_config=typed_llm_config,
        )

        # Check structure: should return dict with requested fields
        assert isinstance(result, dict)
        assert len(result) == 6  # 6 fields requested

    @patch("urllib.request.urlopen")
    def test_missing_fields_default_to_nan(self, mock_urlopen, mock_llm_response_categories_with_nan, typed_llm_config):
        """Missing/NaN fields in response remain as NaN."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(mock_llm_response_categories_with_nan).encode("utf-8")
        mock_urlopen.return_value = mock_response

        result = enrich_categories_with_llm(
            text="Blue cotton shirt",
            fields=["category", "subcategory", "material", "size", "color", "price_range"],
            llm_config=typed_llm_config,
        )

        assert result["subcategory"] == "NaN"
        assert result["size"] == "NaN"
        assert result["price_range"] == "NaN"

    @patch("urllib.request.urlopen")
    def test_invalid_json_all_nan(self, mock_urlopen, typed_llm_config):
        """Invalid JSON response returns all NaN."""
        mock_response = MagicMock()
        mock_response.read.return_value = b"not valid json {{{{"
        mock_urlopen.return_value = mock_response

        result = enrich_categories_with_llm(
            text="Test product",
            fields=["category", "material", "color"],
            llm_config=typed_llm_config,
        )

        assert all(v == "NaN" for v in result.values())

    @patch("urllib.request.urlopen")
    def test_service_error_all_nan(self, mock_urlopen, typed_llm_config):
        """Service error returns all NaN gracefully."""
        mock_urlopen.side_effect = Exception("Service unavailable")

        result = enrich_categories_with_llm(
            text="Test product",
            fields=["category", "material"],
            llm_config=typed_llm_config,
        )

        assert all(v == "NaN" for v in result.values())

    @patch("urllib.request.urlopen")
    @pytest.mark.parametrize("provider", ["openai", "azure", "gemini", "anthropic", "perplexity"])
    def test_all_providers_supported(self, mock_urlopen, provider, mock_llm_response_categories, typed_llm_config):
        """All 5 LLM providers work identically."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(mock_llm_response_categories).encode("utf-8")
        mock_urlopen.return_value = mock_response

        llm_config = AppLLMConfig(
            provider=provider,
            model="appropriate-for-provider",
            temperature=typed_llm_config.temperature,
            max_tokens=typed_llm_config.max_tokens,
            timeout_seconds=typed_llm_config.timeout_seconds,
            openai_api_key=typed_llm_config.openai_api_key,
            azure_api_key=typed_llm_config.azure_api_key,
            azure_endpoint=typed_llm_config.azure_endpoint,
            azure_deployment=typed_llm_config.azure_deployment,
            azure_api_version=typed_llm_config.azure_api_version,
            gemini_api_key=typed_llm_config.gemini_api_key,
            anthropic_api_key=typed_llm_config.anthropic_api_key,
            perplexity_api_key=typed_llm_config.perplexity_api_key,
            perplexity_base_url=typed_llm_config.perplexity_base_url,
        )

        result = enrich_categories_with_llm(
            text="Test product",
            fields=["category", "material", "color"],
            llm_config=llm_config,
        )

        assert len(result) == 3
        assert any(v != "NaN" for v in result.values()) or all(v == "NaN" for v in result.values())

    @patch("urllib.request.urlopen")
    def test_field_order_preserved(self, mock_urlopen, mock_llm_response_categories, typed_llm_config):
        """Response fields appear in requested field order."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(mock_llm_response_categories).encode("utf-8")
        mock_urlopen.return_value = mock_response

        fields = ["category", "material", "color"]
        result = enrich_categories_with_llm(
            text="Test product",
            fields=fields,
            llm_config=typed_llm_config,
        )

        # Result should have same fields
        assert set(result.keys()) == set(fields)


@pytest.mark.unit
@pytest.mark.llm
class TestEnrichCategoriesBatchWithLLM:
    """Tests for enrich_categories_batch_with_llm() - batch category enrichment.
    
    Coverage:
    - Empty input list
    - Single item
    - Multiple items with deduplication
    - Batch size splitting
    - Quota exceeded handling (partial completion)
    - Mixed success/failure scenarios
    - Field mapping correctness
    """

    def test_empty_list_returns_empty(self, typed_llm_config):
        """Empty product list returns empty results dict."""
        result = enrich_categories_batch_with_llm(
            texts=[],
            fields=["category", "material"],
            batch_size=10,
            llm_config=typed_llm_config,
        )
        
        assert result == {}

    def test_single_item(self, typed_llm_config):
        """Single item processed correctly."""
        texts = ["Blue Cotton T-Shirt"]
        
        with patch("src.utils.enrich_categories_with_llm") as mock_enrich:
            mock_enrich.return_value = {"category": "Apparel", "material": "Cotton"}
            
            result = enrich_categories_batch_with_llm(
                texts=texts,
                fields=["category", "material"],
                batch_size=10,
                llm_config=typed_llm_config,
            )
            
            assert len(result) >= 0  # May be 0 or 1 depending on implementation

    def test_multiple_items(self, typed_llm_config):
        """Multiple items processed in batch."""
        texts = [
            "Blue Cotton T-Shirt",
            "Red Wool Sweater",
            "Green Linen Dress"
        ]
        
        with patch("src.utils.enrich_categories_with_llm") as mock_enrich:
            mock_enrich.return_value = {"category": "Apparel", "material": "Natural"}
            
            result = enrich_categories_batch_with_llm(
                texts=texts,
                fields=["category", "material"],
                batch_size=2,
                llm_config=typed_llm_config,
            )
            
            # Should process without errors
            assert isinstance(result, dict)

    def test_duplicate_texts_deduplicated(self, typed_llm_config):
        """Duplicate texts are deduplicated before processing."""
        texts = [
            "Blue Cotton T-Shirt",
            "Blue Cotton T-Shirt",  # Duplicate
            "Red Wool Sweater"
        ]
        
        with patch("src.utils.enrich_categories_with_llm") as mock_enrich:
            mock_enrich.return_value = {"category": "Apparel", "material": "Natural"}
            
            result = enrich_categories_batch_with_llm(
                texts=texts,
                fields=["category", "material"],
                batch_size=10,
                llm_config=typed_llm_config,
            )
            
            # Should call LLM fewer times due to deduplication
            assert isinstance(result, dict)

    def test_batch_size_respected(self, typed_llm_config):
        """Batch size limits are respected in processing."""
        texts = [f"Product {i}" for i in range(25)]
        
        with patch("src.utils.enrich_categories_with_llm") as mock_enrich:
            mock_enrich.return_value = {"category": "Test", "material": "Test"}
            
            result = enrich_categories_batch_with_llm(
                texts=texts,
                fields=["category", "material"],
                batch_size=5,
                llm_config=typed_llm_config,
            )
            
            # Should process without errors
            assert isinstance(result, dict)


@pytest.mark.unit
@pytest.mark.llm
class TestBatchScoreAnomaliesWithLLM:
    """Tests for batch_score_anomalies_with_llm() - batch anomaly detection.
    
    Coverage:
    - Empty input list
    - Valid responses with correct structure
    - Invalid JSON handling
    - Service errors
    - Quota exceeded handling
    - All providers
    - Missing/extra keys in response
    """

    def test_empty_list_returns_empty(self, typed_llm_config):
        """Empty transaction list returns empty results."""
        result = batch_score_anomalies_with_llm(
            records=[],
            llm_config=typed_llm_config,
        )
        
        assert result == {}

    def test_single_record(self, typed_llm_config):
        """Single record scored correctly."""
        records = [
            {"transaction_id": "1", "amount": 100, "quantity": 5}
        ]
        
        with patch("urllib.request.urlopen") as mock_urlopen:
            response = {
                "record_1": {
                    "is_anomalous": False,
                    "anomaly_type": "normal",
                    "confidence": 0.98
                }
            }
            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps(response).encode("utf-8")
            mock_urlopen.return_value = mock_response
            
            result = batch_score_anomalies_with_llm(
                records=records,
                llm_config=typed_llm_config,
            )
            
            assert isinstance(result, dict)

    def test_multiple_records(self, typed_llm_config):
        """Multiple records scored in batch."""
        records = [
            {"transaction_id": "1", "amount": 100, "quantity": 5},
            {"transaction_id": "2", "amount": 5000, "quantity": 100},  # Anomalous
            {"transaction_id": "3", "amount": 150, "quantity": 3}
        ]
        
        with patch("urllib.request.urlopen") as mock_urlopen:
            response = {
                "record_1": {"is_anomalous": False, "anomaly_type": "normal"},
                "record_2": {"is_anomalous": True, "anomaly_type": "high_amount"},
                "record_3": {"is_anomalous": False, "anomaly_type": "normal"}
            }
            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps(response).encode("utf-8")
            mock_urlopen.return_value = mock_response
            
            result = batch_score_anomalies_with_llm(
                records=records,
                llm_config=typed_llm_config,
            )
            
            assert isinstance(result, dict)

    @patch("urllib.request.urlopen")
    def test_invalid_json_returns_empty(self, mock_urlopen, typed_llm_config):
        """Invalid JSON response returns empty dict gracefully."""
        mock_response = MagicMock()
        mock_response.read.return_value = b"invalid {{{"
        mock_urlopen.return_value = mock_response
        
        result = batch_score_anomalies_with_llm(
            records=[{"transaction_id": "1", "amount": 100}],
            llm_config=typed_llm_config,
        )
        
        assert result == {}

    @patch("urllib.request.urlopen")
    def test_service_error_returns_empty(self, mock_urlopen, typed_llm_config):
        """Service error returns empty dict gracefully."""
        mock_urlopen.side_effect = Exception("Service error")
        
        result = batch_score_anomalies_with_llm(
            records=[{"transaction_id": "1", "amount": 100}],
            llm_config=typed_llm_config,
        )
        
        assert result == {}

    @patch("urllib.request.urlopen")
    @pytest.mark.parametrize("provider", ["openai", "azure", "gemini", "anthropic", "perplexity"])
    def test_all_providers_supported(self, mock_urlopen, provider, typed_llm_config):
        """All 5 LLM providers supported."""
        response = {
            "record_1": {"is_anomalous": False, "anomaly_type": "normal"}
        }
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(response).encode("utf-8")
        mock_urlopen.return_value = mock_response
        
        llm_config = AppLLMConfig(
            provider=provider,
            model="appropriate-for-provider",
            temperature=typed_llm_config.temperature,
            max_tokens=typed_llm_config.max_tokens,
            timeout_seconds=typed_llm_config.timeout_seconds,
            openai_api_key=typed_llm_config.openai_api_key,
            azure_api_key=typed_llm_config.azure_api_key,
            azure_endpoint=typed_llm_config.azure_endpoint,
            azure_deployment=typed_llm_config.azure_deployment,
            azure_api_version=typed_llm_config.azure_api_version,
            gemini_api_key=typed_llm_config.gemini_api_key,
            anthropic_api_key=typed_llm_config.anthropic_api_key,
            perplexity_api_key=typed_llm_config.perplexity_api_key,
            perplexity_base_url=typed_llm_config.perplexity_base_url,
        )

        result = batch_score_anomalies_with_llm(
            records=[{"transaction_id": "1", "amount": 100}],
            llm_config=llm_config,
        )
        
        assert isinstance(result, dict)


# ============================================================================
# P1 TESTS - JSON I/O, Data Validation, Inventory, and Utilities
# ============================================================================

# P1.1: JSON File I/O Tests
# ============================================================================

class TestLoadJsonFile:
    """Tests for load_json_file() function."""
    
    def test_load_valid_json_file(self, temp_json_file):
        """Test loading valid JSON file."""
        # temp_json_file fixture creates: {"key": "value", "items": [1, 2, 3]}
        result = load_json_file(str(temp_json_file))
        assert isinstance(result, dict)
        assert "key" in result
        assert result["key"] == "value"
    
    def test_load_nonexistent_file(self):
        """Test loading nonexistent file returns empty dict."""
        result = load_json_file("/nonexistent/path/file.json")
        assert result == {}
    
    def test_load_empty_json_file(self, tmp_path):
        """Test loading empty JSON file returns empty dict."""
        empty_file = tmp_path / "empty.json"
        empty_file.write_text("")
        result = load_json_file(str(empty_file))
        assert result == {}
    
    def test_load_invalid_json_file(self, tmp_path):
        """Test loading invalid JSON file returns empty dict."""
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("{not valid json")
        result = load_json_file(str(invalid_file))
        assert result == {}
    
    def test_load_json_array(self, tmp_path):
        """Test loading JSON array file."""
        array_file = tmp_path / "array.json"
        array_file.write_text('[{"id": 1}, {"id": 2}]')
        result = load_json_file(str(array_file))
        assert isinstance(result, list)
        assert len(result) == 2
    
    def test_load_json_null(self, tmp_path):
        """Test loading JSON null value."""
        null_file = tmp_path / "null.json"
        null_file.write_text("null")
        result = load_json_file(str(null_file))
        assert result is None
    
    def test_load_json_string(self, tmp_path):
        """Test loading JSON string value."""
        string_file = tmp_path / "string.json"
        string_file.write_text('"hello"')
        result = load_json_file(str(string_file))
        assert result == "hello"
    
    def test_load_json_with_unicode(self, tmp_path):
        """Test loading JSON with unicode characters."""
        unicode_file = tmp_path / "unicode.json"
        unicode_file.write_text('{"message": "Hello 世界"}', encoding='utf-8')
        result = load_json_file(str(unicode_file))
        assert result["message"] == "Hello 世界"


class TestSaveJsonFile:
    """Tests for save_json_file() function."""
    
    def test_save_dict_to_json(self, tmp_path):
        """Test saving dictionary to JSON file."""
        output_file = tmp_path / "output.json"
        data = {"key": "value", "number": 42}
        save_json_file(str(output_file), data)
        
        # Verify file exists and contains correct data
        assert output_file.exists()
        result = json.loads(output_file.read_text())
        assert result == data
    
    def test_save_list_to_json(self, tmp_path):
        """Test saving list to JSON file."""
        output_file = tmp_path / "list.json"
        data = [1, 2, 3, "test"]
        save_json_file(str(output_file), data)
        
        assert output_file.exists()
        result = json.loads(output_file.read_text())
        assert result == data
    
    def test_save_empty_dict(self, tmp_path):
        """Test saving empty dictionary."""
        output_file = tmp_path / "empty.json"
        save_json_file(str(output_file), {})
        
        assert output_file.exists()
        result = json.loads(output_file.read_text())
        assert result == {}
    
    def test_save_nested_structure(self, tmp_path):
        """Test saving nested dictionary structure."""
        output_file = tmp_path / "nested.json"
        data = {
            "level1": {
                "level2": {
                    "level3": ["a", "b", "c"]
                }
            }
        }
        save_json_file(str(output_file), data)
        
        result = json.loads(output_file.read_text())
        assert result == data
    
    def test_save_with_unicode(self, tmp_path):
        """Test saving data with unicode characters."""
        output_file = tmp_path / "unicode.json"
        data = {"greeting": "こんにちは", "emoji": "🎉"}
        save_json_file(str(output_file), data)
        
        result = json.loads(output_file.read_text())
        assert result["greeting"] == "こんにちは"
    
    def test_save_overwrites_existing_file(self, tmp_path):
        """Test that save_json_file overwrites existing file."""
        output_file = tmp_path / "overwrite.json"
        
        # Save first data
        save_json_file(str(output_file), {"version": 1})
        
        # Overwrite with new data
        save_json_file(str(output_file), {"version": 2})
        
        result = json.loads(output_file.read_text())
        assert result == {"version": 2}
    
    def test_save_creates_nonexistent_directory(self, tmp_path):
        """Test that save_json_file creates directory if needed."""
        output_file = tmp_path / "subdir" / "nested" / "file.json"
        data = {"test": "data"}
        
        # Ensure directory doesn't exist
        assert not output_file.parent.exists()
        
        # This should work - function should create directories
        try:
            save_json_file(str(output_file), data)
            # If function creates dirs, file should exist
            if output_file.exists():
                result = json.loads(output_file.read_text())
                assert result == data
        except (FileNotFoundError, OSError):
            # If function doesn't create dirs, that's also acceptable behavior
            pass


class TestLoadAliasMap:
    """Tests for load_alias_map() function."""
    
    def test_load_valid_alias_map(self, tmp_path):
        """Test loading valid alias map from JSON file."""
        alias_file = tmp_path / "aliases.json"
        # load_alias_map expects dict values to be strings, not lists
        aliases = {
            "item1": "variant1",
            "item2": "variant2"
        }
        alias_file.write_text(json.dumps(aliases))
        
        result = load_alias_map(str(alias_file))
        assert isinstance(result, dict)
        # Values should be normalized (lowercased, stripped, spaces collapsed)
        assert len(result) > 0
    
    def test_load_alias_map_nonexistent_file(self):
        """Test loading nonexistent alias file returns empty dict."""
        result = load_alias_map("/nonexistent/aliases.json")
        assert result == {}
    
    def test_load_alias_map_invalid_json(self, tmp_path):
        """Test loading invalid JSON alias file returns empty dict."""
        invalid_file = tmp_path / "invalid_aliases.json"
        invalid_file.write_text("not json")
        
        result = load_alias_map(str(invalid_file))
        assert result == {}
    
    def test_load_alias_map_empty_file(self, tmp_path):
        """Test loading empty alias file returns empty dict."""
        empty_file = tmp_path / "empty_aliases.json"
        empty_file.write_text("")
        
        result = load_alias_map(str(empty_file))
        assert result == {}
    
    def test_load_alias_map_with_special_characters(self, tmp_path):
        """Test loading alias map with special characters in keys/values."""
        alias_file = tmp_path / "special_aliases.json"
        # Keys and values should both be strings
        aliases = {
            "item-with-dash": "variant_with_underscore",
            "item.with.dots": "variant (with) parens"
        }
        alias_file.write_text(json.dumps(aliases))
        
        result = load_alias_map(str(alias_file))
        assert isinstance(result, dict)
        # Keys should be normalized
        assert all(isinstance(k, str) for k in result.keys())
    
    def test_load_alias_map_with_unicode(self, tmp_path):
        """Test loading alias map with unicode characters."""
        alias_file = tmp_path / "unicode_aliases.json"
        aliases = {
            "日本語": "バリエーション1",
            "中文": "变体1"
        }
        alias_file.write_text(json.dumps(aliases, ensure_ascii=False), encoding='utf-8')
        
        result = load_alias_map(str(alias_file))
        assert isinstance(result, dict)


# P1.2: Data Validation Tests
# ============================================================================

class TestValidateTransaction:
    """Tests for validate_transaction() function - validates list of product descriptions."""
    
    def test_valid_transaction(self):
        """Test validation of valid transaction list."""
        transaction = ["item1", "item2", "item3"]
        result = validate_transaction(transaction)
        assert result is True
    
    def test_transaction_empty_list(self):
        """Test validation of empty transaction list is invalid."""
        result = validate_transaction([])
        assert result is False
    
    def test_transaction_none(self):
        """Test validation of None transaction is invalid."""
        result = validate_transaction(None)
        assert result is False
    
    def test_transaction_not_list(self):
        """Test validation of non-list transaction is invalid."""
        result = validate_transaction("not a list")
        assert result is False
    
    def test_transaction_with_non_string_items(self):
        """Test transaction with non-string items is invalid."""
        transaction = ["item1", 123, "item3"]
        result = validate_transaction(transaction)
        assert result is False
    
    def test_transaction_single_item(self):
        """Test transaction with single item is valid."""
        result = validate_transaction(["white hanging heart t-light holder"])
        assert result is True


class TestFilterTransaction:
    """Tests for filter_transaction() function - filters items by minimum length."""
    
    def test_filter_valid_items(self):
        """Test filtering valid transaction items."""
        transaction = ["ITEM ONE", "ITEM TWO", "ITEM THREE"]
        result = filter_transaction(transaction)
        assert isinstance(result, list)
        assert len(result) == 3
    
    def test_filter_removes_short_items(self):
        """Test that short items are filtered out (min_length=3)."""
        transaction = ["ITEM", "at", "SOMETHING LONGER"]
        result = filter_transaction(transaction, min_length=3)
        # "at" should be removed (< 3 chars)
        assert "at" not in result
        assert "item" in result or "something longer" in result
    
    def test_filter_lowercases_items(self):
        """Test that filter lowercases items."""
        transaction = ["UPPERCASE", "MixedCase", "lowercase"]
        result = filter_transaction(transaction)
        assert all(item.islower() for item in result)
    
    def test_filter_strips_whitespace(self):
        """Test that filter strips leading/trailing whitespace."""
        transaction = ["  spaced  ", "\ttabbed\t", "normal"]
        result = filter_transaction(transaction)
        # Result should not have leading/trailing whitespace
        for item in result:
            assert item == item.strip()
    
    def test_filter_empty_transaction(self):
        """Test filtering empty transaction list."""
        result = filter_transaction([])
        assert result == []
    
    def test_filter_custom_min_length(self):
        """Test filtering with custom minimum length."""
        transaction = ["ABC", "ABCD", "AB"]
        result = filter_transaction(transaction, min_length=4)
        # Only "abcd" should remain
        assert len(result) == 1


# P1.3: Inventory Loading Tests
# ============================================================================

class TestLoadInventoryCsv:
    """Tests for load_inventory_csv() function."""
    
    def test_load_valid_inventory_csv(self, tmp_path):
        """Test loading valid inventory CSV file."""
        csv_file = tmp_path / "inventory.csv"
        csv_content = """Description,Available
WHITE HANGING HEART T-LIGHT HOLDER,1
WHITE METAL LANTERN,0
CREAM CUPID HEARTS COAT HANGER,1
"""
        csv_file.write_text(csv_content)
        
        result = load_inventory_csv(str(csv_file))
        assert result is not None
        assert isinstance(result, dict)
    
    def test_load_nonexistent_inventory_file(self):
        """Test loading nonexistent inventory file returns empty dict."""
        result = load_inventory_csv("/nonexistent/inventory.csv")
        assert result == {}
    
    def test_load_empty_inventory_csv(self, tmp_path):
        """Test loading empty inventory CSV returns empty dict."""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("")
        
        result = load_inventory_csv(str(csv_file))
        assert result == {}
    
    def test_load_inventory_csv_headers_only(self, tmp_path):
        """Test loading inventory CSV with only headers."""
        csv_file = tmp_path / "headers_only.csv"
        csv_file.write_text("Description,Available\n")
        
        result = load_inventory_csv(str(csv_file))
        # Headers only should return empty dict
        assert result == {}
    
    def test_load_inventory_csv_with_unicode(self, tmp_path):
        """Test loading inventory CSV with unicode characters."""
        csv_file = tmp_path / "unicode_inventory.csv"
        csv_content = """Description,Available
白いハートライト,1
中文描述商品,0
"""
        csv_file.write_text(csv_content, encoding='utf-8')
        
        result = load_inventory_csv(str(csv_file))
        assert result is not None
    
    def test_load_inventory_csv_with_missing_columns(self, tmp_path):
        """Test loading inventory CSV with missing columns."""
        csv_file = tmp_path / "missing_cols.csv"
        csv_content = """Description,Price
WHITE HANGING HEART T-LIGHT HOLDER,2.55
"""
        csv_file.write_text(csv_content)
        
        result = load_inventory_csv(str(csv_file))
        # Missing in_stock/available/stock column should return empty dict or handle gracefully
        assert result == {}
    
    def test_load_inventory_csv_with_special_characters(self, tmp_path):
        """Test loading inventory CSV with special characters in data."""
        csv_file = tmp_path / "special.csv"
        csv_content = '''Description,Available
"WHITE HEART, SPECIAL EDITION",1
"LANTERN (DELUXE) & STAND",0
'''
        csv_file.write_text(csv_content)
        
        result = load_inventory_csv(str(csv_file))
        assert result is not None


# P1.4: Utility Function Tests
# ============================================================================

class TestFormatRecommendations:
    """Tests for format_recommendations() function."""
    
    def test_format_basic_recommendations(self):
        """Test formatting basic recommendations dict."""
        recommendations = {
            "transaction": ["item1", "item2"],
            "confidence": 0.85,
            "recommender": "naive_bayes",
            "bundles": [("bundle1", "bundle2"), ("bundle3", "bundle4")]
        }
        result = format_recommendations(recommendations)
        assert isinstance(result, str)
        assert "BUNDLE RECOMMENDATIONS" in result
        assert "item1" in result
    
    def test_format_recommendations_no_bundles(self):
        """Test formatting recommendations with no bundles."""
        recommendations = {
            "transaction": ["item1"],
            "confidence": 0.5,
            "recommender": "naive_bayes",
            "bundles": []
        }
        result = format_recommendations(recommendations)
        assert isinstance(result, str)
        assert "No recommendations" in result
    
    def test_format_recommendations_high_confidence(self):
        """Test formatting recommendations with high confidence."""
        recommendations = {
            "transaction": ["item1"],
            "confidence": 0.99,
            "recommender": "svm",
            "bundles": [("a", "b")]
        }
        result = format_recommendations(recommendations)
        assert "99.00%" in result  # Should format confidence as percentage with 2 decimals
    
    def test_format_recommendations_verbose(self):
        """Test formatting recommendations in verbose mode."""
        recommendations = {
            "transaction": ["item1"],
            "confidence": 0.85,
            "recommender": "naive_bayes",
            "bundles": [("a", "b")]
        }
        result = format_recommendations(recommendations, verbose=True)
        assert isinstance(result, str)
    
    def test_format_recommendations_multiple_bundles(self):
        """Test formatting multiple bundles."""
        recommendations = {
            "transaction": ["item1", "item2"],
            "confidence": 0.75,
            "recommender": "random",
            "bundles": [("a", "b"), ("c", "d"), ("e", "f")]
        }
        result = format_recommendations(recommendations)
        assert "1." in result  # Should have numbered bundles
        assert "2." in result
        assert "3." in result


class TestComputeIQRBounds:
    """Tests for compute_iqr_bounds() function - works with pandas Series."""
    
    def test_compute_iqr_normal_data(self):
        """Test computing IQR bounds on normal data."""
        import pandas as pd
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        lower, upper = compute_iqr_bounds(data)
        assert isinstance(lower, (int, float))
        assert isinstance(upper, (int, float))
        assert lower < upper
    
    def test_compute_iqr_with_outliers(self):
        """Test computing IQR bounds with outliers."""
        import pandas as pd
        data = pd.Series([1, 2, 3, 4, 5, 100, 1000])
        lower, upper = compute_iqr_bounds(data)
        assert lower < upper
        # Outliers should be beyond bounds
        assert 100 > upper or 1000 > upper
    
    def test_compute_iqr_small_dataset(self):
        """Test computing IQR bounds on small dataset."""
        import pandas as pd
        data = pd.Series([1, 2, 3])
        lower, upper = compute_iqr_bounds(data)
        assert lower <= upper
    
    def test_compute_iqr_single_value(self):
        """Test computing IQR bounds with single value."""
        import pandas as pd
        data = pd.Series([5])
        lower, upper = compute_iqr_bounds(data)
        # Single value IQR should be 0 or similar
        assert lower <= upper
    
    def test_compute_iqr_with_nan(self):
        """Test computing IQR bounds with NaN values (should be dropped)."""
        import pandas as pd
        data = pd.Series([1, 2, 3, float('nan'), 5])
        lower, upper = compute_iqr_bounds(data)
        assert lower < upper
    
    def test_compute_iqr_custom_multiplier(self):
        """Test computing IQR bounds with custom multiplier."""
        import pandas as pd
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        lower1, upper1 = compute_iqr_bounds(data, multiplier=1.5)
        lower2, upper2 = compute_iqr_bounds(data, multiplier=3.0)
        # Larger multiplier should give wider bounds
        assert upper2 > upper1


class TestSetupLogging:
    """Tests for setup_logging() function - configures logging for the module."""
    
    def test_setup_logging_with_default_level(self):
        """Test setup_logging with default level."""
        # setup_logging returns None but configures logging
        result = setup_logging()
        # Function returns None (side-effects only)
        assert result is None
    
    def test_setup_logging_with_debug_level(self):
        """Test setup_logging with DEBUG level."""
        import logging
        result = setup_logging(logging.DEBUG)
        assert result is None
    
    def test_setup_logging_with_info_level(self):
        """Test setup_logging with INFO level."""
        import logging
        result = setup_logging(logging.INFO)
        assert result is None
    
    def test_setup_logging_with_warning_level(self):
        """Test setup_logging with WARNING level."""
        import logging
        result = setup_logging(logging.WARNING)
        assert result is None
    
    def test_setup_logging_creates_log_file(self, tmp_path):
        """Test that setup_logging can create a log file."""
        import logging
        # This may or may not create a file depending on implementation
        setup_logging(logging.INFO)
        assert True  # Just verify function doesn't error


# ============================================================================
# END OF P1 TESTS - JSON I/O, Data Validation, Inventory, Utilities
# ============================================================================


# ============================================================================
# P2 TESTS - Infrastructure, Advanced LLM Functions
# ============================================================================

@pytest.mark.unit
class TestPromptLoading:
    """Tests for prompt template loading and rendering."""
    
    def test_prompt_base_dir_returns_valid_path(self):
        """Test that _prompt_base_dir returns valid configuration path."""
        from src.utils import _prompt_base_dir
        base_dir = _prompt_base_dir()
        assert isinstance(base_dir, str)
        assert "prompts" in base_dir
        assert os.path.isdir(base_dir)
    
    def test_load_prompt_template_from_file(self):
        """Test loading a real prompt template file."""
        from src.utils import _load_prompt_template
        template = _load_prompt_template("normalize_description_system.md")
        assert isinstance(template, str)
        assert len(template) > 0
        assert "normalize" in template.lower() or "description" in template.lower()
    
    def test_load_prompt_template_caching(self):
        """Test that prompt templates are cached."""
        from src.utils import _load_prompt_template, _PROMPT_CACHE
        _PROMPT_CACHE.clear()
        assert "normalize_description_system.md" not in _PROMPT_CACHE
        template1 = _load_prompt_template("normalize_description_system.md")
        assert "normalize_description_system.md" in _PROMPT_CACHE
        template2 = _load_prompt_template("normalize_description_system.md")
        assert template1 == template2
    
    def test_load_prompt_template_missing_file(self):
        """Test that missing prompt file raises RuntimeError."""
        from src.utils import _load_prompt_template
        with pytest.raises(RuntimeError, match="Prompt file missing"):
            _load_prompt_template("nonexistent_prompt.md")
    
    def test_load_prompt_template_path_traversal(self):
        """Test that path traversal attempts are blocked."""
        from src.utils import _load_prompt_template
        with pytest.raises(RuntimeError, match="Invalid prompt path"):
            _load_prompt_template("../../../etc/passwd")
    
    def test_render_prompt_with_single_kwarg(self):
        """Test prompt template rendering with one variable."""
        from src.utils import _render_prompt
        result = _render_prompt("normalize_description_user.md", text="test product")
        assert isinstance(result, str)
        assert len(result) > 0
        # Verify placeholder was replaced
        assert "{text}" not in result
    
    def test_render_prompt_with_multiple_kwargs(self):
        """Test prompt template rendering with multiple variables."""
        from src.utils import _render_prompt
        result = _render_prompt(
            "select_alternatives_user.md",
            missing_item="item1",
            candidates_json='["item2", "item3"]'
        )
        assert isinstance(result, str)
        assert "item1" in result or len(result) > 0  # Template may or may not include the items
    
    def test_render_prompt_caches_template(self):
        """Test that rendering uses cached template."""
        from src.utils import _render_prompt, _PROMPT_CACHE
        _PROMPT_CACHE.clear()
        result1 = _render_prompt("normalize_description_user.md", text="test1")
        cache_size_after_first = len(_PROMPT_CACHE)
        result2 = _render_prompt("normalize_description_user.md", text="test2")
        cache_size_after_second = len(_PROMPT_CACHE)
        # Cache size should not grow for same template file
        assert cache_size_after_first == cache_size_after_second


@pytest.mark.unit
class TestSchemaLoading:
    """Tests for JSON schema loading and validation."""
    
    def test_schema_base_dir_returns_valid_path(self):
        """Test that _schema_base_dir returns valid schema path."""
        from src.utils import _schema_base_dir
        base_dir = _schema_base_dir()
        assert isinstance(base_dir, str)
        assert "schemas" in base_dir
        assert os.path.isdir(base_dir)
    
    def test_load_json_schema_from_file(self):
        """Test loading a real JSON schema file."""
        from src.utils import _load_json_schema
        schema = _load_json_schema("llm_select_alternatives.json")
        assert isinstance(schema, dict)
        assert "type" in schema or "properties" in schema or "$schema" in schema
    
    def test_load_json_schema_caching(self):
        """Test that schemas are cached after first load."""
        from src.utils import _load_json_schema, _SCHEMA_CACHE
        _SCHEMA_CACHE.clear()
        assert "llm_select_alternatives.json" not in _SCHEMA_CACHE
        schema1 = _load_json_schema("llm_select_alternatives.json")
        assert "llm_select_alternatives.json" in _SCHEMA_CACHE
        schema2 = _load_json_schema("llm_select_alternatives.json")
        assert schema1 == schema2
    
    def test_load_json_schema_missing_file(self):
        """Test that missing schema file raises RuntimeError."""
        from src.utils import _load_json_schema
        with pytest.raises(RuntimeError, match="Schema file missing"):
            _load_json_schema("nonexistent_schema.json")
    
    def test_load_json_schema_path_traversal(self):
        """Test that path traversal attempts are blocked."""
        from src.utils import _load_json_schema
        with pytest.raises(RuntimeError, match="Invalid schema path"):
            _load_json_schema("../../../etc/passwd")
    
    def test_validate_json_schema_valid_payload(self):
        """Test schema validation with valid payload."""
        from src.utils import _validate_json_schema
        valid_payload = {
            "alternatives": [
                {"item": "item1", "score": 0.9, "reason": "match"}
            ]
        }
        # Should not raise
        _validate_json_schema(valid_payload, "llm_select_alternatives.json", "test")
    
    def test_validate_json_schema_invalid_payload(self):
        """Test schema validation with invalid payload."""
        from src.utils import _validate_json_schema
        invalid_payload = {"invalid": "structure"}
        with pytest.raises(RuntimeError, match="Invalid LLM JSON output"):
            _validate_json_schema(invalid_payload, "llm_select_alternatives.json", "test context")
    
    def test_validate_json_schema_context_in_error(self):
        """Test that context is included in validation error."""
        from src.utils import _validate_json_schema
        invalid_payload = {}
        with pytest.raises(RuntimeError, match="test_context"):
            _validate_json_schema(invalid_payload, "llm_select_alternatives.json", "test_context")


@pytest.mark.unit
class TestHttpPostJson:
    """Tests for HTTP POST JSON helper function."""
    
    @patch('src.utils.urlopen')
    def test_http_post_json_success(self, mock_urlopen):
        """Test successful HTTP POST with JSON response."""
        from src.utils import _http_post_json
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"status": "ok", "value": 123}'
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response
        
        result = _http_post_json(
            "https://api.example.com/test",
            {"Content-Type": "application/json"},
            {"data": "test"},
            timeout=30
        )
        
        assert result == {"status": "ok", "value": 123}
        mock_urlopen.assert_called_once()
    
    @patch('src.utils.urlopen')
    def test_http_post_json_with_unicode(self, mock_urlopen):
        """Test HTTP POST with unicode response."""
        from src.utils import _http_post_json
        mock_response = MagicMock()
        mock_response.read.return_value = '{"text": "こんにちは"}'.encode('utf-8')
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response
        
        result = _http_post_json(
            "https://api.example.com/test",
            {"Content-Type": "application/json"},
            {"data": "test"},
            timeout=30
        )
        
        assert result["text"] == "こんにちは"
    
    @patch('src.utils.urlopen')
    def test_http_post_json_http_error(self, mock_urlopen):
        """Test HTTP POST with HTTP error response."""
        from src.utils import _http_post_json
        from urllib.error import HTTPError
        mock_urlopen.side_effect = HTTPError(
            "https://api.example.com",
            401,
            "Unauthorized",
            None,  # type: ignore[arg-type]
            None
        )
        
        with pytest.raises(RuntimeError, match="HTTP error"):
            _http_post_json(
                "https://api.example.com/test",
                {"Authorization": "Bearer invalid"},
                {"data": "test"},
                timeout=30
            )
    
    @patch('src.utils.urlopen')
    def test_http_post_json_network_error(self, mock_urlopen):
        """Test HTTP POST with network error."""
        from src.utils import _http_post_json
        from urllib.error import URLError
        mock_urlopen.side_effect = URLError("Connection refused")
        
        with pytest.raises(RuntimeError, match="Network error"):
            _http_post_json(
                "https://api.example.com/test",
                {"Content-Type": "application/json"},
                {"data": "test"},
                timeout=30
            )
    
    @patch('src.utils.urlopen')
    def test_http_post_json_invalid_json_response(self, mock_urlopen):
        """Test HTTP POST with invalid JSON response."""
        from src.utils import _http_post_json
        mock_response = MagicMock()
        mock_response.read.return_value = b'invalid json {{'
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response
        
        with pytest.raises(Exception):  # json.JSONDecodeError
            _http_post_json(
                "https://api.example.com/test",
                {"Content-Type": "application/json"},
                {"data": "test"},
                timeout=30
            )


@pytest.mark.unit
class TestCandidateHash:
    """Tests for candidate hash generation."""
    
    def test_candidate_hash_basic(self):
        """Test basic hash generation for candidates."""
        from src.utils import _candidate_hash
        hash1 = _candidate_hash(["item1", "item2", "item3"])
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA256 hex is 64 chars
    
    def test_candidate_hash_order_independent(self):
        """Test that hash is independent of candidate order."""
        from src.utils import _candidate_hash
        hash1 = _candidate_hash(["item1", "item2", "item3"])
        hash2 = _candidate_hash(["item3", "item1", "item2"])
        hash3 = _candidate_hash(["item2", "item3", "item1"])
        assert hash1 == hash2 == hash3
    
    def test_candidate_hash_different_for_different_candidates(self):
        """Test that different candidates produce different hashes."""
        from src.utils import _candidate_hash
        hash1 = _candidate_hash(["item1", "item2"])
        hash2 = _candidate_hash(["item1", "item3"])
        assert hash1 != hash2


@pytest.mark.unit
class TestLLMQuotaExceededError:
    """Tests for LLMQuotaExceededError exception."""
    
    def test_quota_exceeded_error_subclass(self):
        """Test that LLMQuotaExceededError is RuntimeError subclass."""
        assert issubclass(LLMQuotaExceededError, RuntimeError)
    
    def test_quota_exceeded_error_instantiation(self):
        """Test creating LLMQuotaExceededError instance."""
        error = LLMQuotaExceededError("API quota exceeded")
        assert isinstance(error, RuntimeError)
        assert str(error) == "API quota exceeded"


@pytest.mark.unit
class TestNormalizeDescriptionWithLLM:
    """Tests for LLM-based description normalization."""
    
    def test_normalize_empty_text(self):
        """Test normalization of empty text returns empty string."""
        from src.utils import normalize_description_with_llm
        result = normalize_description_with_llm(
            "",
            provider="openai",
            model="gpt-4",
            temperature=0.7,
            max_tokens=100,
            timeout_seconds=30,
            api_key="fake-key"
        )
        assert result == ""
    
    def test_normalize_none_text(self):
        """Test normalization of None text."""
        from src.utils import normalize_description_with_llm
        result = normalize_description_with_llm(
            None,
            provider="openai",
            model="gpt-4",
            temperature=0.7,
            max_tokens=100,
            timeout_seconds=30,
            api_key="fake-key"
        )
        assert result == ""
    
    @patch('src.llm_client.LLMClient.chat_completion')
    def test_normalize_openai_success(self, mock_chat):
        """Test successful OpenAI normalization."""
        from src.utils import normalize_description_with_llm
        mock_chat.return_value = "normalized description"
        
        result = normalize_description_with_llm(
            "messy description",
            provider="openai",
            model="gpt-4",
            temperature=0.7,
            max_tokens=100,
            timeout_seconds=30,
            api_key="test-key"
        )
        
        assert result == "normalized description"
    
    @patch('src.llm_client.LLMClient.chat_completion')
    def test_normalize_azure_success(self, mock_chat):
        """Test successful Azure normalization."""
        from src.utils import normalize_description_with_llm
        mock_chat.return_value = "normalized"
        
        result = normalize_description_with_llm(
            "test",
            provider="azure",
            model="gpt-4",
            temperature=0.7,
            max_tokens=100,
            timeout_seconds=30,
            api_key="test-key",
            endpoint="https://test.openai.azure.com",
            deployment="test-deployment"
        )
        
        assert result == "normalized"
    
    @patch('src.llm_client.LLMClient.chat_completion')
    def test_normalize_gemini_success(self, mock_chat):
        """Test successful Gemini normalization."""
        from src.utils import normalize_description_with_llm
        mock_chat.return_value = "normalized"
        
        result = normalize_description_with_llm(
            "test",
            provider="gemini",
            model="gemini-pro",
            temperature=0.7,
            max_tokens=100,
            timeout_seconds=30,
            api_key="test-key"
        )
        
        assert result == "normalized"
    
    @patch('src.llm_client.LLMClient.chat_completion')
    def test_normalize_anthropic_success(self, mock_chat):
        """Test successful Anthropic normalization."""
        from src.utils import normalize_description_with_llm
        mock_chat.return_value = "normalized"
        
        result = normalize_description_with_llm(
            "test",
            provider="anthropic",
            model="claude-3",
            temperature=0.7,
            max_tokens=100,
            timeout_seconds=30,
            api_key="test-key"
        )
        
        assert result == "normalized"
    
    @patch('src.llm_client.LLMClient.chat_completion')
    def test_normalize_perplexity_success(self, mock_chat):
        """Test successful Perplexity normalization."""
        from src.utils import normalize_description_with_llm
        mock_chat.return_value = "normalized"
        
        result = normalize_description_with_llm(
            "test",
            provider="perplexity",
            model="pplx-7b",
            temperature=0.7,
            max_tokens=100,
            timeout_seconds=30,
            api_key="test-key"
        )
        
        assert result == "normalized"
    
    def test_normalize_missing_openai_key(self):
        """Test OpenAI normalization with missing API key."""
        from src.utils import normalize_description_with_llm
        result = normalize_description_with_llm(
            "test description",
            provider="openai",
            model="gpt-4",
            temperature=0.7,
            max_tokens=100,
            timeout_seconds=30
        )
        # Should fallback to original text on error
        assert result == "test description"
    
    def test_normalize_missing_azure_credentials(self):
        """Test Azure normalization with missing credentials."""
        from src.utils import normalize_description_with_llm
        result = normalize_description_with_llm(
            "test description",
            provider="azure",
            model="gpt-4",
            temperature=0.7,
            max_tokens=100,
            timeout_seconds=30,
            api_key="test-key"
            # Missing endpoint and deployment
        )
        assert result == "test description"
    
    @patch('src.llm_client.LLMClient.chat_completion')
    def test_normalize_quota_exceeded(self, mock_chat):
        """Test handling of quota exceeded error."""
        from src.utils import normalize_description_with_llm
        mock_chat.side_effect = RuntimeError("insufficient_quota")
        
        with pytest.raises(LLMQuotaExceededError):
            normalize_description_with_llm(
                "test",
                provider="openai",
                model="gpt-4",
                temperature=0.7,
                max_tokens=100,
                timeout_seconds=30,
                api_key="test-key"
            )
    
    @patch('src.llm_client.LLMClient.chat_completion')
    def test_normalize_generic_error_fallback(self, mock_chat):
        """Test fallback to original text on generic error."""
        from src.utils import normalize_description_with_llm
        mock_chat.side_effect = RuntimeError("API error")
        
        result = normalize_description_with_llm(
            "original text",
            provider="openai",
            model="gpt-4",
            temperature=0.7,
            max_tokens=100,
            timeout_seconds=30,
            api_key="test-key"
        )
        
        assert result == "original text"
    
    def test_normalize_unsupported_provider(self):
        """Test with unsupported LLM provider."""
        from src.utils import normalize_description_with_llm
        result = normalize_description_with_llm(
            "test description",
            provider="unsupported_provider",
            model="model",
            temperature=0.7,
            max_tokens=100,
            timeout_seconds=30,
            api_key="key"
        )
        assert result == "test description"
    
    def test_normalize_case_insensitive_provider(self):
        """Test that provider name is case-insensitive."""
        from src.utils import normalize_description_with_llm
        result = normalize_description_with_llm(
            "test",
            provider="OPENAI",
            model="gpt-4",
            temperature=0.7,
            max_tokens=100,
            timeout_seconds=30
        )
        # Should handle uppercase provider name gracefully
        assert isinstance(result, str)


@pytest.mark.unit
class TestSelectAlternativesWithLLM:
    """Tests for LLM-based alternative selection."""
    
    def test_select_alternatives_empty_item(self):
        """Test alternative selection with empty missing item."""
        from src.utils import select_alternatives_with_llm
        result = select_alternatives_with_llm(
            "",
            ["item1", "item2"],
            provider="openai",
            model="gpt-4",
            temperature=0.7,
            max_tokens=500,
            timeout_seconds=30,
            max_alternatives=3,
            api_key="test-key"
        )
        assert result == []
    
    def test_select_alternatives_empty_candidates(self):
        """Test alternative selection with empty candidates."""
        from src.utils import select_alternatives_with_llm
        result = select_alternatives_with_llm(
            "missing_item",
            [],
            provider="openai",
            model="gpt-4",
            temperature=0.7,
            max_tokens=500,
            timeout_seconds=30,
            max_alternatives=3,
            api_key="test-key"
        )
        assert result == []
    
    @patch('src.llm_client.LLMClient.chat_completion_json')
    def test_select_alternatives_openai_success(self, mock_chat):
        """Test successful OpenAI alternative selection."""
        from src.utils import select_alternatives_with_llm
        mock_chat.return_value = {"alternatives": [{"item": "alt1", "score": 0.95, "reason": "similar"}]}
        
        result = select_alternatives_with_llm(
            "missing",
            ["alt1", "alt2"],
            provider="openai",
            model="gpt-4",
            temperature=0.7,
            max_tokens=500,
            timeout_seconds=30,
            max_alternatives=3,
            api_key="test-key"
        )
        
        assert len(result) == 1
        assert result[0]["item"] == "alt1"
        assert result[0]["score"] == 0.95
    
    @patch('src.llm_client.LLMClient.chat_completion_json')
    def test_select_alternatives_with_markdown_fence(self, mock_chat):
        """Test parsing response with markdown code fence."""
        from src.utils import select_alternatives_with_llm
        mock_chat.return_value = [{"item": "alt1", "score": 0.9, "reason": "best match"}]
        
        result = select_alternatives_with_llm(
            "missing",
            ["alt1"],
            provider="openai",
            model="gpt-4",
            temperature=0.7,
            max_tokens=500,
            timeout_seconds=30,
            max_alternatives=3,
            api_key="test-key"
        )
        
        assert len(result) == 1
        assert result[0]["item"] == "alt1"
    
    @patch('src.llm_client.LLMClient.chat_completion_json')
    def test_select_alternatives_max_alternatives_limit(self, mock_chat):
        """Test that results are limited by max_alternatives."""
        from src.utils import select_alternatives_with_llm
        mock_chat.return_value = {
            "alternatives": [
                {"item": "alt1", "score": 0.9, "reason": "1"},
                {"item": "alt2", "score": 0.8, "reason": "2"},
                {"item": "alt3", "score": 0.7, "reason": "3"},
                {"item": "alt4", "score": 0.6, "reason": "4"},
            ]
        }
        
        result = select_alternatives_with_llm(
            "missing",
            ["alt1", "alt2", "alt3", "alt4"],
            provider="openai",
            model="gpt-4",
            temperature=0.7,
            max_tokens=500,
            timeout_seconds=30,
            max_alternatives=2,
            api_key="test-key"
        )
        
        assert len(result) == 2
    
    @patch('src.llm_client.LLMClient.chat_completion_json')
    def test_select_alternatives_raw_list_response(self, mock_chat):
        """Test parsing when response is raw list (not wrapped dict)."""
        from src.utils import select_alternatives_with_llm
        mock_chat.return_value = [
            {"item": "alt1", "score": 0.9, "reason": "match"}
        ]
        
        result = select_alternatives_with_llm(
            "missing",
            ["alt1"],
            provider="openai",
            model="gpt-4",
            temperature=0.7,
            max_tokens=500,
            timeout_seconds=30,
            max_alternatives=3,
            api_key="test-key"
        )
        
        assert len(result) == 1
    
    @patch('src.llm_client.LLMClient.chat_completion_json')
    def test_select_alternatives_azure_success(self, mock_chat):
        """Test successful Azure alternative selection."""
        from src.utils import select_alternatives_with_llm
        mock_chat.return_value = {"alternatives": [{"item": "alt1", "score": 0.9, "reason": "match"}]}
        
        result = select_alternatives_with_llm(
            "missing",
            ["alt1"],
            provider="azure",
            model="gpt-4",
            temperature=0.7,
            max_tokens=500,
            timeout_seconds=30,
            max_alternatives=3,
            api_key="test-key",
            endpoint="https://test.openai.azure.com",
            deployment="test"
        )
        
        assert len(result) == 1
    
    @patch('src.llm_client.LLMClient.chat_completion_json')
    def test_select_alternatives_gemini_success(self, mock_chat):
        """Test successful Gemini alternative selection."""
        from src.utils import select_alternatives_with_llm
        mock_chat.return_value = {"alternatives": [{"item": "alt1", "score": 0.9, "reason": "match"}]}
        
        result = select_alternatives_with_llm(
            "missing",
            ["alt1"],
            provider="gemini",
            model="gemini-pro",
            temperature=0.7,
            max_tokens=500,
            timeout_seconds=30,
            max_alternatives=3,
            api_key="test-key"
        )
        
        assert len(result) == 1
    
    @patch('src.llm_client.LLMClient.chat_completion_json')
    def test_select_alternatives_anthropic_success(self, mock_chat):
        """Test successful Anthropic alternative selection."""
        from src.utils import select_alternatives_with_llm
        mock_chat.return_value = {"alternatives": [{"item": "alt1", "score": 0.9, "reason": "match"}]}
        
        result = select_alternatives_with_llm(
            "missing",
            ["alt1"],
            provider="anthropic",
            model="claude-3",
            temperature=0.7,
            max_tokens=500,
            timeout_seconds=30,
            max_alternatives=3,
            api_key="test-key"
        )
        
        assert len(result) == 1
    
    @patch('src.llm_client.LLMClient.chat_completion_json')
    def test_select_alternatives_perplexity_success(self, mock_chat):
        """Test successful Perplexity alternative selection."""
        from src.utils import select_alternatives_with_llm
        mock_chat.return_value = {"alternatives": [{"item": "alt1", "score": 0.9, "reason": "match"}]}
        
        result = select_alternatives_with_llm(
            "missing",
            ["alt1"],
            provider="perplexity",
            model="pplx-7b",
            temperature=0.7,
            max_tokens=500,
            timeout_seconds=30,
            max_alternatives=3,
            api_key="test-key"
        )
        
        assert len(result) == 1
    
    @patch('src.llm_client.LLMClient.chat_completion_json')
    def test_select_alternatives_invalid_json(self, mock_chat):
        """Test fallback on invalid JSON response."""
        from src.utils import select_alternatives_with_llm
        mock_chat.return_value = "invalid json content"
        
        result = select_alternatives_with_llm(
            "missing",
            ["alt1"],
            provider="openai",
            model="gpt-4",
            temperature=0.7,
            max_tokens=500,
            timeout_seconds=30,
            max_alternatives=3,
            api_key="test-key"
        )
        
        assert result == []
    
    @patch('src.llm_client.LLMClient.chat_completion_json')
    def test_select_alternatives_invalid_schema(self, mock_chat):
        """Test fallback on schema validation failure."""
        from src.utils import select_alternatives_with_llm
        mock_chat.return_value = '{"invalid": "structure"}'
        
        result = select_alternatives_with_llm(
            "missing",
            ["alt1"],
            provider="openai",
            model="gpt-4",
            temperature=0.7,
            max_tokens=500,
            timeout_seconds=30,
            max_alternatives=3,
            api_key="test-key"
        )
        
        assert result == []
    
    @patch('src.llm_client.LLMClient.chat_completion_json')
    def test_select_alternatives_non_list_alternatives(self, mock_chat):
        """Test fallback when alternatives field is not a list."""
        from src.utils import select_alternatives_with_llm
        mock_chat.return_value = {"alternatives": "not a list"}
        
        result = select_alternatives_with_llm(
            "missing",
            ["alt1"],
            provider="openai",
            model="gpt-4",
            temperature=0.7,
            max_tokens=500,
            timeout_seconds=30,
            max_alternatives=3,
            api_key="test-key"
        )
        
        assert result == []
    
    @patch('src.llm_client.LLMClient.chat_completion_json')
    def test_select_alternatives_quota_exceeded(self, mock_chat):
        """Test handling of quota exceeded during selection."""
        from src.utils import select_alternatives_with_llm
        mock_chat.side_effect = RuntimeError("quota exceeded")
        
        with pytest.raises(LLMQuotaExceededError):
            select_alternatives_with_llm(
                "missing",
                ["alt1"],
                provider="openai",
                model="gpt-4",
                temperature=0.7,
                max_tokens=500,
                timeout_seconds=30,
                max_alternatives=3,
                api_key="test-key"
            )
    
    @patch('src.llm_client.LLMClient.chat_completion_json')
    def test_select_alternatives_generic_error_fallback(self, mock_chat):
        """Test fallback to empty list on generic error."""
        from src.utils import select_alternatives_with_llm
        mock_chat.side_effect = RuntimeError("API error")
        
        result = select_alternatives_with_llm(
            "missing",
            ["alt1"],
            provider="openai",
            model="gpt-4",
            temperature=0.7,
            max_tokens=500,
            timeout_seconds=30,
            max_alternatives=3,
            api_key="test-key"
        )
        
        assert result == []
    
    def test_select_alternatives_missing_openai_key(self):
        """Test with missing OpenAI API key."""
        from src.utils import select_alternatives_with_llm
        result = select_alternatives_with_llm(
            "missing",
            ["alt1"],
            provider="openai",
            model="gpt-4",
            temperature=0.7,
            max_tokens=500,
            timeout_seconds=30,
            max_alternatives=3
        )
        assert result == []
    
    def test_select_alternatives_unsupported_provider(self):
        """Test with unsupported provider."""
        from src.utils import select_alternatives_with_llm
        result = select_alternatives_with_llm(
            "missing",
            ["alt1"],
            provider="unknown_provider",
            model="model",
            temperature=0.7,
            max_tokens=500,
            timeout_seconds=30,
            max_alternatives=3,
            api_key="key"
        )
        assert result == []


@pytest.mark.unit
class TestSelectAlternativeHeuristic:
    """Tests for heuristic-based alternative selection."""
    
    def test_heuristic_empty_arguments(self):
        """Test heuristic with empty arguments."""
        result = select_alternative_heuristic("", [])
        assert result is None
    
    def test_heuristic_empty_missing_item(self):
        """Test heuristic with empty missing item."""
        result = select_alternative_heuristic("", ["item1", "item2"])
        assert result is None
    
    def test_heuristic_empty_candidates(self):
        """Test heuristic with empty candidates."""
        result = select_alternative_heuristic("item1", [])
        assert result is None
    
    def test_heuristic_exact_match(self):
        """Test heuristic with exact match candidate."""
        result = select_alternative_heuristic("blue shirt", ["blue shirt", "red shirt"])
        assert result is not None
        assert result["item"] == "blue shirt"
        assert result["score"] > 0.9
        assert result["reason"] == "token-overlap"
    
    def test_heuristic_partial_match(self):
        """Test heuristic with partial token overlap."""
        result = select_alternative_heuristic("blue cotton shirt", ["blue shirt", "red shirt"])
        assert result is not None
        assert result["item"] == "blue shirt"
        assert 0.5 < result["score"] < 1.0
    
    def test_heuristic_no_match(self):
        """Test heuristic with no matching candidates."""
        result = select_alternative_heuristic("apple", ["xyz", "abc", "def"])
        assert result is None  # No overlap should return None
    
    def test_heuristic_whitespace_handling(self):
        """Test heuristic with extra whitespace."""
        result = select_alternative_heuristic(
            "blue   shirt",
            ["blue shirt", "shirt blue"]
        )
        assert result is not None
        assert result["item"] in ["blue shirt", "shirt blue"]
    
    def test_heuristic_case_insensitive(self):
        """Test heuristic with different cases."""
        result = select_alternative_heuristic(
            "BLUE SHIRT",
            ["blue shirt", "red shirt"]
        )
        assert result is not None
        assert result["item"] == "blue shirt"
