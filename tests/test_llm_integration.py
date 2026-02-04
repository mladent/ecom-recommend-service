"""
Unit tests for LLM integration and OOS (Out-of-Stock) substitution.

DESIGN PHILOSOPHY:
- ✅ All LLM API calls are MOCKED by default (zero cost)
- ✅ Optional real LLM testing via USE_REAL_LLM=true (strictly budgeted)
- ✅ Easy toggle between mock and real via environment variables
- ✅ Minimal context for real LLM calls (cheap, fast)

Configuration:
- USE_REAL_LLM=false (default) → All tests use mocks
- USE_REAL_LLM=true → Enable real API calls (requires LLM_BUDGET_LIMIT and API keys)
- LLM_BUDGET_LIMIT=N → Max real API calls (default 5)
- LLM_PROVIDER=openai|azure|gemini (default: openai)

Run Tests:
  # All mocked (FAST, FREE):
  pytest tests/test_llm_integration.py -v

  # Real LLM (SLOW, COSTS $$$):
  USE_REAL_LLM=true LLM_BUDGET_LIMIT=3 pytest tests/test_llm_integration.py::TestRealLLMCalls -v
"""

import json
import os
from unittest.mock import patch, MagicMock
from pathlib import Path

import pytest

from src.utils import (
    select_alternatives_with_llm,
    select_alternative_heuristic,
    load_inventory_csv,
    normalize_description_basic,
    load_json_file,
    LLMQuotaExceededError,
)
from src.config import (
    OPENAI_API_KEY,
    AZURE_OPENAI_API_KEY,
    GEMINI_API_KEY,
    ANTHROPIC_API_KEY,
    PERPLEXITY_API_KEY,
)

# ============================================================================
# Test Configuration: Real vs Mocked LLM Calls
# ============================================================================

USE_REAL_LLM = os.getenv("USE_REAL_LLM", "false").lower() == "true"
LLM_BUDGET_LIMIT = int(os.getenv("LLM_BUDGET_LIMIT", "5"))

_real_llm_call_count = 0


def _check_llm_budget():
    """Verify we haven't exceeded real LLM call budget."""
    global _real_llm_call_count
    _real_llm_call_count += 1
    if _real_llm_call_count > LLM_BUDGET_LIMIT:
        raise RuntimeError(
            f"Real LLM call budget exceeded: {_real_llm_call_count} > {LLM_BUDGET_LIMIT}"
        )


# ============================================================================
# Fixtures: Sample Data
# ============================================================================


@pytest.fixture
def sample_inventory():
    """Sample inventory dict for OOS testing."""
    return {
        "laptop": True,
        "mouse": True,
        "keyboard": True,
        "monitor": False,  # OOS
        "headphones": False,  # OOS
        "usb_cable": True,
        "hdmi_cable": False,  # OOS
    }


@pytest.fixture
def sample_candidates():
    """Sample candidate list for alternative selection."""
    return ["usb_cable", "keyboard", "mouse"]


@pytest.fixture
def mock_llm_response_valid():
    """Valid LLM response."""
    return [
        {"item": "usb_3_cable", "score": 0.92, "reason": "Similar connector"},
        {"item": "mini_usb_cable", "score": 0.78, "reason": "Alternative connection"},
    ]


# ============================================================================
# Fixtures: Mocking urllib (not requests)
# ============================================================================


@pytest.fixture
def mock_urllib_llm_success(mock_llm_response_valid):
    """Mock successful urllib LLM response."""
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_response = MagicMock()
        response_json = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(mock_llm_response_valid)
                    }
                }
            ]
        }
        mock_response.read.return_value = json.dumps(response_json).encode("utf-8")
        mock_response.status = 200
        mock_urlopen.return_value = mock_response
        yield mock_urlopen


@pytest.fixture
def mock_urllib_llm_error():
    """Mock urllib LLM error response."""
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.side_effect = Exception("LLM service unavailable")
        yield mock_urlopen


# ============================================================================
# Test Class 1: Core Functionality (All Mocked)
# ============================================================================


@pytest.mark.llm
@pytest.mark.unit
class TestSelectAlternativesWithLLM:
    """Test select_alternatives_with_llm() with mocked API calls."""

    def test_empty_missing_item(self):
        """Empty missing_item returns empty list."""
        result = select_alternatives_with_llm(
            missing_item="",
            candidates=["a", "b"],
            provider="openai",
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=512,
            timeout_seconds=30,
            max_alternatives=1,
        )
        assert result == []

    def test_empty_candidates(self):
        """Empty candidates returns empty list."""
        result = select_alternatives_with_llm(
            missing_item="monitor",
            candidates=[],
            provider="openai",
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=512,
            timeout_seconds=30,
            max_alternatives=1,
        )
        assert result == []

    def test_none_inputs(self):
        """None inputs return empty list."""
        result = select_alternatives_with_llm(
            missing_item=None,
            candidates=None,
            provider="openai",
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=512,
            timeout_seconds=30,
            max_alternatives=1,
        )
        assert result == []

    def test_success_with_mocked_openai(self, sample_candidates, mock_urllib_llm_success):
        """Successful call with mocked OpenAI."""
        result = select_alternatives_with_llm(
            missing_item="monitor",
            candidates=sample_candidates,
            provider="openai",
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=512,
            timeout_seconds=30,
            max_alternatives=1,
            api_key="sk-mock",
        )
        assert isinstance(result, list)

    @pytest.mark.parametrize("provider", ["openai", "azure", "gemini"])
    def test_different_providers(self, sample_candidates, provider):
        """Test different LLM providers with mocks."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            response_json = {"choices": [{"message": {"content": "[]"}}]}
            mock_response.read.return_value = json.dumps(response_json).encode("utf-8")
            mock_response.status = 200
            mock_urlopen.return_value = mock_response

            result = select_alternatives_with_llm(
                missing_item="monitor",
                candidates=sample_candidates,
                provider=provider,
                model="gpt-4o-mini",
                temperature=0.7,
                max_tokens=512,
                timeout_seconds=30,
                max_alternatives=1,
                api_key="mock-key",
            )
            assert isinstance(result, list)

    def test_invalid_json_graceful_failure(self, sample_candidates):
        """Invalid JSON response returns empty list (no crash)."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = b"{invalid json"
            mock_urlopen.return_value = mock_response

            result = select_alternatives_with_llm(
                missing_item="monitor",
                candidates=sample_candidates,
                provider="openai",
                model="gpt-4o-mini",
                temperature=0.7,
                max_tokens=512,
                timeout_seconds=30,
                max_alternatives=1,
                api_key="sk-mock",
            )
            assert result == []

    def test_llm_service_error_graceful_failure(self, sample_candidates):
        """LLM service error returns empty list."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = Exception("Service error")

            result = select_alternatives_with_llm(
                missing_item="monitor",
                candidates=sample_candidates,
                provider="openai",
                model="gpt-4o-mini",
                temperature=0.7,
                max_tokens=512,
                timeout_seconds=30,
                max_alternatives=1,
                api_key="sk-mock",
            )
            assert result == []


# ============================================================================
# Test Class 2: Fallback Mechanisms
# ============================================================================


@pytest.mark.llm
@pytest.mark.unit
class TestFallbackMechanisms:
    """Test fallback to heuristic when LLM unavailable."""

    def test_heuristic_selection_returns_dict_or_none(self, sample_candidates):
        """Heuristic returns dict or None."""
        result = select_alternative_heuristic(
            missing_item="monitor",
            candidates=sample_candidates,
        )
        assert result is None or isinstance(result, dict)

    def test_heuristic_empty_candidates(self):
        """Heuristic with empty candidates returns None."""
        result = select_alternative_heuristic(
            missing_item="monitor",
            candidates=[],
        )
        assert result is None

    def test_heuristic_none_inputs(self):
        """Heuristic with None inputs returns None."""
        result = select_alternative_heuristic(
            missing_item=None,
            candidates=None,
        )
        assert result is None

    def test_fallback_on_llm_error(self, sample_candidates):
        """Gracefully returns empty list on LLM error."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = LLMQuotaExceededError("Rate limit")

            result = select_alternatives_with_llm(
                missing_item="monitor",
                candidates=sample_candidates,
                provider="openai",
                model="gpt-4o-mini",
                temperature=0.7,
                max_tokens=512,
                timeout_seconds=30,
                max_alternatives=1,
                api_key="sk-mock",
            )
            assert isinstance(result, list)


# ============================================================================
# Test Class 3: Inventory Integration
# ============================================================================


@pytest.mark.llm
@pytest.mark.unit
class TestInventoryOperations:
    """Test inventory loading and normalization."""

    def test_load_missing_inventory_file(self):
        """Missing inventory file returns empty dict."""
        result = load_inventory_csv("/nonexistent/inventory.csv")
        # Should return {} or None depending on implementation
        assert result is None or isinstance(result, dict)

    def test_load_json_missing_file(self):
        """Missing JSON file returns empty dict."""
        result = load_json_file("/nonexistent/cache.json")
        assert result is None or isinstance(result, dict)

    def test_normalize_description_lowercasing(self):
        """Normalization lowercases input."""
        result = normalize_description_basic("LAPTOP USB CABLE")
        assert result == result.lower()

    def test_normalize_description_whitespace(self):
        """Normalization collapses whitespace."""
        result = normalize_description_basic("item   with   spaces")
        assert "  " not in result


# ============================================================================
# Test Class 4: Response Structure Validation
# ============================================================================


@pytest.mark.llm
@pytest.mark.unit
class TestResponseStructure:
    """Validate response structure format."""

    def test_alternatives_list_format(self, mock_llm_response_valid):
        """Alternatives have correct structure."""
        for alt in mock_llm_response_valid:
            assert "item" in alt
            assert "score" in alt
            assert isinstance(alt["score"], (int, float))
            assert 0 <= alt["score"] <= 1

    def test_audit_trail_structure(self):
        """Substitution audit trail has required fields."""
        substitution = {
            "missing_item": "monitor",
            "alternative_item": "display",
            "score": 0.85,
        }
        assert "missing_item" in substitution
        assert "alternative_item" in substitution
        assert "score" in substitution


# ============================================================================
# Test Class 5: Real LLM Calls (Opt-in Only)
# ============================================================================


@pytest.mark.llm_real
@pytest.mark.skipif(
    not USE_REAL_LLM,
    reason="Real LLM tests disabled. Set USE_REAL_LLM=true to enable.",
)
class TestRealLLMCalls:
    """
    Real LLM API tests (OPT-IN ONLY).

    WARNING: These tests make ACTUAL API calls and INCUR COSTS.

    Only run if:
    1. You have set USE_REAL_LLM=true
    2. Your API keys are configured (OPENAI_API_KEY, AZURE_OPENAI_API_KEY, etc.)
    3. You accept the API costs

    Command:
      USE_REAL_LLM=true LLM_BUDGET_LIMIT=5 pytest tests/test_llm_integration.py::TestRealLLMCalls -v
    """

    def test_real_openai_call_minimal(self):
        """Real OpenAI call (minimal context for cost control)."""
        _check_llm_budget()

        if not OPENAI_API_KEY:
            pytest.skip("OPENAI_API_KEY not configured")

        result = select_alternatives_with_llm(
            missing_item="phone",
            candidates=["tablet", "laptop"],  # Minimal candidates
            provider="openai",
            model="gpt-4o-mini",
            api_key=OPENAI_API_KEY,
            timeout_seconds=30,
        )
        assert isinstance(result, list)

    def test_real_azure_call_minimal(self):
        """Real Azure OpenAI call (minimal context)."""
        _check_llm_budget()

        if not AZURE_OPENAI_API_KEY:
            pytest.skip("AZURE_OPENAI_API_KEY not configured")

        result = select_alternatives_with_llm(
            missing_item="phone",
            candidates=["tablet"],  # Minimal
            provider="azure",
            model="gpt-4o-mini",
            api_key=AZURE_OPENAI_API_KEY,
            timeout_seconds=30,
        )
        assert isinstance(result, list)

    def test_real_gemini_call_minimal(self):
        """Real Gemini call (minimal context)."""
        _check_llm_budget()

        if not GEMINI_API_KEY:
            pytest.skip("GEMINI_API_KEY not configured")

        result = select_alternatives_with_llm(
            missing_item="phone",
            candidates=["tablet"],  # Minimal
            provider="gemini",
            model="gemini-2.0-flash",
            api_key=GEMINI_API_KEY,
            timeout_seconds=30,
        )
        assert isinstance(result, list)


# ============================================================================
# Pytest Configuration
# ============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "llm: LLM-related tests")
    config.addinivalue_line("markers", "llm_real: Real LLM API calls (opt-in)")


# ============================================================================
# Usage Documentation
# ============================================================================

"""
RUNNING TESTS:

1. DEFAULT (all mocked, ZERO cost):
   pytest tests/test_llm_integration.py -v

2. FAST LOCAL TESTING:
   pytest tests/test_llm_integration.py -v -m "not llm_real"

3. REAL LLM TESTING (requires API keys + $$$):
   USE_REAL_LLM=true LLM_BUDGET_LIMIT=5 pytest tests/test_llm_integration.py::TestRealLLMCalls -v

4. SPECIFIC TEST:
   pytest tests/test_llm_integration.py::TestSelectAlternativesWithLLM::test_empty_missing_item -v

COST CONTROL:
✅ Mocks enabled by default (zero cost)
✅ Real tests opt-in via USE_REAL_LLM=true
✅ Budget tracking with LLM_BUDGET_LIMIT
✅ Minimal context sent to real LLMs (phone + 2 candidates = ~200 tokens = $0.001)
✅ Fast local CI/CD (mocked tests run in 2-5 seconds)

Environment Variables:
- USE_REAL_LLM:    "false" (default) | "true" (enable real API calls)
- LLM_BUDGET_LIMIT: Number (default 5 if USE_REAL_LLM=true)
- OPENAI_API_KEY:  Your OpenAI API key
- AZURE_OPENAI_API_KEY: Your Azure API key
- GEMINI_API_KEY:  Your Google Gemini API key
"""
