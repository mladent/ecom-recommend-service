"""
Unified LLM client abstraction layer supporting multiple providers.

This module provides a clean abstraction over various LLM providers (OpenAI, Azure,
Gemini, Anthropic, Perplexity) with a unified interface for chat completions.

Architecture:
- BaseLLMProvider: Abstract base class defining the provider interface
- Concrete providers: OpenAIProvider, AzureProvider, etc.
- LLMProviderFactory: Registry-based factory for provider instantiation
- LLMClient: Main unified interface for all LLM operations

Usage:
    client = LLMClient(provider="openai", model="gpt-4o-mini", config=config)
    response = client.chat_completion(system_prompt, user_prompt)
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Type
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

logger = logging.getLogger(__name__)


def _http_post_json(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    """POST JSON and return parsed JSON response."""
    data = json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"HTTP error {exc.code}: {exc.read().decode('utf-8')}") from exc
    except URLError as exc:
        raise RuntimeError(f"Network error: {exc}") from exc


@dataclass
class LLMConfig:
    """Configuration for LLM provider credentials and settings."""
    
    # Provider selection
    provider: str
    model: str
    
    # Generation parameters
    temperature: float = 0.0
    max_tokens: int = 128
    timeout_seconds: int = 20
    
    # OpenAI credentials
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    
    # Azure credentials
    azure_api_key: Optional[str] = None
    azure_endpoint: Optional[str] = None
    azure_deployment: Optional[str] = None
    azure_api_version: Optional[str] = "2024-06-01"
    
    # Google Gemini credentials
    gemini_api_key: Optional[str] = None
    
    # Anthropic credentials
    anthropic_api_key: Optional[str] = None
    
    # Perplexity credentials
    perplexity_api_key: Optional[str] = None
    perplexity_base_url: Optional[str] = "https://api.perplexity.ai"


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    All provider implementations must extend this class and implement:
    - call_api(): Make API call to provider
    - validate_credentials(): Check if required credentials are available
    - parse_response(): Extract text from provider-specific response format
    """
    
    def __init__(self, model: str, config: LLMConfig):
        """
        Initialize the provider.
        
        Args:
            model: Model name/identifier for the provider
            config: LLM configuration with credentials and settings
        """
        self.model = model
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def call_api(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """
        Make API call to the LLM provider.
        
        Args:
            system_prompt: System/instruction prompt
            user_prompt: User message/query
            
        Returns:
            Raw JSON response from the provider
            
        Raises:
            RuntimeError: If credentials are missing or API call fails
        """
        pass
    
    @abstractmethod
    def validate_credentials(self) -> bool:
        """
        Validate that required credentials are available.
        
        Returns:
            True if all required credentials are present, False otherwise
        """
        pass
    
    @abstractmethod
    def parse_response(self, response_data: Dict[str, Any]) -> str:
        """
        Parse provider-specific response format to extract generated text.
        
        Args:
            response_data: Raw JSON response from provider
            
        Returns:
            Extracted text content
            
        Raises:
            KeyError: If expected fields are missing from response
        """
        pass
    
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        High-level method to generate completion with error handling.
        
        Args:
            system_prompt: System/instruction prompt
            user_prompt: User message/query
            
        Returns:
            Generated text response
            
        Raises:
            RuntimeError: If credentials missing or API call fails
        """
        if not self.validate_credentials():
            raise RuntimeError(f"Missing credentials for {self.__class__.__name__}")
        
        try:
            response_data = self.call_api(system_prompt, user_prompt)
            return self.parse_response(response_data)
        except Exception as exc:
            self.logger.error(f"API call failed: {exc}")
            raise


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider (gpt-4, gpt-3.5-turbo, etc.)."""
    
    def validate_credentials(self) -> bool:
        """Check if OpenAI API key is available."""
        return self.config.openai_api_key is not None
    
    def call_api(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Call OpenAI chat completion API."""
        if not self.config.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY missing")
        
        base_url = self.config.openai_base_url or "https://api.openai.com"
        url = f"{base_url}/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.config.openai_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        
        return _http_post_json(url, headers, payload, self.config.timeout_seconds)
    
    def parse_response(self, response_data: Dict[str, Any]) -> str:
        """Extract text from OpenAI response format."""
        return response_data["choices"][0]["message"]["content"].strip()


class AzureProvider(BaseLLMProvider):
    """Azure OpenAI Service provider."""
    
    def validate_credentials(self) -> bool:
        """Check if Azure credentials are available."""
        return all([
            self.config.azure_api_key,
            self.config.azure_endpoint,
            self.config.azure_deployment
        ])
    
    def call_api(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Call Azure OpenAI API."""
        if not self.validate_credentials():
            raise RuntimeError("Azure OpenAI credentials or endpoint missing")
        
        api_version = self.config.azure_api_version or "2024-06-01"
        url = (
            f"{self.config.azure_endpoint}/openai/deployments/"
            f"{self.config.azure_deployment}/chat/completions"
            f"?api-version={api_version}"
        )
        
        headers = {
            "api-key": self.config.azure_api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        
        return _http_post_json(url, headers, payload, self.config.timeout_seconds)
    
    def parse_response(self, response_data: Dict[str, Any]) -> str:
        """Extract text from Azure OpenAI response format."""
        return response_data["choices"][0]["message"]["content"].strip()


class GeminiProvider(BaseLLMProvider):
    """Google Gemini API provider."""
    
    def validate_credentials(self) -> bool:
        """Check if Gemini API key is available."""
        return self.config.gemini_api_key is not None
    
    def call_api(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Call Google Gemini API."""
        if not self.config.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY missing")
        
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.config.gemini_api_key}"
        )
        
        headers = {"Content-Type": "application/json"}
        
        # Gemini combines system and user prompts
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        payload = {
            "contents": [{"parts": [{"text": combined_prompt}]}],
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": self.config.max_tokens
            },
        }
        
        return _http_post_json(url, headers, payload, self.config.timeout_seconds)
    
    def parse_response(self, response_data: Dict[str, Any]) -> str:
        """Extract text from Gemini response format."""
        return response_data["candidates"][0]["content"]["parts"][0]["text"].strip()


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API provider."""
    
    def validate_credentials(self) -> bool:
        """Check if Anthropic API key is available."""
        return self.config.anthropic_api_key is not None
    
    def call_api(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Call Anthropic Claude API."""
        if not self.config.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY missing")
        
        url = "https://api.anthropic.com/v1/messages"
        
        headers = {
            "x-api-key": self.config.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        
        # Anthropic combines system and user prompts differently
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        payload = {
            "model": self.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": [{"role": "user", "content": combined_prompt}],
        }
        
        return _http_post_json(url, headers, payload, self.config.timeout_seconds)
    
    def parse_response(self, response_data: Dict[str, Any]) -> str:
        """Extract text from Anthropic response format."""
        return response_data["content"][0]["text"].strip()


class PerplexityProvider(BaseLLMProvider):
    """Perplexity AI API provider."""
    
    def validate_credentials(self) -> bool:
        """Check if Perplexity API key is available."""
        return self.config.perplexity_api_key is not None
    
    def call_api(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Call Perplexity AI API."""
        if not self.config.perplexity_api_key:
            raise RuntimeError("PERPLEXITY_API_KEY missing")
        
        base_url = (self.config.perplexity_base_url or "https://api.perplexity.ai").rstrip("/")
        url = f"{base_url}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.config.perplexity_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        
        return _http_post_json(url, headers, payload, self.config.timeout_seconds)
    
    def parse_response(self, response_data: Dict[str, Any]) -> str:
        """Extract text from Perplexity response format."""
        return response_data["choices"][0]["message"]["content"].strip()


class LLMProviderFactory:
    """
    Factory for creating LLM provider instances with registry pattern.
    
    Supports registration of new providers at runtime and provides
    a unified interface for provider instantiation.
    """
    
    _registry: Dict[str, Type[BaseLLMProvider]] = {
        "openai": OpenAIProvider,
        "azure": AzureProvider,
        "gemini": GeminiProvider,
        "anthropic": AnthropicProvider,
        "perplexity": PerplexityProvider,
    }
    
    @classmethod
    def register(cls, name: str, provider_class: Type[BaseLLMProvider]) -> None:
        """
        Register a new provider class.
        
        Args:
            name: Provider identifier (e.g., "openai")
            provider_class: Provider class extending BaseLLMProvider
        """
        cls._registry[name.lower()] = provider_class
        logger.info(f"Registered LLM provider: {name}")
    
    @classmethod
    def create(cls, provider: str, model: str, config: LLMConfig) -> BaseLLMProvider:
        """
        Create a provider instance.
        
        Args:
            provider: Provider name (openai, azure, gemini, anthropic, perplexity)
            model: Model identifier
            config: LLM configuration with credentials
            
        Returns:
            Instantiated provider
            
        Raises:
            ValueError: If provider is not registered
        """
        provider_lower = provider.lower()
        if provider_lower not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown LLM provider: {provider}\n"
                f"Available providers: {available}"
            )
        
        provider_class = cls._registry[provider_lower]
        return provider_class(model=model, config=config)
    
    @classmethod
    def list_providers(cls) -> list:
        """List all registered providers."""
        return list(cls._registry.keys())


class LLMClient:
    """
    Unified client interface for LLM operations.
    
    Provides a high-level API over all supported LLM providers with:
    - Automatic provider instantiation
    - Credential validation
    - Response parsing (JSON, markdown code blocks)
    - Error handling
    - Logging
    
    Example:
        config = LLMConfig(
            provider="openai",
            model="gpt-4o-mini",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        client = LLMClient(config)
        response = client.chat_completion(
            "You are a helpful assistant.",
            "What is the capital of France?"
        )
    """
    
    def __init__(self, config: LLMConfig):
        """
        Initialize LLM client.
        
        Args:
            config: LLM configuration with provider, model, and credentials
        """
        self.config = config
        self.provider = LLMProviderFactory.create(
            provider=config.provider,
            model=config.model,
            config=config
        )
        self.logger = logging.getLogger(f"{__name__}.LLMClient")
    
    def validate_credentials(self) -> bool:
        """
        Check if credentials are available for the configured provider.
        
        Returns:
            True if credentials are valid, False otherwise
        """
        return self.provider.validate_credentials()
    
    def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        parse_json: bool = False
    ) -> str:
        """
        Generate chat completion.
        
        Args:
            system_prompt: System/instruction message
            user_prompt: User query
            parse_json: If True, extract JSON from markdown code blocks
            
        Returns:
            Generated text response
            
        Raises:
            RuntimeError: If credentials missing or API call fails
        """
        response_text = self.provider.generate(system_prompt, user_prompt)
        
        if parse_json:
            response_text = self._extract_json_from_markdown(response_text)
        
        return response_text
    
    def chat_completion_json(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> Dict[str, Any]:
        """
        Generate chat completion and parse as JSON.
        
        Args:
            system_prompt: System/instruction message
            user_prompt: User query
            
        Returns:
            Parsed JSON object
            
        Raises:
            RuntimeError: If credentials missing or API call fails
            json.JSONDecodeError: If response is not valid JSON
        """
        response_text = self.chat_completion(system_prompt, user_prompt, parse_json=True)
        return json.loads(response_text)
    
    @staticmethod
    def _extract_json_from_markdown(text: str) -> str:
        """
        Extract JSON from markdown code blocks.
        
        Some models wrap JSON responses in markdown, e.g.:
        ```json
        {"key": "value"}
        ```
        
        Args:
            text: Raw response text
            
        Returns:
            Cleaned JSON string
        """
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        return text
    
    def get_provider_name(self) -> str:
        """Get the name of the current provider."""
        return self.config.provider
    
    def get_model_name(self) -> str:
        """Get the name of the current model."""
        return self.config.model
    
    def __repr__(self) -> str:
        """String representation of the client."""
        return (
            f"LLMClient(provider={self.config.provider}, "
            f"model={self.config.model})"
        )


# ============================================================================
# LLM Operation Tracking - Aggregates metrics for MLflow logging
# ============================================================================

from threading import Lock
import time


@dataclass
class LLMOperationStats:
    """Statistics for a single LLM operation type."""
    
    operation_name: str  # e.g., 'enrich_categories', 'select_alternatives'
    total_calls: int = 0
    cache_hits: int = 0
    total_latency_ms: float = 0.0
    error_count: int = 0
    provider_distribution: Dict[str, int] = field(default_factory=dict)
    
    def add_call(self, latency_ms: float, cached: bool = False, provider: str = "unknown", error: bool = False) -> None:
        """Record a single LLM operation call."""
        self.total_calls += 1
        self.total_latency_ms += latency_ms
        if cached:
            self.cache_hits += 1
        if error:
            self.error_count += 1
        self.provider_distribution[provider] = self.provider_distribution.get(provider, 0) + 1
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate as percentage."""
        if self.total_calls == 0:
            return 0.0
        return (self.cache_hits / self.total_calls) * 100.0
    
    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency in milliseconds."""
        if self.total_calls == 0:
            return 0.0
        return self.total_latency_ms / self.total_calls


class LLMOperationTracker:
    """Thread-safe tracker for LLM operation metrics.
    
    Aggregates metrics across all LLM operations (enrich_categories, extract_contexts,
    batch_score_anomalies, select_alternatives) for logging to MLflow.
    
    Usage:
        tracker = LLMOperationTracker()
        tracker.record_operation("enrich_categories", latency_ms=150, cached=True, provider="openai")
        tracker.record_operation("select_alternatives", latency_ms=200, cached=False, provider="gemini")
        stats = tracker.get_aggregated_stats()  # Dict[str, LLMOperationStats]
    """
    
    _instance: Optional['LLMOperationTracker'] = None
    _lock: Lock = Lock()
    
    def __new__(cls) -> 'LLMOperationTracker':
        """Implement singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self) -> None:
        """Initialize the tracker (singleton)."""
        if not self._initialized:
            self._stats: Dict[str, LLMOperationStats] = {}
            self._operation_lock = Lock()
            self._initialized = True
    
    def record_operation(
        self,
        operation_name: str,
        latency_ms: float,
        cached: bool = False,
        provider: str = "unknown",
        error: bool = False
    ) -> None:
        """Record a single LLM operation call.
        
        Args:
            operation_name: Name of operation (e.g., 'enrich_categories')
            latency_ms: Latency in milliseconds
            cached: Whether result was from cache
            provider: LLM provider name
            error: Whether the operation failed
        """
        with self._operation_lock:
            if operation_name not in self._stats:
                self._stats[operation_name] = LLMOperationStats(operation_name=operation_name)
            self._stats[operation_name].add_call(latency_ms, cached, provider, error)
    
    def get_aggregated_stats(self) -> Dict[str, LLMOperationStats]:
        """Get all aggregated statistics.
        
        Returns:
            Dictionary mapping operation names to their LLMOperationStats
        """
        with self._operation_lock:
            return {k: v for k, v in self._stats.items()}
    
    def get_operation_stats(self, operation_name: str) -> Optional[LLMOperationStats]:
        """Get statistics for a specific operation.
        
        Args:
            operation_name: Name of operation to retrieve
            
        Returns:
            LLMOperationStats if operation was tracked, None otherwise
        """
        with self._operation_lock:
            return self._stats.get(operation_name)
    
    def get_total_calls(self) -> int:
        """Get total number of all LLM calls across all operations."""
        with self._operation_lock:
            return sum(stats.total_calls for stats in self._stats.values())
    
    def get_total_cache_hits(self) -> int:
        """Get total number of cache hits across all operations."""
        with self._operation_lock:
            return sum(stats.cache_hits for stats in self._stats.values())
    
    def get_overall_cache_hit_rate(self) -> float:
        """Get overall cache hit rate as percentage."""
        total_calls = self.get_total_calls()
        if total_calls == 0:
            return 0.0
        total_hits = self.get_total_cache_hits()
        return (total_hits / total_calls) * 100.0
    
    def get_provider_distribution(self) -> Dict[str, int]:
        """Get distribution of calls across providers.
        
        Returns:
            Dictionary mapping provider names to call counts
        """
        with self._operation_lock:
            distribution: Dict[str, int] = {}
            for stats in self._stats.values():
                for provider, count in stats.provider_distribution.items():
                    distribution[provider] = distribution.get(provider, 0) + count
            return distribution
    
    def reset(self) -> None:
        """Reset all tracked statistics."""
        with self._operation_lock:
            self._stats.clear()
    
    def to_mlflow_params(self) -> Dict[str, Any]:
        """Convert tracked stats to MLflow parameter dictionary.
        
        Returns:
            Dictionary of metrics suitable for MLflow.set_params() and log_metric()
        """
        params = {}
        with self._operation_lock:
            # Overall metrics
            total_calls = self.get_total_calls()
            params["llm_total_calls"] = str(total_calls)
            params["llm_cache_hit_rate_percent"] = f"{self.get_overall_cache_hit_rate():.2f}"
            
            # Provider distribution as comma-separated provider=count format
            provider_dist = self.get_provider_distribution()
            if provider_dist:
                dist_str = ",".join(f"{p}={c}" for p, c in sorted(provider_dist.items()))
                params["llm_provider_distribution"] = dist_str
            
            # Per-operation metrics
            for operation_name, stats in self._stats.items():
                prefix = f"llm_{operation_name}"
                params[f"{prefix}_calls"] = str(stats.total_calls)
                params[f"{prefix}_cache_hit_rate_percent"] = f"{stats.cache_hit_rate:.2f}"
                params[f"{prefix}_avg_latency_ms"] = f"{stats.avg_latency_ms:.2f}"
                params[f"{prefix}_error_count"] = str(stats.error_count)
        
        return params
    
    def to_mlflow_metrics(self) -> Dict[str, float]:
        """Convert tracked stats to MLflow metrics dictionary.
        
        Returns:
            Dictionary of numeric metrics suitable for MLflow.log_metrics()
        """
        metrics = {}
        with self._operation_lock:
            # Overall metrics
            metrics["llm_total_calls"] = float(self.get_total_calls())
            metrics["llm_cache_hit_rate_percent"] = self.get_overall_cache_hit_rate()
            metrics["llm_total_latency_ms"] = sum(stats.total_latency_ms for stats in self._stats.values())
            metrics["llm_total_errors"] = sum(stats.error_count for stats in self._stats.values())
            
            # Per-operation metrics
            for operation_name, stats in self._stats.items():
                prefix = f"llm_{operation_name}"
                metrics[f"{prefix}_calls"] = float(stats.total_calls)
                metrics[f"{prefix}_cache_hit_rate_percent"] = stats.cache_hit_rate
                metrics[f"{prefix}_avg_latency_ms"] = stats.avg_latency_ms
                metrics[f"{prefix}_error_count"] = float(stats.error_count)
        
        return metrics
