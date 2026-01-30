"""Utility functions for the recommendation service."""

import json
import logging
import os
import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

logger = logging.getLogger(__name__)

_PROMPT_CACHE: Dict[str, str] = {}


def _prompt_base_dir() -> str:
    return os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "config", "prompts"))


def _load_prompt_template(filename: str) -> str:
    if filename in _PROMPT_CACHE:
        return _PROMPT_CACHE[filename]
    base_dir = _prompt_base_dir()
    path = os.path.normpath(os.path.join(base_dir, filename))
    if not path.startswith(base_dir):
        raise RuntimeError(f"Invalid prompt path: {filename}")
    if not os.path.exists(path):
        raise RuntimeError(f"Prompt file missing: {filename}")
    with open(path, "r", encoding="utf-8") as handle:
        _PROMPT_CACHE[filename] = handle.read().strip()
    return _PROMPT_CACHE[filename]


def _render_prompt(filename: str, **kwargs: Any) -> str:
    template = _load_prompt_template(filename)
    return template.format(**kwargs)


class LLMQuotaExceededError(RuntimeError):
    """Raised when LLM provider reports insufficient quota."""


def setup_logging(level: int = logging.INFO) -> None:
    """
    Setup logging configuration.

    Args:
        level: Logging level
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("recommendation_service.log"),
        ],
    )


def validate_transaction(transaction: List[str]) -> bool:
    """
    Validate a transaction.

    Args:
        transaction: List of product descriptions

    Returns:
        bool: True if valid
    """
    if not transaction or not isinstance(transaction, list):
        logger.warning("Invalid transaction: empty or not a list")
        return False

    if not all(isinstance(item, str) for item in transaction):
        logger.warning("Invalid transaction: not all items are strings")
        return False

    return True


def filter_transaction(transaction: List[str], min_length: int = 3) -> List[str]:
    """
    Filter and clean transaction items.

    Args:
        transaction: List of product descriptions
        min_length: Minimum string length to keep

    Returns:
        List of filtered items
    """
    filtered = []
    for item in transaction:
        # Strip whitespace
        item = item.strip().lower()
        # Filter by length
        if len(item) >= min_length:
            filtered.append(item)

    return filtered


def format_recommendations(recommendations: Dict[str, Any], verbose: bool = False) -> str:
    """
    Format recommendations for display.

    Args:
        recommendations: Recommendations dictionary
        verbose: Include detailed information

    Returns:
        str: Formatted recommendations
    """
    output = []
    output.append("=" * 60)
    output.append("BUNDLE RECOMMENDATIONS")
    output.append("=" * 60)

    output.append(f"Transaction Items: {', '.join(recommendations['transaction'])}")
    output.append(f"Confidence Score: {recommendations['confidence']:.2%}")
    output.append(f"Recommender: {recommendations['recommender']}")
    output.append("")

    if recommendations["bundles"]:
        output.append("Recommended Bundles:")
        for i, bundle in enumerate(recommendations["bundles"], 1):
            output.append(f"  {i}. {', '.join(bundle)}")
    else:
        output.append("No recommendations available for this transaction.")

    output.append("=" * 60)

    return "\n".join(output)


def normalize_description_basic(text: str) -> str:
    """Basic normalization: lowercase, strip, collapse spaces, normalize units."""
    if text is None:
        return ""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\b(inches|inch|in\.)\b", "in", text)
    text = re.sub(r"\b(centimeters|centimetres|cm\.)\b", "cm", text)
    text = re.sub(r"\b(grams|gram|g\.)\b", "g", text)
    text = re.sub(r"\b(kilograms|kilogram|kg\.)\b", "kg", text)
    return text


def load_json_file(path: str) -> Dict[str, Any]:
    """Load a JSON file safely; return empty dict if missing."""
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning(f"Failed to load JSON file {path}: {exc}")
        return {}


def save_json_file(path: str, data: Dict[str, Any]) -> None:
    """Save a JSON file safely."""
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_alias_map(path: str) -> Dict[str, str]:
    """Load alias map for canonical description mapping."""
    raw = load_json_file(path)
    if not isinstance(raw, dict):
        return {}
    return {normalize_description_basic(k): normalize_description_basic(v) for k, v in raw.items()}


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


def normalize_description_with_llm(
    text: str,
    provider: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout_seconds: int,
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    deployment: Optional[str] = None,
    api_version: Optional[str] = None,
    base_url: Optional[str] = None,
) -> str:
    """
    Normalize a product description with an LLM provider. Falls back to input on failure.
    """
    if not text:
        return ""

    prompt = _render_prompt("normalize_description_user.md", text=text)
    system_prompt = _load_prompt_template("normalize_description_system.md")

    try:
        provider = (provider or "").lower()
        if provider == "openai":
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY missing")
            url = (base_url or "https://api.openai.com") + "/v1/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            }
            data = _http_post_json(url, headers, payload, timeout_seconds)
            return data["choices"][0]["message"]["content"].strip()

        if provider == "azure":
            if not api_key or not endpoint or not deployment:
                raise RuntimeError("Azure OpenAI credentials or endpoint missing")
            api_version = api_version or "2024-06-01"
            url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
            headers = {"api-key": api_key, "Content-Type": "application/json"}
            payload = {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            }
            data = _http_post_json(url, headers, payload, timeout_seconds)
            return data["choices"][0]["message"]["content"].strip()

        if provider == "gemini":
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY missing")
            url = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model}:generateContent?key={api_key}"
            )
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
            }
            data = _http_post_json(url, headers, payload, timeout_seconds)
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()

        if provider == "anthropic":
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY missing")
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
            data = _http_post_json(url, headers, payload, timeout_seconds)
            return data["content"][0]["text"].strip()

        if provider == "perplexity":
            if not api_key:
                raise RuntimeError("PERPLEXITY_API_KEY missing")
            base_url = base_url or "https://api.perplexity.ai"
            url = base_url.rstrip("/") + "/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            }
            data = _http_post_json(url, headers, payload, timeout_seconds)
            return data["choices"][0]["message"]["content"].strip()

        raise RuntimeError(f"Unsupported LLM provider: {provider}")
    except Exception as exc:
        message = str(exc)
        if "insufficient_quota" in message.lower() or "quota" in message.lower() and "exceeded" in message.lower():
            raise LLMQuotaExceededError(message) from exc
        logger.warning(f"LLM normalization failed ({provider}): {exc}")
        return text


def enrich_categories_with_llm(
    text: str,
    fields: List[str],
    provider: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout_seconds: int,
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    deployment: Optional[str] = None,
    api_version: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Dict[str, str]:
    """
    Extract category attributes from a product description using an LLM provider.
    
    Args:
        text: Product description to enrich
        fields: List of fields to extract (e.g., ['category', 'material', 'size', 'theme'])
        provider: LLM provider name
        model: Model identifier
        temperature: Generation temperature
        max_tokens: Max tokens for response
        timeout_seconds: Request timeout
        api_key: API key for the provider
        endpoint: API endpoint (Azure)
        deployment: Deployment name (Azure)
        api_version: API version (Azure)
        base_url: Base URL (OpenAI, Perplexity)
    
    Returns:
        Dict mapping field names to extracted values (NaN string for missing values)
    """
    if not text:
        return {field: "NaN" for field in fields}

    # Build prompt requesting JSON output
    fields_str = ", ".join(fields)
    prompt = _render_prompt("enrich_categories_user.md", fields=fields_str, text=text)
    system_prompt = _load_prompt_template("enrich_categories_system.md")

    try:
        provider = (provider or "").lower()
        response_text = ""
        
        if provider == "openai":
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY missing")
            url = (base_url or "https://api.openai.com") + "/v1/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            }
            data = _http_post_json(url, headers, payload, timeout_seconds)
            response_text = data["choices"][0]["message"]["content"].strip()

        elif provider == "azure":
            if not api_key or not endpoint or not deployment:
                raise RuntimeError("Azure OpenAI credentials or endpoint missing")
            api_version = api_version or "2024-06-01"
            url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
            headers = {"api-key": api_key, "Content-Type": "application/json"}
            payload = {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            }
            data = _http_post_json(url, headers, payload, timeout_seconds)
            response_text = data["choices"][0]["message"]["content"].strip()

        elif provider == "gemini":
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY missing")
            url = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model}:generateContent?key={api_key}"
            )
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
            }
            data = _http_post_json(url, headers, payload, timeout_seconds)
            response_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()

        elif provider == "anthropic":
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY missing")
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
            data = _http_post_json(url, headers, payload, timeout_seconds)
            response_text = data["content"][0]["text"].strip()

        elif provider == "perplexity":
            if not api_key:
                raise RuntimeError("PERPLEXITY_API_KEY missing")
            base_url = base_url or "https://api.perplexity.ai"
            url = base_url.rstrip("/") + "/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            }
            data = _http_post_json(url, headers, payload, timeout_seconds)
            response_text = data["choices"][0]["message"]["content"].strip()

        else:
            raise RuntimeError(f"Unsupported LLM provider: {provider}")

        # Parse JSON from response
        # Some models may wrap JSON in markdown code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        result = json.loads(response_text)
        
        # Ensure all requested fields are present
        enriched = {}
        for field in fields:
            enriched[field] = result.get(field, "NaN")
        
        return enriched

    except Exception as exc:
        message = str(exc)
        if "insufficient_quota" in message.lower() or "quota" in message.lower() and "exceeded" in message.lower():
            raise LLMQuotaExceededError(message) from exc
        logger.warning(f"LLM category enrichment failed ({provider}): {exc}")
        return {field: "NaN" for field in fields}


def compute_iqr_bounds(series, multiplier: float = 1.5) -> Tuple[float, float]:
    """Compute IQR-based lower and upper bounds for a numeric series."""
    clean = series.dropna()
    if clean.empty:
        return float("-inf"), float("inf")
    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return lower, upper


def batch_score_anomalies_with_llm(
    records: List[Dict[str, Any]],
    provider: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout_seconds: int,
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    deployment: Optional[str] = None,
    api_version: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Dict[str, Dict[str, str]]:
    """
    Batch score anomalies using an LLM.

    Returns a mapping of record key -> {"anomaly_type": str, "anomaly_reason": str}
    """
    if not records:
        return {}

    prompt = _render_prompt(
        "batch_score_anomalies_user.md",
        records_json=json.dumps(records, ensure_ascii=False),
    )
    system_prompt = _load_prompt_template("batch_score_anomalies_system.md")

    try:
        provider = (provider or "").lower()
        response_text = ""

        if provider == "openai":
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY missing")
            url = (base_url or "https://api.openai.com") + "/v1/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            }
            data = _http_post_json(url, headers, payload, timeout_seconds)
            response_text = data["choices"][0]["message"]["content"].strip()

        elif provider == "azure":
            if not api_key or not endpoint or not deployment:
                raise RuntimeError("Azure OpenAI credentials or endpoint missing")
            api_version = api_version or "2024-06-01"
            url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
            headers = {"api-key": api_key, "Content-Type": "application/json"}
            payload = {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            }
            data = _http_post_json(url, headers, payload, timeout_seconds)
            response_text = data["choices"][0]["message"]["content"].strip()

        elif provider == "gemini":
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY missing")
            url = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model}:generateContent?key={api_key}"
            )
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
            }
            data = _http_post_json(url, headers, payload, timeout_seconds)
            response_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()

        elif provider == "anthropic":
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY missing")
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
            data = _http_post_json(url, headers, payload, timeout_seconds)
            response_text = data["content"][0]["text"].strip()

        elif provider == "perplexity":
            if not api_key:
                raise RuntimeError("PERPLEXITY_API_KEY missing")
            base_url = base_url or "https://api.perplexity.ai"
            url = base_url.rstrip("/") + "/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            }
            data = _http_post_json(url, headers, payload, timeout_seconds)
            response_text = data["choices"][0]["message"]["content"].strip()

        else:
            raise RuntimeError(f"Unsupported LLM provider: {provider}")

        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        parsed = json.loads(response_text)
        result: Dict[str, Dict[str, str]] = {}
        for item in parsed:
            key = str(item.get("key", ""))
            if not key:
                continue
            result[key] = {
                "anomaly_type": str(item.get("anomaly_type", "none")),
                "anomaly_reason": str(item.get("anomaly_reason", "")),
            }
        return result

    except Exception as exc:
        message = str(exc)
        if "insufficient_quota" in message.lower() or "quota" in message.lower() and "exceeded" in message.lower():
            raise LLMQuotaExceededError(message) from exc
        logger.warning(f"LLM anomaly scoring failed ({provider}): {exc}")
        return {}


def extract_contexts_with_llm(
    text: str,
    max_contexts: int,
    provider: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout_seconds: int,
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    deployment: Optional[str] = None,
    api_version: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Extract usage contexts from a product description using an LLM provider.
    
    Args:
        text: Product description to extract contexts from
        max_contexts: Maximum number of contexts to extract
        provider: LLM provider name
        model: Model identifier
        temperature: Generation temperature
        max_tokens: Max tokens for response
        timeout_seconds: Request timeout
        api_key: API key for the provider
        endpoint: API endpoint (Azure)
        deployment: Deployment name (Azure)
        api_version: API version (Azure)
        base_url: Base URL (OpenAI, Perplexity)
    
    Returns:
        Dict with "contexts" key containing list of {context, confidence} objects
    """
    if not text:
        return {"contexts": []}

    # Build prompt requesting JSON output
    prompt = _render_prompt("extract_contexts_user.md", max_contexts=max_contexts, text=text)
    system_prompt = _load_prompt_template("extract_contexts_system.md")

    try:
        provider = (provider or "").lower()
        response_text = ""
        
        if provider == "openai":
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY missing")
            url = (base_url or "https://api.openai.com") + "/v1/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            }
            data = _http_post_json(url, headers, payload, timeout_seconds)
            response_text = data["choices"][0]["message"]["content"].strip()

        elif provider == "azure":
            if not api_key or not endpoint or not deployment:
                raise RuntimeError("Azure OpenAI credentials or endpoint missing")
            api_version = api_version or "2024-06-01"
            url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
            headers = {"api-key": api_key, "Content-Type": "application/json"}
            payload = {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            }
            data = _http_post_json(url, headers, payload, timeout_seconds)
            response_text = data["choices"][0]["message"]["content"].strip()

        elif provider == "gemini":
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY missing")
            url = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model}:generateContent?key={api_key}"
            )
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
            }
            data = _http_post_json(url, headers, payload, timeout_seconds)
            response_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()

        elif provider == "anthropic":
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY missing")
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
            data = _http_post_json(url, headers, payload, timeout_seconds)
            response_text = data["content"][0]["text"].strip()

        elif provider == "perplexity":
            if not api_key:
                raise RuntimeError("PERPLEXITY_API_KEY missing")
            base_url = base_url or "https://api.perplexity.ai"
            url = base_url.rstrip("/") + "/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            }
            data = _http_post_json(url, headers, payload, timeout_seconds)
            response_text = data["choices"][0]["message"]["content"].strip()

        else:
            raise RuntimeError(f"Unsupported LLM provider: {provider}")

        # Parse JSON from response
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        result = json.loads(response_text)
        
        # Ensure result has contexts array
        contexts = result.get("contexts", [])
        if not isinstance(contexts, list):
            contexts = []
        
        return {"contexts": contexts}

    except Exception as exc:
        message = str(exc)
        if "insufficient_quota" in message.lower() or "quota" in message.lower() and "exceeded" in message.lower():
            raise LLMQuotaExceededError(message) from exc
        logger.warning(f"LLM context extraction failed ({provider}): {exc}")
        return {"contexts": []}


def load_inventory_csv(path: str) -> Dict[str, bool]:
    """
    Load inventory availability from a local CSV file.

    Expected columns (case-insensitive):
    - description
    - in_stock / available / stock

    Note: In a real production system, this should be replaced with an external
    inventory API integration instead of reading a local CSV file.
    """
    if not path or not os.path.exists(path):
        return {}
    try:
        import pandas as pd

        df = pd.read_csv(path)
        if df.empty:
            return {}

        columns = {c.lower(): c for c in df.columns}
        desc_col = columns.get("description") or columns.get("item") or columns.get("product")
        stock_col = columns.get("in_stock") or columns.get("available") or columns.get("stock")

        if not desc_col or not stock_col:
            logger.warning("Inventory CSV missing required columns: description + in_stock/available/stock")
            return {}

        inventory = {}
        for _, row in df.iterrows():
            desc = str(row[desc_col]).strip().lower()
            if not desc:
                continue
            raw = str(row[stock_col]).strip().lower()
            in_stock = raw in {"1", "true", "yes", "y", "in_stock", "available"}
            inventory[desc] = in_stock

        return inventory
    except Exception as exc:
        logger.warning(f"Failed to load inventory CSV {path}: {exc}")
        return {}


def _candidate_hash(candidates: List[str]) -> str:
    payload = "|".join(sorted(candidates))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_alternatives_with_llm(
    missing_item: str,
    candidates: List[str],
    provider: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout_seconds: int,
    max_alternatives: int,
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    deployment: Optional[str] = None,
    api_version: Optional[str] = None,
    base_url: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Use LLM to select alternatives for an out-of-stock item from candidates.
    Returns list of {item, score, reason}.
    """
    if not missing_item or not candidates:
        return []

    prompt = _render_prompt(
        "select_alternatives_user.md",
        missing_item=missing_item,
        candidates_json=json.dumps(candidates[:200]),
    )
    system_prompt = _load_prompt_template("select_alternatives_system.md")

    try:
        provider = (provider or "").lower()
        response_text = ""

        if provider == "openai":
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY missing")
            url = (base_url or "https://api.openai.com") + "/v1/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            }
            data = _http_post_json(url, headers, payload, timeout_seconds)
            response_text = data["choices"][0]["message"]["content"].strip()

        elif provider == "azure":
            if not api_key or not endpoint or not deployment:
                raise RuntimeError("Azure OpenAI credentials or endpoint missing")
            api_version = api_version or "2024-06-01"
            url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
            headers = {"api-key": api_key, "Content-Type": "application/json"}
            payload = {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            }
            data = _http_post_json(url, headers, payload, timeout_seconds)
            response_text = data["choices"][0]["message"]["content"].strip()

        elif provider == "gemini":
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY missing")
            url = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model}:generateContent?key={api_key}"
            )
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
            }
            data = _http_post_json(url, headers, payload, timeout_seconds)
            response_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()

        elif provider == "anthropic":
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY missing")
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
            data = _http_post_json(url, headers, payload, timeout_seconds)
            response_text = data["content"][0]["text"].strip()

        elif provider == "perplexity":
            if not api_key:
                raise RuntimeError("PERPLEXITY_API_KEY missing")
            base_url = base_url or "https://api.perplexity.ai"
            url = base_url.rstrip("/") + "/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            }
            data = _http_post_json(url, headers, payload, timeout_seconds)
            response_text = data["choices"][0]["message"]["content"].strip()

        else:
            raise RuntimeError(f"Unsupported LLM provider: {provider}")

        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        result = json.loads(response_text)
        alternatives = result.get("alternatives", []) if isinstance(result, dict) else result
        if not isinstance(alternatives, list):
            return []
        return alternatives[:max_alternatives]

    except Exception as exc:
        message = str(exc)
        if "insufficient_quota" in message.lower() or "quota" in message.lower() and "exceeded" in message.lower():
            raise LLMQuotaExceededError(message) from exc
        logger.warning(f"LLM alternative selection failed ({provider}): {exc}")
        return []


def select_alternative_heuristic(missing_item: str, candidates: List[str]) -> Optional[Dict[str, Any]]:
    """Fallback: choose candidate with highest token overlap similarity."""
    if not missing_item or not candidates:
        return None
    missing_tokens = set(normalize_description_basic(missing_item).split())
    if not missing_tokens:
        return None

    best = None
    best_score = 0.0
    for cand in candidates:
        cand_tokens = set(normalize_description_basic(cand).split())
        if not cand_tokens:
            continue
        score = len(missing_tokens & cand_tokens) / max(len(missing_tokens | cand_tokens), 1)
        if score > best_score:
            best_score = score
            best = {"item": cand, "score": score, "reason": "token-overlap"}
    return best
