"""Utility functions for the recommendation service."""

import json
import logging
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

logger = logging.getLogger(__name__)


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

    prompt = (
        "Normalize the product description for catalog matching. "
        "Fix casing, spelling, and standardize units. "
        "Return only the canonical product name without quotes or extra text.\n"
        f"Description: {text}"
    )

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
                    {"role": "system", "content": "You normalize product descriptions."},
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
                    {"role": "system", "content": "You normalize product descriptions."},
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
                    {"role": "system", "content": "You normalize product descriptions."},
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
    prompt = (
        f"Extract the following attributes from this product description: {fields_str}. "
        "Return a JSON object with each field as a key. "
        'Use "NaN" for any attribute that cannot be determined. '
        "Be concise and specific.\n"
        f"Description: {text}\n"
        "JSON:"
    )

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
                    {"role": "system", "content": "You extract product attributes and return JSON."},
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
                    {"role": "system", "content": "You extract product attributes and return JSON."},
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
                    {"role": "system", "content": "You extract product attributes and return JSON."},
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

    prompt = (
        "You are an anomaly detector for e-commerce transactions. "
        "Classify each record as one of: bot-like, mispriced, invalid, or none. "
        "Use the provided statistical outlier indicators and transaction details. "
        "Return ONLY JSON array of objects with keys: key, anomaly_type, anomaly_reason. "
        'Use "none" if not suspicious.\n\n'
        "Records:\n"
        f"{json.dumps(records, ensure_ascii=False)}"
    )

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
                    {"role": "system", "content": "You detect transaction anomalies and return JSON."},
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
                    {"role": "system", "content": "You detect transaction anomalies and return JSON."},
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
                    {"role": "system", "content": "You detect transaction anomalies and return JSON."},
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

