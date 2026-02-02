"""
Example: Batch LLM Category Enrichment

This example demonstrates batch processing of category enrichment,
which optimizes LLM API usage by submitting multiple requests efficiently.
"""

import logging
import os
from src.utils import enrich_categories_batch_with_llm
from src.config import (
    LLM_PROVIDER,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_TIMEOUT_SECONDS,
    ENRICHMENT_BATCH_SIZE,
    ENRICHMENT_FIELDS,
    OPENAI_API_KEY,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_API_VERSION,
    GEMINI_API_KEY,
    ANTHROPIC_API_KEY,
    PERPLEXITY_API_KEY,
    PERPLEXITY_BASE_URL,
)
from src.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def example_batch_enrichment():
    """Example: Batch category enrichment for multiple products."""

    logger.info("Example: Batch Category Enrichment")
    logger.info("=" * 60)

    # Sample product descriptions
    products = [
        "WHITE HANGING HEART T-LIGHT HOLDER",
        "WHITE METAL LANTERN",
        "CREAM CUPID HEARTS COAT HANGER",
        "RED WOOLLY HOTTIE WHITE LETTER",
        "SET 3 REKLAMEBAK + TINPLATE SIGN",
        "WORLD WAR 2 GLUED JIGSAW PUZZLE",
        "PLAYING CARDS, HISTORICAL",
        "PACK OF 72 RETROSPOT TEA TOWELS",
        "ASSORTED COLOUR TEARDROP HEART",
        "REGENCY CAKESTAND 3 TIER",
    ]

    logger.info(f"Processing {len(products)} products in batches")
    logger.info(f"Batch size: {ENRICHMENT_BATCH_SIZE}")
    logger.info(f"LLM Provider: {LLM_PROVIDER}")
    logger.info("")

    provider = LLM_PROVIDER.lower() if LLM_PROVIDER else "openai"

    try:
        # Process batch enrichment
        results = enrich_categories_batch_with_llm(
            texts=products,
            fields=ENRICHMENT_FIELDS,
            provider=provider,
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            timeout_seconds=LLM_TIMEOUT_SECONDS,
            batch_size=ENRICHMENT_BATCH_SIZE,
            api_key=(
                OPENAI_API_KEY
                if provider == "openai"
                else AZURE_OPENAI_API_KEY
                if provider == "azure"
                else GEMINI_API_KEY
                if provider == "gemini"
                else ANTHROPIC_API_KEY
                if provider == "anthropic"
                else PERPLEXITY_API_KEY
            ),
            endpoint=AZURE_OPENAI_ENDPOINT,
            deployment=AZURE_OPENAI_DEPLOYMENT,
            api_version=AZURE_OPENAI_API_VERSION,
            base_url=PERPLEXITY_BASE_URL if provider == "perplexity" else None,
        )

        # Display results
        logger.info("=" * 60)
        logger.info("Batch Enrichment Results")
        logger.info("=" * 60)

        for product, enrichment in results.items():
            logger.info(f"\nProduct: {product}")
            for field, value in enrichment.items():
                display_value = value if value != "NaN" else "Not determined"
                logger.info(f"  {field.capitalize()}: {display_value}")

    except Exception as exc:
        logger.error(f"Batch enrichment failed: {exc}")
        logger.info("Note: LLM enrichment requires configured API credentials")
        logger.info("Set environment variables or update .env file with your API keys")


if __name__ == "__main__":
    example_batch_enrichment()
