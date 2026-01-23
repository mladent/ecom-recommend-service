"""Utility functions for the recommendation service."""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


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
