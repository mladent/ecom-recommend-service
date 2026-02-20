"""
Example 2: Cross-Sell Product Recommendations

This example demonstrates how to get cross-sell product suggestions
based on customer's current transaction items.
"""

import logging
import os
from src.config import load_config
from src.recommendation_engine import BundleRecommendationEngine
from src.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def example_cross_sell_recommendations():
    """Example: Cross-sell product recommendations."""

    logger.info("Example 2: Cross-Sell Product Recommendations")
    logger.info("=" * 60)

    # Load trained model
    models_path = os.path.join(os.path.dirname(__file__), "models", "recommendation_engine.pkl")

    if not os.path.exists(models_path):
        logger.error(f"Trained model not found at {models_path}")
        logger.info("Please train the model first using: python main.py --train")
        return

    pipeline_config, engine_config, _, _, _ = load_config()
    engine = BundleRecommendationEngine(
        engine_config=engine_config,
        pipeline_config=pipeline_config,
    )
    if not engine.load_model(models_path):
        logger.error("Failed to load model")
        return

    # Example customer transactions
    transactions = [
        ["white hanging heart t-light holder"],
        ["regency cakestand 3 tier"],
        ["world war 2 glued jigsaw puzzle"],
        ["playing cards, historical"],
    ]

    # Get cross-sell recommendations
    for transaction in transactions:
        logger.info(f"\nCustomer has: {', '.join(transaction)}")
        logger.info("Cross-sell suggestions:")

        cross_sell = engine.get_cross_sell_products(transaction, top_n=5)

        if cross_sell:
            for i, (product, score) in enumerate(cross_sell, 1):
                logger.info(f"  {i}. {product} (Affinity Score: {score:.2f})")
        else:
            logger.info("  No cross-sell suggestions available")


if __name__ == "__main__":
    example_cross_sell_recommendations()
