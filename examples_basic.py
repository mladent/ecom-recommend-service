"""
Example 1: Basic Bundle Recommendations

This example demonstrates how to load the trained recommendation engine
and get bundle recommendations for customer transactions.
"""

import logging
import os
from src.config import load_config
from src.recommendation_engine import BundleRecommendationEngine
from src.utils import setup_logging, format_recommendations

setup_logging()
logger = logging.getLogger(__name__)


def example_basic_recommendations():
    """Example: Basic bundle recommendations."""

    logger.info("Example 1: Basic Bundle Recommendations")
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
        ["white hanging heart t-light holder", "regency cakestand 3 tier"],
        ["world war 2 glued jigsaw puzzle", "playing cards, historical"],
        ["pack of 72 retrospot tea towels", "assorted colour teardrop heart"],
    ]

    # Get recommendations
    for transaction in transactions:
        logger.info(f"\nCustomer Transaction: {transaction}")

        # Get recommendations using ensemble (all models)
        recs = engine.recommend_bundles(transaction, threshold=0.3)
        logger.info(format_recommendations(recs))

        # Get recommendations using specific model
        recs_nb = engine.recommend_bundles(transaction, recommender_name="naive_bayes", threshold=0.3)
        logger.info(f"Naive Bayes Confidence: {recs_nb['confidence']:.2%}")

        recs_svm = engine.recommend_bundles(transaction, recommender_name="svm", threshold=0.3)
        logger.info(f"SVM Confidence: {recs_svm['confidence']:.2%}")


if __name__ == "__main__":
    example_basic_recommendations()
