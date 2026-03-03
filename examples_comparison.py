"""
Example 3: Model Comparison

This example demonstrates how to compare recommendations from different
algorithms (Naive Bayes vs SVM) on the same transactions.
"""

import logging
import os
import pandas as pd
from src.config import load_config
from src.recommendation_engine import BundleRecommendationEngine
from src.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def example_model_comparison():
    """Example: Compare different recommendation models."""

    logger.info("Example 3: Model Comparison")
    logger.info("=" * 60)

    # Load trained model
    models_path = os.path.join(os.path.dirname(__file__), "models", "recommendation_engine.pkl")

    if not os.path.exists(models_path):
        logger.error(f"Trained model not found at {models_path}")
        logger.info("Please train the model first using: python main.py --train")
        return

    pipeline_config, engine_config, _, _, _, _ = load_config()
    engine = BundleRecommendationEngine(
        engine_config=engine_config,
        pipeline_config=pipeline_config,
    )
    if not engine.load_model(models_path):
        logger.error("Failed to load model")
        return

    # Example transactions
    transactions = [
        ["white hanging heart t-light holder", "regency cakestand 3 tier"],
        ["world war 2 glued jigsaw puzzle", "playing cards, historical"],
        ["pack of 72 retrospot tea towels"],
    ]

    # Compare models
    comparison_data = []

    for transaction in transactions:
        logger.info(f"\nTransaction: {', '.join(transaction[:2])}")

        rec_nb = engine.recommend_bundles(
            transaction, recommender_name="naive_bayes", threshold=0.0
        )
        rec_svm = engine.recommend_bundles(
            transaction, recommender_name="svm", threshold=0.0
        )
        rec_ensemble = engine.recommend_bundles(
            transaction, threshold=0.0
        )

        comparison_data.append(
            {
                "Transaction": ", ".join(transaction),
                "Naive_Bayes": f"{rec_nb['confidence']:.2%}",
                "SVM": f"{rec_svm['confidence']:.2%}",
                "Ensemble": f"{rec_ensemble['confidence']:.2%}",
            }
        )

        logger.info(f"  Naive Bayes Confidence:  {rec_nb['confidence']:.2%}")
        logger.info(f"  SVM Confidence:          {rec_svm['confidence']:.2%}")
        logger.info(f"  Ensemble Confidence:     {rec_ensemble['confidence']:.2%}")

    # Display comparison table
    logger.info("\n" + "=" * 60)
    logger.info("Confidence Comparison Table:")
    logger.info("=" * 60)

    df_comparison = pd.DataFrame(comparison_data)
    logger.info("\n" + df_comparison.to_string(index=False))


if __name__ == "__main__":
    example_model_comparison()
