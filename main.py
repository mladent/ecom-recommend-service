"""Main entry point for the e-commerce recommendation service."""

import os
import sys
import logging
import argparse
from pathlib import Path

from src.config import validate_config, RAW_DATA_PATH, PROCESSED_DATA_PATH
from src.data_pipeline import DataPipeline
from src.recommendation_engine import (
    BundleRecommendationEngine,
    NaiveBayesBundleRecommender,
    SVMBundleRecommender,
)
from src.utils import setup_logging, format_recommendations

logger = logging.getLogger(__name__)


def download_data():
    """Download dataset from Kaggle."""
    logger.info("Starting data download...")
    pipeline = DataPipeline()

    if not pipeline.download_kaggle_data():
        logger.error("Failed to download data from Kaggle")
        logger.info("Please download manually from: https://www.kaggle.com/datasets/carrie1/ecommerce-data")
        return False

    return True


def prepare_data(force_reprocess: bool = False):
    """Prepare and process data."""
    logger.info("Preparing data...")

    # Check if processed data exists
    if os.path.exists(PROCESSED_DATA_PATH) and not force_reprocess:
        logger.info(f"Processed data found at {PROCESSED_DATA_PATH}")
        pipeline = DataPipeline()
        if pipeline.load_processed_data(PROCESSED_DATA_PATH):
            logger.info("Using cached processed data")
            return pipeline

    # Process raw data
    pipeline = DataPipeline()

    if not os.path.exists(RAW_DATA_PATH):
        logger.error(f"Raw data not found at {RAW_DATA_PATH}")
        logger.info("Please download data first using: python main.py --download")
        return None

    logger.info("Loading raw data...")
    pipeline.load_raw_data()

    logger.info("Exploring data...")
    stats = pipeline.explore_data()
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")

    logger.info("Preprocessing data...")
    pipeline.preprocess()

    logger.info("Creating transaction baskets...")
    pipeline.create_transaction_baskets()

    logger.info("Generating bundles...")
    pipeline.generate_product_bundles()

    logger.info("Saving processed data...")
    pipeline.save_processed_data()

    bundle_stats = pipeline.get_bundle_statistics()
    logger.info(f"Bundle statistics: {bundle_stats}")

    return pipeline


def train_recommenders(pipeline: DataPipeline):
    """Train recommendation engine."""
    logger.info("Training recommendation engine...")

    # Prepare data
    transactions = pipeline.transactions["Items"].tolist()
    bundles = pipeline.bundles

    if not transactions or not bundles:
        logger.error("No transactions or bundles available")
        return None

    # Initialize engine
    engine = BundleRecommendationEngine()

    # Add Naive Bayes recommender
    logger.info("Adding Naive Bayes recommender...")
    nb_recommender = NaiveBayesBundleRecommender(model_type="multinomial")
    engine.add_recommender("naive_bayes", nb_recommender)

    # Add SVM recommender with different kernels
    logger.info("Adding SVM recommender...")
    svm_recommender = SVMBundleRecommender(kernel="rbf", C=1.0)
    engine.add_recommender("svm_rbf", svm_recommender)

    # Train all recommenders
    logger.info("Fitting models...")
    metrics = engine.fit_all(transactions, bundles)

    logger.info("Training metrics:")
    for recommender_name, metric_dict in metrics.items():
        logger.info(f"  {recommender_name}:")
        for metric, value in metric_dict.items():
            logger.info(f"    {metric}: {value:.4f}")

    # Save model
    logger.info("Saving trained model...")
    models_path = os.path.join(os.path.dirname(__file__), "..", "models", "recommendation_engine.pkl")
    engine.save_model(models_path)

    logger.info(f"Engine statistics: {engine.get_engine_stats()}")

    return engine


def demo_recommendations(engine: BundleRecommendationEngine):
    """Run demo recommendations."""
    logger.info("Running demo recommendations...")

    # Example transactions
    sample_transactions = [
        ["white hanging heart t-light holder", "regency cakestand 3 tier"],
        ["world war 2 glued jigsaw puzzle", "playing cards, historical"],
        ["pack of 72 retrospot tea towels", "assorted colour teardrop heart"],
    ]

    for transaction in sample_transactions:
        logger.info(f"\nProcessing transaction: {transaction}")

        # Get recommendations
        recs = engine.recommend_bundles(transaction, threshold=0.3)
        logger.info(format_recommendations(recs))

        # Get cross-sell products
        cross_sell = engine.get_cross_sell_products(transaction, top_n=3)
        if cross_sell:
            logger.info(f"Cross-sell suggestions:")
            for product, score in cross_sell:
                logger.info(f"  - {product} (score: {score:.2f})")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="E-commerce Bundle Recommendation Service")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download dataset from Kaggle",
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Prepare and process data",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train recommendation models",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo recommendations",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run complete pipeline (download, prepare, train, demo)",
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Force reprocessing of data even if cached version exists",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)

    logger.info("E-Commerce Bundle Recommendation Service")
    logger.info("=" * 60)

    # Validate configuration
    validate_config()

    # Execute pipeline
    if args.download or args.full:
        if not download_data():
            return 1

    if args.prepare or args.full:
        pipeline = prepare_data(force_reprocess=args.reprocess)
        if not pipeline:
            return 1

    if args.train or args.full:
        pipeline = prepare_data()  # Ensure data is prepared
        if not pipeline:
            return 1
        engine = train_recommenders(pipeline)
        if not engine:
            return 1

    if args.demo or args.full:
        # Load trained engine
        models_path = os.path.join(os.path.dirname(__file__), "..", "models", "recommendation_engine.pkl")
        if not os.path.exists(models_path):
            logger.error(f"Trained model not found at {models_path}")
            logger.info("Please train the model first using: python main.py --train")
            return 1

        engine = BundleRecommendationEngine()
        engine.load_model(models_path)
        demo_recommendations(engine)

    if not any([args.download, args.prepare, args.train, args.demo, args.full]):
        logger.info("No action specified. Use --help for options.")
        logger.info("Quick start: python main.py --full")
        return 0

    logger.info("=" * 60)
    logger.info("Service completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
