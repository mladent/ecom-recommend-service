"""Main entry point for the e-commerce recommendation service."""

import os
import sys
import logging
import argparse
from pathlib import Path

from src.config import validate_config, RAW_DATA_PATH, PROCESSED_DATA_PATH, SVM_KERNEL, SVM_C, MODELS_PATH
from src.data_pipeline import DataPipeline
from src.data_evaluator import DataEvaluator
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
        logger.info(f"Processed data cache found at: {PROCESSED_DATA_PATH}")
        pipeline = DataPipeline(force_reprocess=False)
        if pipeline.load_processed_data(PROCESSED_DATA_PATH):
            logger.info("Using cached processed data - skipping all preprocessing and LLM operations")
            return pipeline

    # Process raw data (bypass all LLM caches if reprocessing)
    if force_reprocess:
        logger.info("Force reprocessing enabled - bypassing all caches")
    pipeline = DataPipeline(force_reprocess=force_reprocess)

    if not os.path.exists(RAW_DATA_PATH):
        logger.error(f"Raw data not found at {RAW_DATA_PATH}")
        logger.info("Please download data first using: python main.py --download")
        return None


    logger.info("Convert csv to tsv, and save...")
    pipeline.convert_csv_to_tsv()

    logger.info("Loading raw data...")
    pipeline.load_raw_data()

    logger.info("Exploring raw data...")
    stats = pipeline.explore_data()
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")

    logger.info("Preprocessing data...")
    pipeline.preprocess()

    logger.info("Exploring processed data...")
    stats = pipeline.explore_data()
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")

    logger.info("Creating transaction baskets...")
    pipeline.create_transaction_baskets()

    logger.info("Generating bundles...")
    pipeline.generate_product_bundles()

    logger.info("Saving processed data...")
    pipeline.save_processed_data()

    bundle_stats = pipeline.get_bundle_statistics()
    logger.info(f"Bundle statistics: {bundle_stats}")

    return pipeline


def regenerate_bundles_only():
    """Regenerate bundles from preprocessed data only.
    
    Reuses cached preprocessed data (processed_data.pkl) and skips all
    preprocessing steps. Useful for experimenting with different bundle
    generation parameters (MIN_SUPPORT, MIN_CONFIDENCE, MAX_BUNDLE_SIZE).
    
    Returns:
        DataPipeline: Pipeline with regenerated bundles or None if failed
    """
    logger.info("Regenerating bundles from preprocessed data...")
    
    if not os.path.exists(PROCESSED_DATA_PATH):
        logger.error(f"Processed data not found at {PROCESSED_DATA_PATH}")
        logger.info("Please prepare data first using: python main.py --prepare")
        return None
    
    try:
        # Load preprocessed data
        pipeline = DataPipeline(force_reprocess=False)
        if not pipeline.load_processed_data(PROCESSED_DATA_PATH):
            logger.error("Failed to load processed data")
            return None
        
        logger.info("Preprocessed data loaded successfully")
        
        # Regenerate bundles
        logger.info("Creating transaction baskets...")
        pipeline.create_transaction_baskets()
        
        logger.info("Generating bundles...")
        pipeline.generate_product_bundles()
        
        logger.info("Saving processed data...")
        pipeline.save_processed_data()
        
        bundle_stats = pipeline.get_bundle_statistics()
        logger.info(f"Bundle statistics: {bundle_stats}")
        
        return pipeline
        
    except Exception as e:
        logger.error(f"Failed to regenerate bundles: {e}")
        logger.exception(e)
        return None


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

    # Add SVM recommender with configured kernel
    logger.info(f"Adding SVM recommender with kernel={SVM_KERNEL}...")
    svm_recommender = SVMBundleRecommender(kernel=SVM_KERNEL, C=SVM_C)
    engine.add_recommender("svm", svm_recommender)

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
    model_file = os.path.join(MODELS_PATH, "recommendation_engine.pkl")
    engine.save_model(model_file)

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


def evaluate_data():
    """Evaluate and analyze raw dataset."""
    logger.info("Starting data evaluation and analysis...")

    # Load raw data
    if not os.path.exists(RAW_DATA_PATH):
        logger.error(f"Raw data not found at {RAW_DATA_PATH}")
        logger.info("Please download data first using: python main.py --download")
        return False

    try:
        import pandas as pd
        logger.info(f"Loading data from {RAW_DATA_PATH}...")
        df = pd.read_csv(RAW_DATA_PATH, encoding="latin1")
        logger.info(f"Loaded {len(df):,} rows and {len(df.columns)} columns")

        # Run evaluation
        evaluator = DataEvaluator(df, data_path="data")
        report = evaluator.evaluate()

        # Save report
        report_path = evaluator.save_report()
        logger.info(f"Data evaluation complete!")
        logger.info(f"Report saved to: {report_path}")
        logger.info(f"Category files saved to: data/categories_*.txt")

        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 80)

        overview = report["dataset_overview"]
        logger.info(f"Rows: {overview['total_rows']:,} | Columns: {overview['total_columns']}")
        logger.info(f"Memory: {overview['memory_usage_mb']:.2f} MB")

        quality = report["data_quality"]
        logger.info(f"\nData Quality Issues:")
        logger.info(f"  Duplicate rows: {quality['duplicate_rows']}")
        logger.info(f"  Fully null columns: {len(quality['fully_null_columns'])}")

        missing_count = sum(quality["missing_values"].values())
        logger.info(f"  Total missing values: {missing_count}")

        logger.info("\nColumn Analysis Summary:")
        for col in df.columns:
            col_report = report["columns"][col]
            null_pct = col_report["null_percentage"]
            unique = col_report["unique_count"]
            logger.info(f"  {col:20s}: {unique:6,} unique, {null_pct:5.1f}% missing")

        logger.info("=" * 80)
        return True

    except Exception as e:
        logger.error(f"Error during data evaluation: {e}")
        logger.exception(e)
        return False


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
        "--evaluate-data",
        action="store_true",
        help="Evaluate and analyze raw dataset (generates data_report.md)",
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Force reprocessing of data even if cached version exists",
    )
    parser.add_argument(
        "--bundles-only",
        action="store_true",
        help="Regenerate bundles from preprocessed data (skips all preprocessing)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="Launch REST API server for bundle recommendations",
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

    if args.bundles_only:
        pipeline = regenerate_bundles_only()
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
        model_file = os.path.join(MODELS_PATH, "recommendation_engine.pkl")
        if not os.path.exists(model_file):
            logger.error(f"Trained model not found at {model_file}")
            logger.info("Please train the model first using: python main.py --train")
            return 1

        engine = BundleRecommendationEngine()
        engine.load_model(model_file)
        demo_recommendations(engine)

    if args.evaluate_data:
        if not evaluate_data():
            return 1

    if args.api:
        logger.info("Launching REST API server...")
        from src.api import app
        # Read port from environment variable with fallback to 5000
        port = int(os.getenv("PORT", "5000"))
        logger.info(f"API server starting on http://0.0.0.0:{port}")
        logger.info(f"Available endpoints:")
        logger.info(f"  GET  /health")
        logger.info(f"  GET  /api/v1/recommenders")
        logger.info(f"  GET  /api/v1/bundles?product_description=X")
        logger.info(f"  POST /api/v1/bundles/batch")
        logger.info(f"  GET  /api/v1/cross-sell?product_description=X")
        logger.info(f"  GET  /api/v1/stats")
        logger.info(f"  GET  /              (Web UI)")
        app.run(host="0.0.0.0", port=port, debug=args.verbose)
        return 0

    if not any([args.download, args.prepare, args.train, args.demo, args.full, args.evaluate_data, args.bundles_only, args.api]):
        logger.info("No action specified. Use --help for options.")
        logger.info("Quick start: python main.py --full")
        return 0

    logger.info("=" * 60)
    logger.info("Service completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
