"""Main entry point for the e-commerce recommendation service."""

import os
import sys
import csv
import json
import logging
import argparse
import subprocess
import hashlib
import tempfile
from pathlib import Path

from src.config import validate_config, RAW_DATA_PATH, PROCESSED_DATA_PATH, SVM_KERNEL, SVM_C, MODELS_PATH
from src.data_pipeline import DataPipeline
from src.data_evaluator import DataEvaluator
from src.recommendation_engine import (
    BundleRecommendationEngine,
    NaiveBayesBundleRecommender,
    SVMBundleRecommender,
)
from src.utils import setup_logging, format_recommendations, init_mlflow_tracking
from src.llm_client import LLMOperationTracker

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


def train_recommenders(pipeline: DataPipeline, mlflow_tracker=None):
    """Train recommendation engine.
    
    Args:
        pipeline: DataPipeline instance with processed data
        mlflow_tracker: Optional MLflowExperimentTracker for experiment logging
    """
    logger.info("Training recommendation engine...")

    # Validate pipeline data
    if pipeline.transactions is None:
        logger.error("Pipeline has no transactions. Please run data preparation first.")
        logger.info("Run: python main.py --prepare")
        return None
    
    if pipeline.bundles is None:
        logger.error("Pipeline has no bundles. Please run data preparation first.")
        logger.info("Run: python main.py --prepare")
        return None

    # Prepare data
    transactions = pipeline.transactions["Items"].tolist()
    bundles = pipeline.bundles

    if not transactions or not bundles:
        logger.error("No transactions or bundles available")
        return None

    # Initialize engine with MLflow tracker
    engine = BundleRecommendationEngine(mlflow_tracker=mlflow_tracker)

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
    
    # Log model artifact to MLflow
    if mlflow_tracker and mlflow_tracker.enabled:
        mlflow_tracker.log_artifact(model_file, "models")

        # Log dataset statistics and data lineage
        unique_items = sorted({item for trans in transactions for item in trans})
        dataset_preview = []
        if pipeline.transactions is not None and "Items" in pipeline.transactions:
            dataset_preview = pipeline.transactions["Items"].head(20).tolist()

        transactions_digest = hashlib.sha256(
            json.dumps(transactions, sort_keys=False).encode("utf-8")
        ).hexdigest()

        dataset_stats = {
            "n_transactions": len(transactions),
            "n_bundles": len(bundles),
            "n_unique_items": len(unique_items),
            "feature_count": len(unique_items),
            "dataset_hash_sha256": transactions_digest,
            "preprocessing_steps": [
                "convert_csv_to_tsv",
                "load_raw_data",
                "preprocess",
                "create_transaction_baskets",
                "generate_product_bundles",
            ],
        }
        mlflow_tracker.log_dict(dataset_stats, "dataset_stats.json")

        bundle_size_distribution = {}
        for bundle in bundles:
            size_key = str(len(bundle))
            bundle_size_distribution[size_key] = bundle_size_distribution.get(size_key, 0) + 1

        bundle_stats = {
            "bundle_count": len(bundles),
            "bundle_size_distribution": bundle_size_distribution,
            "max_bundle_size": max((len(bundle) for bundle in bundles), default=0),
            "min_bundle_size": min((len(bundle) for bundle in bundles), default=0),
            "avg_bundle_size": (
                sum(len(bundle) for bundle in bundles) / len(bundles)
                if bundles
                else 0.0
            ),
        }
        mlflow_tracker.log_dict(bundle_stats, "bundle_stats.json")

        mlflow_tracker.log_dict(
            {
                "sample_size": len(dataset_preview),
                "items_preview": dataset_preview,
            },
            "dataset_snapshot.json",
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix="_evaluation.csv", delete=False, newline="") as tmp_csv:
            writer = csv.writer(tmp_csv)
            writer.writerow(["model", "accuracy", "precision", "recall", "f1", "roc_auc"])
            for model_name, metric_dict in metrics.items():
                writer.writerow(
                    [
                        model_name,
                        metric_dict.get("accuracy", 0.0),
                        metric_dict.get("precision", 0.0),
                        metric_dict.get("recall", 0.0),
                        metric_dict.get("f1", 0.0),
                        metric_dict.get("roc_auc", 0.0),
                    ]
                )
            evaluation_csv_path = tmp_csv.name
        mlflow_tracker.log_artifact(evaluation_csv_path, "evaluation")
        try:
            os.remove(evaluation_csv_path)
        except OSError:
            logger.debug(f"Could not remove temporary file: {evaluation_csv_path}")
        
        # Log aggregated LLM operation metrics
        llm_tracker = LLMOperationTracker()
        llm_metrics = llm_tracker.to_mlflow_metrics()
        if llm_metrics:
            logger.info(f"Logging {len(llm_metrics)} LLM operation metrics to MLflow...")
            mlflow_tracker.log_metrics(llm_metrics)
            
            # Log operation-level details in MLflow params for reference
            llm_params = llm_tracker.to_mlflow_params()
            if llm_params:
                mlflow_tracker.log_params(llm_params)

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
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Enable MLflow experiment tracking for training runs",
    )
    parser.add_argument(
        "--mlflow-ui",
        action="store_true",
        help="Launch MLflow UI after training (implies --mlflow)",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)

    logger.info("E-Commerce Bundle Recommendation Service")
    logger.info("=" * 60)

    # Validate configuration
    validate_config()
    
    # Initialize MLflow tracker if requested
    mlflow_tracker = None
    mlflow_tracking_uri = "mlruns"
    if args.mlflow or args.mlflow_ui:
        mlflow_tracker = init_mlflow_tracking(enabled_override=True)
        if mlflow_tracker and mlflow_tracker.config is not None:
            mlflow_tracking_uri = mlflow_tracker.config.tracking_uri

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
        
        # Wrap training in MLflow run if enabled
        if mlflow_tracker and mlflow_tracker.enabled:
            with mlflow_tracker.start_run(run_name="training_run"):
                mlflow_tracker.set_tag("pipeline_stage", "training")
                mlflow_tracker.set_tag("model_type", "bundle_recommendation")
                engine = train_recommenders(pipeline, mlflow_tracker=mlflow_tracker)
        else:
            engine = train_recommenders(pipeline)
        
        if not engine:
            return 1
        
        # Launch MLflow UI if requested
        if args.mlflow_ui:
            logger.info("Launching MLflow UI...")
            logger.info("Access the UI at: http://127.0.0.1:5000")
            logger.info("Press Ctrl+C to stop the UI server")
            try:
                subprocess.run(["mlflow", "ui", "--backend-store-uri", mlflow_tracking_uri])
            except KeyboardInterrupt:
                logger.info("MLflow UI stopped")
            except FileNotFoundError:
                logger.error("MLflow CLI not found. Ensure mlflow is installed: pip install mlflow")

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
