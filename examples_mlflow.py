"""
MLflow Integration Examples

Demonstrates various MLflow tracking patterns for the bundle recommendation engine.

Usage:
    python examples_mlflow.py --training                    # Basic training run
    python examples_mlflow.py --tuning                      # Hyperparameter tuning
    python examples_mlflow.py --llm-analysis                # LLM cache analysis
    python examples_mlflow.py --comparison                  # Model comparison
    python examples_mlflow.py --training --mlflow-ui        # Launch UI after training
"""

import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path

from src.config import load_config, MODELS_PATH
from src.data_pipeline import DataPipeline
from src.recommendation_engine import (
    BundleRecommendationEngine,
    NaiveBayesBundleRecommender,
    SVMBundleRecommender,
)
from src.mlflow_client import MLflowExperimentTracker
from src.utils import setup_logging

logger = logging.getLogger(__name__)


def basic_training_run(mlflow_tracker, data_path="data/data_sample.csv"):
    """
    Example 1: Basic Training Run with MLflow Logging
    
    Demonstrates:
    - Creating an MLflow run
    - Logging hyperparameters
    - Logging training metrics
    - Logging model artifacts
    """
    logger.info("="*60)
    logger.info("Example 1: Basic Training Run with MLflow")
    logger.info("="*60)
    
    # Load and prepare data
    pipeline = DataPipeline()
    pipeline.load_raw_data(data_path)
    pipeline.preprocess()
    transactions_df = pipeline.create_transaction_baskets()
    bundles = pipeline.generate_product_bundles()
    
    if transactions_df is None or bundles is None:
        logger.error("Failed to load or process data")
        return
    
    transactions = [list(items) for items in transactions_df["Items"].values]
    bundle_list = [tuple(b) for b in bundles]
    
    # Start MLflow run
    with mlflow_tracker.start_run(run_name="basic_training"):
        mlflow_tracker.set_tags({
            "mode": "training",
            "dataset": Path(data_path).name,
            "example": "basic_training_run",
        })
        
        # Initialize engine with MLflow tracking
        engine = BundleRecommendationEngine(mlflow_tracker=mlflow_tracker)
        
        # Add models
        nb_recommender = NaiveBayesBundleRecommender(model_type="multinomial")
        svm_recommender = SVMBundleRecommender(kernel="linear", C=1.0)
        engine.add_recommender("naive_bayes", nb_recommender)
        engine.add_recommender("svm", svm_recommender)
        
        # Train and log metrics
        logger.info("Training models...")
        metrics = engine.fit_all(transactions, bundle_list)
        
        # Log additional dataset info
        mlflow_tracker.log_dict({
            "n_transactions": len(transactions),
            "n_bundles": len(bundle_list),
            "avg_transaction_size": sum(len(t) for t in transactions) / len(transactions),
        }, "dataset_info.json")
        
        # Save and log model artifact
        model_file = os.path.join(MODELS_PATH, "mlflow_example_model.pkl")
        engine.save_model(model_file)
        mlflow_tracker.log_artifact(model_file, "models")
        
        logger.info("Training complete. Metrics logged to MLflow.")
        logger.info(f"Naive Bayes accuracy: {metrics['naive_bayes']['accuracy']:.4f}")
        logger.info(f"SVM accuracy: {metrics['svm']['accuracy']:.4f}")


def hyperparameter_tuning(mlflow_tracker, data_path="data/data_sample.csv"):
    """
    Example 2: Hyperparameter Tuning Sweep
    
    Demonstrates:
    - Running multiple experiments with different hyperparameters
    - Comparing results across runs
    - Finding best configuration
    """
    logger.info("="*60)
    logger.info("Example 2: Hyperparameter Tuning Sweep")
    logger.info("="*60)
    
    # Load data once
    pipeline = DataPipeline()
    pipeline.load_raw_data(data_path)
    pipeline.preprocess()
    transactions_df = pipeline.create_transaction_baskets()
    bundles = pipeline.generate_product_bundles()
    
    if transactions_df is None or bundles is None:
        logger.error("Failed to load or process data")
        return
    
    transactions = [list(items) for items in transactions_df["Items"].values]
    bundle_list = [tuple(b) for b in bundles]
    
    # Define hyperparameter grid
    svm_kernels = ["linear", "rbf", "poly"]
    svm_c_values = [0.1, 1.0, 10.0]
    
    logger.info(f"Testing {len(svm_kernels)} kernels x {len(svm_c_values)} C values = {len(svm_kernels) * len(svm_c_values)} configurations")
    
    best_accuracy = 0
    best_config = {}
    
    for kernel in svm_kernels:
        for c_value in svm_c_values:
            run_name = f"svm_tuning_{kernel}_C{c_value}"
            
            with mlflow_tracker.start_run(run_name=run_name):
                mlflow_tracker.set_tags({
                    "mode": "tuning",
                    "model": "svm",
                    "example": "hyperparameter_tuning",
                })
                
                mlflow_tracker.log_params({
                    "svm_kernel": kernel,
                    "svm_c": c_value,
                })
                
                # Train SVM with these hyperparameters
                engine = BundleRecommendationEngine(mlflow_tracker=mlflow_tracker)
                svm_recommender = SVMBundleRecommender(kernel=kernel, C=c_value)
                engine.add_recommender("svm", svm_recommender)
                
                metrics = engine.fit_all(transactions, bundle_list)
                accuracy = metrics["svm"]["accuracy"]
                
                logger.info(f"  {kernel} kernel, C={c_value}: accuracy={accuracy:.4f}")
                
                # Track best configuration
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_config = {"kernel": kernel, "C": c_value, "accuracy": accuracy}
    
    logger.info(f"\nBest configuration:")
    logger.info(f"  Kernel: {best_config['kernel']}")
    logger.info(f"  C: {best_config['C']}")
    logger.info(f"  Accuracy: {best_config['accuracy']:.4f}")
    logger.info("\nView results in MLflow UI to compare all runs")


def llm_cache_analysis(mlflow_tracker, data_path="data/data_sample.csv"):
    """
    Example 3: LLM Operations and Cache Analysis
    
    Demonstrates:
    - Tracking LLM API calls
    - Logging cache hit rates
    - Monitoring LLM costs and latency
    """
    logger.info("="*60)
    logger.info("Example 3: LLM Cache Analysis")
    logger.info("="*60)
    
    # Load configuration
    pipeline_config, _, _, _, _, _ = load_config()
    
    with mlflow_tracker.start_run(run_name="llm_analysis"):
        mlflow_tracker.set_tags({
            "mode": "llm_analysis",
            "example": "llm_cache_analysis",
        })
        
        # Load data with LLM operations enabled (if configured)
        pipeline = DataPipeline()
        pipeline.load_raw_data(data_path)
        
        # Track LLM enablement
        mlflow_tracker.log_params({
            "normalization_enabled": pipeline_config.normalization_enabled,
            "enrichment_enabled": pipeline_config.enrichment_enabled,
            "outlier_enabled": pipeline_config.outlier_enabled,
            "context_enabled": pipeline_config.context_enabled,
            "llm_provider": pipeline_config.llm_config.provider,
            "llm_model": pipeline_config.llm_config.model,
        })
        
        # Preprocess with potential LLM calls
        logger.info("Preprocessing data (LLM operations may be cached)...")
        pipeline.preprocess()
        
        # Log cache statistics
        cache_stats = {
            "normalization_cache_first": pipeline_config.cache_config.normalization_cache_first,
            "enrichment_cache_first": pipeline_config.cache_config.enrichment_cache_first,
            "outlier_cache_first": pipeline_config.cache_config.outlier_cache_first,
        }
        mlflow_tracker.log_params(cache_stats)
        
        # Simulate LLM metrics (in real usage, these would be tracked during LLM calls)
        simulated_llm_stats = {
            "llm_total_calls": 0,
            "llm_cache_hit_rate": 1.0,  # All cached
            "llm_avg_latency_ms": 0.0,
        }
        mlflow_tracker.log_metrics(simulated_llm_stats)
        
        logger.info("LLM analysis complete.")
        logger.info(f"Normalization cache enabled: {cache_stats['normalization_cache_first']}")
        logger.info(f"Cache hit rate: {simulated_llm_stats['llm_cache_hit_rate']:.2%}")


def model_comparison(mlflow_tracker, data_path="data/data_sample.csv"):
    """
    Example 4: Side-by-Side Model Comparison
    
    Demonstrates:
    - Comparing multiple models in a single run
    - Logging comparative metrics
    - Creating evaluation reports
    """
    logger.info("="*60)
    logger.info("Example 4: Model Comparison")
    logger.info("="*60)
    
    # Load data
    pipeline = DataPipeline()
    pipeline.load_raw_data(data_path)
    pipeline.preprocess()
    transactions_df = pipeline.create_transaction_baskets()
    bundles = pipeline.generate_product_bundles()
    
    if transactions_df is None or bundles is None:
        logger.error("Failed to load or process data")
        return
    
    transactions = [list(items) for items in transactions_df["Items"].values]
    bundle_list = [tuple(b) for b in bundles]
    
    with mlflow_tracker.start_run(run_name="model_comparison"):
        mlflow_tracker.set_tags({
            "mode": "comparison",
            "example": "model_comparison",
        })
        
        # Initialize engine
        engine = BundleRecommendationEngine(mlflow_tracker=mlflow_tracker)
        
        # Add all models
        nb_multinomial = NaiveBayesBundleRecommender(model_type="multinomial")
        nb_gaussian = NaiveBayesBundleRecommender(model_type="gaussian")
        svm_linear = SVMBundleRecommender(kernel="linear", C=1.0)
        svm_rbf = SVMBundleRecommender(kernel="rbf", C=1.0)
        
        engine.add_recommender("nb_multinomial", nb_multinomial)
        engine.add_recommender("nb_gaussian", nb_gaussian)
        engine.add_recommender("svm_linear", svm_linear)
        engine.add_recommender("svm_rbf", svm_rbf)
        
        # Train all models
        logger.info("Training all models for comparison...")
        metrics = engine.fit_all(transactions, bundle_list)
        
        # Create comparison report
        comparison_data = []
        for model_name, model_metrics in metrics.items():
            comparison_data.append({
                "model": model_name,
                "accuracy": model_metrics["accuracy"],
                "precision": model_metrics["precision"],
                "recall": model_metrics["recall"],
                "f1": model_metrics["f1"],
            })
        
        # Log comparison as artifact
        mlflow_tracker.log_dict(comparison_data, "model_comparison.json")
        
        # Print comparison table
        logger.info("\nModel Comparison Results:")
        logger.info("-" * 80)
        logger.info(f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        logger.info("-" * 80)
        for data in comparison_data:
            logger.info(
                f"{data['model']:<20} "
                f"{data['accuracy']:>10.4f} "
                f"{data['precision']:>10.4f} "
                f"{data['recall']:>10.4f} "
                f"{data['f1']:>10.4f}"
            )
        logger.info("-" * 80)
        
        # Identify best model
        best_model = max(comparison_data, key=lambda x: x['f1'])
        logger.info(f"\nBest model (by F1 score): {best_model['model']}")
        mlflow_tracker.set_tag("best_model", best_model['model'])


def main():
    """Main entry point for MLflow examples."""
    parser = argparse.ArgumentParser(
        description="MLflow Integration Examples for Bundle Recommendation Engine"
    )
    
    # Example modes
    parser.add_argument(
        "--training",
        action="store_true",
        help="Run basic training example with MLflow logging",
    )
    parser.add_argument(
        "--tuning",
        action="store_true",
        help="Run hyperparameter tuning sweep example",
    )
    parser.add_argument(
        "--llm-analysis",
        action="store_true",
        help="Run LLM cache analysis example",
    )
    parser.add_argument(
        "--comparison",
        action="store_true",
        help="Run model comparison example",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all examples sequentially",
    )
    
    # Data options
    parser.add_argument(
        "--data",
        type=str,
        default="data/data_sample.csv",
        help="Path to data file (default: data/data_sample.csv)",
    )
    
    # MLflow options
    parser.add_argument(
        "--mlflow-ui",
        action="store_true",
        help="Launch MLflow UI after running examples",
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
    
    # Load MLflow configuration
    _, _, _, _, _, mlflow_config = load_config()
    mlflow_config.enabled = True  # Always enable for examples
    
    # Initialize MLflow tracker
    mlflow_tracker = MLflowExperimentTracker(mlflow_config)
    
    if not mlflow_tracker.enabled:
        logger.error("MLflow not available. Please install: pip install mlflow")
        return 1
    
    logger.info("MLflow Examples - Bundle Recommendation Engine")
    logger.info(f"Experiment: {mlflow_config.experiment_name}")
    logger.info(f"Tracking URI: {mlflow_config.tracking_uri}")
    logger.info("")
    
    # Ensure data directory exists
    if not os.path.exists(args.data):
        logger.error(f"Data file not found: {args.data}")
        logger.info("Please ensure data is prepared first: python main.py --prepare")
        return 1
    
    # Run selected examples
    if args.all:
        basic_training_run(mlflow_tracker, args.data)
        hyperparameter_tuning(mlflow_tracker, args.data)
        llm_cache_analysis(mlflow_tracker, args.data)
        model_comparison(mlflow_tracker, args.data)
    else:
        if args.training:
            basic_training_run(mlflow_tracker, args.data)
        if args.tuning:
            hyperparameter_tuning(mlflow_tracker, args.data)
        if args.llm_analysis:
            llm_cache_analysis(mlflow_tracker, args.data)
        if args.comparison:
            model_comparison(mlflow_tracker, args.data)
        
        if not any([args.training, args.tuning, args.llm_analysis, args.comparison]):
            logger.info("No example specified. Use --help for options.")
            logger.info("\nQuick start:")
            logger.info("  python examples_mlflow.py --training")
            logger.info("  python examples_mlflow.py --comparison")
            logger.info("  python examples_mlflow.py --all")
            return 0
    
    logger.info("\n" + "="*60)
    logger.info("Examples complete!")
    logger.info(f"View results: mlflow ui --backend-store-uri {mlflow_config.tracking_uri}")
    logger.info("="*60)
    
    # Launch MLflow UI if requested
    if args.mlflow_ui:
        logger.info("\nLaunching MLflow UI...")
        logger.info("Press Ctrl+C to stop the UI server")
        try:
            subprocess.run(["mlflow", "ui", "--backend-store-uri", mlflow_config.tracking_uri])
        except KeyboardInterrupt:
            logger.info("MLflow UI stopped")
        except FileNotFoundError:
            logger.error("MLflow CLI not found. Ensure mlflow is installed: pip install mlflow")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
