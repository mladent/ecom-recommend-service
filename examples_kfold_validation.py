"""
Example demonstrating k-fold cross-validation for the recommendation engine.

This example shows how to:
1. Load processed data
2. Train models using 10-fold cross-validation
3. Compare results with random split
4. Display cross-validation statistics

Usage:
    python examples_kfold_validation.py              # Run with Naive Bayes only (fast)
    python examples_kfold_validation.py --all        # Run with all models (slow)
    python examples_kfold_validation.py --svm        # Run with SVM only
    python examples_kfold_validation.py --naive-bayes # Run with Naive Bayes only
    python examples_kfold_validation.py --quick      # Quick demo with 20% of data
    python examples_kfold_validation.py --quick --all # Quick demo with all models
"""

import logging
import sys
from src.config import load_config
from src.recommendation_engine import (
    BundleRecommendationEngine,
    NaiveBayesBundleRecommender,
    SVMBundleRecommender,
)
from src.data_pipeline import DataPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def format_metrics_table(metrics_dict: dict, enabled_models: dict) -> str:
    """
    Format metrics dictionary as a markdown-style table with separate std columns.
    
    Args:
        metrics_dict: Dictionary mapping model names to their metrics
        enabled_models: Dictionary mapping model names to boolean (enabled/disabled)
    
    Returns:
        Formatted markdown table string
    """
    # Filter to only enabled models
    enabled_model_names = [name for name, enabled in enabled_models.items() if enabled and name in metrics_dict]
    
    if not enabled_model_names:
        return "No enabled models to display."
    
    # Define metrics columns (excluding n_splits)
    metric_columns = ["accuracy", "precision", "recall", "f1"]
    
    # Build header
    header = "| Model"
    for metric in metric_columns:
        header += f" | {metric.capitalize()} | Std {metric.capitalize()}"
    header += " |"
    
    # Build separator
    separator = "|" + "|-" * (1 + len(metric_columns) * 2) + "|"
    
    # Build rows
    rows = []
    for model_name in enabled_model_names:
        model_metrics = metrics_dict[model_name]
        row = f"| {model_name}"
        for metric in metric_columns:
            value = model_metrics.get(metric, 0.0)
            std_key = f"std_{metric}"
            std_value = model_metrics.get(std_key, 0.0)
            row += f" | {value:.4f} | {std_value:.4f}"
        row += " |"
        rows.append(row)
    
    return "\n".join([header, separator] + rows)


def format_comparison_table(metrics_random: dict, metrics_kfold: dict, enabled_models: dict) -> str:
    """
    Format side-by-side comparison of random split vs k-fold cross-validation results.
    
    Args:
        metrics_random: Metrics from random split
        metrics_kfold: Metrics from k-fold cross-validation
        enabled_models: Dictionary mapping model names to boolean (enabled/disabled)
    
    Returns:
        Formatted markdown table string
    """
    # Filter to only enabled models that appear in both results
    enabled_model_names = [
        name for name, enabled in enabled_models.items()
        if enabled and name in metrics_random and name in metrics_kfold
    ]
    
    if not enabled_model_names:
        return "No enabled models to compare."
    
    # Build header
    header = "| Model | Random Accuracy | K-Fold Accuracy | K-Fold Std | Difference |"
    separator = "|---|---|---|---|---|"
    
    rows = []
    for model_name in enabled_model_names:
        random_acc = metrics_random[model_name].get("accuracy", 0.0)
        kfold_acc = metrics_kfold[model_name].get("accuracy", 0.0)
        kfold_std = metrics_kfold[model_name].get("std_accuracy", 0.0)
        diff = abs(random_acc - kfold_acc)
        row = f"| {model_name} | {random_acc:.4f} | {kfold_acc:.4f} | {kfold_std:.4f} | {diff:.4f} |"
        rows.append(row)
    
    return "\n".join([header, separator] + rows)


def parse_arguments() -> tuple[dict, bool]:
    """
    Parse command line arguments to determine which models to run and data size.
    
    Returns:
        Tuple of (models_config dict, quick_mode bool)
    """
    default_config = {"naive_bayes": True, "svm": False}
    quick_mode = False
    
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            arg = arg.lower()
            if arg == "--quick":
                quick_mode = True
            elif arg == "--all":
                default_config = {"naive_bayes": True, "svm": True}
            elif arg == "--svm":
                default_config = {"naive_bayes": False, "svm": True}
            elif arg == "--naive-bayes":
                default_config = {"naive_bayes": True, "svm": False}
            else:
                logger.warning(f"Unknown argument: {arg}. Using defaults (Naive Bayes only)")
    
    return default_config, quick_mode


def main():
    """Main execution function."""
    # Parse command line arguments
    models_config, quick_mode = parse_arguments()
    
    logger.info("=" * 80)
    logger.info("E-Commerce Bundle Recommendation - K-Fold Cross-Validation Example")
    logger.info("=" * 80)
    logger.info(f"Running with: Naive Bayes={models_config['naive_bayes']}, SVM={models_config['svm']}")
    if quick_mode:
        logger.info("Quick demo mode: Using 0.5% of data for fastest execution")

    pipeline_config, engine_config, _, _, _ = load_config()

    # Load processed data
    logger.info("\nLoading processed data...")
    pipeline = DataPipeline(config=pipeline_config)
    pipeline.load_raw_data()
    pipeline.preprocess()
    transactions = pipeline.create_transaction_baskets()

    if transactions is None:
        logger.error("Failed to load data. Please run: python main.py --prepare")
        return

    # Apply quick mode sampling BEFORE bundle generation (most time-consuming step)
    if quick_mode:
        import random
        random.seed(42)
        sample_size = max(25, len(transactions) // 200)  # Use 0.5% or minimum 25
        sampled_indices = random.sample(range(len(transactions)), min(sample_size, len(transactions)))
        transactions = transactions.iloc[sampled_indices].reset_index(drop=True)
        # Update pipeline's internal transactions so bundle generation uses the sample
        pipeline.transactions = transactions
        logger.info(f"Sampled {len(transactions)} transactions for quick demo")

    # Now generate bundles on sampled data (much faster)
    bundles = pipeline.generate_product_bundles() if not quick_mode else pipeline.generate_product_bundles(max_size=2)

    if bundles is None:
        logger.error("Failed to generate bundles. Consider lowering MIN_SUPPORT and MIN_CONFIDENCE in .env file.")
        return

    # Convert to required format
    transaction_items = [list(items) for items in transactions["Items"].values]
    bundle_list = [tuple(b) for b in bundles]

    if not bundle_list:
        error_msg = (
            f"No bundles found (sampled {len(transaction_items)} transactions). "
            "To generate bundles with smaller datasets:\n"
            "  1. Edit .env file\n"
            "  2. Lower MIN_SUPPORT (try 0.000298) and MIN_CONFIDENCE (try 0.07)\n"
            "  3. See .env-template comments for recommended values\n"
            "  4. Run again without --quick flag for full dataset, or adjust --quick sampling"
        )
        logger.error(error_msg)
        return

    logger.info(f"Loaded {len(transaction_items)} transactions and {len(bundle_list)} bundles")

    # ========================================================================
    # Example 1: Single Random Split
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 1: Single Random Split (80/20)")
    logger.info("=" * 80)

    engine1 = BundleRecommendationEngine(
        engine_config=engine_config,
        pipeline_config=pipeline_config,
    )
    if models_config["naive_bayes"]:
        engine1.add_recommender(
            "naive_bayes",
            NaiveBayesBundleRecommender(
                config=engine_config,
                default_validation_split=pipeline_config.train_test_split,
            ),
        )
    if models_config["svm"]:
        engine1.add_recommender(
            "svm",
            SVMBundleRecommender(
                config=engine_config,
                default_validation_split=pipeline_config.train_test_split,
            ),
        )

    logger.info("\nTraining with single random split...")
    metrics1 = engine1.fit_all_with_random_split(transaction_items, bundle_list, test_size=0.2)

    logger.info("\nRandom Split Results:")
    for model_name, model_metrics in metrics1.items():
        logger.info(f"\n{model_name}:")
        for metric_name, metric_value in model_metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")
    
    # ========================================================================
    # Summary Table: Random Split Results
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY TABLE: Random Split Metrics")
    logger.info("=" * 80)
    logger.info("\n" + format_metrics_table(metrics1, models_config))

    # ========================================================================
    # Example 2: 10-Fold Cross-Validation
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 2: 10-Fold Cross-Validation")
    logger.info("=" * 80)

    engine2 = BundleRecommendationEngine(
        engine_config=engine_config,
        pipeline_config=pipeline_config,
    )
    if models_config["naive_bayes"]:
        engine2.add_recommender(
            "naive_bayes",
            NaiveBayesBundleRecommender(
                config=engine_config,
                default_validation_split=pipeline_config.train_test_split,
            ),
        )
    if models_config["svm"]:
        engine2.add_recommender(
            "svm",
            SVMBundleRecommender(
                config=engine_config,
                default_validation_split=pipeline_config.train_test_split,
            ),
        )

    logger.info("\nTraining with 10-fold cross-validation...")
    metrics2 = engine2.fit_all_with_kfold(transaction_items, bundle_list, n_splits=10)

    logger.info("\n10-Fold Cross-Validation Results:")
    for model_name, model_metrics in metrics2.items():
        logger.info(f"\n{model_name}:")
        for metric_name, metric_value in model_metrics.items():
            if metric_name.startswith("std_"):
                logger.info(f"  {metric_name}: {metric_value:.4f}")
            elif metric_name == "n_splits":
                logger.info(f"  {metric_name}: {metric_value}")
            else:
                # Print mean ± std for main metrics
                std_key = f"std_{metric_name}"
                if std_key in model_metrics:
                    std_value = model_metrics[std_key]
                    logger.info(f"  {metric_name}: {metric_value:.4f} (± {std_value:.4f})")
                else:
                    logger.info(f"  {metric_name}: {metric_value:.4f}")
    
    # ========================================================================
    # Summary Table: K-Fold Cross-Validation Results
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY TABLE: 10-Fold Cross-Validation Metrics")
    logger.info("=" * 80)
    logger.info("\n" + format_metrics_table(metrics2, models_config))

    # ========================================================================
    # Example 3: Comparison Summary Table
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON: Random Split vs K-Fold Cross-Validation")
    logger.info("=" * 80)
    logger.info("\n" + format_comparison_table(metrics1, metrics2, models_config))

    # ========================================================================
    # Example 4: Testing Recommendations
    # ========================================================================
    if models_config["naive_bayes"] or models_config["svm"]:
        logger.info("\n" + "=" * 80)
        logger.info("EXAMPLE 4: Getting Recommendations with Trained Models")
        logger.info("=" * 80)

        # Get a sample transaction
        sample_transaction = transaction_items[0][:3]  # Use first 3 items
        logger.info(f"\nSample transaction items: {sample_transaction}")

        logger.info("\nRecommendations from Random Split model:")
        recs1 = engine1.recommend_bundles(sample_transaction, threshold=0.3)
        logger.info(f"  Confidence: {recs1['confidence']:.4f}")
        logger.info(f"  Number of bundles recommended: {len(recs1['bundles'])}")
        if recs1['bundles']:
            logger.info(f"  Sample bundle: {recs1['bundles'][0]}")

        logger.info("\nRecommendations from K-Fold model:")
        recs2 = engine2.recommend_bundles(sample_transaction, threshold=0.3)
        logger.info(f"  Confidence: {recs2['confidence']:.4f}")
        logger.info(f"  Number of bundles recommended: {len(recs2['bundles'])}")
        if recs2['bundles']:
            logger.info(f"  Sample bundle: {recs2['bundles'][0]}")

    logger.info("\n" + "=" * 80)
    logger.info("Example completed successfully!")

    logger.info("=" * 80)


if __name__ == "__main__":
    main()
