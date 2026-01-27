# Data Splitting Reference

## Overview

The recommendation engine supports flexible data splitting for model training and evaluation:

- **Single Random Split**: Quick training with configurable train-test ratio (e.g., 80/20)
- **K-Fold Cross-Validation**: Robust evaluation with mean and standard deviation metrics across folds

## Classes

### RandomSplit
Located in [src/data_splitter.py](src/data_splitter.py)

Single random train-test split strategy. Useful for quick iteration during development and hyperparameter tuning.

**Parameters:**
- `test_size`: Fraction of data for testing (default: 0.2)
- `random_state`: Random seed for reproducibility (default: 42)

**Key Methods:**
- `split(X, y)`: Generator yielding single (X_train, X_test, y_train, y_test) tuple
- `get_split_info()`: Returns dictionary with splitting strategy metadata

### KFoldSplit
Located in [src/data_splitter.py](src/data_splitter.py)

K-fold cross-validation splitting strategy. Provides robust statistical evaluation with variance estimates.

**Parameters:**
- `n_splits`: Number of folds (default: 10, must be ≥ 2)
- `random_state`: Random seed for reproducibility (default: 42)
- `shuffle`: Whether to shuffle data before splitting (default: True)

**Key Methods:**
- `split(X, y)`: Generator yielding (X_train, X_test, y_train, y_test) tuples for each fold
- `get_split_info()`: Returns dictionary with splitting strategy metadata

### BundleDataPreprocessor
Located in [src/data_splitter.py](src/data_splitter.py)

Utility for encoding transactions and generating labels from bundles.

**Key Methods:**
- `preprocess(transactions, bundles)`: Fits encoder and returns (X, y) with features and binary labels
- `transform(transactions)`: Applies fitted encoder to new data
- `get_feature_names()`: Returns array of feature names

## Engine Methods

### BundleRecommendationEngine

Located in [src/recommendation_engine.py](src/recommendation_engine.py)

**fit_all_with_random_split(transactions, bundles, test_size=0.2)**
- Trains all recommenders using single random split
- Returns: Dictionary mapping recommender names to metrics

**fit_all_with_kfold(transactions, bundles, n_splits=10)**
- Trains all recommenders using k-fold cross-validation
- Returns: Dictionary mapping recommender names to aggregated metrics with std values

**fit_all(transactions, bundles, validation_split=0.8)**
- Original method for backward compatibility
- Trains all recommenders using fixed train-test split

## Returned Metrics

### Random Split Metrics
```
{
  'model_name': {
    'accuracy': float,
    'precision': float,
    'recall': float,
    'f1': float
  }
}
```

### K-Fold Metrics
```
{
  'model_name': {
    'accuracy': float,          # mean across folds
    'precision': float,
    'recall': float,
    'f1': float,
    'std_accuracy': float,      # standard deviation
    'std_precision': float,
    'std_recall': float,
    'std_f1': float,
    'n_splits': int
  }
}
```

## Usage Patterns

**Quick Model Testing**
- Use `RandomSplit` with small test_size for rapid iteration
- Fast training, minimal memory overhead

**Hyperparameter Tuning**
- Use `RandomSplit` to quickly evaluate different configurations
- Switch to `KFoldSplit` for final parameter selection

**Production Evaluation**
- Use `KFoldSplit` with n_splits=10 for robust metrics
- Report mean ± std metrics for reliability
- Variance indicates model stability

**Model Comparison**
- Train models with same splitter for fair comparison
- Use `KFoldSplit` to account for dataset variance

## Best Practices

1. **Use fixed random_state** for reproducible results
2. **Use KFoldSplit for final evaluation** to get statistical significance
3. **Monitor standard deviation** in k-fold results - high variance indicates model instability
4. **Use RandomSplit during development** for speed
5. **Ensure balanced data** in folds for classification tasks

## Examples

See [examples_kfold_validation.py](examples_kfold_validation.py) for complete working examples:
- Single random split training
- 10-fold cross-validation training
- Results comparison
- Recommendation generation with trained models

## Tests

Comprehensive test suite in [tests/test_data_splitter.py](tests/test_data_splitter.py):
- RandomSplit initialization and reproducibility
- KFoldSplit fold generation and validation
- BundleDataPreprocessor encoding and transformation
- Edge cases and error handling

Run tests:
```bash
pytest tests/test_data_splitter.py -v
```

## Performance Notes

| Aspect | Random Split | K-Fold (10) |
|--------|-------------|-----------|
| Training Time | 1x | ~10x slower |
| Statistical Robustness | Low | High |
| Memory Usage | Low | Low-Moderate |
| Use Case | Development | Production |

## See Also

- [README.md](README.md) - Project overview
- [examples_kfold_validation.py](examples_kfold_validation.py) - Complete working examples
- [src/data_splitter.py](src/data_splitter.py) - Implementation details
- [src/recommendation_engine.py](src/recommendation_engine.py) - Engine integration
- [tests/test_data_splitter.py](tests/test_data_splitter.py) - Test examples
