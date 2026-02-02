# Architecture and Design

## System Overview

The E-Commerce Bundle Recommendation Service is designed as a modular, scalable system for generating product bundle recommendations using machine learning. The architecture supports current functionality while being prepared for future expansion with LLM capabilities.

```
┌─────────────────────────────────────────────────────────────┐
│                     Web/API Layer (Future)                  │
│                  LLM Integration Module (Future)            │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│          Bundle Recommendation Engine                        │
│  ┌──────────────────┬──────────────────┬─────────────────┐  │
│  │  Naive Bayes     │  SVM (RBF/Linear)│ Ensemble Manager│  │
│  └──────────────────┴──────────────────┴─────────────────┘  │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    Data Pipeline                            │
│  ┌──────────────────┬──────────────────┬─────────────────┐  │
│  │  Load & Explore  │ Clean & Process  │ Bundle Generator│  │
│  └──────────────────┴──────────────────┴─────────────────┘  │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│              Kaggle E-Commerce Dataset                      │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Data Pipeline (`src/data_pipeline.py`)

**Responsibility**: Load, clean, and preprocess e-commerce data

**Key Classes**:
- `DataPipeline`: Main class for data operations

**Key Methods**:
- `load_raw_data()`: Load CSV from Kaggle
- `explore_data()`: Generate dataset statistics
- `preprocess()`: Clean and normalize data
- `create_transaction_baskets()`: Group items by transaction
- `generate_product_bundles()`: Discover frequent item combinations
- `save_processed_data()`: Serialize to pickle for caching
- `load_processed_data()`: Deserialize cached data

**Data Flow**:
```
Raw CSV → Exploration → Cleaning → Preprocessing → 
Basket Creation → Bundle Generation → Serialization
```

**Key Algorithms**:
- Apriori-like frequent itemset mining for bundle discovery
- Support and confidence thresholds for filtering

### 2. Recommendation Engine (`src/recommendation_engine.py`)

**Responsibility**: Provide ML-based bundle recommendations

**Architecture Layers**:

#### Base Layer: `BaseRecommender` (Abstract)
Defines the interface for all recommenders:
- `fit(X, y)`: Train the model
- `predict(X)`: Make predictions
- `predict_proba(X)`: Get prediction probabilities

#### Algorithm Layer: Concrete Recommenders

**NaiveBayesBundleRecommender**:
- Uses Multinomial NB (for binary features) or Gaussian NB
- Advantages: Fast training, works well with high-dimensional data
- Training: Converts transactions to binary feature vectors
- Prediction: Estimates probability of bundle recommendation

**SVMBundleRecommender**:
- Uses Support Vector Machines with configurable kernels
- Supports: RBF (default), Linear, Polynomial
- Features: Hyperparameter tuning via C parameter
- Preprocessing: StandardScaler for feature normalization
- Prediction: SVC with probability=True

#### Engine Layer: `BundleRecommendationEngine`
Orchestrates all recommenders:
- Manages multiple recommender instances
- Provides unified prediction interface
- Supports ensemble methods (averaging probabilities)
- Implements cross-sell recommendations

**Key Methods**:
- `add_recommender()`: Register a recommender
- `fit_all()`: Train all recommenders
- `recommend_bundles()`: Get recommendations
- `get_cross_sell_products()`: Find complementary products
- `save_model()`: Persist trained models
- `load_model()`: Load from disk

### 3. Configuration (`src/config.py`)

**Responsibility**: Centralized configuration management

**Features**:
- Loads environment variables from `.env`
- Provides defaults for all configuration
- Manages file paths
- Validates critical settings

**Key Configuration Items**:
- Data paths and file names
- Bundle generation thresholds
- Model hyperparameters
- Training parameters

### 4. Utilities (`src/utils.py`)

**Responsibility**: Common helper functions

**Functions**:
- `setup_logging()`: Configure logging
- `validate_transaction()`: Input validation
- `filter_transaction()`: Clean transaction data
- `format_recommendations()`: Pretty-print results

## Data Flow

### Training Pipeline

```
1. Download Dataset (Kaggle API)
   ↓
2. Load Raw Data (CSV)
   ↓
3. Explore Statistics
   ↓
4. Data Cleaning
   - Remove nulls
   - Filter invalid entries
   - Standardize formats
   ↓
5. Feature Engineering
   - Create baskets
   - Group by transaction
   ↓
6. Bundle Generation
   - Frequent itemset mining
   - Apply support/confidence thresholds
   ↓
7. Model Training
   - Naive Bayes: Direct fit on feature vectors
   - SVM: Scale features, fit with probability
   ↓
8. Model Evaluation
   - Accuracy, Precision, Recall, F1-Score
   ↓
9. Model Serialization (Pickle)
```

### Recommendation Pipeline

```
1. Load Trained Models (Pickle)
   ↓
2. Accept Customer Transaction
   ↓
3. Feature Encoding
   - Use stored MultiLabelBinarizer
   - Convert items to binary vector
   ↓
4. Model Prediction
   - Naive Bayes: Get prediction probability
   - SVM: Scale features, get prediction probability
   - Ensemble: Average all predictions
   ↓
5. Bundle Filtering
   - Apply confidence threshold
   - Find overlapping bundles
   ↓
6. Result Formatting
   - Return top N recommendations
   - Include confidence scores
```

## Algorithm Explanations

### Naive Bayes for Bundle Recommendation

**Approach**:
1. Encode each transaction as binary feature vector (items present/absent)
2. Train classifier to predict: "Is this transaction likely to contain a bundle?"
3. For new transaction: Calculate P(bundle_present | items) using Bayes' theorem

**Advantages**:
- Fast training and prediction
- Handles high-dimensional data well
- Provides probability estimates

**Implementation Details**:
- MultiLabelBinarizer: Converts item lists to sparse binary matrix
- Handles class imbalance via sample weighting

### SVM for Bundle Recommendation

**Approach**:
1. Encode transactions as binary feature vectors
2. Find optimal hyperplane separating bundled vs non-bundled transactions
3. Use kernel trick (RBF) for non-linear decision boundaries

**Advantages**:
- Powerful non-linear decision boundaries
- Effective with smaller datasets
- Hyperparameter tuning allows customization

**Implementation Details**:
- StandardScaler: Normalizes features (critical for SVM)
- Kernel options:
  - RBF: Non-linear, flexible (default)
  - Linear: Fast, interpretable
  - Polynomial: Mid-complexity

### Ensemble Approach

**Strategy**:
- Train multiple models with different algorithms
- Predictions are averaged (soft voting)
- More robust than single model

**Advantages**:
- Reduces overfitting risk
- Combines strengths of multiple algorithms
- Better generalization

## Feature Engineering

### Transaction Representation

**Original Data**:
```python
transactions = [
    ["product_a", "product_b", "product_c"],
    ["product_b", "product_d"],
    ["product_a", "product_e"]
]
```

**After Encoding** (using MultiLabelBinarizer):
```
     a  b  c  d  e
0    1  1  1  0  0
1    0  1  0  1  0
2    1  0  0  0  1
```

### Bundle Discovery

**Input**: Transaction baskets
**Process**:
1. Count item frequencies
2. Filter items below support threshold
3. Generate 2-itemsets from frequent items
4. Recursively generate larger itemsets
5. Apply confidence filtering

**Output**: List of bundles (tuples of products)

## Scalability Considerations

### Current Limitations
- All data loaded into memory
- Sequential processing
- Limited to single-machine execution

### Future Improvements
- Distributed processing (Spark)
- Streaming data support
- Incremental model updates
- Batch prediction API

### Performance Optimization
- Feature caching
- Model versioning
- Parallel model predictions
- Efficient serialization (joblib vs pickle)

## Extension Points

### 1. LLM Integration (Future)

**Planned Modules**:
- `src/llm_bundle_namer.py`: Generate bundle descriptions
- `src/llm_bundle_personalizer.py`: Personalize recommendations
- API integration with OpenAI/Anthropic

**Configuration**:
```yaml
llm:
  enabled: true
  provider: openai
  model: gpt-3.5-turbo
  temperature: 0.7
```

### 2. Additional Algorithms

**Potential Additions**:
- Logistic Regression
- Random Forest
- Gradient Boosting
- Neural Networks
- Collaborative Filtering

**Implementation Pattern**:
```python
class NewRecommender(BaseRecommender):
    def __init__(self, ...):
        super().__init__(name="NewRecommender")
        self.model = ...
    
    def fit(self, transactions, bundles):
        # Implementation
        pass
    
    def predict(self, transactions):
        # Implementation
        pass
```

### 3. API Layer (Future)

**REST API Endpoints**:
```
POST /api/v1/recommend
  - Input: Customer transaction
  - Output: Bundle recommendations

GET /api/v1/engine/stats
  - Output: Engine statistics

POST /api/v1/retrain
  - Input: New training data
  - Output: Updated models
```

## Testing Strategy

### Unit Tests
- Individual component testing
- Mock data usage
- Isolated functionality

### Integration Tests
- End-to-end pipeline
- Model training and prediction
- File I/O operations

### Performance Tests
- Benchmark prediction speed
- Memory usage monitoring
- Scalability testing

## Deployment Considerations

### Model Versioning
- Timestamp-based versioning
- Metadata tracking
- Rollback capability

### Monitoring
- Prediction distribution
- Recommendation acceptance rates
- Model drift detection

### A/B Testing
- Compare recommendations across models
- Measure business impact
- Statistical significance testing

## Data Splitting Strategies

The recommendation engine supports flexible data splitting for model training and evaluation to ensure robust performance metrics and prevent overfitting.

### RandomSplit

Single random train-test split strategy useful for quick iteration during development and hyperparameter tuning.

**Location**: [src/data_splitter.py](src/data_splitter.py)

**Parameters**:
- `test_size`: Fraction of data for testing (default: 0.2)
- `random_state`: Random seed for reproducibility (default: 42)

**Key Methods**:
- `split(X, y)`: Generator yielding single (X_train, X_test, y_train, y_test) tuple
- `get_split_info()`: Returns dictionary with splitting strategy metadata

**Use Cases**:
- Quick model testing with small test_size for rapid iteration
- Hyperparameter tuning to quickly evaluate different configurations
- Development phase before final model evaluation

### KFoldSplit

K-fold cross-validation splitting strategy providing robust statistical evaluation with variance estimates.

**Location**: [src/data_splitter.py](src/data_splitter.py)

**Parameters**:
- `n_splits`: Number of folds (default: 10, must be ≥ 2)
- `random_state`: Random seed for reproducibility (default: 42)
- `shuffle`: Whether to shuffle data before splitting (default: True)

**Key Methods**:
- `split(X, y)`: Generator yielding (X_train, X_test, y_train, y_test) tuples for each fold
- `get_split_info()`: Returns dictionary with splitting strategy metadata

**Use Cases**:
- Production evaluation with n_splits=10 for robust metrics
- Model comparison with same splitter for fair comparison
- Final parameter selection to account for dataset variance

### BundleDataPreprocessor

Utility for encoding transactions and generating labels from bundles.

**Location**: [src/data_splitter.py](src/data_splitter.py)

**Key Methods**:
- `preprocess(transactions, bundles)`: Fits encoder and returns (X, y) with features and binary labels
- `transform(transactions)`: Applies fitted encoder to new data
- `get_feature_names()`: Returns array of feature names

### Engine Training Methods

Located in [src/recommendation_engine.py](src/recommendation_engine.py)

**fit_all_with_random_split(transactions, bundles, test_size=0.2)**
- Trains all recommenders using single random split
- Returns: Dictionary mapping recommender names to metrics
- Metrics include: accuracy, precision, recall, f1

**fit_all_with_kfold(transactions, bundles, n_splits=10)**
- Trains all recommenders using k-fold cross-validation
- Returns: Dictionary mapping recommender names to aggregated metrics with std values
- Metrics include: accuracy, precision, recall, f1 plus std_accuracy, std_precision, std_recall, std_f1, n_splits

**fit_all(transactions, bundles, validation_split=0.8)**
- Original method for backward compatibility
- Trains all recommenders using fixed train-test split

### Returned Metrics Structure

**Random Split Metrics**:
```python
{
  'model_name': {
    'accuracy': float,
    'precision': float,
    'recall': float,
    'f1': float
  }
}
```

**K-Fold Metrics**:
```python
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

### Performance Comparison

| Aspect | Random Split | K-Fold (10) |
|--------|-------------|-----------|
| Training Time | 1x | ~10x slower |
| Statistical Robustness | Low | High |
| Memory Usage | Low | Low-Moderate |
| Use Case | Development | Production |

### Best Practices

1. **Use fixed random_state** for reproducible results
2. **Use KFoldSplit for final evaluation** to get statistical significance
3. **Monitor standard deviation** in k-fold results - high variance indicates model instability
4. **Use RandomSplit during development** for speed
5. **Ensure balanced data** in folds for classification tasks

### Working Example

See [examples_kfold_validation.py](examples_kfold_validation.py) for complete demonstrations:
- Single random split training
- 10-fold cross-validation training
- Results comparison
- Recommendation generation with trained models

### Testing

Comprehensive test suite in [tests/test_data_splitter.py](tests/test_data_splitter.py) covering:
- RandomSplit initialization and reproducibility
- KFoldSplit fold generation and validation
- BundleDataPreprocessor encoding and transformation
- Edge cases and error handling

Run tests: `pytest tests/test_data_splitter.py -v`

## Algorithm Implementations

### 1. Naive Bayes Classifier

**Type**: Probabilistic classifier based on Bayes' theorem

**Approach**:
1. Encode each transaction as binary feature vector (items present/absent)
2. Train classifier to predict: "Is this transaction likely to contain a bundle?"
3. For new transaction: Calculate P(bundle_present | items) using Bayes' theorem

**Variants**:
- **Multinomial NB**: Best for discrete counts and frequency data
- **Gaussian NB**: For continuous feature distributions

**Advantages**:
- Fast training and prediction
- Handles high-dimensional data well
- Provides probability estimates
- Works well with smaller datasets

**Implementation Details**:
- MultiLabelBinarizer: Converts item lists to sparse binary matrix
- Handles class imbalance via sample weighting
- No feature scaling required

**Use Case**: Initial quick recommendations, baseline model

### 2. Support Vector Machine (SVM)

**Type**: Discriminative classifier finding optimal hyperplane

**Approach**:
1. Encode transactions as binary feature vectors
2. Find optimal hyperplane separating bundled vs non-bundled transactions
3. Use kernel trick for non-linear decision boundaries
4. Probability estimates via Platt scaling

**Kernels**:
- **RBF (Radial Basis Function)**: Non-linear, flexible (default) - best for complex patterns
- **Linear**: Fast, interpretable - good for linearly separable data
- **Polynomial**: Mid-complexity non-linear relationships

**Advantages**:
- Powerful non-linear decision boundaries
- Effective with smaller datasets
- Hyperparameter tuning allows customization
- Robust to overfitting with proper regularization

**Implementation Details**:
- StandardScaler: Normalizes features (critical for SVM performance)
- Probability calibration enabled for confidence scores
- Configurable C (regularization) and gamma (kernel coefficient) parameters

**Use Case**: More accurate recommendations with hyperparameter tuning, production model

### 3. Ensemble Method

**Type**: Soft voting ensemble averaging probabilities from multiple models

**Strategy**:
- Train multiple models with different algorithms
- Predictions are averaged (soft voting)
- Each model contributes equally to final prediction

**Advantages**:
- Reduces overfitting risk
- Combines strengths of multiple algorithms
- Better generalization across different data patterns
- More robust than single model

**Implementation**:
```python
ensemble_proba = (nb_proba + svm_proba) / 2
```

**Use Case**: Production recommendations combining complementary algorithm strengths

### 4. Frequent Itemset Mining (Apriori-like)

**Type**: Market basket analysis algorithm for bundle discovery

**Purpose**: Automatically discover product bundles from historical transaction data

**Algorithm**:
1. Count item frequency across all transactions
2. Generate candidate itemsets (pairs, triples, etc.)
3. Filter by minimum support threshold (frequency)
4. Calculate confidence for association rules
5. Filter by minimum confidence threshold

**Parameters**:
- **MIN_SUPPORT** (default: 0.02): Minimum percentage of transactions containing bundle
- **MIN_CONFIDENCE** (default: 0.5): Minimum confidence for bundle rules
- **MAX_BUNDLE_SIZE** (default: 5): Maximum products per bundle

**Output**: List of frequent product combinations that co-occur above thresholds

**Example**:
```
Support({Milk, Bread}) = 0.15  → 15% of transactions contain both
Confidence(Milk → Bread) = 0.75 → 75% of Milk purchases also include Bread
```

**Use Case**: Discovering bundles from transaction history for training data generation

## Performance Characteristics

### System Performance

| Operation | Time | Memory |
|-----------|------|--------|
| Download dataset | ~2-3 min | Network dependent |
| Data processing | ~2-3 min | ~500MB |
| Model training (Random Split) | ~1-2 min | ~700MB |
| Model training (10-Fold) | ~10-15 min | ~700MB |
| Per-prediction | <100ms | ~10MB |
| Full pipeline | ~10-15 min | ~1GB peak |

### Algorithm Comparison

| Algorithm | Training Speed | Prediction Speed | Memory | Accuracy |
|-----------|---------------|------------------|---------|----------|
| Naive Bayes | Fast | Very Fast | Low | Good |
| SVM (Linear) | Moderate | Fast | Moderate | Good |
| SVM (RBF) | Slower | Moderate | Moderate | Best |
| Ensemble | Slowest | Moderate | Higher | Most Robust |

## Security Considerations

### Sensitive Data
- Kaggle credentials in .env (not git-tracked)
- Customer data privacy
- Model privacy (proprietary algorithms)

### Input Validation
- Transaction format validation
- Product description sanitization
- Threshold bounds checking

### Access Control
- API authentication (future)
- Rate limiting (future)
- Audit logging (future)
