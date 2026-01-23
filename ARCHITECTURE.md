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
