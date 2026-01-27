# E-Commerce Bundle Recommendation Service

A machine learning-based recommendation engine that suggests product bundles to customers using Naive Bayes and SVM algorithms.

## Features

- **Data Pipeline**: Automated Pandas-based data loading, cleaning, and preprocessing
- **Bundle Recommendation**: Naive Bayes and SVM-based algorithms for product bundle recommendations
- **Scalable Architecture**: Modular design prepared for future LLM API integrations
- **Kaggle Dataset Integration**: Direct download and integration of e-commerce data

## Project Structure

```
ecom-recommend-service/
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py          # Data loading and preprocessing
│   ├── recommendation_engine.py   # Bundle recommendation algorithms
│   ├── config.py                  # Configuration management
│   └── utils.py                   # Utility functions
├── data/
│   └── .gitkeep
├── config/
│   └── settings.yaml              # Default configuration
├── tests/
│   ├── __init__.py
│   └── test_*.py                  # Unit tests
├── notebooks/
│   └── exploratory_analysis.ipynb # EDA and experimentation
├── requirements.txt               # Python dependencies
├── .env-template                  # Environment template
├── .gitignore                     # Git ignore rules
├── main.py                        # Entry point
└── README.md                      # This file
```

## Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git
- Kaggle account (for downloading the dataset)

### Installation Steps

1. **Set up virtual environment**:
   ```bash
   # Navigate to project directory
   cd ecom-recommend-service

   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Kaggle API**:
   
   **Option A: Using .kaggle/kaggle.json**
   - Go to https://www.kaggle.com/settings/account and click "Create New Token"
   - Place `kaggle.json` in your home directory:
     - Linux/Mac: `~/.kaggle/kaggle.json`
     - Windows: `C:\Users\<YourUsername>\.kaggle\kaggle.json`
   - Set permissions (Linux/Mac only): `chmod 600 ~/.kaggle/kaggle.json`

   **Option B: Using .env File**
   ```bash
   cp .env-template .env
   ```
   Edit `.env` and add your credentials:
   ```
   KAGGLE_USERNAME=your_username
   KAGGLE_KEY=your_api_key
   ```

4. **Download and prepare dataset**:
   ```bash
   # Download the e-commerce dataset
   python main.py --download

   # Process and prepare the dataset
   python main.py --prepare

   # Train the recommendation models
   python main.py --train

   # Run demo recommendations
   python main.py --demo
   ```

## Usage

### Data Pipeline

The data pipeline processes raw e-commerce transaction data through multiple stages:

**Data Loading** ([`src/data_pipeline.py:load_raw_data()`](src/data_pipeline.py#L69))
- Loads raw CSV data from Kaggle (541,909 records, 8 columns)
- Uses ISO-8859-1 encoding to handle special characters
- Raw columns: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country

**Data Filtering and Cleaning** ([`src/data_pipeline.py:preprocess()`](src/data_pipeline.py#L109))

Records are filtered based on these criteria:
1. **Remove missing CustomerID**: Removes 135,080 records (~25%) for transactions without customer identification
2. **Remove missing Description**: Removes 1,454 records without product descriptions
3. **Remove cancellations**: Filters out negative quantities (returns/cancellations)
4. **Remove invalid prices**: Removes zero or negative UnitPrice entries
5. **Remove duplicates**: Removes duplicate invoice-product combinations

After filtering: ~406,000 valid transaction records remain

**Data Transformations**:
- Convert `InvoiceDate` to datetime format
- Standardize product descriptions (lowercase, strip whitespace)
- Create `TransactionValue` column: `Quantity × UnitPrice`

**Transaction Baskets** ([`src/data_pipeline.py:create_transaction_baskets()`](src/data_pipeline.py#L177))
- Groups items by `InvoiceNo` to create shopping baskets
- Filters baskets to only include transactions with 2+ items (for bundle analysis)
- Creates ~17,000+ transaction baskets with transaction metadata (customer, date, total value)

**Bundle Generation** ([`src/data_pipeline.py:generate_product_bundles()`](src/data_pipeline.py#L211))
- Uses Apriori-like frequent itemset mining to identify product combinations that appear frequently together
- Filters based on configurable thresholds:
  - `MIN_SUPPORT` (default: 2%): Item must appear in at least 2% of transactions
  - `MIN_CONFIDENCE` (default: 50%): Bundle confidence threshold
  - `MAX_BUNDLE_SIZE` (default: 5): Maximum products in a bundle
- Generates ~46 product bundles (2-5 items per bundle)

**Label Creation** (Implicit in model training):
- Binary classification target: Does a transaction include a given bundle?
- Each transaction is labeled as 1 if it contains ALL items in a bundle, 0 otherwise
- Used in [`src/recommendation_engine.py:fit()`](src/recommendation_engine.py#L91) for both Naive Bayes and SVM

**Usage Example**:
```python
from src.data_pipeline import DataPipeline

# Initialize pipeline
pipeline = DataPipeline()

# Download from Kaggle
pipeline.download_kaggle_data()

# Load and preprocess
pipeline.load_raw_data()
stats = pipeline.explore_data()
pipeline.preprocess()

# Create transaction baskets
baskets = pipeline.create_transaction_baskets()

# Generate bundles
bundles = pipeline.generate_product_bundles(
    min_support=0.02,
    min_confidence=0.5
)

# Access processed data
transactions = pipeline.transactions  # DataFrame with baskets
products = pipeline.products          # List of unique products
bundles = pipeline.bundles            # List of product bundles
```

### Recommendation Engine

The recommendation engine uses machine learning algorithms to predict which product bundles customers will purchase based on their current transaction items.

**Algorithms**:

1. **Naive Bayes** ([`src/recommendation_engine.py:NaiveBayesBundleRecommender`](src/recommendation_engine.py#L56))
   - Model type: Multinomial (default) or Gaussian
   - Feature encoding: MultiLabelBinarizer converts product lists into binary feature vectors
   - Training: Uses transaction items as features (X) and bundle membership as labels (y)

2. **Support Vector Machine** ([`src/recommendation_engine.py:SVMBundleRecommender`](src/recommendation_engine.py#L178))
   - Kernel: Linear (default, changed from RBF for faster training)
   - Feature scaling: StandardScaler applied to dense feature vectors
   - Regularization parameter C: 1.0 (default)

**Feature Engineering** ([`src/recommendation_engine.py`](src/recommendation_engine.py)):
- **Input**: Transaction items (product descriptions, lowercase)
- **Feature Transformation**: MultiLabelBinarizer creates binary feature vector (1 if product present, 0 otherwise)
- **Output**: Feature vector for each transaction

**Training Process** ([`src/recommendation_engine.py:fit_all()`](src/recommendation_engine.py#L320)):

For each recommender:
1. **Feature Preparation** (lines 346-350):
   - Transforms transaction items into binary features using MultiLabelBinarizer
   - Creates binary labels (1 = bundle present in transaction, 0 = not present)

2. **Data Splitting** (lines 355-359):
   - Train-test split: 80% training, 20% validation (configurable via `TRAIN_TEST_SPLIT`)
   - Uses fixed random seed for reproducibility

3. **Model Training** (lines 361-362):
   - Naive Bayes: Learns probability distributions of features per class
   - SVM: Finds optimal hyperplane separating bundle vs. non-bundle transactions

4. **Evaluation** (lines 365-371):
   - Calculates accuracy, precision, recall, and F1-score on test set

**Prediction / Recommendation Process** ([`src/recommendation_engine.py:recommend_bundles()`](src/recommendation_engine.py#L339)):

1. **Input**: Customer's current transaction items (list of product descriptions)

2. **Feature Transformation** (lines 365-375):
   - Converts transaction items to binary feature vector using fitted MultiLabelBinarizer
   - For SVM: Applies StandardScaler normalization

3. **Confidence Scoring** (lines 377-388):
   - **Single Recommender**: Uses specified model's probability prediction
   - **Ensemble (default)**: Averages probability predictions from all recommenders (Naive Bayes + SVM)
   - Output: Confidence score (0.0 to 1.0) indicating likelihood of bundle purchase

4. **Bundle Matching** (lines 390-404):
   - Compares confidence score against threshold (default: 0.5 or 50%)
   - If confidence >= threshold: Identifies applicable bundles with overlap in current transaction
   - Returns top 5 applicable bundles for recommendation

5. **Output**: Recommendation dictionary containing:
   - `transaction`: Current items
   - `confidence`: Probability score (0.0-1.0)
   - `bundles`: List of recommended product bundles
   - `recommender`: Which model made the prediction (or "ensemble")

**Usage Example**:
```python
from src.recommendation_engine import BundleRecommendationEngine

# Load trained model
engine = BundleRecommendationEngine()
engine.load_model("models/recommendation_engine.pkl")

# Customer's current transaction
customer_items = ["white hanging heart t-light holder", "regency cakestand 3 tier"]

# Get recommendations with 50% confidence threshold
recommendations = engine.recommend_bundles(
    customer_items, 
    threshold=0.5
)

# Output:
# {
#   "transaction": ["white hanging heart t-light holder", "regency cakestand 3 tier"],
#   "confidence": 0.238,  # 23.8% likelihood of bundle
#   "bundles": [bundle1, bundle2, ...],  # Top 5 applicable bundles
#   "recommender": "ensemble"  # Using both Naive Bayes and SVM
# }

# Get cross-sell product recommendations
cross_sell = engine.get_cross_sell_products(customer_items, top_n=5)
# Returns: [("product_a", 0.95), ("product_b", 0.87), ...]
```

**Configuration Files**:
- [`config/settings.yaml`](config/settings.yaml): Algorithm settings, hyperparameters, thresholds
- [`.env-template`](.env-template): Environment variables for Kaggle credentials, data paths

## Configuration

Edit `.env` file to customize:
- Kaggle credentials
- Data file paths
- Model hyperparameters
- Bundle recommendation thresholds

## Future Enhancements

- LLM-based bundle explanations and naming
- Real-time recommendation API
- A/B testing framework
- Advanced feature engineering with LLM embeddings
- Recommendation personalization with customer profiling



## Quick Start (All Steps in One)

```bash
python main.py --full
```

This runs the complete pipeline: download → prepare → train → demo

## Usage Examples

### Using in Python Code

```python
from src.recommendation_engine import BundleRecommendationEngine

# Load trained model
engine = BundleRecommendationEngine()
engine.load_model("models/recommendation_engine.pkl")

# Get recommendations
transaction = ["product_a", "product_b"]
recommendations = engine.recommend_bundles(transaction, threshold=0.5)

# Get cross-sell products
cross_sell = engine.get_cross_sell_products(transaction, top_n=5)
```

### Running Example Scripts

```bash
# Example 1: Basic recommendations
python examples_basic.py

# Example 2: Cross-sell recommendations
python examples_crosssell.py

# Example 3: Model comparison
python examples_comparison.py
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_recommendation_engine.py -v
```

## Configuration

### Environment Variables (.env)

- `KAGGLE_USERNAME`: Your Kaggle username
- `KAGGLE_KEY`: Your Kaggle API key
- `MIN_SUPPORT`: Minimum support for bundle generation (default: 0.02)
- `MIN_CONFIDENCE`: Minimum confidence for bundles (default: 0.5)
- `MAX_BUNDLE_SIZE`: Maximum items in a bundle (default: 5)
- `TRAIN_TEST_SPLIT`: Train-test split ratio (default: 0.8)

### Configuration File (config/settings.yaml)

The YAML configuration file allows detailed control over:
- Data processing parameters
- Bundle generation settings
- Model hyperparameters
- Recommendation thresholds
- Logging configuration

## Troubleshooting

### Kaggle API Authentication Error

If you get authentication errors:

1. Verify your `kaggle.json` is in the correct location
2. Check file permissions: `chmod 600 ~/.kaggle/kaggle.json`
3. Try using .env file instead with explicit credentials

### Out of Memory Error

If processing large datasets:

1. Reduce batch size in config
2. Use `--prepare --reprocess` with lower `MAX_BUNDLE_SIZE`
3. Filter data before processing

### Model Not Found Error

If models aren't found when running demos:

1. Ensure training completed: `python main.py --train`
2. Check `models/` directory exists
3. Verify file path in code matches actual location

## Project Structure Reference

```
ecom-recommend-service/
├── src/                           # Source code
│   ├── __init__.py
│   ├── config.py                  # Configuration management
│   ├── data_pipeline.py           # Data loading and preprocessing
│   ├── recommendation_engine.py   # ML recommendation algorithms
│   └── utils.py                   # Utility functions
├── data/                          # Dataset storage
│   └── .gitkeep
├── models/                        # Trained models (created after training)
├── config/                        # Configuration files
│   └── settings.yaml
├── tests/                         # Unit tests
│   ├── __init__.py
│   └── test_recommendation_engine.py
├── notebooks/                     # Jupyter notebooks for exploration
├── examples_*.py                  # Example usage scripts
├── main.py                        # Main entry point
├── requirements.txt               # Python dependencies
├── .env-template                  # Environment template
├── .gitignore                     # Git ignore rules
├── README.md                      # Project overview
└── SETUP.md                       # This file
```

## Next Steps

- Explore the example scripts to understand the API
- Review the test suite to see usage patterns
- Check configuration files to customize behavior
- Review ARCHITECTURE.md for system design details

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review example scripts for correct usage
3. Examine test files for API patterns
4. Check logs: `recommendation_service.log`

## Future Enhancements

- LLM-based bundle descriptions
- Real-time recommendation API
- Performance optimization for large datasets
- Advanced feature engineering
- A/B testing framework
- Customer segmentation
