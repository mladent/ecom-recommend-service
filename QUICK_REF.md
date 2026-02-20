# Quick Reference Guide

Complete command reference, API examples, configuration options, and troubleshooting for the E-Commerce Bundle Recommendation Engine.

## 📊 Project Status

**Phase 1: LLM Provider Consolidation** ✅ **COMPLETE** (19 Feb 2026)
- LLM provider dispatch logic consolidated into 5 reusable helper methods
- Data pipeline refactored with 82 lines eliminated (6.9% reduction)
- Error handling standardized across all LLM integration points
- All tests passing (7/7 LLM + helper method tests)
- Zero breaking changes - fully backward compatible
- **Next Phase:** Config refactoring (Phase 2) - Global config elimination and dependency injection

See [to-do-list.md](to-do-list.md) for detailed progress tracking and upcoming phases.

---

## One-Liner Commands

### Main CLI Commands

```bash
# Complete pipeline (download → prepare → train → demo)
python main.py --full

# Individual pipeline steps
python main.py --download              # Download dataset from Kaggle
python main.py --prepare               # Process and prepare data
python main.py --train                 # Train all models
python main.py --demo                  # Run demonstration

# Force reprocessing (ignore cache)
python main.py --prepare --reprocess

# Verbose logging
python main.py --demo -v               # Verbose
python main.py --full -v -v            # Extra verbose
```

### Example Scripts

```bash
# Basic usage examples
python examples_basic.py               # Basic bundle recommendations
python examples_crosssell.py           # Cross-sell product suggestions
python examples_comparison.py          # Compare model performance
python examples_kfold_validation.py    # K-fold cross-validation
python examples_category_enrichment.py # LLM-powered category enrichment
```

### Testing Commands

```bash
# Run all tests
pytest tests/ -v

# Run specific test files
pytest tests/test_recommendation_engine.py -v
pytest tests/test_data_splitter.py -v

# Run specific test class or method
pytest tests/test_recommendation_engine.py::TestBundleRecommendationEngine -v
pytest tests/test_recommendation_engine.py::TestNaiveBayesRecommender::test_fit -v

# Coverage report
pytest tests/ --cov=src --cov-report=html
```

### Setup Commands

```bash
# Virtual environment setup
python -m venv venv
source venv/bin/activate              # Linux/Mac
venv\Scripts\activate                 # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env-template .env
# Edit .env with your Kaggle credentials
```

**SVM**:
- `kernel`: "rbf" (non-linear, default), "linear" (fast), "poly" (polynomial)
- `C`: Regularization parameter (smaller = more regularization)
- `gamma`: Kernel coefficient ("scale", "auto", or float value)

**Ensemble**:
- `ensemble_method`: "average" (soft voting) or "voting" (hard voting)

## API Quick Reference

### Load and Get Recommendations

```python
from src.recommendation_engine import BundleRecommendationEngine

# Load model
engine = BundleRecommendationEngine()
engine.load_model("models/recommendation_engine.pkl")

# Get bundle recommendations
recs = engine.recommend_bundles(
    customer_transaction=["product_a", "product_b"],
    recommender_name="naive_bayes",  # or "svm_rbf" or None for ensemble
    threshold=0.5
)

# Access results
print(f"Bundles: {recs['bundles']}")
print(f"Confidence: {recs['confidence']:.2%}")

# Get cross-sell products
cross_sell = engine.get_cross_sell_products(
    customer_transaction=["product_a"],
    top_n=5,
    recommender_name="naive_bayes"
)

for product, score in cross_sell:
    print(f"- {product}: {score:.2f}")
```

### Data Pipeline Usage

```python
from src.data_pipeline import DataPipeline

pipeline = DataPipeline()

# Download from Kaggle
pipeline.download_kaggle_data()

# Load and explore
pipeline.load_raw_data()
stats = pipeline.explore_data()

# Process
pipeline.preprocess()
pipeline.create_transaction_baskets()
pipeline.generate_product_bundles()

# Save
pipeline.save_processed_data()

# Get stats
bundle_stats = pipeline.get_bundle_statistics()
```

### Train Custom Models

```python
from src.recommendation_engine import (
    BundleRecommendationEngine,
    NaiveBayesBundleRecommender,
    SVMBundleRecommender
)

engine = BundleRecommendationEngine()

# Add Naive Bayes
nb = NaiveBayesBundleRecommender(model_type="multinomial")
engine.add_recommender("my_nb", nb)

# Add SVM with different kernel
svm = SVMBundleRecommender(kernel="linear", C=0.5)
engine.add_recommender("my_svm", svm)

# Train
metrics = engine.fit_all(transactions, bundles, validation_split=0.8)

# Save
engine.save_model("models/my_engine.pkl")
```

## Configuration Reference

### Environment Variables (.env)

```bash
# Kaggle API Credentials (Required for dataset download)
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key

# Bundle Generation Parameters
MIN_SUPPORT=0.02              # Minimum % of transactions containing bundle (0.0-1.0)
MIN_CONFIDENCE=0.5            # Minimum confidence for bundle rules (0.0-1.0)
MAX_BUNDLE_SIZE=5             # Maximum products per bundle (2+)

# Training Parameters
TRAIN_TEST_SPLIT=0.8          # Training data ratio (0.0-1.0)
RANDOM_STATE=42               # Random seed for reproducibility
N_JOBS=-1                     # CPU cores: -1 (all cores) or specific number

# Data Paths
DATA_PATH=./data              # Dataset directory
RAW_DATA_FILE=data.csv        # Raw CSV filename
PROCESSED_DATA_FILE=processed_data.pkl  # Cached processed data
MODEL_DIR=./models            # Trained model storage directory

# Logging
LOG_LEVEL=INFO                # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE=recommendation_service.log
```

### YAML Configuration (config/settings.yaml)

```yaml
data:
  path: ./data
  raw_file: data.csv
  processed_file: processed_data.pkl
  kaggle_dataset: carrie1/ecommerce-data

bundles:
  min_support: 0.02
  min_confidence: 0.5
  max_bundle_size: 5

training:
  test_size: 0.2
  random_state: 42
  n_jobs: -1
  validation_split: 0.8

algorithms:
  naive_bayes:
    model_type: multinomial    # multinomial or gaussian
  svm:
    kernel: rbf               # rbf, linear, poly
    C: 1.0                    # Regularization parameter
    gamma: scale              # Kernel coefficient

recommendation:
  confidence_threshold: 0.5
  max_recommendations: 5
  ensemble_method: average    # average or voting

logging:
  level: INFO
  file: recommendation_service.log
```

### Model Configuration Options

**Naive Bayes**:
- `model_type`: "multinomial" (discrete counts) or "gaussian" (continuous)

## Project Layout

```
ecom-recommend-service/
├── src/                    # Core modules
├── data/                   # Dataset directory
├── models/                 # Trained models (after training)
├── tests/                  # Unit tests
├── config/                 # Configuration
├── examples_*.py           # Usage examples
├── main.py                 # Entry point
├── requirements.txt        # Dependencies
├── .env-template           # Config template
├── README.md               # Overview
├── SETUP.md                # Installation guide
├── ARCHITECTURE.md         # Design details
└── this file               # Quick reference
```

## Common Workflows

### Workflow 1: Fresh Start

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run everything
python main.py --full

# Explore examples
python examples_basic.py
```

### Workflow 2: Iterative Development

```bash
# Data already downloaded, reuse cache
python main.py --demo

# Make changes to code
# Test
pytest tests/ -v

# Retrain with new changes
python main.py --train

# Test again
python main.py --demo
```

### Workflow 3: Custom Analysis

```python
# Python script
from src.recommendation_engine import BundleRecommendationEngine

engine = BundleRecommendationEngine()
engine.load_model("models/recommendation_engine.pkl")

# Your custom logic here
my_transaction = ["product_x", "product_y"]
recs = engine.recommend_bundles(my_transaction)

# Process results
for bundle in recs['bundles']:
    print(bundle)
```

### Workflow 4: Retraining

```bash
# Force reprocessing (ignore cache)
python main.py --prepare --reprocess

# Retrain models
python main.py --train

# Verify with demo
python main.py --demo
```

## Debugging Tips

### Check Data Loading

```python
from src.data_pipeline import DataPipeline

pipeline = DataPipeline()
raw = pipeline.load_raw_data()
print(f"Loaded: {len(raw)} rows")
print(raw.head())

pipeline.preprocess()
print(f"After preprocessing: {len(pipeline.processed_data)} rows")
```

### Check Model Training

```python
from src.recommendation_engine import NaiveBayesBundleRecommender

nb = NaiveBayesBundleRecommender()
metrics = nb.fit(transactions, bundles)

print(f"Accuracy: {metrics['accuracy']:.2%}")
print(f"Precision: {metrics['precision']:.2%}")
print(f"Recall: {metrics['recall']:.2%}")
print(f"F1 Score: {metrics['f1']:.2%}")
```

### Check Predictions

```python
engine = BundleRecommendationEngine()
engine.load_model("models/recommendation_engine.pkl")

# Try prediction
test_transaction = ["product_a", "product_b"]
proba = engine.recommenders["naive_bayes"].predict_proba([test_transaction])
print(f"Probabilities: {proba}")

# Try with low threshold
recs = engine.recommend_bundles(test_transaction, threshold=0.0)
print(f"All bundles: {len(recs['bundles'])}")
print(f"Confidence: {recs['confidence']}")
```

### Enable Verbose Logging

```bash
python main.py --demo -v      # Verbose logging
python main.py --full -v -v   # Extra verbose
```

## Troubleshooting Guide

### Common Issues & Solutions

#### Issue: Kaggle Authentication Error

**Symptoms**:
```
OSError: Could not find kaggle.json
```

**Solutions**:
1. **Option 1 - Environment variables**:
   ```bash
   cp .env-template .env
   # Edit .env and add:
   KAGGLE_USERNAME=your_username
   KAGGLE_KEY=your_api_key
   ```

2. **Option 2 - Kaggle JSON file**:
   ```bash
   mkdir -p ~/.kaggle
   cp kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

3. **Option 3 - Manual download**:
   - Download dataset from https://www.kaggle.com/carrie1/ecommerce-data
   - Place `data.csv` in `data/` directory

#### Issue: Out of Memory During Training

**Symptoms**:
```
MemoryError: Unable to allocate array
```

**Solutions**:
1. Reduce `MAX_BUNDLE_SIZE` in `.env`:
   ```bash
   MAX_BUNDLE_SIZE=3  # Reduce from 5 to 3
   ```

2. Increase `MIN_SUPPORT` to filter more bundles:
   ```bash
   MIN_SUPPORT=0.05  # Increase from 0.02 to 0.05
   ```

3. Use random split instead of k-fold:
   ```python
   # Instead of
   metrics = engine.fit_all_with_kfold(txns, bundles, n_splits=10)
   # Use
   metrics = engine.fit_all_with_random_split(txns, bundles, test_size=0.2)
   ```

4. Sample the data before training:
   ```python
   sampled_txns = transactions[:10000]  # Use first 10k transactions
   ```

#### Issue: Model File Not Found

**Symptoms**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'models/recommendation_engine.pkl'
```

**Solution**:
```bash
# Train models first
python main.py --train

# Verify models directory exists
ls -la models/
```

#### Issue: Low Recommendation Quality

**Symptoms**:
- Few or no recommendations returned
- Low confidence scores
- Poor accuracy metrics

**Solutions**:
1. **Adjust confidence threshold**:
   ```python
   # Lower threshold to get more recommendations
   recs = engine.recommend_bundles(transaction, threshold=0.3)  # Instead of 0.5
   ```

2. **Tune bundle generation parameters**:
   ```bash
   # In .env
   MIN_SUPPORT=0.01        # Lower to find more bundles
   MIN_CONFIDENCE=0.3      # Lower to include more associations
   ```

3. **Use ensemble instead of single model**:
   ```python
   # Don't specify recommender_name to use ensemble
   recs = engine.recommend_bundles(transaction)  # Ensemble (best)
   ```

4. **Check data quality**:
   ```python
   from src.data_pipeline import DataPipeline
   pipeline = DataPipeline()
   pipeline.load_raw_data()
   stats = pipeline.explore_data()
   print(stats)  # Check for sufficient data
   ```

#### Issue: Import Errors

**Symptoms**:
```
ModuleNotFoundError: No module named 'sklearn'
ImportError: cannot import name 'BundleRecommendationEngine'
```

**Solutions**:
1. **Reinstall dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify virtual environment is activated**:
   ```bash
   which python  # Should point to venv/bin/python
   ```

3. **Add project root to PYTHONPATH**:
   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

#### Issue: Tests Failing

**Symptoms**:
```
FAILED tests/test_recommendation_engine.py::test_something
```

**Solutions**:
1. **Run specific failing test with verbose output**:
   ```bash
   pytest tests/test_recommendation_engine.py::test_something -v -s
   ```

2. **Check for data dependencies**:
   ```bash
   # Ensure data is prepared
   python main.py --prepare
   ```

3. **Clean and rebuild**:
   ```bash
   # Remove cache files
   rm -rf __pycache__ src/__pycache__ tests/__pycache__
   rm -rf .pytest_cache
   
   # Reinstall
   pip install -e .
   
   # Run tests again
   pytest tests/ -v
   ```

#### Issue: Slow Performance

**Symptoms**:
- Training takes >30 minutes
- Predictions take >1 second

**Solutions**:
1. **Enable parallel processing**:
   ```bash
   # In .env
   N_JOBS=-1  # Use all CPU cores
   ```

2. **Use cached data**:
   ```bash
   # Don't use --reprocess flag unnecessarily
   python main.py --train  # Uses cached data
   ```

3. **Use RandomSplit for development**:
   ```python
   # Faster than k-fold
   metrics = engine.fit_all_with_random_split(txns, bundles)
   ```

4. **Profile your code**:
   ```python
   import cProfile
   cProfile.run('engine.fit_all(txns, bundles)')
   ```

### Verification Checklist

Before reporting an issue, verify:

- [ ] Virtual environment activated? (`which python` points to venv)
- [ ] All dependencies installed? (`pip list` shows scikit-learn, pandas, etc.)
- [ ] `.env` file exists and configured with Kaggle credentials?
- [ ] Dataset downloaded? (`ls data/data.csv`)
- [ ] Data processed? (`ls data/processed_data.pkl`)
- [ ] Models trained? (`ls models/*.pkl`)
- [ ] Logs checked? (`cat recommendation_service.log`)
- [ ] Tests passing? (`pytest tests/ -v`)
- [ ] Python version ≥ 3.8? (`python --version`)

### Debugging Workflow

1. **Enable verbose logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Check data pipeline step-by-step**:
   ```python
   from src.data_pipeline import DataPipeline
   
   pipeline = DataPipeline()
   print("Step 1: Loading data...")
   df = pipeline.load_raw_data()
   print(f"Loaded {len(df)} rows")
   
   print("Step 2: Preprocessing...")
   pipeline.preprocess()
   print(f"After preprocessing: {len(pipeline.processed_data)} rows")
   
   print("Step 3: Creating baskets...")
   baskets = pipeline.create_transaction_baskets()
   print(f"Created {len(baskets)} transaction baskets")
   
   print("Step 4: Generating bundles...")
   bundles = pipeline.generate_product_bundles()
   print(f"Generated {len(bundles)} bundles")
   ```

3. **Test model individually**:
   ```python
   from src.recommendation_engine import NaiveBayesBundleRecommender
   
   nb = NaiveBayesBundleRecommender()
   metrics = nb.fit(transactions, bundles)
   
   print(f"Accuracy: {metrics['accuracy']:.2%}")
   print(f"Precision: {metrics['precision']:.2%}")
   print(f"Recall: {metrics['recall']:.2%}")
   print(f"F1 Score: {metrics['f1']:.2%}")
   ```

4. **Test prediction with debug output**:
   ```python
   engine = BundleRecommendationEngine()
   engine.load_model("models/recommendation_engine.pkl")
   
   test_transaction = ["product_a", "product_b"]
   
   # Check probability scores
   proba = engine.recommenders["naive_bayes"].predict_proba([test_transaction])
   print(f"Probabilities: {proba}")
   
   # Try with low threshold to see all possible bundles
   recs = engine.recommend_bundles(test_transaction, threshold=0.0)
   print(f"All bundles (threshold=0): {len(recs['bundles'])}")
   
   # Try with normal threshold
   recs = engine.recommend_bundles(test_transaction, threshold=0.5)
   print(f"Filtered bundles (threshold=0.5): {len(recs['bundles'])}")
   print(f"Confidence: {recs['confidence']}")
   ```

## Troubleshooting Checklist

- [ ] Virtual environment activated?
- [ ] All dependencies installed? `pip install -r requirements.txt`
- [ ] `.env` file configured with Kaggle credentials?
- [ ] Dataset downloaded? `python main.py --download`
- [ ] Data processed? `python main.py --prepare`
- [ ] Models trained? `python main.py --train`
- [ ] Check logs: `recommendation_service.log`
- [ ] Run tests: `pytest tests/ -v`

## Performance Notes

- **First run**: ~5-10 minutes (includes download)
- **Prepare**: ~2-3 minutes (dataset processing)
- **Train**: ~1-2 minutes (model training)
- **Prediction**: <100ms per transaction
- **Memory**: ~500MB-1GB for full dataset

## Performance Notes & Benchmarks

### Typical Performance Metrics

| Operation | Time | Memory | Notes |
|-----------|------|--------|-------|
| Download dataset | 2-3 min | Network dependent | One-time operation |
| Data processing | 2-3 min | ~500MB | Cached after first run |
| Model training (Random Split) | 1-2 min | ~700MB | Single train-test split |
| Model training (10-Fold) | 10-15 min | ~700MB | More robust metrics |
| Per-prediction | <100ms | ~10MB | Real-time capable |
| Full pipeline (first run) | 10-15 min | ~1GB peak | Includes download |
| Full pipeline (cached) | 5-7 min | ~1GB peak | Reuses cached data |

### Optimization Tips

1. **Use caching**: Don't use `--reprocess` unnecessarily
2. **Parallel processing**: Set `N_JOBS=-1` in `.env`
3. **Development iteration**: Use `RandomSplit` during development, `KFoldSplit` for production
4. **Data sampling**: Sample transactions for quick experiments
5. **Model persistence**: Save and load models instead of retraining

### System Requirements

- **Minimum**: 4GB RAM, 2 CPU cores, 2GB disk space
- **Recommended**: 8GB RAM, 4+ CPU cores, 5GB disk space
- **Python**: 3.8 or higher

## Project File Structure

```
ecom-recommend-service/
├── src/                          # Core modules
│   ├── __init__.py              # Package initialization
│   ├── api.py                   # FastAPI REST API (150 lines)
│   ├── config.py                # Configuration management (55 lines)
│   ├── data_evaluator.py        # Data quality evaluation
│   ├── data_pipeline.py         # Data processing (340 lines)
│   ├── data_splitter.py         # Train-test splitting strategies
│   ├── recommendation_engine.py # ML algorithms (420 lines)
│   ├── utils.py                 # Helper functions (55 lines)
│   └── web/                     # Web UI components
│       ├── index.html           # Frontend interface
│       ├── app.js               # JavaScript application
│       └── products.tsv         # Sample product data
├── data/                         # Dataset storage
│   ├── data.csv                 # Raw dataset (after download)
│   ├── processed_data.pkl       # Cached processed data
│   ├── About_Dataset.md         # Dataset documentation
│   └── data_report.md           # Data quality report
├── models/                       # Trained models (after training)
│   └── recommendation_engine.pkl
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── test_recommendation_engine.py  # Engine tests (180 lines)
│   └── test_data_splitter.py    # Splitter tests
├── config/                       # Configuration files
│   ├── settings.yaml            # YAML configuration
│   ├── prompts/                 # LLM prompts
│   └── schemas/                 # JSON schemas
├── notebooks/                    # Jupyter notebooks
├── examples_*.py                 # Usage examples
│   ├── examples_basic.py        # Basic recommendations
│   ├── examples_crosssell.py    # Cross-sell suggestions
│   ├── examples_comparison.py   # Model comparison
│   ├── examples_kfold_validation.py  # K-fold validation
│   └── examples_category_enrichment.py  # LLM features
├── main.py                       # CLI entry point (240 lines)
├── requirements.txt              # Python dependencies
├── .env-template                 # Environment template
├── .env                          # Your configuration (not in git)
├── .gitignore                    # Git ignore rules
├── Dockerfile                    # Docker image definition
├── docker-compose.yml            # Docker orchestration
├── docker-entrypoint.sh          # Docker startup script
├── README.md                     # Project overview
├── QUICK_REF.md                  # This file - Quick reference
├── ARCHITECTURE.md               # System design details
├── DOCKER_README.md              # Docker deployment guide
├── WORKFLOW_DIAGRAMS.md          # Visual system flows
├── CATEGORY_ENRICHMENT.md        # LLM feature documentation
└── AGENTS.md                     # AI agent development guide
```

## Contact & Support

For issues and questions:

1. **Check documentation**:
   - [README.md](README.md) - Project overview and quick start
   - [ARCHITECTURE.md](ARCHITECTURE.md) - System design and algorithms
   - [DOCKER_README.md](DOCKER_README.md) - Docker deployment
   - This file (QUICK_REF.md) - Commands and troubleshooting

2. **Review examples**:
   - See `examples_*.py` files for working code
   - See `tests/` directory for usage patterns

3. **Check logs**:
   - `recommendation_service.log` for application logs
   - Enable verbose mode: `python main.py --demo -v -v`

4. **Run tests**:
   ```bash
   pytest tests/ -v  # Verify everything works
   ```
