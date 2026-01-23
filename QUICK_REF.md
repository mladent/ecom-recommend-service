# Quick Reference Guide

## One-Liner Commands

```bash
# Complete pipeline (download → prepare → train → demo)
python main.py --full

# Just download data
python main.py --download

# Just prepare data
python main.py --prepare

# Train models
python main.py --train

# Run examples
python examples_basic.py
python examples_crosssell.py
python examples_comparison.py

# Run tests
pytest tests/ -v
```

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

## Configuration Quick Reference

### Key Environment Variables

```bash
# Kaggle credentials
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key

# Bundle generation
MIN_SUPPORT=0.02              # Min % of transactions containing bundle
MIN_CONFIDENCE=0.5            # Min confidence for bundle rules
MAX_BUNDLE_SIZE=5             # Max products per bundle

# Training
TRAIN_TEST_SPLIT=0.8          # 80% train, 20% test
RANDOM_STATE=42               # For reproducibility
N_JOBS=-1                     # Use all cores (-1) or specify number

# Data paths
DATA_PATH=./data
RAW_DATA_FILE=data.csv
PROCESSED_DATA_FILE=processed_data.pkl
```

### YAML Configuration (config/settings.yaml)

Key sections:
- `data`: File paths and processing
- `bundles`: Generation thresholds
- `training`: Model training params
- `algorithms`: Algorithm-specific settings
- `recommendation`: Prediction thresholds
- `logging`: Log level and file

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

## Contact & Support

For issues, check:
1. SETUP.md - Installation guide
2. ARCHITECTURE.md - Design details
3. Test files - Usage examples
4. Example scripts - Real-world usage
5. Logs - Error messages
