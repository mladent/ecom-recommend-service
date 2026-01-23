# Installation and Setup Guide

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git
- Kaggle account (for downloading the dataset)

## Step 1: Set Up Virtual Environment

```bash
# Navigate to project directory
cd ecom-recommend-service

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 3: Configure Kaggle API

### Option A: Using .kaggle/kaggle.json

1. Go to https://www.kaggle.com/settings/account
2. Click "Create New Token" to download `kaggle.json`
3. Place the file in your home directory:
   - Linux/Mac: `~/.kaggle/kaggle.json`
   - Windows: `C:\Users\<YourUsername>\.kaggle\kaggle.json`
4. Set permissions (Linux/Mac only): `chmod 600 ~/.kaggle/kaggle.json`

### Option B: Using .env File

1. Copy `.env-template` to `.env`:
   ```bash
   cp .env-template .env
   ```
2. Edit `.env` and add your Kaggle credentials:
   ```
   KAGGLE_USERNAME=your_kaggle_username
   KAGGLE_KEY=your_kaggle_api_key
   ```

## Step 4: Download Dataset

```bash
# Download the e-commerce dataset from Kaggle
python main.py --download
```

The dataset will be saved to the `data/` directory.

## Step 5: Prepare Data

```bash
# Process and prepare the dataset
python main.py --prepare
```

This will:
- Load the raw CSV data
- Clean and preprocess it
- Generate product bundles
- Cache the processed data for faster future runs

## Step 6: Train Models

```bash
# Train the recommendation models
python main.py --train
```

This will:
- Train Naive Bayes recommender
- Train SVM recommender
- Evaluate both models
- Save trained models to `models/` directory

## Step 7: Run Demo

```bash
# Run demo recommendations
python main.py --demo
```

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
