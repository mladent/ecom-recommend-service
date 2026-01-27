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

```python
from src.data_pipeline import DataPipeline

pipeline = DataPipeline(config_path='config/settings.yaml')
df = pipeline.load_data()
processed_data = pipeline.preprocess()
bundles = pipeline.generate_bundles()
```

### Recommendation Engine

```python
from src.recommendation_engine import BundleRecommender

recommender = BundleRecommender(algorithm='naive_bayes')
recommender.fit(training_data)
bundles = recommender.recommend(customer_id=123, n_recommendations=5)
```

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
