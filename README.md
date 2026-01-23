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

1. **Clone and install dependencies**:
   ```bash
   cd ecom-recommend-service
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure Kaggle credentials**:
- Create a `.env` file from `.env-template`:
```bash
cp .env-template .env
```
   - Add your Kaggle API credentials (get them from https://www.kaggle.com/settings/account)
   - Or place `kaggle.json` in `~/.kaggle/` directory

3. **Download dataset**:
   ```bash
   python main.py --download
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

## License

MIT
