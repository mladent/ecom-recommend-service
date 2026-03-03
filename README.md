# E-Commerce Bundle Recommendation Service

![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![Status](https://img.shields.io/badge/status-proof--of--concept-yellow.svg)
![ML](https://img.shields.io/badge/ML-scikit--learn-orange.svg)
<!-- ![License](https://img.shields.io/badge/license-MIT-green.svg) -->

A machine learning service that recommends product bundles to e-commerce customers using Naive Bayes and SVM algorithms with ensemble methods. Features include automated data pipelines, multiple training strategies, and LLM integration capabilities.

## ✨ Features

- **Multiple ML Algorithms**: Naive Bayes, SVM (RBF/Linear/Poly kernels), and Ensemble methods
- **Flexible Training**: Single random split or k-fold cross-validation for robust evaluation
- **Automated Data Pipeline**: Kaggle dataset integration with cleaning and preprocessing
- **Bundle Discovery**: Apriori-like frequent itemset mining for automatic bundle generation
- **Cross-sell Recommendations**: Intelligent product suggestions based on purchase patterns
- **Model Persistence**: Save and load trained models for production deployment
- **REST API**: Flask-based web service with JSON endpoints
- **LLM Integration**: Category enrichment and description normalization capabilities
- **Docker Support**: Full containerization for easy deployment
- **Comprehensive Testing**: Unit tests with pytest covering all major components

## 🚀 Quick Start

### Prerequisites

- Python 3.12 or higher
- Kaggle account (for dataset download)
- 4GB+ RAM recommended

### Installation

```bash
# 1. Clone and navigate to project
cd ecom-recommend-service

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure Kaggle credentials
cp .env-template .env
# Edit .env and add your KAGGLE_USERNAME and KAGGLE_KEY

# 5. Run complete pipeline
python main.py --full
```

That's it! The system will download data, train models, and run a demonstration.

### First Recommendation

```python
from src.recommendation_engine import BundleRecommendationEngine

# Load trained model
engine = BundleRecommendationEngine()
engine.load_model("models/recommendation_engine.pkl")

# Get recommendations
recs = engine.recommend_bundles(
    customer_transaction=["WHITE HANGING HEART T-LIGHT HOLDER", "WHITE METAL LANTERN"],
    threshold=0.5
)

print(f"Recommended bundles: {recs['bundles']}")
print(f"Confidence: {recs['confidence']:.2%}")
```

## 📊 Example Scripts

| Script | Description | Run Command |
|--------|-------------|-------------|
| [examples_basic.py](examples_basic.py) | Basic bundle recommendations with single model | `python examples_basic.py` |
| [examples_crosssell.py](examples_crosssell.py) | Cross-sell product suggestions | `python examples_crosssell.py` |
| [examples_comparison.py](examples_comparison.py) | Compare Naive Bayes vs SVM vs Ensemble | `python examples_comparison.py` |
| [examples_kfold_validation.py](examples_kfold_validation.py) | K-fold cross-validation for robust metrics | `python examples_kfold_validation.py` |
| [examples_category_enrichment.py](examples_category_enrichment.py) | LLM-powered category enrichment | `python examples_category_enrichment.py` |
| [examples_batch_enrichment.py](examples_batch_enrichment.py) | Batch processing for optimized LLM enrichment | `python examples_batch_enrichment.py` |

## 🏗️ Project Structure

```
ecom-recommend-service/
├── src/                          # Core modules
│   ├── data_pipeline.py         # Data loading and preprocessing (340 lines)
│   ├── recommendation_engine.py # ML algorithms (420 lines)
│   ├── data_splitter.py         # Train/test splitting strategies
│   ├── api.py                   # Flask REST API
│   ├── config.py                # Configuration management
│   └── utils.py                 # Helper functions
├── data/                         # Dataset storage (created on first run)
├── models/                       # Trained models (created after training)
├── tests/                        # Unit tests
├── config/                       # Configuration files
│   ├── settings.yaml            # YAML configuration
│   ├── prompts/                 # LLM prompts
│   └── schemas/                 # JSON schemas
├── examples_*.py                 # Usage examples
├── main.py                       # CLI entry point
├── requirements.txt              # Python dependencies
├── .env-template                 # Environment template
├── Dockerfile                    # Docker image
├── docker-compose.yml            # Docker orchestration
└── Documentation files (*.md)
```

## 💡 Basic Usage

### Training Models

```python
from src.recommendation_engine import (
    BundleRecommendationEngine,
    NaiveBayesBundleRecommender,
    SVMBundleRecommender
)
from src.data_pipeline import DataPipeline

# Prepare data
pipeline = DataPipeline()
pipeline.load_raw_data()
pipeline.preprocess()
transactions = pipeline.create_transaction_baskets()
bundles = pipeline.generate_product_bundles()

# Create and train engine
engine = BundleRecommendationEngine()
engine.add_recommender("nb", NaiveBayesBundleRecommender())
engine.add_recommender("svm", SVMBundleRecommender(kernel="rbf"))

# Train with k-fold cross-validation for robust metrics
metrics = engine.fit_all_with_kfold(
    [list(t) for t in transactions["Items"]], 
    bundles, 
    n_splits=10
)

# Save trained model
engine.save_model("models/my_engine.pkl")

print(f"Naive Bayes Accuracy: {metrics['nb']['accuracy']:.2%} ± {metrics['nb']['std_accuracy']:.2%}")
print(f"SVM Accuracy: {metrics['svm']['accuracy']:.2%} ± {metrics['svm']['std_accuracy']:.2%}")
```

### Getting Recommendations

```python
# Load model
engine = BundleRecommendationEngine()
engine.load_model("models/recommendation_engine.pkl")

# Get bundle recommendations (uses ensemble by default)
recommendations = engine.recommend_bundles(
    customer_transaction=["JUMBO BAG RED RETROSPOT"],
    threshold=0.5
)

# Get cross-sell products
cross_sell = engine.get_cross_sell_products(
    customer_transaction=["JUMBO BAG RED RETROSPOT"],
    top_n=5
)
```

### Using the REST API

```bash
# Start the API server
python main.py --api

# In another terminal, make requests
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{"items": ["WHITE HANGING HEART T-LIGHT HOLDER", "CREAM CUPID HEARTS COAT HANGER"]}'

# Or open the web UI
open http://localhost:8000
```

## 🧪 Testing

Run `pytest tests/ -v` for comprehensive test suite covering data pipeline, recommendation engine, API endpoints, and integration tests. All LLM functions are mocked for speed and consistency. See [TESTING_GUIDE.md](TESTING_GUIDE.md) for detailed testing documentation, mocking strategy, and available commands.

## ⚙️ Configuration

Edit `.env` to configure bundle generation parameters (MIN_SUPPORT, MIN_CONFIDENCE, MAX_BUNDLE_SIZE), training settings (TRAIN_TEST_SPLIT, RANDOM_STATE), and resource allocation (N_JOBS). Use `python main.py --bundles-only` to quickly test different parameters without full reprocessing.

For complete configuration reference, defaults, and LLM settings, see [QUICK_REF.md](QUICK_REF.md#configuration-reference).

## 📚 Documentation Map

| Goal | Read | Details |
|------|------|----------|
| **Getting started** | This README + [examples_basic.py](examples_basic.py) | Installation, quick run, simple example |
| **Commands & API** | [QUICK_REF.md](QUICK_REF.md) | All CLI commands, Python API, configuration options |
| **System design** | [ARCHITECTURE.md](ARCHITECTURE.md) | Algorithms, data splitting, performance metrics |
| **Web API & UI** | [QUICK_REF.md](QUICK_REF.md), run `python main.py --api` | REST endpoints, example requests |
| **Testing** | [TESTING_GUIDE.md](TESTING_GUIDE.md) | Test organization, mocking strategy, CI/CD |
| **Docker deployment** | [DOCKER_README.md](DOCKER_README.md) | Build, run, and orchestrate containers |
| **LLM features** | [CATEGORY_ENRICHMENT.md](CATEGORY_ENRICHMENT.md) | Category enrichment, out-of-stock handling |
| **System workflows** | [WORKFLOW_DIAGRAMS.md](WORKFLOW_DIAGRAMS.md) | Visual diagrams and process flows |
| **Dataset info** | [data/About_Dataset.md](data/About_Dataset.md) | Schema, features, data preparation |
| **Troubleshooting** | [QUICK_REF.md](QUICK_REF.md#troubleshooting-guide) | Common issues and solutions |



## 🐳 Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Access the service
curl http://localhost:8000/health
```

See [DOCKER_README.md](DOCKER_README.md) for complete Docker documentation.

## 🤝 Contributing

Contributions are welcome! For AI agents working on this codebase, please review [AGENTS.md](AGENTS.md) for best practices and coding guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest tests/ -v`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

<!-- 
## 📄 License
ToDo 
-->

## 🙏 Acknowledgments

- **Dataset**: [E-Commerce Data](https://www.kaggle.com/carrie1/ecommerce-data) from Kaggle by UCI ML Repository
- **Libraries**: scikit-learn, pandas, numpy, Flask
- **ML Algorithms**: Naive Bayes, Support Vector Machines, Ensemble Methods
- **Market Basket Analysis**: Apriori-like frequent itemset mining

## 📞 Support

Having issues? See [QUICK_REF.md](QUICK_REF.md#troubleshooting-guide) for common issues, run `pytest tests/ -v` to verify installation, or review working examples in `examples_*.py` files.

---

**Ready to recommend product bundles to your customers!** 🚀

For questions or feedback, please open an issue on GitHub.
