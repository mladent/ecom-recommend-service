# E-Commerce Bundle Recommendation Service

![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![Status](https://img.shields.io/badge/status-proof--of--concept-yellow.svg)
![ML](https://img.shields.io/badge/ML-scikit--learn-orange.svg)
<!-- ![License](https://img.shields.io/badge/license-MIT-green.svg) -->

A proof-of-concept machine learning service that recommends product bundles to e-commerce customers using Naive Bayes and SVM algorithms with ensemble methods. Features include automated data pipelines, multiple training strategies, and LLM integration capabilities.

## ✨ Features

- **Multiple ML Algorithms**: Naive Bayes, SVM (RBF/Linear/Poly kernels), and Ensemble methods
- **Flexible Training**: Single random split or k-fold cross-validation for robust evaluation
- **Automated Data Pipeline**: Kaggle dataset integration with cleaning and preprocessing
- **Bundle Discovery**: Apriori-like frequent itemset mining for automatic bundle generation
- **Cross-sell Recommendations**: Intelligent product suggestions based on purchase patterns
- **Model Persistence**: Save and load trained models for production deployment
- **REST API**: FastAPI-based web service with interactive UI
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

## 🏗️ Project Structure

```
ecom-recommend-service/
├── src/                          # Core modules
│   ├── data_pipeline.py         # Data loading and preprocessing (340 lines)
│   ├── recommendation_engine.py # ML algorithms (420 lines)
│   ├── data_splitter.py         # Train/test splitting strategies
│   ├── api.py                   # FastAPI REST API
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

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_recommendation_engine.py -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html
```

## ⚙️ Configuration

### Key Parameters

Edit `.env` to configure the system:

```bash
# Bundle Generation
MIN_SUPPORT=0.02              # Minimum frequency for bundles (2% of transactions)
MIN_CONFIDENCE=0.5            # Minimum confidence for association rules
MAX_BUNDLE_SIZE=5             # Maximum products per bundle

# Training
TRAIN_TEST_SPLIT=0.8          # Train/test split ratio
RANDOM_STATE=42               # Random seed for reproducibility
N_JOBS=-1                     # CPU cores (-1 = all available)
```

For complete configuration options, see [QUICK_REF.md](QUICK_REF.md#configuration-reference).

## 📚 Documentation

Comprehensive documentation is available for different use cases:

- **[QUICK_REF.md](QUICK_REF.md)** - Command reference, API examples, troubleshooting, and debugging
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design, algorithms, data splitting strategies, and performance
- **[DOCKER_README.md](DOCKER_README.md)** - Docker deployment and container orchestration
- **[CATEGORY_ENRICHMENT.md](CATEGORY_ENRICHMENT.md)** - LLM-powered category enrichment feature
- **[WORKFLOW_DIAGRAMS.md](WORKFLOW_DIAGRAMS.md)** - Visual system workflows and diagrams
- **[AGENTS.md](AGENTS.md)** - Best practices for AI agents working on this codebase
- **[data/About_Dataset.md](data/About_Dataset.md)** - Dataset information and schema

### Documentation Guide

**New to the project?**
1. Start with this README (you're here!)
2. Run Quick Start above
3. Explore [examples_basic.py](examples_basic.py)
4. Check [QUICK_REF.md](QUICK_REF.md) for commands

**Want to use the API?**
- See [QUICK_REF.md](QUICK_REF.md) for all API examples
- Run `python main.py --api` for REST API
- Check `src/web/index.html` for web UI

**Need technical details?**
- [ARCHITECTURE.md](ARCHITECTURE.md) for system design
- [ARCHITECTURE.md](ARCHITECTURE.md#algorithm-implementations) for algorithm explanations
- [ARCHITECTURE.md](ARCHITECTURE.md#data-splitting-strategies) for training strategies

**Deploying to production?**
- [DOCKER_README.md](DOCKER_README.md) for containerization
- [QUICK_REF.md](QUICK_REF.md#troubleshooting-guide) for common issues
- [ARCHITECTURE.md](ARCHITECTURE.md#performance-characteristics) for performance metrics

## 🛠️ CLI Commands

```bash
# Complete pipeline (recommended for first run)
python main.py --full

# Individual steps
python main.py --download        # Download dataset from Kaggle
python main.py --prepare         # Process and prepare data
python main.py --train           # Train all models
python main.py --demo            # Run demonstration

# Start REST API server
python main.py --api

# Force reprocessing (ignore cache)
python main.py --prepare --reprocess
```

For complete command reference and troubleshooting, see [QUICK_REF.md](QUICK_REF.md).

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
- **Libraries**: scikit-learn, pandas, numpy, FastAPI
- **ML Algorithms**: Naive Bayes, Support Vector Machines, Ensemble Methods
- **Market Basket Analysis**: Apriori-like frequent itemset mining

## 📞 Support

Having issues? Check our troubleshooting resources:

1. **[QUICK_REF.md](QUICK_REF.md#troubleshooting-guide)** - Common issues and solutions
2. **Run tests**: `pytest tests/ -v` to verify installation
3. **Check logs**: `recommendation_service.log` for error details
4. **Review examples**: Working code in `examples_*.py` files

---

**Ready to recommend product bundles to your customers!** 🚀

For questions or feedback, please open an issue on GitHub.
