"""Unit tests for the recommendation service."""

import pytest
import numpy as np
import os
import tempfile
from unittest.mock import patch, MagicMock
from src.data_pipeline import DataPipeline
from src.recommendation_engine import (
    NaiveBayesBundleRecommender,
    SVMBundleRecommender,
    BundleRecommendationEngine,
)
from src.config import EngineConfig, PipelineConfig, LLMConfig, CacheConfig


# ============================================================================
# RECOMMENDATION-SPECIFIC FIXTURES
# ============================================================================

@pytest.fixture
def sample_transactions():
    """Basic transaction data for testing."""
    return [
        ["product_a", "product_b"],
        ["product_b", "product_c"],
        ["product_a", "product_c"],
        ["product_a", "product_b", "product_c"],
    ]


@pytest.fixture
def sample_bundles():
    """Basic bundle data for testing."""
    return [
        ("product_a", "product_b"),
        ("product_b", "product_c"),
    ]


@pytest.fixture
def large_transactions():
    """Larger transaction dataset for robust testing."""
    return [
        ["item_1", "item_2", "item_3"],
        ["item_2", "item_3"],
        ["item_1", "item_3"],
        ["item_1", "item_2"],
        ["item_2", "item_4"],
        ["item_3", "item_4"],
        ["item_1", "item_2", "item_3", "item_4"],
        ["item_1", "item_4"],
        ["item_2", "item_3", "item_4"],
        ["item_1", "item_2", "item_4"],
    ]


@pytest.fixture
def large_bundles():
    """Larger bundle dataset for robust testing."""
    return [
        ("item_1", "item_2"),
        ("item_2", "item_3"),
        ("item_3", "item_4"),
    ]


@pytest.fixture
def mock_inventory():
    """Mock inventory data for OOS testing."""
    return {
        "product_a": True,   # in stock
        "product_b": False,  # out of stock
        "product_c": True,
        "item_1": True,
        "item_2": False,
        "item_3": True,
        "item_4": True,
    }


@pytest.fixture
def mock_llm_alternatives():
    """Mock LLM alternative selection response."""
    return [
        {
            "item": "alternative_product_b",
            "score": 0.85,
            "reason": "Similar electronics category"
        }
    ]


@pytest.fixture
def mock_inventory_csv(tmp_path):
    """Create a temporary inventory CSV file."""
    import pandas as pd
    
    inventory_data = pd.DataFrame({
        "StockCode": ["A", "B", "C", "D"],
        "Description": ["product_a", "product_b", "product_c", "product_d"],
        "InStock": [True, False, True, True],
    })
    
    filepath = tmp_path / "inventory.csv"
    inventory_data.to_csv(filepath, index=False)
    return str(filepath)


@pytest.fixture
def custom_engine_config():
    """Custom engine config for injection tests."""
    return EngineConfig(svm_kernel="rbf", svm_c=2.5, random_state=123, n_jobs=1)


@pytest.fixture
def custom_pipeline_config():
    """Custom pipeline config for injection tests."""
    return PipelineConfig(
        train_test_split=0.73,
        llm_config=LLMConfig(),
        cache_config=CacheConfig(),
        oos_enabled=False,
    )


class TestDataPipeline:
    """Tests for data pipeline."""

    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = DataPipeline()
        assert pipeline.raw_data is None
        assert pipeline.processed_data is None
        assert pipeline.bundles is None

    def test_create_transaction_baskets(self):
        """Test transaction basket creation."""
        pipeline = DataPipeline()

        # Create mock data
        import pandas as pd

        pipeline.processed_data = pd.DataFrame(
            {
                "InvoiceNo": ["INV001", "INV001", "INV002"],
                "Description": ["product_a", "product_b", "product_c"],
                "CustomerID": [1, 1, 2],
                "InvoiceDate": ["2023-01-01", "2023-01-01", "2023-01-02"],
                "TransactionValue": [10, 20, 15],
                "Quantity": [1, 1, 1],
            }
        )

        baskets = pipeline.create_transaction_baskets()
        assert len(baskets) > 0
        assert "Items" in baskets.columns


class TestConfigInjection:
    """Tests for config injection behavior in recommendation components."""

    def test_svm_recommender_uses_injected_engine_config(self, custom_engine_config):
        """SVM recommender defaults should come from injected EngineConfig."""
        recommender = SVMBundleRecommender(config=custom_engine_config)
        assert recommender.kernel == "rbf"
        assert recommender.C == 2.5
        assert recommender.model.random_state == 123

    def test_fit_all_uses_pipeline_split_default(
        self,
        sample_transactions,
        sample_bundles,
        custom_engine_config,
        custom_pipeline_config,
    ):
        """BundleRecommendationEngine.fit_all should default to injected pipeline split."""
        engine = BundleRecommendationEngine(
            engine_config=custom_engine_config,
            pipeline_config=custom_pipeline_config,
        )
        mock_recommender = MagicMock()
        mock_recommender.fit.return_value = {"accuracy": 1.0}
        engine.add_recommender("mock", mock_recommender)

        engine.fit_all(sample_transactions, sample_bundles)

        assert mock_recommender.fit.called
        args = mock_recommender.fit.call_args[0]
        assert args[0] == sample_transactions
        assert args[1] == sample_bundles
        assert args[2] == pytest.approx(0.73)


class TestNaiveBayesRecommender:
    """Tests for Naive Bayes recommender."""

    def test_initialization_default(self):
        """Test recommender initialization with default model type."""
        recommender = NaiveBayesBundleRecommender()
        assert not recommender.is_fitted
        assert recommender.model is not None
        assert recommender.name == "NaiveBayesBundleRecommender"

    def test_initialization_multinomial(self):
        """Test initialization with multinomial model type."""
        recommender = NaiveBayesBundleRecommender(model_type="multinomial")
        assert recommender.model_type == "multinomial"
        
    def test_initialization_gaussian(self):
        """Test initialization with gaussian model type."""
        recommender = NaiveBayesBundleRecommender(model_type="gaussian")
        assert recommender.model_type == "gaussian"
        
    def test_initialization_invalid_model_type(self):
        """Test that invalid model type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model_type"):
            NaiveBayesBundleRecommender(model_type="invalid")

    def test_fit_basic(self, sample_transactions, sample_bundles):
        """Test basic model fitting."""
        recommender = NaiveBayesBundleRecommender()
        metrics = recommender.fit(sample_transactions, sample_bundles)

        assert recommender.is_fitted
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_fit_with_empty_transactions(self, sample_bundles):
        """Test that empty transactions raise ValueError from sklearn."""
        recommender = NaiveBayesBundleRecommender()
        with pytest.raises(ValueError):
            recommender.fit([], sample_bundles)

    def test_fit_with_empty_bundles(self, sample_transactions):
        """Test fitting with empty bundles (all labels become 0)."""
        recommender = NaiveBayesBundleRecommender()
        # Empty bundles means all transactions have y=0, which is valid
        metrics = recommender.fit(sample_transactions, [])
        assert recommender.is_fitted

    def test_fit_with_single_item_bundles(self, sample_transactions):
        """Test fitting with single-item bundles."""
        single_item_bundles = [("product_a",), ("product_b",)]
        recommender = NaiveBayesBundleRecommender()
        metrics = recommender.fit(sample_transactions, single_item_bundles)
        
        assert recommender.is_fitted
        assert "accuracy" in metrics

    def test_fit_all_positive_labels(self):
        """Test fitting when all transactions contain bundles (all y=1)."""
        transactions = [
            ["product_a", "product_b"],
            ["product_a", "product_b"],
            ["product_a", "product_b"],
        ]
        bundles = [("product_a", "product_b")]
        
        recommender = NaiveBayesBundleRecommender()
        metrics = recommender.fit(transactions, bundles)
        
        # Should still fit, even with imbalanced data
        assert recommender.is_fitted

    def test_fit_all_negative_labels(self):
        """Test fitting when no transactions contain bundles (all y=0)."""
        transactions = [
            ["product_c"],
            ["product_d"],
            ["product_e"],
        ]
        bundles = [("product_a", "product_b")]
        
        recommender = NaiveBayesBundleRecommender()
        metrics = recommender.fit(transactions, bundles)
        
        assert recommender.is_fitted

    def test_fit_with_kfold_splitter(self, large_transactions, large_bundles):
        """Test fitting with k-fold cross-validation."""
        from src.data_splitter import KFoldSplit
        
        recommender = NaiveBayesBundleRecommender()
        splitter = KFoldSplit(n_splits=3, random_state=42)
        
        metrics = recommender.fit_with_splitter(large_transactions, large_bundles, splitter)
        
        assert recommender.is_fitted
        assert "accuracy" in metrics
        assert "std_accuracy" in metrics
        assert "std_precision" in metrics
        assert metrics["std_accuracy"] >= 0

    def test_predict_before_fit_raises_error(self, sample_transactions):
        """Test that predict before fit raises ValueError."""
        recommender = NaiveBayesBundleRecommender()
        
        with pytest.raises(ValueError, match="Model not fitted"):
            recommender.predict([sample_transactions[0]])

    def test_predict_proba_before_fit_raises_error(self, sample_transactions):
        """Test that predict_proba before fit raises ValueError."""
        recommender = NaiveBayesBundleRecommender()
        
        with pytest.raises(ValueError, match="Model not fitted"):
            recommender.predict_proba([sample_transactions[0]])

    def test_predict_proba_basic(self, sample_transactions, sample_bundles):
        """Test probability predictions."""
        recommender = NaiveBayesBundleRecommender()
        recommender.fit(sample_transactions, sample_bundles)

        proba = recommender.predict_proba([["product_a", "product_b"]])
        assert proba is not None
        assert proba.shape[0] == 1
        assert proba.shape[1] == len(sample_bundles)
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_predict_proba_with_unknown_items(self, sample_transactions, sample_bundles):
        """Test prediction with items not seen during training."""
        recommender = NaiveBayesBundleRecommender()
        recommender.fit(sample_transactions, sample_bundles)
        
        # Predict with unknown items
        proba = recommender.predict_proba([["unknown_item_1", "unknown_item_2"]])
        assert proba is not None
        assert proba.shape[0] == 1

    def test_predict_proba_empty_transaction_after_filtering(self, sample_transactions, sample_bundles):
        """Test prediction when transaction becomes empty after filtering unknown items."""
        recommender = NaiveBayesBundleRecommender()
        recommender.fit(sample_transactions, sample_bundles)
        
        # All unknown items - should filter to empty
        proba = recommender.predict_proba([["completely_unknown"]])
        assert proba is not None

    def test_gaussian_vs_multinomial(self, large_transactions, large_bundles):
        """Test that both Gaussian and Multinomial NB can fit and predict."""
        for model_type in ["multinomial", "gaussian"]:
            recommender = NaiveBayesBundleRecommender(model_type=model_type)
            metrics = recommender.fit(large_transactions, large_bundles)
            
            assert recommender.is_fitted
            assert "accuracy" in metrics
            
            proba = recommender.predict_proba([large_transactions[0]])
            assert proba is not None


class TestSVMRecommender:
    """Tests for SVM recommender."""

    def test_initialization_default(self):
        """Test recommender initialization with default parameters."""
        recommender = SVMBundleRecommender()
        assert not recommender.is_fitted
        assert recommender.model is not None
        assert recommender.name == "SVMBundleRecommender"
        assert recommender.kernel == "linear"

    def test_initialization_custom_kernel(self):
        """Test initialization with custom kernel."""
        for kernel in ["linear", "rbf", "poly", "sigmoid"]:
            recommender = SVMBundleRecommender(kernel=kernel)
            assert recommender.kernel == kernel

    def test_initialization_custom_c(self):
        """Test initialization with custom C parameter."""
        recommender = SVMBundleRecommender(C=10.0)
        assert recommender.model.C == 10.0

    def test_fit_basic(self, sample_transactions, sample_bundles):
        """Test basic model fitting."""
        recommender = SVMBundleRecommender()
        metrics = recommender.fit(sample_transactions, sample_bundles)

        assert recommender.is_fitted
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_fit_with_empty_transactions(self, sample_bundles):
        """Test that empty transactions raise ValueError from sklearn."""
        recommender = SVMBundleRecommender()
        with pytest.raises(ValueError):
            recommender.fit([], sample_bundles)

    def test_fit_with_empty_bundles(self, sample_transactions):
        """Test fitting with empty bundles (SVM requires >= 2 classes)."""
        recommender = SVMBundleRecommender()
        # Empty bundles means all transactions have y=0 (single class)
        # SVM requires at least 2 classes, so this will raise ValueError
        with pytest.raises(ValueError, match="number of classes"):
            recommender.fit(sample_transactions, [])

    def test_different_kernels(self, large_transactions, large_bundles):
        """Test different SVM kernels."""
        kernels = ["linear", "rbf", "poly"]
        
        for kernel in kernels:
            recommender = SVMBundleRecommender(kernel=kernel)
            metrics = recommender.fit(large_transactions, large_bundles)
            
            assert recommender.is_fitted
            assert "accuracy" in metrics
            assert 0 <= metrics["accuracy"] <= 1

    def test_different_c_values(self, large_transactions, large_bundles):
        """Test different regularization C values."""
        c_values = [0.1, 1.0, 10.0]
        
        for c in c_values:
            recommender = SVMBundleRecommender(C=c)
            metrics = recommender.fit(large_transactions, large_bundles)
            
            assert recommender.is_fitted
            assert "accuracy" in metrics

    def test_feature_scaling(self, sample_transactions, sample_bundles):
        """Test that feature scaling is applied."""
        recommender = SVMBundleRecommender()
        recommender.fit(sample_transactions, sample_bundles)
        
        # Scaler should be fitted
        assert recommender.scaler is not None
        assert hasattr(recommender.scaler, 'mean_')

    def test_fit_with_kfold_splitter(self, large_transactions, large_bundles):
        """Test fitting with k-fold cross-validation."""
        from src.data_splitter import KFoldSplit
        
        recommender = SVMBundleRecommender()
        splitter = KFoldSplit(n_splits=3, random_state=42)
        
        metrics = recommender.fit_with_splitter(large_transactions, large_bundles, splitter)
        
        assert recommender.is_fitted
        assert "accuracy" in metrics
        assert "std_accuracy" in metrics
        assert metrics["std_accuracy"] >= 0

    def test_predict_before_fit_raises_error(self, sample_transactions):
        """Test that predict before fit raises ValueError."""
        recommender = SVMBundleRecommender()
        
        with pytest.raises(ValueError, match="Model not fitted"):
            recommender.predict([sample_transactions[0]])

    def test_predict_proba_before_fit_raises_error(self, sample_transactions):
        """Test that predict_proba before fit raises ValueError."""
        recommender = SVMBundleRecommender()
        
        with pytest.raises(ValueError, match="Model not fitted"):
            recommender.predict_proba([sample_transactions[0]])

    def test_predict_proba_basic(self, sample_transactions, sample_bundles):
        """Test probability predictions."""
        recommender = SVMBundleRecommender()
        recommender.fit(sample_transactions, sample_bundles)

        proba = recommender.predict_proba([["product_a", "product_b"]])
        assert proba is not None
        assert proba.shape[0] == 1
        assert proba.shape[1] == len(sample_bundles)
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_predict_proba_with_unknown_items(self, sample_transactions, sample_bundles):
        """Test prediction with items not seen during training."""
        recommender = SVMBundleRecommender()
        recommender.fit(sample_transactions, sample_bundles)
        
        # Predict with unknown items
        proba = recommender.predict_proba([["unknown_item_1", "unknown_item_2"]])
        assert proba is not None
        assert proba.shape[0] == 1

    def test_scaling_consistency(self, large_transactions, large_bundles):
        """Test that scaling is applied consistently in fit and predict."""
        recommender = SVMBundleRecommender()
        recommender.fit(large_transactions, large_bundles)
        
        # Get predictions
        proba1 = recommender.predict_proba([large_transactions[0]])
        proba2 = recommender.predict_proba([large_transactions[0]])
        
        # Should be identical (deterministic)
        np.testing.assert_array_almost_equal(proba1, proba2)


class TestBundleRecommendationEngine:
    """Tests for recommendation engine."""

    def test_initialization(self):
        """Test engine initialization."""
        engine = BundleRecommendationEngine()
        assert len(engine.recommenders) == 0
        assert engine.bundles is None
        assert engine.transactions is None

    def test_add_recommender(self):
        """Test adding a single recommender."""
        engine = BundleRecommendationEngine()
        recommender = NaiveBayesBundleRecommender()
        engine.add_recommender("nb", recommender)

        assert "nb" in engine.recommenders
        assert engine.recommenders["nb"] == recommender

    def test_add_multiple_recommenders(self):
        """Test adding multiple recommenders."""
        engine = BundleRecommendationEngine()
        engine.add_recommender("nb", NaiveBayesBundleRecommender())
        engine.add_recommender("svm", SVMBundleRecommender())
        
        assert len(engine.recommenders) == 2
        assert "nb" in engine.recommenders
        assert "svm" in engine.recommenders

    def test_fit_all_without_recommenders(self, sample_transactions, sample_bundles):
        """Test that fit_all with no recommenders returns empty metrics."""
        engine = BundleRecommendationEngine()
        
        # Should return empty metrics dict, not raise error
        metrics = engine.fit_all(sample_transactions, sample_bundles)
        assert metrics == {}

    def test_fit_all_with_empty_transactions(self, sample_bundles):
        """Test that fit_all with empty transactions raises ValueError from sklearn."""
        engine = BundleRecommendationEngine()
        engine.add_recommender("nb", NaiveBayesBundleRecommender())
        
        with pytest.raises(ValueError):
            engine.fit_all([], sample_bundles)

    def test_fit_all_with_empty_bundles(self, sample_transactions):
        """Test fit_all with empty bundles (all y=0 for NaiveBayes)."""
        engine = BundleRecommendationEngine()
        engine.add_recommender("nb", NaiveBayesBundleRecommender())
        
        # NB can handle empty bundles (all labels = 0)
        metrics = engine.fit_all(sample_transactions, [])
        assert "nb" in metrics

    def test_fit_all_single_recommender(self, sample_transactions, sample_bundles):
        """Test fitting with a single recommender."""
        engine = BundleRecommendationEngine()
        engine.add_recommender("nb", NaiveBayesBundleRecommender())

        metrics = engine.fit_all(sample_transactions, sample_bundles)
        
        assert "nb" in metrics
        assert "accuracy" in metrics["nb"]
        assert engine.bundles == sample_bundles
        assert engine.transactions == sample_transactions

    def test_fit_all_multiple_recommenders(self, large_transactions, large_bundles):
        """Test fitting with multiple recommenders."""
        engine = BundleRecommendationEngine()
        engine.add_recommender("nb", NaiveBayesBundleRecommender())
        engine.add_recommender("svm", SVMBundleRecommender())

        metrics = engine.fit_all(large_transactions, large_bundles)
        
        assert "nb" in metrics
        assert "svm" in metrics
        assert "accuracy" in metrics["nb"]
        assert "accuracy" in metrics["svm"]

    def test_fit_all_with_kfold(self, large_transactions, large_bundles):
        """Test fit_all_with_kfold method."""
        engine = BundleRecommendationEngine()
        engine.add_recommender("nb", NaiveBayesBundleRecommender())
        engine.add_recommender("svm", SVMBundleRecommender())

        metrics = engine.fit_all_with_kfold(
            large_transactions, 
            large_bundles,
            n_splits=3
        )
        
        assert "nb" in metrics
        assert "svm" in metrics
        assert "std_accuracy" in metrics["nb"]
        assert "std_accuracy" in metrics["svm"]
        assert metrics["nb"]["std_accuracy"] >= 0

    def test_fit_all_with_random_split(self, large_transactions, large_bundles):
        """Test fit_all_with_random_split method."""
        engine = BundleRecommendationEngine()
        engine.add_recommender("nb", NaiveBayesBundleRecommender())
        engine.add_recommender("svm", SVMBundleRecommender())

        metrics = engine.fit_all_with_random_split(
            large_transactions,
            large_bundles,
            test_size=0.2
        )
        
        assert "nb" in metrics
        assert "svm" in metrics
        assert "accuracy" in metrics["nb"]

    def test_recommend_bundles_without_fit(self, sample_transactions):
        """Test that recommend_bundles raises error when not fitted."""
        engine = BundleRecommendationEngine()
        engine.add_recommender("nb", NaiveBayesBundleRecommender())
        
        with pytest.raises(ValueError, match="Bundles not set"):
            engine.recommend_bundles(sample_transactions[0])

    def test_recommend_bundles_single_recommender(self, sample_transactions, sample_bundles):
        """Test bundle recommendations with single recommender."""
        engine = BundleRecommendationEngine()
        engine.add_recommender("nb", NaiveBayesBundleRecommender())
        engine.fit_all(sample_transactions, sample_bundles)

        recs = engine.recommend_bundles(["product_a"], recommender_name="nb", threshold=0.0)
        
        assert "bundles" in recs
        assert "confidence" in recs
        assert "transaction" in recs
        assert "recommender" in recs
        assert recs["recommender"] == "nb"
        assert isinstance(recs["bundles"], list)

    def test_recommend_bundles_ensemble_averaging(self, large_transactions, large_bundles):
        """Test ensemble averaging with multiple recommenders."""
        engine = BundleRecommendationEngine()
        engine.add_recommender("nb", NaiveBayesBundleRecommender())
        engine.add_recommender("svm", SVMBundleRecommender())
        engine.fit_all(large_transactions, large_bundles)

        # Ensemble mode (recommender_name=None)
        recs = engine.recommend_bundles(["item_1", "item_2"], threshold=0.0)
        
        assert "bundles" in recs
        assert "confidence" in recs
        assert "recommender" in recs
        assert recs["recommender"] == "ensemble"
        
        # Confidence should be a float (averaged from both models)
        assert isinstance(recs["confidence"], (int, float))
        assert 0 <= recs["confidence"] <= 1

    def test_recommend_bundles_with_unknown_recommender(self, sample_transactions, sample_bundles):
        """Test that unknown recommender name raises ValueError."""
        engine = BundleRecommendationEngine()
        engine.add_recommender("nb", NaiveBayesBundleRecommender())
        engine.fit_all(sample_transactions, sample_bundles)
        
        with pytest.raises(ValueError, match="not found"):
            engine.recommend_bundles(["product_a"], recommender_name="unknown")

    def test_threshold_filtering_zero(self, large_transactions, large_bundles):
        """Test threshold=0.0 returns all bundles."""
        engine = BundleRecommendationEngine()
        engine.add_recommender("nb", NaiveBayesBundleRecommender())
        engine.fit_all(large_transactions, large_bundles)

        recs = engine.recommend_bundles(["item_1"], threshold=0.0)
        
        # With threshold=0, should return bundles (up to top 5)
        assert len(recs["bundles"]) <= 5

    def test_threshold_filtering_high(self, large_transactions, large_bundles):
        """Test threshold=0.9 filters aggressively."""
        engine = BundleRecommendationEngine()
        engine.add_recommender("nb", NaiveBayesBundleRecommender())
        engine.fit_all(large_transactions, large_bundles)

        recs = engine.recommend_bundles(["item_1"], threshold=0.9)
        
        # Should filter bundles based on confidence
        assert isinstance(recs["bundles"], list)
        assert isinstance(recs["confidence"], (int, float))

    def test_threshold_filtering_medium(self, large_transactions, large_bundles):
        """Test threshold=0.5 for moderate filtering."""
        engine = BundleRecommendationEngine()
        engine.add_recommender("nb", NaiveBayesBundleRecommender())
        engine.fit_all(large_transactions, large_bundles)

        recs = engine.recommend_bundles(["item_1"], threshold=0.5)
        
        assert isinstance(recs["confidence"], (int, float))

    def test_recommend_bundles_top_n_limit(self, large_transactions, large_bundles):
        """Test that only top 5 bundles are returned."""
        engine = BundleRecommendationEngine()
        engine.add_recommender("nb", NaiveBayesBundleRecommender())
        engine.fit_all(large_transactions, large_bundles)

        recs = engine.recommend_bundles(["item_1"], threshold=0.0)
        
        # Should return at most 5 bundles
        assert len(recs["bundles"]) <= 5

    @patch('src.recommendation_engine.OOS_ENABLED', True)
    @patch('src.recommendation_engine.load_inventory_csv')
    @patch('src.recommendation_engine.select_alternatives_with_llm')
    def test_oos_substitution_enabled(
        self, 
        mock_select_llm, 
        mock_load_inv,
        large_transactions, 
        large_bundles,
        mock_inventory,
        mock_llm_alternatives
    ):
        """Test OOS substitution when enabled."""
        mock_load_inv.return_value = mock_inventory
        mock_select_llm.return_value = mock_llm_alternatives
        
        engine = BundleRecommendationEngine()
        engine.add_recommender("nb", NaiveBayesBundleRecommender())
        engine.fit_all(large_transactions, large_bundles)

        recs = engine.recommend_bundles(["item_1"], threshold=0.0)
        
        assert "bundles" in recs
        assert "confidence" in recs
        # bundle_substitutions only appears if OOS items were substituted
        # It may not appear if no items were out of stock

    @patch('src.recommendation_engine.OOS_ENABLED', False)
    def test_oos_substitution_disabled(
        self,
        large_transactions,
        large_bundles
    ):
        """Test that OOS is not called when disabled."""
        engine = BundleRecommendationEngine()
        engine.add_recommender("nb", NaiveBayesBundleRecommender())
        engine.fit_all(large_transactions, large_bundles)

        recs = engine.recommend_bundles(["item_1"], threshold=0.0)
        
        # Should not have bundle_substitutions when disabled
        assert "bundle_substitutions" not in recs

    @patch('src.recommendation_engine.OOS_ENABLED', True)
    @patch('src.recommendation_engine.load_inventory_csv')
    @patch('src.recommendation_engine.select_alternatives_with_llm')
    @patch('src.recommendation_engine.select_alternative_heuristic')
    def test_oos_fallback_to_heuristic_on_llm_error(
        self,
        mock_heuristic,
        mock_select_llm,
        mock_load_inv,
        large_transactions,
        large_bundles,
        mock_inventory
    ):
        """Test fallback to heuristic when LLM fails."""
        from src.utils import LLMQuotaExceededError
        
        mock_load_inv.return_value = mock_inventory
        mock_select_llm.side_effect = LLMQuotaExceededError("Quota exceeded")
        mock_heuristic.return_value = {"item": "heuristic_item", "score": 0.6}
        
        engine = BundleRecommendationEngine()
        engine.add_recommender("nb", NaiveBayesBundleRecommender())
        engine.fit_all(large_transactions, large_bundles)

        # Should not crash even with LLM error
        recs = engine.recommend_bundles(["item_1"], threshold=0.0)
        
        assert "bundles" in recs

    @patch('src.recommendation_engine.OOS_ENABLED', True)
    @patch('src.recommendation_engine.load_inventory_csv')
    def test_oos_missing_inventory_file(
        self,
        mock_load_inv,
        large_transactions,
        large_bundles
    ):
        """Test handling of missing inventory file."""
        mock_load_inv.return_value = {}
        
        engine = BundleRecommendationEngine()
        engine.add_recommender("nb", NaiveBayesBundleRecommender())
        engine.fit_all(large_transactions, large_bundles)

        # Should not crash with empty inventory
        recs = engine.recommend_bundles(["item_1"], threshold=0.0)
        
        assert "bundles" in recs

    def test_cross_sell_products_basic(self, large_transactions, large_bundles):
        """Test basic cross-sell functionality."""
        engine = BundleRecommendationEngine()
        engine.add_recommender("nb", NaiveBayesBundleRecommender())
        engine.fit_all(large_transactions, large_bundles)

        cross_sell = engine.get_cross_sell_products(["item_1"], top_n=3)
        
        assert isinstance(cross_sell, list)
        assert len(cross_sell) <= 3
        
        for item, score in cross_sell:
            assert isinstance(item, str)
            assert isinstance(score, (int, float))
            assert item != "item_1"  # Should not recommend item already in basket

    def test_cross_sell_without_bundles(self, sample_transactions):
        """Test that cross_sell raises error when bundles not set."""
        engine = BundleRecommendationEngine()
        engine.add_recommender("nb", NaiveBayesBundleRecommender())
        # Don't fit
        
        with pytest.raises(ValueError, match="Bundles not set"):
            engine.get_cross_sell_products(["product_a"])

    def test_cross_sell_empty_transaction(self, large_transactions, large_bundles):
        """Test cross-sell with empty transaction."""
        engine = BundleRecommendationEngine()
        engine.add_recommender("nb", NaiveBayesBundleRecommender())
        engine.fit_all(large_transactions, large_bundles)

        cross_sell = engine.get_cross_sell_products([])
        
        # Should return something (all bundle items are valid)
        assert isinstance(cross_sell, list)

    def test_cross_sell_top_n_respected(self, large_transactions, large_bundles):
        """Test that top_n parameter is respected."""
        engine = BundleRecommendationEngine()
        engine.add_recommender("nb", NaiveBayesBundleRecommender())
        engine.fit_all(large_transactions, large_bundles)

        for n in [1, 2, 5]:
            cross_sell = engine.get_cross_sell_products(["item_1"], top_n=n)
            assert len(cross_sell) <= n

    def test_model_persistence_save_load(self, tmp_path, large_transactions, large_bundles):
        """Test saving and loading model."""
        engine = BundleRecommendationEngine()
        engine.add_recommender("nb", NaiveBayesBundleRecommender())
        engine.add_recommender("svm", SVMBundleRecommender())
        engine.fit_all(large_transactions, large_bundles)

        # Save model
        model_path = tmp_path / "test_model.pkl"
        result = engine.save_model(str(model_path))
        
        assert result is True
        assert model_path.exists()

        # Load model into new engine
        loaded_engine = BundleRecommendationEngine()
        result = loaded_engine.load_model(str(model_path))
        
        assert result is True
        assert len(loaded_engine.recommenders) == 2
        assert "nb" in loaded_engine.recommenders
        assert "svm" in loaded_engine.recommenders
        assert loaded_engine.bundles == large_bundles

    def test_model_persistence_prediction_consistency(
        self, 
        tmp_path, 
        large_transactions, 
        large_bundles
    ):
        """Test that predictions are consistent after save/load."""
        engine = BundleRecommendationEngine()
        engine.add_recommender("nb", NaiveBayesBundleRecommender())
        engine.fit_all(large_transactions, large_bundles)

        # Get original prediction
        recs_original = engine.recommend_bundles(["item_1", "item_2"], threshold=0.0)

        # Save and load
        model_path = tmp_path / "test_model.pkl"
        engine.save_model(str(model_path))
        
        loaded_engine = BundleRecommendationEngine()
        loaded_engine.load_model(str(model_path))

        # Get prediction from loaded model
        recs_loaded = loaded_engine.recommend_bundles(["item_1", "item_2"], threshold=0.0)

        # Should be identical
        assert recs_original["bundles"] == recs_loaded["bundles"]
        assert abs(recs_original["confidence"] - recs_loaded["confidence"]) < 0.001

    def test_get_stats_basic(self, large_transactions, large_bundles):
        """Test get_engine_stats returns correct structure."""
        engine = BundleRecommendationEngine()
        engine.add_recommender("nb", NaiveBayesBundleRecommender())
        engine.add_recommender("svm", SVMBundleRecommender())
        engine.fit_all(large_transactions, large_bundles)

        stats = engine.get_engine_stats()
        
        assert "num_recommenders" in stats
        assert "recommender_names" in stats
        assert "num_bundles" in stats
        assert "num_transactions" in stats
        
        assert stats["num_recommenders"] == 2
        assert set(stats["recommender_names"]) == {"nb", "svm"}
        assert stats["num_bundles"] == len(large_bundles)
        assert stats["num_transactions"] == len(large_transactions)

    def test_get_stats_before_fit(self):
        """Test get_engine_stats before fitting."""
        engine = BundleRecommendationEngine()
        engine.add_recommender("nb", NaiveBayesBundleRecommender())

        stats = engine.get_engine_stats()
        
        assert stats["num_recommenders"] == 1
        assert stats["num_bundles"] == 0
        assert stats["num_transactions"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
