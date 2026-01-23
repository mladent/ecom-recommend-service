"""Unit tests for the recommendation service."""

import pytest
import numpy as np
from src.data_pipeline import DataPipeline
from src.recommendation_engine import (
    NaiveBayesBundleRecommender,
    SVMBundleRecommender,
    BundleRecommendationEngine,
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


class TestNaiveBayesRecommender:
    """Tests for Naive Bayes recommender."""

    def test_initialization(self):
        """Test recommender initialization."""
        recommender = NaiveBayesBundleRecommender()
        assert not recommender.is_fitted
        assert recommender.model is not None

    def test_fit_and_predict(self):
        """Test model fitting and prediction."""
        transactions = [
            ["product_a", "product_b"],
            ["product_b", "product_c"],
            ["product_a", "product_c"],
        ]
        bundles = [("product_a", "product_b")]

        recommender = NaiveBayesBundleRecommender()
        metrics = recommender.fit(transactions, bundles)

        assert recommender.is_fitted
        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_predict_proba(self):
        """Test probability predictions."""
        transactions = [
            ["product_a", "product_b"],
            ["product_b", "product_c"],
            ["product_a", "product_c"],
        ]
        bundles = [("product_a", "product_b")]

        recommender = NaiveBayesBundleRecommender()
        recommender.fit(transactions, bundles)

        proba = recommender.predict_proba([["product_a", "product_b"]])
        assert proba is not None
        assert proba.shape[0] == 1


class TestSVMRecommender:
    """Tests for SVM recommender."""

    def test_initialization(self):
        """Test recommender initialization."""
        recommender = SVMBundleRecommender()
        assert not recommender.is_fitted
        assert recommender.model is not None

    def test_fit_and_predict(self):
        """Test model fitting and prediction."""
        transactions = [
            ["product_a", "product_b"],
            ["product_b", "product_c"],
            ["product_a", "product_c"],
        ]
        bundles = [("product_a", "product_b")]

        recommender = SVMBundleRecommender()
        metrics = recommender.fit(transactions, bundles)

        assert recommender.is_fitted
        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_different_kernels(self):
        """Test different SVM kernels."""
        transactions = [
            ["product_a", "product_b"],
            ["product_b", "product_c"],
            ["product_a", "product_c"],
        ]
        bundles = [("product_a", "product_b")]

        for kernel in ["linear", "rbf", "poly"]:
            recommender = SVMBundleRecommender(kernel=kernel)
            metrics = recommender.fit(transactions, bundles)
            assert recommender.is_fitted


class TestBundleRecommendationEngine:
    """Tests for recommendation engine."""

    def test_initialization(self):
        """Test engine initialization."""
        engine = BundleRecommendationEngine()
        assert len(engine.recommenders) == 0

    def test_add_recommender(self):
        """Test adding recommender."""
        engine = BundleRecommendationEngine()
        recommender = NaiveBayesBundleRecommender()
        engine.add_recommender("nb", recommender)

        assert "nb" in engine.recommenders

    def test_fit_all(self):
        """Test fitting all recommenders."""
        transactions = [
            ["product_a", "product_b"],
            ["product_b", "product_c"],
            ["product_a", "product_c"],
        ]
        bundles = [("product_a", "product_b")]

        engine = BundleRecommendationEngine()
        engine.add_recommender("nb", NaiveBayesBundleRecommender())
        engine.add_recommender("svm", SVMBundleRecommender())

        metrics = engine.fit_all(transactions, bundles)
        assert "nb" in metrics
        assert "svm" in metrics

    def test_recommend_bundles(self):
        """Test bundle recommendations."""
        transactions = [
            ["product_a", "product_b"],
            ["product_b", "product_c"],
            ["product_a", "product_c"],
        ]
        bundles = [("product_a", "product_b")]

        engine = BundleRecommendationEngine()
        engine.add_recommender("nb", NaiveBayesBundleRecommender())
        engine.fit_all(transactions, bundles)

        recs = engine.recommend_bundles(["product_a"], recommender_name="nb", threshold=0.3)
        assert "bundles" in recs
        assert "confidence" in recs

    def test_cross_sell_products(self):
        """Test cross-sell recommendations."""
        transactions = [
            ["product_a", "product_b"],
            ["product_b", "product_c"],
            ["product_a", "product_c"],
        ]
        bundles = [
            ("product_a", "product_b"),
            ("product_b", "product_c"),
        ]

        engine = BundleRecommendationEngine()
        engine.add_recommender("nb", NaiveBayesBundleRecommender())
        engine.fit_all(transactions, bundles)

        cross_sell = engine.get_cross_sell_products(["product_a"], top_n=2)
        assert isinstance(cross_sell, list)
        assert len(cross_sell) <= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
