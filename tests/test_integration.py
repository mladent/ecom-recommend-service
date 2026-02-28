"""Integration tests for end-to-end workflow coverage."""

import json
import os
import tempfile
from unittest.mock import patch

import pytest

from src.api import app
from src.config import load_config
from src.data_pipeline import DataPipeline
from src.llm_client import LLMOperationTracker
from src.recommendation_engine import (
    BundleRecommendationEngine,
    NaiveBayesBundleRecommender,
)


@pytest.fixture
def sample_transactions_csv() -> str:
    """Create a small realistic CSV for end-to-end pipeline tests."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as handle:
        handle.write(
            """InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country
INV001,S001,Widget A,2,2023-01-01,10.0,10001,UK
INV001,S002,Widget B,1,2023-01-01,15.0,10001,UK
INV002,S001,Widget A,1,2023-01-02,10.0,10002,UK
INV002,S003,Widget C,1,2023-01-02,20.0,10002,UK
INV003,S002,Widget B,2,2023-01-03,15.0,10003,UK
INV003,S003,Widget C,1,2023-01-03,20.0,10003,UK
INV004,S001,Widget A,1,2023-01-04,10.0,10004,UK
INV004,S002,Widget B,1,2023-01-04,15.0,10004,UK
"""
        )
        path = handle.name
    try:
        yield path
    finally:
        if os.path.exists(path):
            os.unlink(path)


@pytest.fixture
def trained_engine(sample_transactions_csv: str) -> BundleRecommendationEngine:
    """Train a small engine for integration scenarios."""
    pipeline_config, engine_config, _, _, _, _ = load_config()

    pipeline = DataPipeline(config=pipeline_config)
    pipeline.load_raw_data(sample_transactions_csv)
    pipeline.preprocess()
    pipeline.create_transaction_baskets()
    bundles = pipeline.generate_product_bundles(min_support=0.2)

    engine = BundleRecommendationEngine(
        engine_config=engine_config,
        pipeline_config=pipeline_config,
    )
    engine.add_recommender("naive_bayes", NaiveBayesBundleRecommender())
    engine.fit_all(pipeline.transactions["Items"].tolist(), bundles)
    return engine


@pytest.fixture
def api_client():
    """Flask API test client."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestEndToEndWorkflow:
    """Covers: load -> preprocess -> train -> evaluate -> recommend."""

    def test_full_pipeline_workflow(self, sample_transactions_csv: str):
        pipeline_config, engine_config, _, _, _, _ = load_config()

        pipeline = DataPipeline(config=pipeline_config)
        raw = pipeline.load_raw_data(sample_transactions_csv)
        assert len(raw) > 0

        processed = pipeline.preprocess()
        assert not processed.empty

        baskets = pipeline.create_transaction_baskets()
        assert not baskets.empty

        bundles = pipeline.generate_product_bundles(min_support=0.2)
        assert len(bundles) > 0

        engine = BundleRecommendationEngine(
            engine_config=engine_config,
            pipeline_config=pipeline_config,
        )
        engine.add_recommender("naive_bayes", NaiveBayesBundleRecommender())

        metrics = engine.fit_all(baskets["Items"].tolist(), bundles)
        assert "naive_bayes" in metrics
        assert "accuracy" in metrics["naive_bayes"]

        recommendation = engine.recommend_bundles(["Widget A"], threshold=0.0)
        assert "bundles" in recommendation
        assert "confidence" in recommendation


class TestModelPersistence:
    """Covers save -> load -> predict consistency."""

    def test_save_load_consistency(self, trained_engine: BundleRecommendationEngine):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as handle:
            model_path = handle.name
        try:
            assert trained_engine.save_model(model_path)

            pipeline_config, engine_config, _, _, _, _ = load_config()
            loaded_engine = BundleRecommendationEngine(
                engine_config=engine_config,
                pipeline_config=pipeline_config,
            )
            assert loaded_engine.load_model(model_path)

            before = trained_engine.recommend_bundles(["Widget A"], threshold=0.0)
            after = loaded_engine.recommend_bundles(["Widget A"], threshold=0.0)

            assert before["bundles"] == after["bundles"]
            assert pytest.approx(before["confidence"], rel=1e-9) == after["confidence"]
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)


class TestApiEngineIntegration:
    """Covers API + trained engine integration."""

    def test_bundles_endpoint_with_trained_engine(self, api_client, trained_engine: BundleRecommendationEngine):
        with patch("src.api.get_engine", return_value=trained_engine):
            response = api_client.get(
                "/api/v1/bundles",
                query_string={"product_description": "Widget A", "threshold": 0.0, "top_n": 3},
            )

        assert response.status_code == 200
        payload = json.loads(response.data)
        assert payload["status"] == "success"
        assert "recommendations" in payload
        assert payload["total_models"] >= 1


class TestLlmAndFallbackIntegration:
    """Covers LLM tracker usage and fallback behavior."""

    def test_llm_tracker_aggregation(self):
        tracker = LLMOperationTracker()
        tracker.reset()

        tracker.record_operation("enrich_categories", latency_ms=120.0, cached=False, provider="openai")
        tracker.record_operation("enrich_categories", latency_ms=80.0, cached=True, provider="openai")
        tracker.record_operation("select_alternatives", latency_ms=150.0, cached=False, provider="gemini")

        assert tracker.get_total_calls() == 3
        assert tracker.get_total_cache_hits() == 1

        operation_stats = tracker.get_operation_stats("enrich_categories")
        assert operation_stats is not None
        assert operation_stats.total_calls == 2
        assert operation_stats.cache_hits == 1

    def test_recommendation_fallback_with_missing_inventory(self, trained_engine: BundleRecommendationEngine):
        pipeline_config, _, _, _, _, _ = load_config()
        pipeline_config.oos_enabled = True
        pipeline_config.oos_inventory_path = "/tmp/does-not-exist.csv"

        result = trained_engine.recommend_bundles(["Widget A"], threshold=0.0)
        assert "bundles" in result
        assert "confidence" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
