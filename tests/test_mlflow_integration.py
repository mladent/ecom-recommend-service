"""
Tests for MLflow Integration

Tests MLflow experiment tracking, decorators, and integration with training pipelines.
"""

import os
import sys
import pytest
import tempfile
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.config import MLflowConfig
from src.mlflow_client import MLflowExperimentTracker, _NoOpContext
from src.recommendation_engine import (
    BundleRecommendationEngine,
    NaiveBayesBundleRecommender,
    SVMBundleRecommender,
)
from src.utils import init_mlflow_tracking


# ============================================================================
# MLFLOW CONFIG TESTS
# ============================================================================

class TestMLflowConfig:
    """Tests for MLflowConfig dataclass."""
    
    def test_mlflow_config_defaults(self):
        """Test default MLflow configuration."""
        config = MLflowConfig()
        
        assert config.enabled is False
        assert config.tracking_uri == "mlruns"
        assert config.experiment_name == "bundle-recommendation-engine"
        assert config.run_name_prefix == ""
        assert config.log_system_metrics is True
    
    def test_mlflow_config_custom_values(self):
        """Test MLflow configuration with custom values."""
        config = MLflowConfig(
            enabled=True,
            tracking_uri="http://localhost:5000",
            experiment_name="test_experiment",
            run_name_prefix="test_",
            log_system_metrics=False,
        )
        
        assert config.enabled is True
        assert config.tracking_uri == "http://localhost:5000"
        assert config.experiment_name == "test_experiment"
        assert config.run_name_prefix == "test_"
        assert config.log_system_metrics is False
    
    def test_mlflow_config_validation_tracking_uri(self):
        """Test validation of tracking URI when enabled."""
        with pytest.raises(ValueError, match="tracking_uri must be set"):
            MLflowConfig(enabled=True, tracking_uri="")
    
    def test_mlflow_config_validation_experiment_name(self):
        """Test validation of experiment name when enabled."""
        with pytest.raises(ValueError, match="experiment_name must be set"):
            MLflowConfig(enabled=True, experiment_name="")


# ============================================================================
# MLFLOW TRACKER TESTS
# ============================================================================

class TestMLflowExperimentTracker:
    """Tests for MLflowExperimentTracker class."""
    
    def test_tracker_initialization_disabled(self):
        """Test tracker initialization when disabled."""
        config = MLflowConfig(enabled=False)
        tracker = MLflowExperimentTracker(config)
        
        assert tracker.enabled is False
        assert tracker.config == config
    
    def test_tracker_initialization_no_config(self):
        """Test tracker initialization with no config."""
        tracker = MLflowExperimentTracker(None)
        
        assert tracker.enabled is False
        assert tracker.config is None
    
    @patch('src.mlflow_client.MLFLOW_AVAILABLE', False)
    def test_tracker_initialization_mlflow_unavailable(self):
        """Test tracker initialization when MLflow is not installed."""
        config = MLflowConfig(enabled=True)
        tracker = MLflowExperimentTracker(config)
        
        assert tracker.enabled is False
    
    def test_no_op_context(self):
        """Test no-op context manager."""
        context = _NoOpContext()
        
        with context as c:
            assert c is None
        # Should not raise any exceptions
    
    def test_start_run_disabled(self):
        """Test start_run returns no-op context when disabled."""
        config = MLflowConfig(enabled=False)
        tracker = MLflowExperimentTracker(config)
        
        context = tracker.start_run("test_run")
        
        assert isinstance(context, _NoOpContext)
    
    def test_log_params_disabled(self):
        """Test log_params is no-op when disabled."""
        config = MLflowConfig(enabled=False)
        tracker = MLflowExperimentTracker(config)
        
        # Should not raise any exceptions
        tracker.log_params({"param1": "value1", "param2": 2})
    
    def test_log_param_disabled(self):
        """Test log_param is no-op when disabled."""
        config = MLflowConfig(enabled=False)
        tracker = MLflowExperimentTracker(config)
        
        # Should not raise any exceptions
        tracker.log_param("key", "value")
    
    def test_log_metrics_disabled(self):
        """Test log_metrics is no-op when disabled."""
        config = MLflowConfig(enabled=False)
        tracker = MLflowExperimentTracker(config)
        
        # Should not raise any exceptions
        tracker.log_metrics({"metric1": 0.5, "metric2": 0.8})
    
    def test_log_metric_disabled(self):
        """Test log_metric is no-op when disabled."""
        config = MLflowConfig(enabled=False)
        tracker = MLflowExperimentTracker(config)
        
        # Should not raise any exceptions
        tracker.log_metric("accuracy", 0.92)
    
    def test_log_artifact_disabled(self):
        """Test log_artifact is no-op when disabled."""
        config = MLflowConfig(enabled=False)
        tracker = MLflowExperimentTracker(config)
        
        # Should not raise any exceptions
        tracker.log_artifact("/path/to/file", "model")
    
    def test_set_tags_disabled(self):
        """Test set_tags is no-op when disabled."""
        config = MLflowConfig(enabled=False)
        tracker = MLflowExperimentTracker(config)
        
        # Should not raise any exceptions
        tracker.set_tags({"tag1": "value1", "tag2": "value2"})
    
    def test_set_tag_disabled(self):
        """Test set_tag is no-op when disabled."""
        config = MLflowConfig(enabled=False)
        tracker = MLflowExperimentTracker(config)
        
        # Should not raise any exceptions
        tracker.set_tag("key", "value")
    
    def test_flatten_dict(self):
        """Test dictionary flattening for MLflow parameters."""
        nested_dict = {
            "model": {
                "kernel": "linear",
                "C": 1.0,
            },
            "data": {
                "train_size": 0.8,
            },
        }
        
        flattened = MLflowExperimentTracker._flatten_dict(nested_dict)
        
        assert flattened["model_kernel"] == "linear"
        assert flattened["model_C"] == 1.0
        assert flattened["data_train_size"] == 0.8


class TestMLflowInitializationHelper:
    """Tests for init_mlflow_tracking utility helper."""

    @patch('src.mlflow_client.MLflowExperimentTracker')
    @patch('src.config.load_config')
    def test_init_mlflow_tracking_enabled_override(self, mock_load_config, mock_tracker_class):
        """Test helper enables MLflow when override is True."""
        mlflow_cfg = MLflowConfig(enabled=False)
        mock_load_config.return_value = (None, None, None, None, None, mlflow_cfg)

        tracker_instance = Mock()
        tracker_instance.enabled = True
        tracker_instance.config = MLflowConfig(enabled=True, tracking_uri="mlruns", experiment_name="test")
        mock_tracker_class.return_value = tracker_instance

        tracker = init_mlflow_tracking(enabled_override=True)

        assert tracker is tracker_instance
        assert mlflow_cfg.enabled is True
        mock_tracker_class.assert_called_once()

    @patch('src.config.load_config')
    def test_init_mlflow_tracking_returns_none_when_disabled(self, mock_load_config):
        """Test helper returns None when MLflow remains disabled."""
        mlflow_cfg = MLflowConfig(enabled=False)
        mock_load_config.return_value = (None, None, None, None, None, mlflow_cfg)

        tracker = init_mlflow_tracking()

        assert tracker is None


# ============================================================================
# RECOMMENDATION ENGINE MLflow INTEGRATION TESTS
# ============================================================================

class TestBundleRecommendationEngineMLflow:
    """Tests for MLflow integration with BundleRecommendationEngine."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample transaction and bundle data."""
        transactions = [
            ["item1", "item2"],
            ["item2", "item3"],
            ["item1", "item3"],
            ["item1", "item2", "item3"],
            ["item2", "item3", "item4"],
        ]
        bundles = [("item1", "item2"), ("item2", "item3")]
        return transactions, bundles
    
    @pytest.fixture
    def disabled_tracker(self):
        """Create disabled MLflow tracker."""
        config = MLflowConfig(enabled=False)
        return MLflowExperimentTracker(config)
    
    def test_engine_with_disabled_tracker(self, sample_data, disabled_tracker):
        """Test engine works normally with disabled tracker."""
        transactions, bundles = sample_data
        
        engine = BundleRecommendationEngine(mlflow_tracker=disabled_tracker)
        engine.add_recommender("nb", NaiveBayesBundleRecommender())
        
        metrics = engine.fit_all(transactions, bundles)
        
        assert "nb" in metrics
        assert "accuracy" in metrics["nb"]
        assert "precision" in metrics["nb"]
        assert "recall" in metrics["nb"]
        assert "f1" in metrics["nb"]
        assert "roc_auc" in metrics["nb"]
    
    def test_engine_without_tracker(self, sample_data):
        """Test engine works without tracker."""
        transactions, bundles = sample_data
        
        engine = BundleRecommendationEngine()  # No tracker
        engine.add_recommender("nb", NaiveBayesBundleRecommender())
        
        metrics = engine.fit_all(transactions, bundles)
        
        assert "nb" in metrics
        assert "accuracy" in metrics["nb"]
        assert "roc_auc" in metrics["nb"]
    
    @patch('src.mlflow_client.MLFLOW_AVAILABLE', False)
    def test_engine_with_unavailable_mlflow(self, sample_data):
        """Test engine when MLflow is not installed."""
        transactions, bundles = sample_data
        config = MLflowConfig(enabled=True)
        tracker = MLflowExperimentTracker(config)
        
        assert tracker.enabled is False
        
        engine = BundleRecommendationEngine(mlflow_tracker=tracker)
        engine.add_recommender("nb", NaiveBayesBundleRecommender())
        
        # Should work with disabled tracker
        metrics = engine.fit_all(transactions, bundles)
        assert "nb" in metrics


# ============================================================================
# MLFLOW TRACKER WITH MOCKED MLFLOW TESTS
# ============================================================================

class TestMLflowTrackerWithMockedMLflow:
    """Tests for MLflow tracker when MLflow is available (mocked)."""
    
    @patch('src.mlflow_client.mlflow')
    @patch.object(MLflowExperimentTracker, '_flatten_dict')
    def test_tracker_with_mlflow_available(self, mock_flatten, mock_mlflow):
        """Test tracker with MLflow available."""
        mock_flatten.return_value = {"param1": "value1"}
        
        config = MLflowConfig(enabled=True)
        
        with patch('src.mlflow_client.MLFLOW_AVAILABLE', True):
            tracker = MLflowExperimentTracker(config)
            
            assert tracker.enabled is True
    
    @patch('src.mlflow_client.mlflow')
    def test_log_params_with_mlflow_available(self, mock_mlflow):
        """Test log_params with MLflow available."""
        config = MLflowConfig(enabled=True)
        
        with patch('src.mlflow_client.MLFLOW_AVAILABLE', True):
            tracker = MLflowExperimentTracker(config)
            
            params = {"kernel": "linear", "C": 1.0}
            tracker.log_params(params)
            
            # Should have called mlflow.log_params
            if tracker.enabled:
                assert mock_mlflow.log_params.called
    
    @patch('src.mlflow_client.mlflow')
    def test_log_metrics_with_mlflow_available(self, mock_mlflow):
        """Test log_metrics with MLflow available."""
        config = MLflowConfig(enabled=True)
        
        with patch('src.mlflow_client.MLFLOW_AVAILABLE', True):
            tracker = MLflowExperimentTracker(config)
            
            metrics = {"accuracy": 0.92, "f1": 0.89}
            tracker.log_metrics(metrics)
            
            if tracker.enabled:
                assert mock_mlflow.log_metrics.called
    
    @patch('src.mlflow_client.mlflow')
    def test_set_tags_with_mlflow_available(self, mock_mlflow):
        """Test set_tags with MLflow available."""
        config = MLflowConfig(enabled=True)
        
        with patch('src.mlflow_client.MLFLOW_AVAILABLE', True):
            tracker = MLflowExperimentTracker(config)
            
            tags = {"model": "nb", "dataset": "sample"}
            tracker.set_tags(tags)
            
            if tracker.enabled:
                assert mock_mlflow.set_tags.called


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestMLflowIntegration:
    """Integration tests for complete MLflow workflows."""
    
    def test_full_training_pipeline_without_mlflow(self):
        """Test full training pipeline works without MLflow interference."""
        # Create sample data
        transactions = [
            ["a", "b"],
            ["b", "c"],
            ["a", "c"],
            ["a", "b", "c"],
        ]
        bundles = [("a", "b"), ("b", "c")]
        
        # Train without MLflow
        engine = BundleRecommendationEngine()
        engine.add_recommender("nb", NaiveBayesBundleRecommender())
        engine.add_recommender("svm", SVMBundleRecommender())
        
        metrics = engine.fit_all(transactions, bundles)
        
        # Validate results
        assert "nb" in metrics
        assert "svm" in metrics
        assert "accuracy" in metrics["nb"]
        assert "accuracy" in metrics["svm"]
        assert 0.0 <= metrics["nb"]["accuracy"] <= 1.0
        assert 0.0 <= metrics["svm"]["accuracy"] <= 1.0
    
    def test_full_training_pipeline_with_disabled_mlflow(self):
        """Test full training pipeline with disabled MLflow."""
        transactions = [
            ["a", "b"],
            ["b", "c"],
            ["a", "c"],
            ["a", "b", "c"],
        ]
        bundles = [("a", "b"), ("b", "c")]
        
        # Train with disabled MLflow tracker
        config = MLflowConfig(enabled=False)
        tracker = MLflowExperimentTracker(config)
        
        engine = BundleRecommendationEngine(mlflow_tracker=tracker)
        engine.add_recommender("nb", NaiveBayesBundleRecommender())
        engine.add_recommender("svm", SVMBundleRecommender())
        
        metrics = engine.fit_all(transactions, bundles)
        
        # Should produce same results
        assert "nb" in metrics
        assert "svm" in metrics
        assert 0.0 <= metrics["nb"]["accuracy"] <= 1.0
        assert 0.0 <= metrics["svm"]["accuracy"] <= 1.0
    
    def test_kfold_validation_without_mlflow(self):
        """Test k-fold cross-validation without MLflow."""
        from src.data_splitter import KFoldSplit
        
        transactions = [
            ["a", "b"],
            ["b", "c"],
            ["a", "c"],
            ["a", "b", "c"],
            ["b", "c", "d"],
            ["a", "d"],
        ]
        bundles = [("a", "b"), ("b", "c")]
        
        engine = BundleRecommendationEngine()
        engine.add_recommender("nb", NaiveBayesBundleRecommender())
        
        metrics = engine.fit_all_with_kfold(transactions, bundles, n_splits=2)
        
        assert "nb" in metrics
        assert "accuracy" in metrics["nb"]
        assert "std_accuracy" in metrics["nb"]
        assert "roc_auc" in metrics["nb"]
        assert "std_roc_auc" in metrics["nb"]
        assert "n_splits" in metrics["nb"]


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================

class TestMLflowEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_flatten_dict_single_level(self):
        """Test flattening single-level dictionary."""
        data = {"a": 1, "b": 2}
        flattened = MLflowExperimentTracker._flatten_dict(data)
        
        assert flattened == data
    
    def test_flatten_dict_deeply_nested(self):
        """Test flattening deeply nested dictionary."""
        data = {
            "level1": {
                "level2": {
                    "level3": "value"
                }
            }
        }
        flattened = MLflowExperimentTracker._flatten_dict(data)
        
        assert flattened["level1_level2_level3"] == "value"
    
    def test_flatten_dict_custom_separator(self):
        """Test flattening with custom separator."""
        data = {"a": {"b": 1}}
        flattened = MLflowExperimentTracker._flatten_dict(data, sep="/")
        
        assert "a/b" in flattened
        assert flattened["a/b"] == 1
    
    def test_tracker_with_none_config_operations(self):
        """Test tracker operations with None config."""
        tracker = MLflowExperimentTracker(None)
        
        # All operations should be no-ops
        tracker.log_params({"a": 1})
        tracker.log_metrics({"m": 0.5})
        tracker.set_tags({"tag": "value"})
        
        # Should not raise exceptions


# ============================================================================
# LLM OPERATION TRACKER TESTS
# ============================================================================

class TestLLMOperationTracker:
    """Tests for LLMOperationTracker singleton."""
    
    def setup_method(self):
        """Reset tracker before each test."""
        from src.llm_client import LLMOperationTracker
        # Get singleton and reset it
        tracker = LLMOperationTracker()
        tracker.reset()
    
    def test_tracker_singleton_pattern(self):
        """Test that LLMOperationTracker is a singleton."""
        from src.llm_client import LLMOperationTracker
        tracker1 = LLMOperationTracker()
        tracker2 = LLMOperationTracker()
        
        assert tracker1 is tracker2
    
    def test_record_single_operation(self):
        """Test recording a single LLM operation."""
        from src.llm_client import LLMOperationTracker
        tracker = LLMOperationTracker()
        
        tracker.record_operation("enrich_categories", latency_ms=150.0, cached=False, provider="openai")
        
        stats = tracker.get_aggregated_stats()
        assert "enrich_categories" in stats
        assert stats["enrich_categories"].total_calls == 1
        assert stats["enrich_categories"].total_latency_ms == 150.0
    
    def test_record_multiple_operations(self):
        """Test recording multiple different operations."""
        from src.llm_client import LLMOperationTracker
        tracker = LLMOperationTracker()
        
        tracker.record_operation("enrich_categories", latency_ms=100.0, cached=False, provider="openai")
        tracker.record_operation("select_alternatives", latency_ms=200.0, cached=False, provider="gemini")
        tracker.record_operation("extract_contexts", latency_ms=150.0, cached=False, provider="openai")
        
        stats = tracker.get_aggregated_stats()
        assert len(stats) == 3
        assert stats["enrich_categories"].total_calls == 1
        assert stats["select_alternatives"].total_calls == 1
        assert stats["extract_contexts"].total_calls == 1
    
    def test_cache_hit_tracking(self):
        """Test tracking cache hits."""
        from src.llm_client import LLMOperationTracker
        tracker = LLMOperationTracker()
        
        # Record 10 calls: 3 cached, 7 uncached
        tracker.record_operation("enrich_categories", latency_ms=50.0, cached=True, provider="openai")
        tracker.record_operation("enrich_categories", latency_ms=50.0, cached=True, provider="openai")
        tracker.record_operation("enrich_categories", latency_ms=50.0, cached=True, provider="openai")
        for _ in range(7):
            tracker.record_operation("enrich_categories", latency_ms=150.0, cached=False, provider="openai")
        
        stats = tracker.get_operation_stats("enrich_categories")
        assert stats.total_calls == 10
        assert stats.cache_hits == 3
        assert stats.cache_hit_rate == 30.0  # 3/10 * 100
    
    def test_provider_distribution(self):
        """Test tracking provider distribution."""
        from src.llm_client import LLMOperationTracker
        tracker = LLMOperationTracker()
        
        tracker.record_operation("enrich_categories", latency_ms=100.0, provider="openai")
        tracker.record_operation("enrich_categories", latency_ms=100.0, provider="openai")
        tracker.record_operation("enrich_categories", latency_ms=100.0, provider="gemini")
        tracker.record_operation("select_alternatives", latency_ms=200.0, provider="openai")
        
        stats = tracker.get_operation_stats("enrich_categories")
        assert stats.provider_distribution["openai"] == 2
        assert stats.provider_distribution["gemini"] == 1
        
        overall_dist = tracker.get_provider_distribution()
        assert overall_dist["openai"] == 3
        assert overall_dist["gemini"] == 1
    
    def test_error_tracking(self):
        """Test tracking operation errors."""
        from src.llm_client import LLMOperationTracker
        tracker = LLMOperationTracker()
        
        tracker.record_operation("enrich_categories", latency_ms=100.0, error=False)
        tracker.record_operation("enrich_categories", latency_ms=50.0, error=True)
        tracker.record_operation("enrich_categories", latency_ms=100.0, error=False)
        
        stats = tracker.get_operation_stats("enrich_categories")
        assert stats.total_calls == 3
        assert stats.error_count == 1
    
    def test_average_latency_calculation(self):
        """Test average latency calculation."""
        from src.llm_client import LLMOperationTracker
        tracker = LLMOperationTracker()
        
        tracker.record_operation("enrich_categories", latency_ms=100.0)
        tracker.record_operation("enrich_categories", latency_ms=200.0)
        tracker.record_operation("enrich_categories", latency_ms=300.0)
        
        stats = tracker.get_operation_stats("enrich_categories")
        assert stats.avg_latency_ms == 200.0  # (100 + 200 + 300) / 3
    
    def test_overall_cache_hit_rate(self):
        """Test overall cache hit rate calculation."""
        from src.llm_client import LLMOperationTracker
        tracker = LLMOperationTracker()
        
        # Operation 1: 2 cached out of 5
        for _ in range(2):
            tracker.record_operation("enrich_categories", latency_ms=50.0, cached=True)
        for _ in range(3):
            tracker.record_operation("enrich_categories", latency_ms=150.0, cached=False)
        
        # Operation 2: 1 cached out of 4
        for _ in range(1):
            tracker.record_operation("select_alternatives", latency_ms=50.0, cached=True)
        for _ in range(3):
            tracker.record_operation("select_alternatives", latency_ms=200.0, cached=False)
        
        # Overall: 3 cached out of 9 = 33.33%
        overall_rate = tracker.get_overall_cache_hit_rate()
        assert overall_rate == pytest.approx(33.33, abs=0.1)
    
    def test_to_mlflow_params(self):
        """Test conversion to MLflow parameters."""
        from src.llm_client import LLMOperationTracker
        tracker = LLMOperationTracker()
        
        tracker.record_operation("enrich_categories", latency_ms=100.0, cached=True, provider="openai")
        tracker.record_operation("enrich_categories", latency_ms=150.0, cached=False, provider="openai")
        tracker.record_operation("select_alternatives", latency_ms=200.0, cached=False, provider="gemini")
        
        params = tracker.to_mlflow_params()
        
        # Check overall metrics
        assert params["llm_total_calls"] == "3"
        assert "llm_cache_hit_rate_percent" in params
        assert "llm_provider_distribution" in params
        
        # Check per-operation metrics
        assert "llm_enrich_categories_calls" in params
        assert params["llm_enrich_categories_calls"] == "2"
        assert "llm_select_alternatives_calls" in params
        assert params["llm_select_alternatives_calls"] == "1"
    
    def test_to_mlflow_metrics(self):
        """Test conversion to MLflow metrics (numeric)."""
        from src.llm_client import LLMOperationTracker
        tracker = LLMOperationTracker()
        
        tracker.record_operation("enrich_categories", latency_ms=100.0, cached=True)
        tracker.record_operation("enrich_categories", latency_ms=150.0, cached=False)
        
        metrics = tracker.to_mlflow_metrics()
        
        # Check that all values are numeric
        assert metrics["llm_total_calls"] == 2.0
        assert isinstance(metrics["llm_cache_hit_rate_percent"], float)
        assert isinstance(metrics["llm_enrich_categories_calls"], float)
        assert isinstance(metrics["llm_enrich_categories_avg_latency_ms"], float)
    
    def test_tracker_reset(self):
        """Test tracker reset functionality."""
        from src.llm_client import LLMOperationTracker
        tracker = LLMOperationTracker()
        
        tracker.record_operation("enrich_categories", latency_ms=100.0)
        assert tracker.get_total_calls() == 1
        
        tracker.reset()
        assert tracker.get_total_calls() == 0
        assert len(tracker.get_aggregated_stats()) == 0
    
    def test_empty_tracker_metrics(self):
        """Test metrics from empty tracker."""
        from src.llm_client import LLMOperationTracker
        tracker = LLMOperationTracker()
        tracker.reset()  # Ensure empty
        
        assert tracker.get_total_calls() == 0
        assert tracker.get_overall_cache_hit_rate() == 0.0
        assert tracker.to_mlflow_metrics()["llm_total_calls"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
