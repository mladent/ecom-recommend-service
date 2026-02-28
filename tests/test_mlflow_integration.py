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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
