"""MLflow experiment tracking and logging utilities."""

import time
import functools
from typing import Dict, Any, Optional, Callable, TYPE_CHECKING, cast

if TYPE_CHECKING:
    from .config import MLflowConfig

try:
    import mlflow  # type: ignore[import]
    import mlflow.sklearn as mlflow_sklearn  # type: ignore[import]
    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None  # type: ignore[assignment]
    mlflow_sklearn = None  # type: ignore[assignment]
    MLFLOW_AVAILABLE = False


def _get_mlflow() -> Any:
    """Return mlflow module reference for typed access in guarded code paths."""
    if mlflow is None:
        raise RuntimeError("MLflow is not available. Install with: pip install mlflow")
    return cast(Any, mlflow)


def _get_mlflow_sklearn() -> Any:
    """Return mlflow.sklearn reference for typed access in guarded code paths."""
    if mlflow_sklearn is None:
        raise RuntimeError("MLflow sklearn flavor is not available")
    return cast(Any, mlflow_sklearn)


class MLflowExperimentTracker:
    """
    Wrapper for MLflow experiment tracking with graceful degradation.
    
    Provides context managers and decorators for logging metrics, parameters,
    and artifacts. When MLflow is disabled or unavailable, all operations
    become no-ops.
    """
    
    def __init__(self, config: Optional['MLflowConfig'] = None):
        """
        Initialize MLflow tracker.
        
        Args:
            config: MLflowConfig instance. If None, tracking is disabled.
        """
        self.enabled = False
        self.config = config
        
        if config and config.enabled and MLFLOW_AVAILABLE:
            mlflow_mod = _get_mlflow()
            self.enabled = True
            mlflow_mod.set_tracking_uri(config.tracking_uri)
            mlflow_mod.set_experiment(config.experiment_name)
            if config.log_system_metrics:
                mlflow_mod.enable_system_metrics_logging()
        
    def start_run(self, run_name: Optional[str] = None, **kwargs):
        """
        Start an MLflow run (context manager).
        
        Args:
            run_name: Optional name for the run
            **kwargs: Additional arguments to pass to mlflow.start_run()
            
        Returns:
            Context manager (MLflow run or no-op)
        """
        if not self.enabled:
            return _NoOpContext()
        
        if run_name and self.config and self.config.run_name_prefix:
            run_name = f"{self.config.run_name_prefix}{run_name}"
        
        return _get_mlflow().start_run(run_name=run_name, **kwargs)
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters to MLflow.
        
        Args:
            params: Dictionary of parameter names and values
        """
        if not self.enabled:
            return
        
        # MLflow can't handle nested dicts, so flatten them
        flat_params = self._flatten_dict(params)
        _get_mlflow().log_params(flat_params)
    
    def log_param(self, key: str, value: Any):
        """
        Log a single parameter to MLflow.
        
        Args:
            key: Parameter name
            value: Parameter value
        """
        if not self.enabled:
            return
        _get_mlflow().log_param(key, value)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to MLflow.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number for the metrics
        """
        if not self.enabled:
            return
        _get_mlflow().log_metrics(metrics, step=step)
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """
        Log a single metric to MLflow.
        
        Args:
            key: Metric name
            value: Metric value
            step: Optional step number
        """
        if not self.enabled:
            return
        _get_mlflow().log_metric(key, value, step=step)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log a file or directory as an artifact.
        
        Args:
            local_path: Path to the file or directory
            artifact_path: Optional subdirectory in the artifact store
        """
        if not self.enabled:
            return
        _get_mlflow().log_artifact(local_path, artifact_path=artifact_path)
    
    def log_dict(self, dictionary: Dict[str, Any], artifact_file: str):
        """
        Log a dictionary as a JSON artifact.
        
        Args:
            dictionary: Dictionary to log
            artifact_file: Name for the artifact file
        """
        if not self.enabled:
            return
        _get_mlflow().log_dict(dictionary, artifact_file)
    
    def log_model(self, model, artifact_path: str, **kwargs):
        """
        Log a model to MLflow.
        
        Args:
            model: Model object to log
            artifact_path: Path within the artifact store
            **kwargs: Additional arguments for model logging
        """
        if not self.enabled:
            return
        
        # Use sklearn flavor for scikit-learn models
        from sklearn.base import BaseEstimator
        if isinstance(model, BaseEstimator):
            _get_mlflow_sklearn().log_model(model, artifact_path, **kwargs)
        else:
            # Generic pickle logging
            import pickle
            import tempfile
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
                pickle.dump(model, f)
                temp_path = f.name
            self.log_artifact(temp_path, artifact_path)
    
    def set_tags(self, tags: Dict[str, Any]):
        """
        Set tags for the current run.
        
        Args:
            tags: Dictionary of tag names and values
        """
        if not self.enabled:
            return
        _get_mlflow().set_tags(tags)
    
    def set_tag(self, key: str, value: Any):
        """
        Set a single tag for the current run.
        
        Args:
            key: Tag name
            value: Tag value
        """
        if not self.enabled:
            return
        _get_mlflow().set_tag(key, value)
    
    @staticmethod
    def _flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """
        Flatten nested dictionary for MLflow logging.
        
        Args:
            d: Dictionary to flatten
            parent_key: Prefix for nested keys
            sep: Separator between nested keys
            
        Returns:
            Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(MLflowExperimentTracker._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


class _NoOpContext:
    """No-op context manager for when MLflow is disabled."""
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        return False


# ============================================================================
# DECORATORS
# ============================================================================

def log_model_training(tracker: MLflowExperimentTracker):
    """
    Decorator to log model training metrics and parameters.
    
    Usage:
        @log_model_training(tracker)
        def fit_model(model, X, y):
            # Training code
            return metrics_dict
            
    The decorated function should return a dict with keys:
        - metrics: Dict[str, float] - Model performance metrics
        - params: Dict[str, Any] - Model hyperparameters (optional)
        - model_name: str - Name of the model (optional)
    
    Args:
        tracker: MLflowExperimentTracker instance
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            if not tracker.enabled:
                return result
            
            # Extract metrics and parameters from result
            if isinstance(result, dict):
                metrics = result.get('metrics', {})
                params = result.get('params', {})
                model_name = result.get('model_name', func.__name__)
                
                # Log parameters
                if params:
                    tracker.log_params({f"{model_name}_{k}": v for k, v in params.items()})
                
                # Log metrics
                if metrics:
                    tracker.log_metrics({f"{model_name}_{k}": v for k, v in metrics.items()})
            
            return result
        
        return wrapper
    return decorator


def log_training_duration(tracker: MLflowExperimentTracker, metric_name: str = "training_time"):
    """
    Decorator to log execution duration as a metric.
    
    Usage:
        @log_training_duration(tracker, "model_training_time")
        def train_model():
            # Training code
            pass
    
    Args:
        tracker: MLflowExperimentTracker instance
        metric_name: Name for the duration metric
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            if tracker.enabled:
                tracker.log_metric(metric_name, duration)
            
            return result
        
        return wrapper
    return decorator


def log_llm_operation(tracker: MLflowExperimentTracker, operation_name: str):
    """
    Decorator to log LLM operation statistics.
    
    Usage:
        @log_llm_operation(tracker, "enrich_categories")
        def enrich_with_llm(items):
            # LLM operation
            return result, stats
            
    The decorated function should return a tuple:
        (result, stats_dict)
    Where stats_dict contains:
        - calls: int - Number of API calls
        - cache_hits: int - Number of cache hits
        - latency_ms: float - Average latency in milliseconds
        - errors: int - Number of errors (optional)
    
    Args:
        tracker: MLflowExperimentTracker instance
        operation_name: Name of the LLM operation
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            if not tracker.enabled:
                return result
            
            # Extract stats from result tuple
            if isinstance(result, tuple) and len(result) == 2:
                actual_result, stats = result
                
                if isinstance(stats, dict):
                    # Log operation-specific metrics
                    prefix = f"llm_{operation_name}"
                    
                    if 'calls' in stats:
                        tracker.log_metric(f"{prefix}_calls", stats['calls'])
                    if 'cache_hits' in stats:
                        tracker.log_metric(f"{prefix}_cache_hits", stats['cache_hits'])
                    if 'latency_ms' in stats:
                        tracker.log_metric(f"{prefix}_latency_ms", stats['latency_ms'])
                    if 'errors' in stats:
                        tracker.log_metric(f"{prefix}_errors", stats['errors'])
                
                return actual_result
            
            return result
        
        return wrapper
    return decorator


def with_mlflow_run(tracker: MLflowExperimentTracker, run_name: Optional[str] = None):
    """
    Decorator to wrap function execution in an MLflow run.
    
    Usage:
        @with_mlflow_run(tracker, "experiment_run")
        def run_experiment():
            # Experiment code with MLflow logging
            pass
    
    Args:
        tracker: MLflowExperimentTracker instance
        run_name: Optional name for the run
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with tracker.start_run(run_name=run_name):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator
