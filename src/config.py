"""Configuration management for the recommendation service."""

import os
import yaml
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Load settings from YAML
def _load_yaml_config(config_path="config/settings.yaml"):
    """Load configuration from YAML file."""
    config_file = PROJECT_ROOT / config_path
    if config_file.exists():
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    return {}


def _env_bool(name: str, default: bool) -> bool:
    """Parse environment variable as boolean with fallback default."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_path(path_value: str) -> str:
    """Resolve relative paths against the project root."""
    if not path_value:
        return path_value
    return path_value if os.path.isabs(path_value) else str(PROJECT_ROOT / path_value)


# ============================================================================
# CONFIGURATION DATACLASSES (Type-Safe Configuration Objects)
# ============================================================================

@dataclass
class LLMConfig:
    """Configuration for LLM providers and parameters."""
    
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 64
    timeout_seconds: int = 20
    
    # Provider credentials
    openai_api_key: Optional[str] = None
    azure_api_key: Optional[str] = None
    azure_endpoint: Optional[str] = None
    azure_deployment: Optional[str] = None
    azure_api_version: str = "2024-06-01"
    gemini_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    perplexity_api_key: Optional[str] = None
    perplexity_base_url: str = "https://api.perplexity.ai"
    
    def __post_init__(self):
        """Validate LLM configuration."""
        if self.provider not in {"openai", "azure", "gemini", "anthropic", "perplexity"}:
            raise ValueError(f"Invalid LLM provider: {self.provider}")
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError(f"Temperature must be between 0.0 and 2.0, got {self.temperature}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")


@dataclass
class CacheConfig:
    """Configuration for caching behavior."""
    
    normalization_cache_first: bool = True
    normalization_cache_path: str = "data/normalization_cache.json"
    
    enrichment_cache_first: bool = True
    enrichment_cache_path: str = "data/category_enrichment_cache.json"
    
    outlier_cache_first: bool = True
    outlier_cache_path: str = "data/anomaly_cache.json"
    
    context_cache_first: bool = True
    context_cache_path: str = "data/context_extraction_cache.json"
    
    oos_cache_first: bool = True
    oos_cache_path: str = "data/out_of_stock_alternatives_cache.json"
    
    def resolve_paths(self):
        """Resolve all relative paths against project root."""
        self.normalization_cache_path = _resolve_path(self.normalization_cache_path)
        self.enrichment_cache_path = _resolve_path(self.enrichment_cache_path)
        self.outlier_cache_path = _resolve_path(self.outlier_cache_path)
        self.context_cache_path = _resolve_path(self.context_cache_path)
        self.oos_cache_path = _resolve_path(self.oos_cache_path)


@dataclass
class PipelineConfig:
    """Configuration for data pipeline processing."""
    
    # Data paths
    data_path: str = "data"
    raw_data_file: str = "data.csv"
    processed_data_file: str = "processed_data.pkl"
    
    # Mining parameters
    min_support: float = 0.02
    min_confidence: float = 0.5
    max_bundle_size: int = 5
    
    # Train/test split
    train_test_split: float = 0.8
    random_state: int = 42
    
    # LLM & cache configuration
    llm_config: LLMConfig = field(default_factory=lambda: LLMConfig())
    cache_config: CacheConfig = field(default_factory=lambda: CacheConfig())
    
    # Feature flags
    normalization_enabled: bool = False
    enrichment_enabled: bool = False
    outlier_enabled: bool = False
    context_enabled: bool = False
    oos_enabled: bool = False

    # Category enrichment settings
    enrichment_batch_size: int = 10
    enrichment_fields: List[str] = field(
        default_factory=lambda: ["category", "material", "size", "theme"]
    )

    # Outlier detection settings
    outlier_output_path: str = "data/suspicious_transactions.tsv"
    outlier_batch_size: int = 50
    outlier_iqr_multiplier: float = 1.5
    outlier_fields: List[str] = field(
        default_factory=lambda: ["Quantity", "UnitPrice", "TransactionValue"]
    )

    # Context extraction settings
    context_max_contexts: int = 3
    context_min_confidence: float = 0.6

    # Out-of-stock alternative settings
    oos_inventory_path: str = "data/inventory.csv"
    oos_max_alternatives: int = 1
    oos_min_score: float = 0.3
    
    def __post_init__(self):
        """Validate pipeline configuration."""
        if not (0.0 < self.min_support < 1.0):
            raise ValueError(f"min_support must be between 0 and 1, got {self.min_support}")
        if not (0.0 < self.min_confidence < 1.0):
            raise ValueError(f"min_confidence must be between 0 and 1, got {self.min_confidence}")
        if not (0.0 < self.train_test_split < 1.0):
            raise ValueError(f"train_test_split must be between 0 and 1, got {self.train_test_split}")
        if self.max_bundle_size < 2:
            raise ValueError(f"max_bundle_size must be >= 2, got {self.max_bundle_size}")
        if self.enrichment_batch_size < 1:
            raise ValueError(
                f"enrichment_batch_size must be >= 1, got {self.enrichment_batch_size}"
            )
        if self.outlier_batch_size < 1:
            raise ValueError(f"outlier_batch_size must be >= 1, got {self.outlier_batch_size}")
        if self.outlier_iqr_multiplier <= 0:
            raise ValueError(
                f"outlier_iqr_multiplier must be > 0, got {self.outlier_iqr_multiplier}"
            )
        if self.context_max_contexts < 1:
            raise ValueError(
                f"context_max_contexts must be >= 1, got {self.context_max_contexts}"
            )
        if not (0.0 <= self.context_min_confidence <= 1.0):
            raise ValueError(
                f"context_min_confidence must be between 0.0 and 1.0, got {self.context_min_confidence}"
            )
        if self.oos_max_alternatives < 1:
            raise ValueError(
                f"oos_max_alternatives must be >= 1, got {self.oos_max_alternatives}"
            )
        if not (0.0 <= self.oos_min_score <= 1.0):
            raise ValueError(f"oos_min_score must be between 0.0 and 1.0, got {self.oos_min_score}")
    
    @property
    def raw_data_path(self) -> str:
        """Get full path to raw data file."""
        return os.path.join(self.data_path, self.raw_data_file)
    
    @property
    def processed_data_path(self) -> str:
        """Get full path to processed data file."""
        return os.path.join(self.data_path, self.processed_data_file)


@dataclass
class EngineConfig:
    """Configuration for recommendation engine models."""
    
    # SVM parameters
    svm_kernel: str = "linear"
    svm_c: float = 1.0
    
    # Random state for reproducibility
    random_state: int = 42
    
    # Parallel processing
    n_jobs: int = -1
    
    def __post_init__(self):
        """Validate engine configuration."""
        if self.svm_kernel not in {"linear", "rbf", "poly", "sigmoid"}:
            raise ValueError(f"Invalid SVM kernel: {self.svm_kernel}")
        if self.svm_c <= 0:
            raise ValueError(f"SVM C must be > 0, got {self.svm_c}")


@dataclass
class APIConfig:
    """Configuration for REST API server."""
    
    host: str = "127.0.0.1"
    port: int = 5000
    debug: bool = False
    workers: int = 4


@dataclass
class MLflowConfig:
    """Configuration for MLflow experiment tracking."""
    
    enabled: bool = False
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "bundle-recommendation-engine"
    run_name_prefix: str = ""
    log_system_metrics: bool = True
    
    def __post_init__(self):
        """Validate MLflow configuration."""
        if self.enabled and not self.tracking_uri:
            raise ValueError("tracking_uri must be set when MLflow is enabled")
        if self.enabled and not self.experiment_name:
            raise ValueError("experiment_name must be set when MLflow is enabled")


# ============================================================================
# FACTORY FUNCTION for Loading All Configurations
# ============================================================================

def load_config() -> tuple[PipelineConfig, EngineConfig, APIConfig, LLMConfig, CacheConfig, MLflowConfig]:
    """
    Load all configuration objects from environment and YAML.
    
    Returns:
        Tuple of (PipelineConfig, EngineConfig, APIConfig, LLMConfig, CacheConfig, MLflowConfig)
    """
    yaml_config = _load_yaml_config()
    
    # Extract YAML sections
    llm_cfg = YAML_CONFIG.get("normalization", {}).get("llm", {})
    norm_cfg = YAML_CONFIG.get("normalization", {})
    enrich_cfg = YAML_CONFIG.get("category_enrichment", {})
    outlier_cfg = YAML_CONFIG.get("outlier_detection", {})
    context_cfg = YAML_CONFIG.get("context_extraction", {})
    oos_cfg = YAML_CONFIG.get("out_of_stock_alternatives", {})
    
    # Build LLMConfig from environment and YAML
    llm = LLMConfig(
        provider=os.getenv("LLM_PROVIDER", llm_cfg.get("provider", "openai")).lower(),
        model=os.getenv("LLM_MODEL", llm_cfg.get("model", "gpt-4o-mini")),
        temperature=float(os.getenv("LLM_TEMPERATURE", llm_cfg.get("temperature", 0.0))),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", llm_cfg.get("max_tokens", 64))),
        timeout_seconds=int(os.getenv("LLM_TIMEOUT_SECONDS", llm_cfg.get("timeout_seconds", 20))),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        azure_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        azure_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01"),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        perplexity_api_key=os.getenv("PERPLEXITY_API_KEY"),
        perplexity_base_url=os.getenv("PERPLEXITY_BASE_URL", "https://api.perplexity.ai"),
    )
    
    # Build CacheConfig
    cache = CacheConfig(
        normalization_cache_first=_env_bool("LLM_CACHE_FIRST", norm_cfg.get("cache_first", True)),
        normalization_cache_path=os.getenv("LLM_CACHE_PATH", norm_cfg.get("cache_path", "data/normalization_cache.json")),
        enrichment_cache_first=_env_bool("LLM_CATEGORY_CACHE_FIRST", enrich_cfg.get("cache_first", True)),
        enrichment_cache_path=os.getenv("LLM_CATEGORY_CACHE_PATH", enrich_cfg.get("cache_path", "data/category_enrichment_cache.json")),
        outlier_cache_first=_env_bool("LLM_OUTLIER_CACHE_FIRST", outlier_cfg.get("cache_first", True)),
        outlier_cache_path=os.getenv("LLM_OUTLIER_CACHE_PATH", outlier_cfg.get("cache_path", "data/anomaly_cache.json")),
        context_cache_first=_env_bool("LLM_CONTEXT_CACHE_FIRST", context_cfg.get("cache_first", True)),
        context_cache_path=os.getenv("LLM_CONTEXT_CACHE_PATH", context_cfg.get("cache_path", "data/context_extraction_cache.json")),
        oos_cache_first=_env_bool("LLM_OOS_CACHE_FIRST", oos_cfg.get("cache_first", True)),
        oos_cache_path=os.getenv("LLM_OOS_CACHE_PATH", oos_cfg.get("cache_path", "data/out_of_stock_alternatives_cache.json")),
    )
    cache.resolve_paths()
    
    # Build PipelineConfig
    pipeline = PipelineConfig(
        data_path=os.getenv("DATA_PATH", str(PROJECT_ROOT / "data")),
        raw_data_file=os.getenv("RAW_DATA_FILE", "data.csv"),
        processed_data_file=os.getenv("PROCESSED_DATA_FILE", "processed_data.pkl"),
        min_support=float(os.getenv("MIN_SUPPORT", 0.02)),
        min_confidence=float(os.getenv("MIN_CONFIDENCE", 0.5)),
        max_bundle_size=int(os.getenv("MAX_BUNDLE_SIZE", 5)),
        train_test_split=float(os.getenv("TRAIN_TEST_SPLIT", 0.8)),
        random_state=int(os.getenv("RANDOM_STATE", 42)),
        llm_config=llm,
        cache_config=cache,
        normalization_enabled=_env_bool("LLM_NORMALIZATION_ENABLED", norm_cfg.get("enabled", False)),
        enrichment_enabled=_env_bool("LLM_CATEGORY_ENRICHMENT_ENABLED", enrich_cfg.get("enabled", False)),
        outlier_enabled=_env_bool("LLM_OUTLIER_ENABLED", outlier_cfg.get("enabled", False)),
        context_enabled=_env_bool("LLM_CONTEXT_ENABLED", context_cfg.get("enabled", False)),
        oos_enabled=_env_bool("LLM_OOS_ENABLED", oos_cfg.get("enabled", False)),
        enrichment_batch_size=int(
            os.getenv("LLM_CATEGORY_BATCH_SIZE", enrich_cfg.get("batch_size", 10))
        ),
        enrichment_fields=enrich_cfg.get("fields", ["category", "material", "size", "theme"]),
        outlier_output_path=_resolve_path(
            os.getenv(
                "LLM_OUTLIER_OUTPUT_PATH",
                outlier_cfg.get("output_path", "data/suspicious_transactions.tsv"),
            )
        ),
        outlier_batch_size=int(
            os.getenv("LLM_OUTLIER_BATCH_SIZE", outlier_cfg.get("batch_size", 50))
        ),
        outlier_iqr_multiplier=float(
            os.getenv("LLM_OUTLIER_IQR_MULTIPLIER", outlier_cfg.get("iqr_multiplier", 1.5))
        ),
        outlier_fields=outlier_cfg.get("fields", ["Quantity", "UnitPrice", "TransactionValue"]),
        context_max_contexts=int(
            os.getenv("LLM_CONTEXT_MAX_CONTEXTS", context_cfg.get("max_contexts", 3))
        ),
        context_min_confidence=float(
            os.getenv("LLM_CONTEXT_MIN_CONFIDENCE", context_cfg.get("min_confidence", 0.6))
        ),
        oos_inventory_path=_resolve_path(
            os.getenv("LLM_OOS_INVENTORY_PATH", oos_cfg.get("inventory_path", "data/inventory.csv"))
        ),
        oos_max_alternatives=int(
            os.getenv("LLM_OOS_MAX_ALTERNATIVES", oos_cfg.get("max_alternatives", 1))
        ),
        oos_min_score=float(
            os.getenv("LLM_OOS_MIN_SCORE", oos_cfg.get("min_score", 0.3))
        ),
    )
    
    # Build EngineConfig
    engine = EngineConfig(
        svm_kernel=os.getenv("SVM_KERNEL", yaml_config.get("algorithms", {}).get("svm", {}).get("kernel", "linear")),
        svm_c=float(os.getenv("SVM_C", yaml_config.get("algorithms", {}).get("svm", {}).get("C", 1.0))),
        random_state=int(os.getenv("RANDOM_STATE", 42)),
        n_jobs=int(os.getenv("N_JOBS", -1)),
    )
    
    # Build APIConfig
    api = APIConfig(
        host=os.getenv("API_HOST", "127.0.0.1"),
        port=int(os.getenv("API_PORT", 5000)),
        debug=_env_bool("API_DEBUG", False),
        workers=int(os.getenv("API_WORKERS", 4)),
    )
    
    # Build MLflowConfig
    mlflow_cfg = yaml_config.get("mlflow", {})
    mlflow = MLflowConfig(
        enabled=_env_bool("MLFLOW_ENABLED", mlflow_cfg.get("enabled", False)),
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db")),
        experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", mlflow_cfg.get("experiment_name", "bundle-recommendation-engine")),
        run_name_prefix=os.getenv("MLFLOW_RUN_NAME_PREFIX", mlflow_cfg.get("run_name_prefix", "")),
        log_system_metrics=_env_bool("MLFLOW_LOG_SYSTEM_METRICS", mlflow_cfg.get("log_system_metrics", True)),
    )
    
    return pipeline, engine, api, llm, cache, mlflow


YAML_CONFIG = _load_yaml_config()


# Data Configuration
DATA_PATH = os.getenv("DATA_PATH", str(PROJECT_ROOT / "data"))
RAW_DATA_FILE = os.getenv("RAW_DATA_FILE", "data.csv")
PROCESSED_DATA_FILE = os.getenv("PROCESSED_DATA_FILE", "processed_data.pkl")

# Paths
RAW_DATA_PATH = os.path.join(DATA_PATH, RAW_DATA_FILE)
PROCESSED_DATA_PATH = os.path.join(DATA_PATH, PROCESSED_DATA_FILE)
MODELS_PATH = os.path.join(PROJECT_ROOT, "models")

# Recommendation Engine Configuration
MIN_SUPPORT = float(os.getenv("MIN_SUPPORT", 0.02))
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", 0.5))
MAX_BUNDLE_SIZE = int(os.getenv("MAX_BUNDLE_SIZE", 5))

# Model Configuration
TRAIN_TEST_SPLIT = float(os.getenv("TRAIN_TEST_SPLIT", 0.8))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))
N_JOBS = int(os.getenv("N_JOBS", -1))

# SVM Configuration
SVM_KERNEL = os.getenv("SVM_KERNEL", YAML_CONFIG.get("algorithms", {}).get("svm", {}).get("kernel", "linear"))
SVM_C = float(os.getenv("SVM_C", YAML_CONFIG.get("algorithms", {}).get("svm", {}).get("C", 1.0)))

# Kaggle Configuration
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY = os.getenv("KAGGLE_KEY")

# LLM Normalization Configuration
NORMALIZATION_CONFIG = YAML_CONFIG.get("normalization", {})
NORMALIZATION_ENABLED = _env_bool("LLM_NORMALIZATION_ENABLED", NORMALIZATION_CONFIG.get("enabled", False))
NORMALIZATION_CACHE_FIRST = _env_bool("LLM_CACHE_FIRST", NORMALIZATION_CONFIG.get("cache_first", True))
NORMALIZATION_CACHE_PATH = _resolve_path(
    os.getenv("LLM_CACHE_PATH", NORMALIZATION_CONFIG.get("cache_path", "data/normalization_cache.json"))
)
NORMALIZATION_ALIAS_MAP_PATH = _resolve_path(
    os.getenv("LLM_ALIAS_MAP_PATH", NORMALIZATION_CONFIG.get("alias_map_path", "data/description_aliases.json"))
)
NORMALIZATION_MIN_LENGTH = int(os.getenv("LLM_MIN_LENGTH", NORMALIZATION_CONFIG.get("min_length", 3)))

LLM_CONFIG = NORMALIZATION_CONFIG.get("llm", {})
LLM_PROVIDER = os.getenv("LLM_PROVIDER", LLM_CONFIG.get("provider", "openai")).lower()
LLM_MODEL = os.getenv("LLM_MODEL", LLM_CONFIG.get("model", "gpt-4o-mini"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", LLM_CONFIG.get("temperature", 0.0)))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", LLM_CONFIG.get("max_tokens", 64)))
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", LLM_CONFIG.get("timeout_seconds", 20)))

# Provider API credentials
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
PERPLEXITY_BASE_URL = os.getenv("PERPLEXITY_BASE_URL", "https://api.perplexity.ai")

# LLM Category Enrichment Configuration
ENRICHMENT_CONFIG = YAML_CONFIG.get("category_enrichment", {})
ENRICHMENT_ENABLED = _env_bool("LLM_CATEGORY_ENRICHMENT_ENABLED", ENRICHMENT_CONFIG.get("enabled", False))
ENRICHMENT_CACHE_FIRST = _env_bool("LLM_CATEGORY_CACHE_FIRST", ENRICHMENT_CONFIG.get("cache_first", True))
ENRICHMENT_CACHE_PATH = _resolve_path(
    os.getenv("LLM_CATEGORY_CACHE_PATH", ENRICHMENT_CONFIG.get("cache_path", "data/category_enrichment_cache.json"))
)
ENRICHMENT_BATCH_SIZE = int(os.getenv("LLM_CATEGORY_BATCH_SIZE", ENRICHMENT_CONFIG.get("batch_size", 10)))
ENRICHMENT_FIELDS = ENRICHMENT_CONFIG.get("fields", ["category", "material", "size", "theme"])

# LLM Outlier Detection Configuration
OUTLIER_CONFIG = YAML_CONFIG.get("outlier_detection", {})
OUTLIER_ENABLED = _env_bool("LLM_OUTLIER_ENABLED", OUTLIER_CONFIG.get("enabled", False))
OUTLIER_CACHE_FIRST = _env_bool("LLM_OUTLIER_CACHE_FIRST", OUTLIER_CONFIG.get("cache_first", True))
OUTLIER_CACHE_PATH = _resolve_path(
    os.getenv("LLM_OUTLIER_CACHE_PATH", OUTLIER_CONFIG.get("cache_path", "data/anomaly_cache.json"))
)
OUTLIER_OUTPUT_PATH = _resolve_path(
    os.getenv("LLM_OUTLIER_OUTPUT_PATH", OUTLIER_CONFIG.get("output_path", "data/suspicious_transactions.tsv"))
)
OUTLIER_BATCH_SIZE = int(os.getenv("LLM_OUTLIER_BATCH_SIZE", OUTLIER_CONFIG.get("batch_size", 50)))
OUTLIER_IQR_MULTIPLIER = float(
    os.getenv("LLM_OUTLIER_IQR_MULTIPLIER", OUTLIER_CONFIG.get("iqr_multiplier", 1.5))
)
OUTLIER_FIELDS = OUTLIER_CONFIG.get("fields", ["Quantity", "UnitPrice", "TransactionValue"])

# LLM Context Extraction Configuration
CONTEXT_CONFIG = YAML_CONFIG.get("context_extraction", {})
CONTEXT_ENABLED = _env_bool("LLM_CONTEXT_ENABLED", CONTEXT_CONFIG.get("enabled", False))
CONTEXT_CACHE_FIRST = _env_bool("LLM_CONTEXT_CACHE_FIRST", CONTEXT_CONFIG.get("cache_first", True))
CONTEXT_CACHE_PATH = _resolve_path(
    os.getenv("LLM_CONTEXT_CACHE_PATH", CONTEXT_CONFIG.get("cache_path", "data/context_extraction_cache.json"))
)
CONTEXT_MAX_CONTEXTS = int(os.getenv("LLM_CONTEXT_MAX_CONTEXTS", CONTEXT_CONFIG.get("max_contexts", 3)))
CONTEXT_MIN_CONFIDENCE = float(os.getenv("LLM_CONTEXT_MIN_CONFIDENCE", CONTEXT_CONFIG.get("min_confidence", 0.6)))

# LLM Out-of-Stock Alternatives Configuration
OOS_CONFIG = YAML_CONFIG.get("out_of_stock_alternatives", {})
OOS_ENABLED = _env_bool("LLM_OOS_ENABLED", OOS_CONFIG.get("enabled", False))
OOS_CACHE_FIRST = _env_bool("LLM_OOS_CACHE_FIRST", OOS_CONFIG.get("cache_first", True))
OOS_CACHE_PATH = _resolve_path(
    os.getenv("LLM_OOS_CACHE_PATH", OOS_CONFIG.get("cache_path", "data/out_of_stock_alternatives_cache.json"))
)
OOS_INVENTORY_PATH = _resolve_path(
    os.getenv("LLM_OOS_INVENTORY_PATH", OOS_CONFIG.get("inventory_path", "data/inventory.csv"))
)
OOS_MAX_ALTERNATIVES = int(os.getenv("LLM_OOS_MAX_ALTERNATIVES", OOS_CONFIG.get("max_alternatives", 1)))
OOS_MIN_SCORE = float(os.getenv("LLM_OOS_MIN_SCORE", OOS_CONFIG.get("min_score", 0.3)))


def validate_config():
    """Validate critical configuration values."""
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH, exist_ok=True)
    if not os.path.exists(MODELS_PATH):
        os.makedirs(MODELS_PATH, exist_ok=True)
    return True
