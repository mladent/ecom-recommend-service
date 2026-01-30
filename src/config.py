"""Configuration management for the recommendation service."""

import os
import yaml
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


def validate_config():
    """Validate critical configuration values."""
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH, exist_ok=True)
    if not os.path.exists(MODELS_PATH):
        os.makedirs(MODELS_PATH, exist_ok=True)
    return True
