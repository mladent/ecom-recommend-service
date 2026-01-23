"""Configuration management for the recommendation service."""

import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

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

# Kaggle Configuration
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY = os.getenv("KAGGLE_KEY")

# LLM Configuration (for future use)
# LLM_API_KEY = os.getenv("LLM_API_KEY")
# LLM_MODEL = os.getenv("LLM_MODEL")
# LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.7))


def validate_config():
    """Validate critical configuration values."""
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH, exist_ok=True)
    if not os.path.exists(MODELS_PATH):
        os.makedirs(MODELS_PATH, exist_ok=True)
    return True
