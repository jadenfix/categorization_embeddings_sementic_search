"""
Configuration settings for the venue categorization system.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
OPENAI_API_KEY: Optional[str] = os.getenv('OPEN_AI_API_KEY')
SERPER_API_KEY: Optional[str] = os.getenv('SERPER_KEY')

# Model Configuration
EMBEDDING_MODEL: str = "text-embedding-3-small"
EMBEDDING_DIMENSION: int = 1536

CHAT_MODEL: str = "gpt-4o-mini"

# API Endpoints
SERPER_SEARCH_URL: str = "https://google.serper.dev/search"

# File Paths
PROJECT_ROOT: str = "/Users/jadenfix/reprompt"
DATA_DIR: str = f"{PROJECT_ROOT}/data"
RESULTS_DIR: str = f"{PROJECT_ROOT}/results"

# Input Files
CATEGORIES_FILE: str = f"{DATA_DIR}/foursquare_categories - foursquare_categories.csv"
VENUES_FILE: str = f"{DATA_DIR}/foursquare_categories - data.csv"

# Output Files
CATEGORY_EMBEDDINGS_BASE: str = f"{RESULTS_DIR}/foursquare_category_embeddings"
VENUE_EMBEDDINGS_BASE: str = f"{RESULTS_DIR}/foursquare_venue_embeddings"
VENUE_MATCHES_FILE: str = f"{RESULTS_DIR}/venue_category_matches.csv"
VENUE_MATCHES_WITH_FALLBACK_FILE: str = f"{RESULTS_DIR}/venue_category_matches_with_llm_fallback.csv"

# Processing Configuration
DEFAULT_RETRY_ATTEMPTS: int = 3
API_DELAY_SECONDS: float = 0.5
SEARCH_RESULTS_LIMIT: int = 3
EXPONENTIAL_BACKOFF_BASE: int = 2

# Logging Configuration
LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Validation
def validate_config() -> None:
    """Validate that required configuration is present."""
    if not OPENAI_API_KEY:
        raise ValueError("OPEN_AI_API_KEY environment variable is required")
    if not SERPER_API_KEY:
        raise ValueError("SERPER_KEY environment variable is required")

# File extensions
NUMPY_EXTENSION: str = "_embeddings.npy"
METADATA_EXTENSION: str = "_metadata.json"
CSV_EXTENSION: str = ".csv"
PICKLE_EXTENSION: str = ".pkl"