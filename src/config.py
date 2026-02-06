"""
Centralized configuration for Knoccs RAG Server.
All settings, constants, and environment variables in one place.
"""

import os
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

class DatabaseConfig:
    HOST = os.getenv('PGVECTOR_HOST', 'localhost')
    PORT = int(os.getenv('PGVECTOR_PORT', '5432'))
    DATABASE = os.getenv('PGVECTOR_DATABASE', 'postgres')
    USER = os.getenv('PGVECTOR_USER', 'postgres')
    PASSWORD = os.getenv('PGVECTOR_PASSWORD', 'postgres')
    SSLMODE = 'require'

    @classmethod
    def get_connection_params(cls) -> dict:
        return {
            'host': cls.HOST,
            'port': cls.PORT,
            'database': cls.DATABASE,
            'user': cls.USER,
            'password': cls.PASSWORD,
            'sslmode': cls.SSLMODE
        }


# =============================================================================
# LLM CONFIGURATION
# =============================================================================

class LLMConfig:
    # Main LLM for RAG responses
    MODEL = os.getenv('LLM_MODEL', 'gpt-4o')
    TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', '0.1'))

    # OpenAI API Key
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

    # DeepEval model (faster, cheaper for evaluation)
    DEEPEVAL_MODEL = os.getenv('DEEPEVAL_MODEL', 'gpt-4o-mini')


# =============================================================================
# EMBEDDING CONFIGURATION
# =============================================================================

class EmbeddingConfig:
    MODEL = os.getenv('EMBEDDING_MODEL', 'all-mpnet-base-v2')
    RERANKER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'


# =============================================================================
# CHUNKING CONFIGURATION
# =============================================================================

class ChunkingConfig:
    SIZE = int(os.getenv('CHUNK_SIZE', '1024'))
    OVERLAP = int(os.getenv('CHUNK_OVERLAP', '256'))


# =============================================================================
# SEARCH CONFIGURATION
# =============================================================================

class SearchConfig:
    # Number of final results to return
    TOP_K = int(os.getenv('SEARCH_TOP_K', '15'))

    # Number of initial candidates before reranking
    INITIAL_K = int(os.getenv('SEARCH_INITIAL_K', '50'))

    # Company extraction similarity threshold (0.0 - 1.0)
    COMPANY_MATCH_THRESHOLD = float(os.getenv('COMPANY_MATCH_THRESHOLD', '0.6'))

    # Ground truth matching threshold
    GROUND_TRUTH_THRESHOLD = float(os.getenv('GROUND_TRUTH_THRESHOLD', '0.6'))


# =============================================================================
# DEEPEVAL CONFIGURATION
# =============================================================================

class DeepEvalConfig:
    # Timeout settings (in seconds)
    PER_TASK_TIMEOUT = int(os.getenv('DEEPEVAL_PER_TASK_TIMEOUT', '180'))
    PER_ATTEMPT_TIMEOUT = int(os.getenv('DEEPEVAL_PER_ATTEMPT_TIMEOUT', '90'))
    RETRY_MAX_ATTEMPTS = int(os.getenv('DEEPEVAL_RETRY_MAX_ATTEMPTS', '1'))

    # Metric thresholds
    METRIC_THRESHOLD = float(os.getenv('DEEPEVAL_METRIC_THRESHOLD', '0.7'))

    # Parallel execution
    MAX_CONCURRENT = int(os.getenv('DEEPEVAL_MAX_CONCURRENT', '3'))


# =============================================================================
# SERVER CONFIGURATION
# =============================================================================

class ServerConfig:
    HOST = os.getenv('SERVER_HOST', '127.0.0.1')
    PORT = int(os.getenv('SERVER_PORT', '8000'))

    # CORS origins - parse from env (comma-separated) or use defaults
    _cors_env = os.getenv('CORS_ORIGINS', '')
    CORS_ORIGINS = [o.strip() for o in _cors_env.split(',') if o.strip()] or [
        "http://localhost:4200",
        "http://localhost:3000",
    ]

    # Data paths
    DATA_FOLDER = os.getenv('DATA_FOLDER', 'data')
    GROUND_TRUTH_FILE = os.getenv('GROUND_TRUTH_FILE', 'ground_truth.json')


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

class LoggingConfig:
    LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


# =============================================================================
# CONVERSATION CONFIGURATION
# =============================================================================

class ConversationConfig:
    # Maximum recent messages to keep in full
    MAX_RECENT_MESSAGES = int(os.getenv('MAX_RECENT_MESSAGES', '10'))


# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================

def validate_config():
    """
    Validate critical environment variables at startup.
    Fails fast if required configuration is missing or insecure.
    """
    errors = []

    # Check OpenAI API key
    if not LLMConfig.OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY is required")

    # Check database password (warn if using default)
    if not DatabaseConfig.PASSWORD or DatabaseConfig.PASSWORD == 'postgres':
        errors.append("PGVECTOR_PASSWORD must be set to a secure value (not 'postgres')")

    if errors:
        raise ValueError(
            "Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors)
        )
