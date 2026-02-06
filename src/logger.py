"""
Centralized logging configuration for Knoccs RAG Server.
"""

import logging
import sys

# Try to import config, fallback to defaults if not available
try:
    from src.config import LoggingConfig
except ImportError:
    try:
        from config import LoggingConfig
    except ImportError:
        # Fallback defaults
        class LoggingConfig:
            LEVEL = 'INFO'
            FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name (typically __name__ of the calling module)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(getattr(logging, LoggingConfig.LEVEL.upper(), logging.INFO))

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, LoggingConfig.LEVEL.upper(), logging.INFO))

        # Formatter
        formatter = logging.Formatter(
            fmt=LoggingConfig.FORMAT,
            datefmt=LoggingConfig.DATE_FORMAT
        )
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

        # Prevent propagation to root logger to avoid duplicate logs
        logger.propagate = False

    return logger


# Pre-configured loggers for common modules
def get_server_logger() -> logging.Logger:
    return get_logger('knoccs.server')


def get_search_logger() -> logging.Logger:
    return get_logger('knoccs.search')


def get_vectorstore_logger() -> logging.Logger:
    return get_logger('knoccs.vectorstore')


def get_deepeval_logger() -> logging.Logger:
    return get_logger('knoccs.deepeval')


def get_dataloader_logger() -> logging.Logger:
    return get_logger('knoccs.dataloader')
