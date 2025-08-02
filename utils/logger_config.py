"""
Logging configuration utilities.
"""

import logging
import sys
from typing import Optional

from config import LOG_LEVEL, LOG_FORMAT


def setup_logging(
    level: Optional[str] = None,
    format_string: Optional[str] = None,
    log_to_file: Optional[str] = None
) -> None:
    """
    Set up logging configuration for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string for log messages
        log_to_file: Optional file path to log to file
    """
    # Use config defaults if not provided
    log_level = level or LOG_LEVEL
    log_format = format_string or LOG_FORMAT
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure logging
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_to_file:
        handlers.append(logging.FileHandler(log_to_file))
    
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        handlers=handlers,
        force=True  # Override any existing configuration
    )
    
    # Set specific loggers to appropriate levels
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with level: {log_level}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)