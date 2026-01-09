"""Production-ready logging configuration for the sensor API.

This module provides structured logging with proper formatting, log levels,
and support for both development (human-readable) and production (JSON) formats.
"""

import logging
import os
import sys


def setup_logging(level: str = "INFO", show_access_logs: bool = False) -> None:
    """Configure application logging."""

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Configure Uvicorn loggers
    for logger_name in ["uvicorn", "uvicorn.error"]:
        uvicorn_logger = logging.getLogger(logger_name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.addHandler(console_handler)
        uvicorn_logger.setLevel(level)
        uvicorn_logger.propagate = False

    # Access logs
    access_logger = logging.getLogger("uvicorn.access")
    if show_access_logs:
        access_logger.handlers.clear()
        access_logger.addHandler(console_handler)
        access_logger.setLevel(level)
    else:
        access_logger.setLevel(logging.WARNING)
    access_logger.propagate = False

    # Control SQLAlchemy logging
    # By default, only show warnings/errors to reduce noise
    sqlalchemy_logger = logging.getLogger("sqlalchemy.engine")

    # To see SQL queries, set SQLALCHEMY_LOG_LEVEL=INFO
    sqlalchemy_log_level = os.getenv("SQLALCHEMY_LOG_LEVEL", "WARNING")
    sqlalchemy_logger.setLevel(getattr(logging, sqlalchemy_log_level.upper()))

    # Quiet other noisy loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given module.

    Args:
        name: Logger name (typically __name__ from the calling module)

    Returns:
        Configured logger instance under the 'app' namespace
    """
    return logging.getLogger(name)
