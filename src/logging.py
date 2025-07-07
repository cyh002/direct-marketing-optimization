from __future__ import annotations

import logging
import logging.config
import os
import yaml

DEFAULT_LOGGING_CONFIG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "conf", "logging.yaml"
)
_logger_initialized = False


def setup_logging(path: str = DEFAULT_LOGGING_CONFIG) -> None:
    """Configure Python logging.

    Args:
        path: Path to a YAML logging configuration file.
    """

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger instance."""

    global _logger_initialized
    if not _logger_initialized:
        setup_logging()
        _logger_initialized = True
    return logging.getLogger(name)
