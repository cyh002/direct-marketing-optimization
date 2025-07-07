"""Utilities for working with MLflow servers."""

from __future__ import annotations

from typing import Optional
import logging
import urllib.request
import urllib.error

from .config_models import MlflowConfig
from .logging import get_logger


def ensure_mlflow_server(
    mlflow_config: MlflowConfig,
    timeout: int = 5,
    logger: Optional[logging.Logger] = None,
) -> MlflowConfig:
    """Ensure MLflow server is reachable, disable if not.

    Args:
        mlflow_config: MLflow configuration to validate.
        timeout: Timeout in seconds for the health check request.
        logger: Optional logger for status messages.

    Returns:
        The (possibly modified) MLflow configuration.
    """
    if logger is None:
        logger = get_logger(__name__)

    if not mlflow_config.enabled:
        return mlflow_config

    try:
        urllib.request.urlopen(mlflow_config.tracking_uri, timeout=timeout)
        logger.debug("MLflow server %s is reachable", mlflow_config.tracking_uri)
    except urllib.error.URLError as exc:
        logger.warning(
            "MLflow server %s unreachable: %s. Disabling MLflow logging.",
            mlflow_config.tracking_uri,
            exc,
        )
        mlflow_config.enabled = False
    return mlflow_config
