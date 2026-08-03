"""Application-wide logging setup."""
from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler

from config import config

_LOGGERS: dict[str, logging.Logger] = {}


def get_logger(name: str = "data_analyst_agent") -> logging.Logger:
    """Return a configured, cached logger.

    Args:
        name: Logger name.

    Returns:
        A logger writing to both console and a rotating file.
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    os.makedirs(config.log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
        )

        console = logging.StreamHandler()
        console.setFormatter(fmt)
        logger.addHandler(console)

        file_handler = RotatingFileHandler(
            os.path.join(config.log_dir, "app.log"),
            maxBytes=1_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    _LOGGERS[name] = logger
    return logger
