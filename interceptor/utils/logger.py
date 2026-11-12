"""logger.py -- stdlib logging setup for scripts and library modules.

All module loggers hang under an ``interceptor`` root logger, so a rotating
file handler attached to the root also captures every child logger
(``orchestrator``, ``train``, ``evaluate``, ...).  Locally (no data dir
configured) the root carries a console-only handler.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

__all__ = ["get_logger", "configure_file_logging"]

_ROOT_LOGGER = "interceptor"
_CONSOLE = logging.StreamHandler(sys.stdout)
_CONSOLE.setFormatter(logging.Formatter(
    "%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))


def _ensure_root() -> logging.Logger:
    root = logging.getLogger(_ROOT_LOGGER)
    root.setLevel(logging.INFO)
    if not root.handlers:
        root.addHandler(_CONSOLE)
    return root


def get_logger(name: str = _ROOT_LOGGER) -> logging.Logger:
    """Return the named logger as a child of the ``interceptor`` root."""
    _ensure_root()
    if not name or name == _ROOT_LOGGER:
        return logging.getLogger(_ROOT_LOGGER)
    return logging.getLogger(f"{_ROOT_LOGGER}.{name}")


_FILE_HANDLER: Optional[RotatingFileHandler] = None
_CONFIGURED_FOR: Optional[str] = None


def configure_file_logging(log_dir: Optional[str]) -> None:
    """Attach a rotating file handler (once) to the ``interceptor`` root.

    ``log_dir`` is created if needed.  A second call for the same directory is
    a no-op; switching directories detaches the previous handler first.
    """
    global _FILE_HANDLER, _CONFIGURED_FOR
    if log_dir is None:
        return
    resolved = str(Path(log_dir).resolve())
    if _CONFIGURED_FOR == resolved:
        return
    root = _ensure_root()
    if _FILE_HANDLER is not None:
        root.removeHandler(_FILE_HANDLER)
        _FILE_HANDLER = None

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    _FILE_HANDLER = RotatingFileHandler(
        str(Path(log_dir) / "interceptor.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=2,
        encoding="utf-8",
    )
    _FILE_HANDLER.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(_FILE_HANDLER)
    _CONFIGURED_FOR = resolved