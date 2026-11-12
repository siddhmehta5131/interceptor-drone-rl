"""Single public logging entry point (plan P1.11).

Implementation lives in :mod:`interceptor.utils.logger` for backwards
compatibility; import from here in new code.
"""

from __future__ import annotations

from ..utils.logger import configure_file_logging, get_logger  # noqa: F401

__all__ = ["get_logger", "configure_file_logging"]
