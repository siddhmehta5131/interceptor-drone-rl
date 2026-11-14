"""Logging and deterministic seeding (plan P1.11).

The 3-stage configuration loader that used to be re-exported here was archived
on 2026-11-14 (``archive/legacy_3stage/``); v2 configs are read by the module
that owns them (``guidance/platform.py``, ``training/algos.py``, ``fc/env.py``).
"""

from .logging import configure_file_logging, get_logger  # noqa: F401
from .seeding import derive_seed, make_rng, seed_everything  # noqa: F401

__all__ = [
    "get_logger", "configure_file_logging",
    "make_rng", "derive_seed", "seed_everything",
]
