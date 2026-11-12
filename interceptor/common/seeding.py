"""Deterministic seeding shared by every environment, trainer and evaluator.

The project standard is one explicit :class:`numpy.random.Generator` per
environment (plan §2), with Python/NumPy/Torch global RNGs seeded only at run
boundaries.  ``seed_everything`` is the boundary helper; ``make_rng`` is the
per-component helper.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

__all__ = ["make_rng", "seed_everything", "derive_seed"]


def make_rng(seed: Optional[int]) -> np.random.Generator:
    """Return a private Generator (``None`` gives fresh OS entropy)."""
    return np.random.default_rng(seed)


def derive_seed(base: int, *parts: object) -> int:
    """Derive a stable child seed from a base seed and component labels.

    Uses SHA-256 rather than :func:`hash` so the value is identical across
    processes and platforms (PYTHONHASHSEED independent).
    """
    import hashlib

    label = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(f"{int(base)}|{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def seed_everything(seed: int, *, torch: bool = False) -> int:
    """Seed Python, NumPy and (optionally) Torch, returning the seed.

    Torch is only imported when requested so the simulation-only install does
    not need it.  ``PYTHONHASHSEED`` cannot be changed at runtime; scripts set
    it before interpreter start when byte-identical dict ordering is required.
    """
    seed = int(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    if torch:
        try:
            import torch as _torch
        except Exception:  # pragma: no cover - optional dependency
            return seed
        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
            _torch.backends.cudnn.deterministic = True
            _torch.backends.cudnn.benchmark = False
    return seed
