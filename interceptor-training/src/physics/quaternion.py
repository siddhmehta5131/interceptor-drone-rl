"""quaternion.py  --  scalar quaternion helpers (body <-> world rotations)."""

from __future__ import annotations

import numpy as np

__all__ = ["quat2rot", "quat_mult", "normalize_quat"]


def quat2rot(q: np.ndarray) -> np.ndarray:
    """[w, x, y, z] quaternion -> 3x3 rotation matrix (body -> world).

    Verbatim port of ``hover_env._ph_quat2rot``.
    """
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z),   2 * (x * y - w * z),       2 * (x * z + w * y)],
        [2 * (x * y + w * z),       1 - 2 * (x * x + z * z),   2 * (y * z - w * x)],
        [2 * (x * z - w * y),       2 * (y * z + w * x),       1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def quat_mult(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions, [w, x, y, z] convention.

    Verbatim port of ``hover_env._ph_quat_mult``.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


def normalize_quat(q: np.ndarray) -> np.ndarray:
    """Renormalise a quaternion (in place safe)."""
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else q