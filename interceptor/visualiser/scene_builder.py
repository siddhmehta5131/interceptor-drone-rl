"""scene_builder.py -- mesh / transform / config helpers for the 3D visualiser.

Keeps all of the "make a PyVista object" glue in one place so ``renderer.py``
stays focused on scene state.  Everything that touches PyVista is guarded by a
lazy import, so importing this module never fails on a machine without VTK.
"""

from __future__ import annotations

from itertools import permutations, product
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "have_pyvista",
    "require_pyvista",
    "load_visualiser_config",
    "quat_to_matrix",
    "transform_matrix",
    "quat_from_direction",
    "load_drone_mesh",
    "placeholder_drone",
    "primitive_mesh",
    "trail_polydata",
    "cube_rotations",
    "rotation_matrix_from",
    "apply_mesh_rotation",
]

_DEFAULT_CONFIG: Dict[str, Any] = {
    "offscreen": True,
    "resolution": [1920, 1080],
    "fps": 30,
    "quality": 10,
    "background": "sky",
    "lighting": {"intensity": 0.8, "shadows": False},
    "ground": {"size": 200.0, "color": "darkseagreen"},
    "models": {
        "interceptor": "models/interceptor_drone.obj",
        "target": "models/target_drone.obj",
        "scale": 0.01,
        "fallback_size": 1.5,
    },
    "colors": {"interceptor": "steelblue", "target": "crimson"},
    "trails": {"max_points": 400, "width": 2, "interceptor": "#4d9fe0", "target": "#e04d4d"},
    "camera": {"mode": "chase", "distance": 15.0, "elevation": 8.0},
    "show": {
        "los_line": True,
        "apn_arrow": True,
        "rl_arrow": True,
        "total_arrow": True,
        "kill_zone": True,
        "hud": True,
        "axes": True,
    },
}


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_visualiser_config(path: str = "configs/visualiser.yaml") -> Dict[str, Any]:
    """Read the visualiser YAML, merged over built-in defaults.

    The returned mapping carries BOTH the nested structure used by this
    package and the flat keys the plan's reference implementation expects
    (``interceptor_model_path``, ``target_model_path``, ``resolution`` ...).
    """
    data: Dict[str, Any] = {}
    p = Path(path)
    if p.exists():
        try:
            import yaml

            loaded = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            if isinstance(loaded, Mapping):
                data = dict(loaded)
        except Exception:
            data = {}
    cfg = _deep_merge(_DEFAULT_CONFIG, data)

    models = cfg.get("models", {})
    cfg.setdefault("interceptor_model_path", models.get("interceptor"))
    cfg.setdefault("target_model_path", models.get("target"))
    cfg.setdefault("interceptor_scale", models.get("interceptor_scale", models.get("scale", 0.01)))
    cfg.setdefault("target_scale", models.get("target_scale", models.get("scale", 0.01)))
    cfg.setdefault("fallback_size", models.get("fallback_size", 1.5))
    cfg.setdefault("interceptor_rotation", models.get("interceptor_rotation", "identity"))
    cfg.setdefault("target_rotation", models.get("target_rotation", "identity"))
    return cfg


def have_pyvista() -> bool:
    try:
        import pyvista  # noqa: F401

        return True
    except Exception:
        return False


def require_pyvista():
    try:
        import pyvista as pv

        return pv
    except Exception as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "The 3D visualiser needs pyvista (and VTK).  Install it with "
            "`pip install pyvista trimesh`."
        ) from exc


# --------------------------------------------------------- model orientation

_CUBE_ROTATIONS_CACHE: Optional[list] = None


def cube_rotations() -> list:
    """All 24 proper rotations of a cube as 3x3 matrices (det = +1).

    Ordering matches ``data/orientation_shots`` so that, e.g., code ``I08``
    refers to ``cube_rotations()[8]``.  Each matrix maps local -> world:
    ``world = R @ local``.
    """
    global _CUBE_ROTATIONS_CACHE
    if _CUBE_ROTATIONS_CACHE is None:
        rots = []
        for perm in permutations(range(3)):
            for signs in product([1, -1], repeat=3):
                m = np.zeros((3, 3), dtype=np.float64)
                for i, (p, s) in enumerate(zip(perm, signs)):
                    m[i, p] = s
                if np.linalg.det(m) > 0:
                    rots.append(m)
        rots.sort(key=lambda m: tuple(m.ravel()))
        _CUBE_ROTATIONS_CACHE = rots
    return _CUBE_ROTATIONS_CACHE


def rotation_matrix_from(value: Any) -> np.ndarray:
    """Resolve a rotation spec into a 3x3 matrix.

    Accepts ``None`` / ``"identity"`` (no rotation), a code such as ``"I08"``
    or ``"T21"`` (indexes :func:`cube_rotations`), or a row-major length-9 list.
    """
    if value is None:
        return np.eye(3)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("", "identity", "none", "i", "t"):
            return np.eye(3)
        digits = "".join(ch for ch in text if ch.isdigit())
        if not digits:
            return np.eye(3)
        idx = int(digits)
        rots = cube_rotations()
        if 0 <= idx < len(rots):
            return np.array(rots[idx], dtype=np.float64)
        return np.eye(3)
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 9:
        return arr.reshape(3, 3)
    return np.eye(3)


def apply_mesh_rotation(mesh, matrix: Optional[np.ndarray]):
    """Return a copy of ``mesh`` with ``points' = matrix @ points``."""
    if matrix is None:
        return mesh
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (3, 3) or np.allclose(m, np.eye(3)):
        return mesh
    out = mesh.copy()
    out.points = (m @ np.asarray(out.points).T).T
    return out


# --------------------------------------------------------------------- math


def quat_to_matrix(quat: Sequence[float]) -> np.ndarray:
    """3x3 rotation matrix from a ``(w, x, y, z)`` quaternion."""
    w, x, y, z = (float(v) for v in quat)
    n = (w * w + x * x + y * y + z * z) ** 0.5
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transform_matrix(position: Sequence[float], quat: Sequence[float], scale: float = 1.0) -> np.ndarray:
    """4x4 affine matrix placing a mesh at ``position`` with orientation ``quat``."""
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = quat_to_matrix(quat) * float(scale)
    m[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return m


def quat_from_direction(direction: Sequence[float], up: Sequence[float] = (0.0, 0.0, 1.0)) -> np.ndarray:
    """Quaternion (w, x, y, z) whose +x axis points along ``direction``."""
    d = np.asarray(direction, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    d = d / n
    x_axis = np.array([1.0, 0.0, 0.0])
    dot = float(np.clip(x_axis @ d, -1.0, 1.0))
    if dot > 1.0 - 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if dot < -1.0 + 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])  # 180 deg about z
    axis = np.cross(x_axis, d)
    axis = axis / (float(np.linalg.norm(axis)) + 1e-12)
    angle = float(np.arccos(dot))
    half = 0.5 * angle
    return np.array([np.cos(half), *(np.sin(half) * axis)])


# -------------------------------------------------------------------- meshes


def _as_mesh(data: Any):
    """Collapse a MultiBlock / CompositeDataSet to a single PolyData."""
    import pyvista as pv

    if isinstance(data, pv.MultiBlock):
        merged = data.combine()
        return merged
    return data


def load_drone_mesh(path: Optional[str], scale: float = 0.01, fallback_size: float = 1.5):
    """Load an OBJ (or any VTK-readable) mesh, falling back to a placeholder.

    Returns ``(mesh, used_placeholder)``.
    """
    pv = require_pyvista()
    if path and str(path).startswith("primitive:"):
        return primitive_mesh(str(path).split(":", 1)[1], fallback_size), True
    if path:
        candidate = Path(path)
        if candidate.exists():
            try:
                mesh = _as_mesh(pv.read(str(candidate)))
                return mesh.scale(float(scale), inplace=False), False
            except Exception:
                pass
    return placeholder_drone(fallback_size), True


def primitive_mesh(kind: str, size: float = 1.5):
    """Meshes built from primitives, nose along +x (no third-party asset, no licence issue)."""
    pv = require_pyvista()
    s = float(size)
    if kind == "missile":
        body = pv.Cylinder(center=(0, 0, 0), direction=(1, 0, 0), radius=0.12 * s, height=2.4 * s)
        nose = pv.Cone(center=(1.5 * s, 0, 0), direction=(1, 0, 0), radius=0.12 * s, height=0.6 * s)
        parts = [nose]
        for ang in (0, 90, 180, 270):
            c, sn = np.cos(np.radians(ang)), np.sin(np.radians(ang))
            parts.append(pv.Cube(center=(-1.0 * s, 0.3 * s * c, 0.3 * s * sn),
                                 x_length=0.5 * s,
                                 y_length=0.6 * s * abs(c) + 0.02 * s,
                                 z_length=0.6 * s * abs(sn) + 0.02 * s))
        return body.merge(parts)
    if kind == "fixed_wing":
        fus = pv.Cylinder(center=(0, 0, 0), direction=(1, 0, 0), radius=0.12 * s, height=2.0 * s)
        nose = pv.Cone(center=(1.2 * s, 0, 0), direction=(1, 0, 0), radius=0.12 * s, height=0.4 * s)
        wing = pv.Cube(center=(0.1 * s, 0, 0), x_length=0.45 * s, y_length=2.4 * s, z_length=0.03 * s)
        tail = pv.Cube(center=(-0.9 * s, 0, 0), x_length=0.3 * s, y_length=0.9 * s, z_length=0.03 * s)
        fin = pv.Cube(center=(-0.9 * s, 0, 0.18 * s), x_length=0.3 * s, y_length=0.03 * s, z_length=0.4 * s)
        return fus.merge([nose, wing, tail, fin])
    return placeholder_drone(size)


def placeholder_drone(size: float = 1.5):
    """A tiny stylised quadcopter built from primitives (no assets required)."""
    pv = require_pyvista()
    body = pv.Cube(center=(0, 0, 0), x_length=size, y_length=size * 0.5, z_length=size * 0.25)
    parts = [body]
    arm = size * 0.7
    for sx in (-1, 1):
        for sy in (-1, 1):
            rotor = pv.Cylinder(
                center=(sx * arm, sy * arm, 0.0),
                direction=(0, 0, 1),
                radius=size * 0.28,
                height=size * 0.06,
            )
            parts.append(rotor)
    combined = parts[0].merge(parts[1:])
    return combined


def trail_polydata(points: Sequence[Sequence[float]]):
    """A poly-line through ``points`` (>= 2 points) else ``None``."""
    pv = require_pyvista()
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < 2:
        return None
    poly = pv.PolyData(pts)
    cells = np.full((pts.shape[0] - 1, 3), 2, dtype=np.int64)
    cells[:, 1] = np.arange(pts.shape[0] - 1)
    cells[:, 2] = np.arange(1, pts.shape[0])
    poly.lines = cells.ravel()
    return poly
