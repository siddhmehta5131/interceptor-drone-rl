"""Airframe limit derivations for P2.4.

Aerosonde point-mass equations and limit derivations: EQ-FW-1.
The AIM5 normal-force limit reproduces the CADAC++ aerodynamics module:
``cnp_max = cd(alpha_max, M) sin(alpha_max) + cl(alpha_max, M) cos(alpha_max)``
and ``g_max = cnp_max q S / (m g)``.  The Aerosonde helpers evaluate the
textbook point-mass formulas.  Data are loaded from ``configs/airframes``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

__all__ = [
    "load_airframe", "bilinear", "aim5_normal_force_coefficient",
    "aim5_gmax", "aerosonde_stall_speed", "aerosonde_load_factor",
    "aerosonde_turn_rate",
]

_ROOT = Path(__file__).resolve().parents[2]


def _resolve(path: "str | Path") -> Path:
    path = Path(path)
    return path if path.exists() else _ROOT / path


def load_airframe(path: "str | Path") -> dict:
    import yaml

    with _resolve(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _leaf(data, path: str):
    node = data
    for key in path.split("."):
        node = node[key]
    return node["value"] if isinstance(node, dict) and "value" in node else node


def bilinear(x: float, y: float, xs: Sequence[float], ys: Sequence[float],
             table: Sequence[Sequence[float]]) -> float:
    """Bilinear interpolation with edge clamping."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    z = np.asarray(table, dtype=float)
    x = float(np.clip(x, xs[0], xs[-1]))
    y = float(np.clip(y, ys[0], ys[-1]))
    i = int(np.clip(np.searchsorted(xs, x) - 1, 0, len(xs) - 2))
    j = int(np.clip(np.searchsorted(ys, y) - 1, 0, len(ys) - 2))
    tx = (x - xs[i]) / (xs[i + 1] - xs[i])
    ty = (y - ys[j]) / (ys[j + 1] - ys[j])
    return float(
        (1 - tx) * (1 - ty) * z[i, j]
        + tx * (1 - ty) * z[i + 1, j]
        + (1 - tx) * ty * z[i, j + 1]
        + tx * ty * z[i + 1, j + 1]
    )


def aim5_aero_coefficients(alpha_deg: float, mach: float,
                           powered: bool = False,
                           config: Optional[dict] = None) -> tuple:
    """Return (cl, cd) interpolated from the CADAC AIM5 deck."""
    cfg = config or load_airframe("configs/airframes/aim5.yaml")
    aero = cfg["aerodynamics"]
    alpha = _leaf(aero, "alpha_grid_deg")
    machs = _leaf(aero, "mach_grid")
    cl = bilinear(alpha_deg, mach, alpha, machs,
                  _leaf(aero, "cl_vs_alpha_mach"))
    cd_key = "cd_powered_vs_alpha_mach" if powered else "cd_unpowered_vs_alpha_mach"
    cd = bilinear(alpha_deg, mach, alpha, machs, _leaf(aero, cd_key))
    return cl, cd


def aim5_normal_force_coefficient(alpha_deg: float, mach: float,
                                  powered: bool = False,
                                  config: Optional[dict] = None) -> float:
    """CADAC ``cnp_max`` in the manoeuvre plane (EQ-MS-1)."""
    cl, cd = aim5_aero_coefficients(alpha_deg, mach, powered, config)
    alpha = np.deg2rad(alpha_deg)
    return float(cd * np.sin(alpha) + cl * np.cos(alpha))


def aim5_gmax(alpha_deg: float, mach: float, dynamic_pressure: float,
              mass: float, *, powered: bool = False,
              area: Optional[float] = None, gravity: float = 9.81,
              config: Optional[dict] = None) -> float:
    """CADAC++ g-limit calculation (EQ-MS-1)."""
    cfg = config or load_airframe("configs/airframes/aim5.yaml")
    if area is None:
        area = _leaf(cfg["physical"], "reference_area")
    cnp = aim5_normal_force_coefficient(alpha_deg, mach, powered, cfg)
    return float(cnp * dynamic_pressure * area / (mass * gravity))


def aerosonde_stall_speed(mass: float, rho: float, wing_area: float,
                          cl_max: float, gravity: float = 9.81) -> float:
    """Level-flight stall speed (EQ-FW-2)."""
    return float(np.sqrt(2.0 * mass * gravity / (rho * wing_area * cl_max)))


def aerosonde_load_factor(speed: float, rho: float, wing_area: float,
                          cl: float, mass: float,
                          gravity: float = 9.81) -> float:
    """Instantaneous load factor n = L/W (EQ-FW-5)."""
    lift = 0.5 * rho * speed ** 2 * wing_area * cl
    return float(lift / (mass * gravity))


def aerosonde_turn_rate(speed: float, load_factor: float,
                        gravity: float = 9.81) -> float:
    """Coordinated-turn rate for a given load factor (EQ-FW-6)."""
    return float(gravity * np.sqrt(max(load_factor ** 2 - 1.0, 0.0))
                 / max(speed, 1e-6))
