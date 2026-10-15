"""aero.py  --  aerodynamic model, ground effect, and wind (scalar, parameterized).

The aerodynamic polynomial model is a verbatim port of ``hover_env._ph_aero``
with three extensions taken from ``swift_rl_env.py``:

* **Domain randomization ready** -- the raw (acceleration-unit) polynomial
  coefficients live in ``PhysicsParams`` and are multiplied by the (possibly
  randomized) mass / inertia at evaluation time.
* **Wind** -- an Ornstein-Uhlenbeck process; wind in the world frame is
  transformed into air-relative body velocity before entering the aero model.
* **Ground effect** -- multiplicative thrust boost near the floor.

When wind is zero and the params are at their nominal values the model is
numerically identical to ``hover_env``.
"""

from __future__ import annotations

import numpy as np

from .constants import (
    GROUND_EFFECT_HEIGHT,
    GROUND_EFFECT_MAX_GAIN,
    PH_J,
    WIND_MEAN_STD,
    WIND_SIGMA_RANGE,
    WIND_THETA_RANGE,
)
from .quaternion import quat2rot

__all__ = [
    "aero_force_torque",
    "ground_effect_gain",
    "wind_ou_step",
    "sample_wind_params",
]

_AERO_CLIP_F = 50.0
_AERO_CLIP_T = 5.0


def ground_effect_gain(z: float) -> float:
    """Multiplicative thrust correction near the floor (1.0 far away).

    Port of ``swift_rl_env.ground_effect_gain`` (scalar).
    """
    h = float(np.clip(z, 0.0, GROUND_EFFECT_HEIGHT))
    return 1.0 + GROUND_EFFECT_MAX_GAIN * (1.0 - h / GROUND_EFFECT_HEIGHT)


def aero_force_torque(
    v_rel_B: np.ndarray,
    Omega: np.ndarray,
    *,
    mass: float,
    J: np.ndarray,
    fx_raw: np.ndarray,
    fy_raw: np.ndarray,
    fz_raw: np.ndarray,
    tx_raw: np.ndarray,
    ty_raw: np.ndarray,
    tz_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Body-frame aerodynamic forces/torques from air-relative velocity.

    Parameters
    ----------
    v_rel_B : (3,)   body-frame air-relative velocity (accounting for wind).
    Omega   : (4,)   motor speeds (used for Omega_bar^2 terms).

    Identical numerics to ``hover_env._ph_aero`` when the raw coefficients
    are the constants' nominals and ``mass``/``J`` are nominal.
    """
    vx, vy, vz = v_rel_B
    v_xy = float(np.sqrt(vx**2 + vy**2))
    ob2 = float(np.mean(Omega**2))  # Omega_bar^2

    fx = float((fx_raw * mass) @ [vx, vx * abs(vx), ob2, vx * ob2])
    fy = float((fy_raw * mass) @ [vy, vy * abs(vy), ob2, vy * ob2])
    fz = float((fz_raw * mass) @ [vz, vz * abs(vz), v_xy**2, v_xy * ob2, vz * ob2, v_xy * vz * ob2])
    tx = float((tx_raw * J[0, 0]) @ [vy, vy * abs(vy), ob2, vy * ob2, vy * abs(vy) * ob2])
    ty = float((ty_raw * J[1, 1]) @ [vx, vx * abs(vx), ob2, vx * ob2, vx * abs(vx) * ob2])
    tz = float((tz_raw * J[2, 2]) @ [vx, vy])

    f_aero = np.clip([fx, fy, fz], -_AERO_CLIP_F, _AERO_CLIP_F)
    tau_aero = np.clip([tx, ty, tz], -_AERO_CLIP_T, _AERO_CLIP_T)
    return np.asarray(f_aero, dtype=np.float64), np.asarray(tau_aero, dtype=np.float64)


def wind_ou_step(wind_W: np.ndarray, mean_W: np.ndarray, theta: float, sigma: float,
                 dt: float, rng: np.random.Generator) -> np.ndarray:
    """One Ornstein-Uhlenbeck step of the 3-axis wind in the world frame."""
    noise = rng.normal(0.0, sigma * np.sqrt(dt), size=(3,))
    return wind_W + theta * (mean_W - wind_W) * dt + noise


def sample_wind_params(rng: np.random.Generator) -> tuple[np.ndarray, float, float]:
    """Sample per-episode wind parameters (mean, theta, sigma)."""
    mean_W = rng.normal(0.0, WIND_MEAN_STD, size=(3,))
    theta = float(rng.uniform(*WIND_THETA_RANGE))
    sigma = float(rng.uniform(*WIND_SIGMA_RANGE))
    return mean_W, theta, sigma


def air_relative_body_velocity(q_WB: np.ndarray, v_WB: np.ndarray, wind_W: np.ndarray) -> np.ndarray:
    """Transform world-frame velocity minus wind into body frame."""
    return quat2rot(q_WB).T @ (v_WB - wind_W)