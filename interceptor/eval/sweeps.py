"""Robustness-sweep and analysis helpers (plan P9.1, P9.4, P9.6).

Evaluation only: no training.  A sweep varies one scalar of the engagement
(seeker noise scale, target acceleration relative to the interceptor's
normal-acceleration limit, target speed relative to the interceptor's maximum
speed) and records kill rate and median miss for one arm on one platform.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from ..guidance.env import UnifiedInterceptEnv
from ..guidance.maneuvers import ManeuverType, PlatformType
from ..guidance.platform import PLATFORMS
from .stats import KillRate, median_iqr
from ..guidance.seeker import load_sensor_noise_config

__all__ = ["SWEEPS", "limits_for", "noise_config_for", "run_episode", "run_sweep", "kill_rate_vs_radius",
           "residual_components"]

#: sweep name -> (axis label, default grid)
SWEEPS = {
    # 1x = Echodyne EchoShield 0.5 deg (configs/seeker.yaml); 2x = EchoGuard ~1 deg; 0.25x ~ pre-2026-11-14 nominal
    "noise_scale": ("seeker noise scale (x nominal)", (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)),
    "target_accel_ratio": ("target a_max / interceptor a_n_max", (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)),
    "target_speed_ratio": ("target v_nominal / interceptor v_max", (0.3, 0.5, 0.7, 0.9, 1.1)),
    # every manoeuvre shape constant (rates, switch times, mixing fractions, gains, angles) scaled together
    "maneuver_shape": ("manoeuvre shape scale (x nominal)", (0.5, 1.0, 1.5)),
    # Interceptor response limits that are project choices (docs/tuned_values_plan.md):
    # tau_accel (first-order acceleration lag), a_t_max / a_t_min (axial envelope) and
    # max_jerk, each scaled with the other limits nominal.  The axial grid reaches 0.06,
    # the fixed-wing value the book propulsion model gives (2.5 / 41.2 m/s^2, BUGS #32).
    "interceptor_lag_scale": ("interceptor tau_accel (x nominal)", (0.5, 1.0, 2.0)),
    "axial_limit_scale": ("interceptor a_t_max, a_t_min (x nominal)", (0.06, 0.25, 0.5, 1.0)),
    "jerk_limit_scale": ("interceptor max_jerk (x nominal)", (0.5, 1.0, 2.0)),
    # The three alpha-beta-gamma seeker gains (configs/seeker.yaml ema_alpha_*, PROJECT_CHOICE)
    # scaled together, each capped at 1.0 (1.0 = raw measurement, no smoothing).
    "seeker_gain_scale": ("seeker filter gains (x nominal, capped at 1)", (0.5, 1.0, 1.5)),
}


def limits_for(platform: PlatformType, sweep: str, value: float):
    """Platform limits with the swept target parameter replaced (noise sweep: unchanged)."""
    lim = PLATFORMS[platform]
    if sweep == "target_accel_ratio":
        return replace(lim, a_target_max=float(value) * lim.a_n_max)
    if sweep == "target_speed_ratio":
        return replace(lim, v_target_nominal=float(value) * lim.v_max)
    if sweep == "interceptor_lag_scale":
        return replace(lim, tau_accel=float(value) * lim.tau_accel)
    if sweep == "axial_limit_scale":
        return replace(lim, a_t_max=float(value) * lim.a_t_max, a_t_min=float(value) * lim.a_t_min)
    if sweep == "jerk_limit_scale":
        return replace(lim, max_jerk=float(value) * lim.max_jerk)
    return lim


def noise_config_for(sweep: str, value: float):
    """Seeker config with the swept filter gains (``None`` = the env loads configs/seeker.yaml)."""
    if sweep != "seeker_gain_scale":
        return None
    cfg = load_sensor_noise_config()
    v = float(value)
    return replace(cfg, ema_alpha_pos=min(1.0, v * cfg.ema_alpha_pos),
                   ema_alpha_vel=min(1.0, v * cfg.ema_alpha_vel),
                   ema_alpha_acc=min(1.0, v * cfg.ema_alpha_acc))


def run_episode(platform: PlatformType, mode: str, behavior: Optional[ManeuverType], seed: int,
                *, limits=None, noise_scale: float = 1.0, shape_scale: float = 1.0,
                noise_config=None, policy=None) -> Dict[str, float]:
    """One episode; ``policy`` is an SB3 model for RL arms, ``None`` otherwise."""
    kwargs = {"fixed_maneuver": behavior} if behavior is not None else {}
    env = UnifiedInterceptEnv(platform=platform, mode=mode, limits=limits,
                              noise_scale=noise_scale, seed=seed, noise_config=noise_config,
                              maneuver_shape_scale=shape_scale, **kwargs)
    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        action = (policy.predict(obs, deterministic=True)[0] if policy is not None
                  else np.zeros(3, dtype=np.float32))
        obs, _, term, trunc, info = env.step(action)
        done = bool(term or trunc)
    return {"killed": bool(info["killed"]), "miss": float(env.min_miss_distance)}


def run_sweep(sweep: str, platform: PlatformType, mode: str, *,
              values: Optional[Sequence[float]] = None,
              behaviors: Iterable[ManeuverType] = tuple(ManeuverType), episodes: int = 5,
              seed0: int = 10000, policy=None) -> List[dict]:
    """One row per grid value: kill rate with Wilson interval, median miss, IQR."""
    if sweep not in SWEEPS:
        raise ValueError(f"unknown sweep {sweep!r}")
    behaviors = list(behaviors)
    rows = []
    for value in (values if values is not None else SWEEPS[sweep][1]):
        kills, misses = [], []
        for b in behaviors:
            for k in range(episodes):
                r = run_episode(platform, mode, b, seed0 + k,
                                limits=limits_for(platform, sweep, value),
                                noise_scale=float(value) if sweep == "noise_scale" else 1.0,
                                shape_scale=float(value) if sweep == "maneuver_shape" else 1.0,
                                noise_config=noise_config_for(sweep, value),
                                policy=policy)
                kills.append(r["killed"])
                misses.append(r["miss"])
        kr, mi = KillRate.from_outcomes(kills), median_iqr(misses)
        rows.append({"sweep": sweep, "platform": platform.value, "mode": mode,
                     "value": float(value), "episodes": len(kills), "kill_rate": kr.rate,
                     "kill_lo": kr.lo, "kill_hi": kr.hi, "median_miss": mi["median"],
                     "iqr_miss": mi["iqr"], "misses": misses})
    return rows


def kill_rate_vs_radius(misses: Sequence[float], radii: Sequence[float]) -> List[dict]:
    """Recompute the kill rate from stored miss distances at several radii (P9.6).

    A kill is defined as ``miss <= radius``, so the kill rate at radius R is the
    empirical CDF of the miss distance evaluated at R.
    """
    m = np.asarray(misses, dtype=float)
    out = []
    for r in radii:
        kr = KillRate.from_outcomes(list(m <= r))
        out.append({"radius": float(r), "kill_rate": kr.rate, "kill_lo": kr.lo, "kill_hi": kr.hi})
    return out


def residual_components(pos: np.ndarray, target: np.ndarray, a_apn: np.ndarray, a_rl: np.ndarray):
    """Split the RL residual into along-LOS and across-LOS parts (P9.4).

    ``a_par = a_rl . l`` with ``l = (p_T - p_M) / |p_T - p_M|``; the across part is
    ``|a_rl - a_par l|``.  Returns two arrays of length ``len(a_rl)``.
    """
    n = len(a_rl)
    los = target[:n] - pos[:n]
    l = los / np.maximum(np.linalg.norm(los, axis=1, keepdims=True), 1e-9)
    par = np.sum(a_rl * l, axis=1)
    perp = np.linalg.norm(a_rl - par[:, None] * l, axis=1)
    return par, perp
