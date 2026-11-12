"""Frame-converted manoeuvre laws must reproduce the original NED module.

Each fixture is a target trajectory recorded from the archived
``evasive_maneuvers.py`` (NED).  The ported module runs in the z-up frame from
the rotated initial conditions; rotating its output back must match
(plan P1.6).  ``T = diag(1, -1, -1)`` is its own inverse.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from interceptor.guidance.maneuvers import (
    EvasiveTargetGenerator, ManeuverType, PlatformType,
)

FIX = Path(__file__).resolve().parents[1] / "fixtures"
T = np.array([1.0, -1.0, -1.0])
TOL = 1e-9

CASES = [(p, m) for p in PlatformType for m in ManeuverType]


@pytest.mark.parametrize("platform,maneuver", CASES,
                         ids=[f"{p.value}-{m.value}" for p, m in CASES])
def test_maneuver_matches_original(platform, maneuver):
    fx = np.load(FIX / f"maneuver_{platform.value}_{maneuver.value}.npz")
    dt = float(fx["dt"])
    tgt = EvasiveTargetGenerator(platform, fx["p_T0"] * T, fx["v_T0"] * T, maneuver)
    p_M = fx["p_M0"] * T
    v_M = fx["v_M"] * T
    P, V, A, E, W = [], [], [], [], []
    t = 0.0
    for _ in range(len(fx["p"])):
        p, v, a, _ = tgt.step(dt, t, p_M, v_M)
        P.append(p); V.append(v); A.append(a)
        E.append(tgt.euler.copy()); W.append(tgt.omega_B.copy())
        p_M = p_M + v_M * dt
        t += dt
    scale = max(1.0, float(np.max(np.abs(fx["p"]))))
    assert np.max(np.abs(np.array(P) * T - fx["p"])) <= TOL * scale
    assert np.max(np.abs(np.array(V) * T - fx["v"])) <= TOL * scale
    assert np.max(np.abs(np.array(A) * T - fx["a"])) <= TOL * scale
    # Euler angles are physical (aerospace convention) and frame-independent.
    d = np.array(E) - fx["euler"]
    assert np.max(np.abs(np.arctan2(np.sin(d), np.cos(d)))) <= 1e-7
