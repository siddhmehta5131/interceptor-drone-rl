"""Run any inner rate controller against a setpoint sequence (P4.4, P4.8, P4.10).

``controller(omega_cmd, collective, omega_meas) -> cmd[4]`` is the common
interface of B1 (``RatePID``), B2 (``FirmwarePID``) and wrapped RL policies.
"""
from __future__ import annotations

import numpy as np

from ..sim import pipeline
from ..sim.constants import PH_OMEGA_HOVER
from . import metrics


def rollout(controller, setpoints, params=None, wind=None, dt=0.002, physics_dt=0.001,
            state0=None, U_bat=4.2, p0=(0.0, 0.0, 10.0)):
    """Simulate ``len(setpoints)`` control steps.  Returns a dict of arrays."""
    params = params or pipeline.PhysicsParams.rotor_drag()
    s = state0 or (np.array(p0, float), np.array([1.0, 0, 0, 0]), np.zeros(3), np.zeros(3),
                   np.full(4, PH_OMEGA_HOVER))
    if hasattr(controller, "reset"):
        controller.reset()
    n = len(setpoints)
    log = {k: np.zeros((n, d)) for k, d in
           (("omega", 3), ("cmd", 4), ("q", 4), ("p", 3), ("v", 3))}
    for k in range(n):
        w = None if wind is None else wind.step(dt)
        cmd = controller(setpoints[k, :3], setpoints[k, 3], s[3])
        s = pipeline.control_step(s, cmd, U_bat, params, control_dt=dt, physics_dt=physics_dt,
                                  wind_W=w)
        log["omega"][k], log["cmd"][k], log["q"][k] = s[3], cmd, s[1]
        log["p"][k], log["v"][k] = s[0], s[2]
    log["t"] = (np.arange(n) + 1) * dt
    log["setpoint"] = np.asarray(setpoints)
    return log


def step_response_metrics(log, axis=0):
    t, y, sp = log["t"], log["omega"][:, axis], log["setpoint"][:, axis]
    k0 = int(np.argmax(np.abs(sp) > 0))
    amp = float(sp[-1])
    seg_t, seg_y = t[k0:], y[k0:]
    return {"rise": metrics.rise_time(seg_t, seg_y, amp),
            "overshoot": metrics.overshoot(seg_y, amp),
            "settling": metrics.settling_time(seg_t, seg_y, amp) - seg_t[0]}
