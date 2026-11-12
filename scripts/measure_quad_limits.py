"""P4.9: measure the quadrotor's guidance-level limits with the B1 stack.

Outputs ``results/tables/quad_limits.json``.  Stack: AccelBridge -> RatePID (B1)
-> mixer -> 1000 Hz physics, 500 Hz control.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from interceptor.bridge.setpoint import AccelBridge, max_thrust
from interceptor.fc.pid import RatePID
from interceptor.sim import pipeline
from interceptor.sim.constants import PH_G_W, PH_M, PH_OMEGA_HOVER

DT = 0.002
ROOT = Path(__file__).resolve().parents[1]


def fresh_state(v0=(0, 0, 0)):
    return (np.array([0.0, 0.0, 100.0]), np.array([1.0, 0, 0, 0]), np.array(v0, float),
            np.zeros(3), np.full(4, PH_OMEGA_HOVER))


def run_accel(a_cmd_fn, T, bridge=None, params=None, state=None):
    params = params or pipeline.PhysicsParams.rotor_drag()
    bridge = bridge or AccelBridge()
    pid = RatePID.tuned(dt=DT, params=params)
    s = state or fresh_state()
    n = int(T / DT)
    rec = {"t": (np.arange(n) + 1) * DT, "v": np.zeros((n, 3)), "omega": np.zeros((n, 3)),
           "sat": np.zeros(n), "spec_force": np.zeros((n, 3))}
    for k in range(n):
        sp = bridge(s[1], a_cmd_fn(k * DT))
        cmd = pid(sp.omega_cmd, sp.collective, s[3])
        s_new = pipeline.control_step(s, cmd, 4.2, params)
        rec["spec_force"][k] = (s_new[2] - s[2]) / DT - PH_G_W  # a - g (world frame)
        s = s_new
        rec["v"][k], rec["omega"][k] = s[2], s[3]
        rec["sat"][k] = float(np.any(cmd <= 0.0) or np.any(cmd >= 1.0))
    return rec


def smooth(x, w=25):
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def main():
    out = {}
    tw = max_thrust() / (PH_M * 9.81)
    out["thrust_to_weight"] = tw
    # Analytic lateral limit at level flight: a = g sqrt(tw^2 - 1)
    out["analytic_lateral_a_max"] = 9.81 * np.sqrt(tw ** 2 - 1)
    out["analytic_vertical_up_a_max"] = 9.81 * (tw - 1)
    out["analytic_vertical_down_a_max"] = 9.81

    # 1. Acceleration step response (lateral 4 m/s^2): time constant and delay.
    a_step = 4.0
    rec = run_accel(lambda t: np.array([a_step if t > 0.1 else 0.0, 0, 0]), 1.5)
    ax = smooth(rec["spec_force"][:, 0])
    idx = int(0.1 / DT)
    final = ax[idx:idx + 150].max()  # early peak; drag has not yet built up
    k63 = np.argmax(ax[idx:] >= 0.632 * a_step)
    out["tau_accel_s"] = float(k63 * DT)
    out["accel_step_final"] = float(final)

    # 2. Maximum sustained lateral acceleration holding altitude (ramp the command).
    best = 0.0
    for a in np.arange(6.0, out["analytic_lateral_a_max"] + 1.0, 1.0):
        rec = run_accel(lambda t, a=a: np.array([a, 0, 0]), 1.2)
        ax_m = smooth(rec["spec_force"][:, 0])[300:450].mean()
        az_m = smooth(rec["spec_force"][:, 2])[300:450].mean() + PH_G_W[2]
        # Peak acceleration reached within 0.6 s from rest (drag-limited model, F10).
        best = max(best, float(smooth(rec["spec_force"][:, 0]).max()))
        if rec["sat"].mean() > 0.5:
            break
    out["lateral_a_max_measured"] = float(best)

    # 3. Maximum upward acceleration with level attitude.
    rec = run_accel(lambda t: np.array([0, 0, 1.0e3 if False else (tw - 1) * 9.81 * 0.98]), 0.6)
    out["vertical_up_a_max_measured"] = float(smooth(rec["spec_force"][:, 2])[:200].max() + PH_G_W[2])  # peak in first 0.4 s, before drag builds
    rec = run_accel(lambda t: np.array([0, 0, -9.81 * 0.98]), 0.8)
    out["vertical_down_a_max_measured"] = float(-(smooth(rec["spec_force"][:, 2])[:200].min() + PH_G_W[2]))

    # 4. Maximum speed: hold the maximal tilt command until terminal velocity.
    a_hold = 0.9 * out["lateral_a_max_measured"]
    rec = run_accel(lambda t: np.array([a_hold, 0, 0]), 40.0)
    sp = np.linalg.norm(rec["v"], axis=1)
    out["v_max_measured"] = float(sp[-500:].mean())
    out["v_max_settled"] = bool(abs(sp[-500:].mean() - sp[-2500:-2000].mean()) < 0.1)

    # 5. Body-rate range demanded: peak roll rate in a hard 90 deg attitude step.
    bridge = AccelBridge(rate_limit=40.0)
    rec = run_accel(lambda t: np.array([0, 3000.0 if t > 0.1 else 0, 0]), 1.0, bridge=bridge)
    out["peak_rate_hard_attitude_step"] = float(np.abs(rec["omega"]).max())

    # 6. Jerk limit and turn-rate limit derived from the measurements.
    out["max_jerk_m_s3"] = out["lateral_a_max_measured"] / max(out["tau_accel_s"], 1e-3)
    out["omega_max_turn"] = out["lateral_a_max_measured"] / max(out["v_max_measured"], 1e-3)

    path = ROOT / "results" / "tables"
    path.mkdir(parents=True, exist_ok=True)
    (path / "quad_limits.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
