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


def measure(*, a_step=4.0, smooth_w=25, lat_T=1.2, lat_step=1.0, v_hold_T=40.0,
            v_hold_frac=0.9, peak_window=200, params=None):
    """Measure the envelope.  The keyword arguments are the procedure settings
    (step size, smoothing window, hold times); the defaults produced
    ``quad_limits.json``.  ``scripts/check_quad_limits_repeatability.py`` varies
    them to show the envelope does not depend on the procedure."""
    sm = lambda x: smooth(x, smooth_w)
    run = lambda fn, T, **kw: run_accel(fn, T, params=params, **kw)
    out = {}
    tw = max_thrust() / (PH_M * 9.81)
    out["thrust_to_weight"] = tw
    # Analytic lateral limit at level flight: a = g sqrt(tw^2 - 1)
    out["analytic_lateral_a_max"] = 9.81 * np.sqrt(tw ** 2 - 1)
    out["analytic_vertical_up_a_max"] = 9.81 * (tw - 1)
    out["analytic_vertical_down_a_max"] = 9.81

    # 1. Acceleration step response: time constant (63.2 % of the commanded step).
    rec = run(lambda t: np.array([a_step if t > 0.1 else 0.0, 0, 0]), 1.5)
    ax = sm(rec["spec_force"][:, 0])
    idx = int(0.1 / DT)
    final = ax[idx:idx + 150].max()  # early peak; drag has not yet built up
    k63 = np.argmax(ax[idx:] >= 0.632 * a_step)
    out["tau_accel_s"] = float(k63 * DT)
    out["accel_step_final"] = float(final)

    # 2. Maximum lateral acceleration: raise the command until the motors saturate;
    #    record the peak acceleration reached from rest (drag-limited model, F10).
    best = 0.0
    for a in np.arange(6.0, out["analytic_lateral_a_max"] + 1.0, lat_step):
        rec = run(lambda t, a=a: np.array([a, 0, 0]), lat_T)
        best = max(best, float(sm(rec["spec_force"][:, 0]).max()))
        if rec["sat"].mean() > 0.5:
            break
    out["lateral_a_max_measured"] = float(best)

    # 3. Maximum vertical acceleration with level attitude (peak before drag builds).
    rec = run(lambda t: np.array([0, 0, (tw - 1) * 9.81 * 0.98]), 0.6)
    out["vertical_up_a_max_measured"] = float(sm(rec["spec_force"][:, 2])[:peak_window].max() + PH_G_W[2])
    rec = run(lambda t: np.array([0, 0, -9.81 * 0.98]), 0.8)
    out["vertical_down_a_max_measured"] = float(-(sm(rec["spec_force"][:, 2])[:peak_window].min() + PH_G_W[2]))

    # 4. Maximum speed: hold the maximal tilt command until terminal velocity.
    a_hold = v_hold_frac * out["lateral_a_max_measured"]
    rec = run(lambda t: np.array([a_hold, 0, 0]), v_hold_T)
    sp = np.linalg.norm(rec["v"], axis=1)
    out["v_max_measured"] = float(sp[-500:].mean())
    out["v_max_settled"] = bool(abs(sp[-500:].mean() - sp[-2500:-2000].mean()) < 0.1)

    # 5. Body-rate range demanded: peak roll rate in the hardest unsaturated attitude step.
    #    The step demands v_hold_frac x the measured lateral limit.  An unreachable demand
    #    (the former 3000 m/s^2) pins the collective at full thrust, and the collective-priority
    #    mixer (EQ-SIX-4) then leaves no roll torque, so the vehicle never tilts (BUGS #37).
    bridge = AccelBridge(rate_limit=40.0)
    a_att = v_hold_frac * out["lateral_a_max_measured"]
    rec = run(lambda t: np.array([0, a_att if t > 0.1 else 0, 0]), 1.0, bridge=bridge)
    out["peak_rate_hard_attitude_step"] = float(np.abs(rec["omega"]).max())

    # 6. Jerk limit and turn-rate limit derived from the measurements.
    out["max_jerk_m_s3"] = out["lateral_a_max_measured"] / max(out["tau_accel_s"], 1e-3)
    out["omega_max_turn"] = out["lateral_a_max_measured"] / max(out["v_max_measured"], 1e-3)
    return out


def main():
    out = measure()
    path = ROOT / "results" / "tables"
    path.mkdir(parents=True, exist_ok=True)
    (path / "quad_limits.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
