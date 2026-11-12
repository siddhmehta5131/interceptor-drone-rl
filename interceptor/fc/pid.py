"""Baseline B1: the project rate PID, re-tuned at 500 Hz (plan P4.4).

Design method (model-based loop shaping, derived in docs/mathematics.md 8.1):
about one axis the plant is ``G(s) = K / (s (tau s + 1))`` where ``K`` is the
angular acceleration per unit mixer output (measured from the model by
:func:`axis_gain`) and ``tau`` the motor lag.  With a PI controller whose zero
sits a factor 5 below crossover and crossover ``wc = 1 / (3 tau)`` the phase
margin is about 60 degrees.
"""
from __future__ import annotations

import numpy as np

from ..sim import pipeline
from ..sim.constants import PH_CUT_THR, PH_C_HOVER, PH_K_MOT


def axis_gain(params: pipeline.PhysicsParams | None = None) -> np.ndarray:
    """K_i = d(omega_dot_i)/d(u_i) at hover, rad/s^2 per unit mixer output."""
    params = params or pipeline.PhysicsParams()
    gains = np.zeros(3)
    du = 0.01
    for i in range(3):
        u = np.zeros(3)
        u[i] = du
        cmd = pipeline.mixer(u, PH_C_HOVER)
        omega_ss, _ = pipeline.esc(cmd, 4.2)
        f, t = pipeline.prop_ft(omega_ss, params)
        _, tau = pipeline.aggregate(f, t)
        gains[i] = tau[i] / params.J[i, i] / du
    return gains


def design_rate_pid(K: np.ndarray, tau: float = PH_K_MOT, wc_tau: float = 1.0 / 3.0,
                    zero_ratio: float = 5.0):
    """Return ``(kp, ki)`` for crossover ``wc = wc_tau / tau`` (EQ-FC-1)."""
    wc = wc_tau / tau
    kp = wc / K
    ki = kp * wc / zero_ratio
    return kp, ki


class RatePID:
    """Rate PID plus saturation-aware mixer.  Call with the rate setpoint,
    collective command and measured body rates; returns four motor commands."""

    def __init__(self, kp, ki, kd=(0.0, 0.0, 0.0), dt: float = 0.002,
                 i_limit: float = 0.25):
        self.kp, self.ki, self.kd = (np.asarray(x, float) for x in (kp, ki, kd))
        self.dt, self.i_limit = dt, i_limit
        self.reset()

    @classmethod
    def tuned(cls, dt: float = 0.002, params=None, wc_tau: float = 1.0 / 3.0):
        kp, ki = design_rate_pid(axis_gain(params), wc_tau=wc_tau)
        return cls(kp, ki, dt=dt)

    def reset(self):
        self.integ = np.zeros(3)
        self.prev = None

    def __call__(self, omega_cmd, collective, omega_meas):
        omega_cmd = np.asarray(omega_cmd, float)
        omega_meas = np.asarray(omega_meas, float)
        if collective < PH_CUT_THR:
            self.integ[:] = 0.0
            self.prev = omega_meas.copy()
            return np.zeros(4)
        e = omega_cmd - omega_meas
        prev = omega_meas if self.prev is None else self.prev
        d = -self.kd * (omega_meas - prev) / self.dt
        self.prev = omega_meas.copy()
        u_unsat = self.kp * e + self.integ + d
        cmd = pipeline.mixer(u_unsat, collective)
        saturated = np.any(cmd <= 0.0) or np.any(cmd >= 1.0)
        if not saturated:  # conditional-integration anti-windup
            self.integ = np.clip(self.integ + self.ki * e * self.dt,
                                 -self.i_limit, self.i_limit)
        return cmd
