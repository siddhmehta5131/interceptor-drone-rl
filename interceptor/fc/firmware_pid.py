"""Baseline B2: Crazyflie firmware rate PID and X-quad mixer (plan P4.5).

Gains are verbatim from ``platform_defaults_cf2.h`` (roll/pitch Kp 250, Ki 500,
Kd 2.5; yaw Kp 120, Ki 16.7).  Firmware units: rates in deg/s, outputs in
16-bit counts (full scale 65535).  Conversion to the simulator's units
(rad/s in, normalised motor command out) is explicit below.

Items to verify against the firmware source (not available offline): the rate
loop runs at 500 Hz; integral limit 33.3 deg/s (``PID_*_RATE_INTEGRATION_LIMIT``);
derivative low-pass 30 Hz at 500 Hz.  They are exposed as arguments.
"""
from __future__ import annotations

import numpy as np

from ..sim import pipeline
from ..sim.constants import PH_CUT_THR

KP_DEG = np.array([250.0, 250.0, 120.0])
KI_DEG = np.array([500.0, 500.0, 16.7])
KD_DEG = np.array([2.5, 2.5, 0.0])
INT_LIMIT_DEG = 33.3
OUT_LIMIT = 32767.0
FULL_SCALE = 65535.0
RAD2DEG = 180.0 / np.pi


def firmware_rate_pid(e_deg, integ, d_filt, kp=KP_DEG, ki=KI_DEG, kd=KD_DEG,
                      dt=0.002, int_limit=INT_LIMIT_DEG):
    """One firmware PID update on rate errors in deg/s.

    ``d_filt`` is the filtered error derivative.  Returns ``(counts, integ)``.
    """
    integ = np.clip(integ + e_deg * dt, -int_limit, int_limit)
    out = kp * e_deg + ki * integ + kd * d_filt
    return np.clip(out, -OUT_LIMIT, OUT_LIMIT), integ


def counts_to_mixer(counts: np.ndarray) -> np.ndarray:
    """Counts -> simulator mixer input.  The firmware mixer applies roll/2 and
    pitch/2 (power_distribution_quadrotor) and full-scale yaw."""
    return np.array([counts[0] / 2.0, counts[1] / 2.0, counts[2]]) / FULL_SCALE


class FirmwarePID:
    def __init__(self, dt: float = 0.002, d_cutoff_hz: float = 30.0):
        self.dt = dt
        self.alpha = dt / (dt + 1.0 / (2.0 * np.pi * d_cutoff_hz))
        self.reset()

    def reset(self):
        self.integ = np.zeros(3)
        self.prev_e = None
        self.d = np.zeros(3)

    def __call__(self, omega_cmd, collective, omega_meas):
        if collective < PH_CUT_THR:
            self.integ[:] = 0.0
            self.prev_e = None
            return np.zeros(4)
        e = (np.asarray(omega_cmd) - np.asarray(omega_meas)) * RAD2DEG
        raw_d = np.zeros(3) if self.prev_e is None else (e - self.prev_e) / self.dt
        self.prev_e = e.copy()
        self.d += self.alpha * (raw_d - self.d)
        counts, self.integ = firmware_rate_pid(e, self.integ, self.d, dt=self.dt)
        return pipeline.mixer(counts_to_mixer(counts), collective)
