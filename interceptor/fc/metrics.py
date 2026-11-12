"""Flight-controller metrics (plan P4.8).  Pure functions on arrays."""
from __future__ import annotations

import numpy as np


def rise_time(t, y, y_final, lo=0.1, hi=0.9):
    """10-90 % rise time of a step response from y[0] toward ``y_final``."""
    y = np.asarray(y)
    span = y_final - y[0]
    if abs(span) < 1e-12:
        return 0.0
    frac = (y - y[0]) / span
    if not (np.any(frac >= lo) and np.any(frac >= hi)):
        return float("inf")
    return float(t[np.argmax(frac >= hi)] - t[np.argmax(frac >= lo)])


def overshoot(y, y_final, y0=0.0):
    """Peak overshoot as a fraction of the step size (>= 0)."""
    span = y_final - y0
    if abs(span) < 1e-12:
        return 0.0
    peak = np.max((np.asarray(y) - y0) / span)
    return float(max(peak - 1.0, 0.0))


def settling_time(t, y, y_final, tol=0.05, y0=0.0):
    """Time after which the response stays within +/- tol * |step| of y_final."""
    band = tol * abs(y_final - y0)
    out = np.where(np.abs(np.asarray(y) - y_final) > band)[0]
    if len(out) == 0:
        return float(t[0])
    if out[-1] == len(t) - 1:
        return float("inf")
    return float(t[out[-1] + 1])


def iae(t, e):
    """Integral of |e| dt (trapezoid), summed over axes."""
    return float(np.sum(np.trapezoid(np.abs(e), t, axis=0)))


def rms(e):
    return float(np.sqrt(np.mean(np.square(e))))


def control_effort(u, dt):
    """Integral of squared deviation of motor commands from their mean."""
    u = np.asarray(u)
    return float(np.sum((u - u.mean(axis=0)) ** 2) * dt)


def action_smoothness(u, dt):
    """RMS of du/dt (1/s)."""
    return rms(np.diff(np.asarray(u), axis=0) / dt)


def saturation_fraction(cmd, lo=0.0, hi=1.0):
    cmd = np.asarray(cmd)
    return float(np.mean((cmd <= lo + 1e-9) | (cmd >= hi - 1e-9)))


def tracking_metrics(t, omega_cmd, omega, cmd):
    """Summary dictionary for one tracking record."""
    e = np.asarray(omega_cmd) - np.asarray(omega)
    dt = float(t[1] - t[0])
    return {"iae": iae(t, e), "rms": rms(e), "effort": control_effort(cmd, dt),
            "smoothness": action_smoothness(cmd, dt), "saturation": saturation_fraction(cmd)}
