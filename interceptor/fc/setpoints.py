"""Rate/collective setpoint generators (plan P4.2).

Every generator returns ``(T, 4)`` arrays ``[omega_x, omega_y, omega_z, collective]``
sampled at ``dt``.  The rate range ``rate_max`` comes from the P4.9
measurement (``configs/fc.yaml``).  All randomness uses an explicit
``numpy.random.Generator``.
"""
from __future__ import annotations

import numpy as np

from ..sim.constants import PH_C_HOVER

KINDS = ("step", "ramp", "doublet", "sinusoid", "chirp", "random")


def _blank(n):
    sp = np.zeros((n, 4))
    sp[:, 3] = PH_C_HOVER
    return sp


def step(n, dt, axis, amp, t0=0.1):
    sp = _blank(n)
    sp[int(t0 / dt):, axis] = amp
    return sp


def ramp(n, dt, axis, amp, t0=0.1, duration=0.5):
    sp = _blank(n)
    t = np.arange(n) * dt
    sp[:, axis] = amp * np.clip((t - t0) / duration, 0.0, 1.0)
    return sp


def doublet(n, dt, axis, amp, t0=0.1, width=0.3):
    sp = _blank(n)
    t = np.arange(n) * dt
    sp[(t >= t0) & (t < t0 + width), axis] = amp
    sp[(t >= t0 + width) & (t < t0 + 2 * width), axis] = -amp
    return sp


def sinusoid(n, dt, axis, amp, freq=1.0):
    sp = _blank(n)
    sp[:, axis] = amp * np.sin(2 * np.pi * freq * np.arange(n) * dt)
    return sp


def chirp(n, dt, axis, amp, f0=0.2, f1=5.0):
    """Linear chirp from f0 to f1 over the record."""
    sp = _blank(n)
    t = np.arange(n) * dt
    T = n * dt
    phase = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) * t ** 2 / T)
    sp[:, axis] = amp * np.sin(phase)
    return sp


def band_limited_random(n, dt, rng, amp, bandwidth=2.0):
    """First-order low-passed Gaussian noise (corner ``bandwidth`` Hz); the
    stationary sigma is amp/3 and the signal is clipped to +/- amp."""
    a = np.exp(-2 * np.pi * bandwidth * dt)
    sp = _blank(n)
    x = np.zeros(3)
    gain = np.sqrt((1 + a) / (1 - a)) * (1 - a)  # unit stationary variance
    for k in range(n):
        x = a * x + gain * rng.normal(size=3)
        sp[k, :3] = np.clip(x * amp / 3.0, -amp, amp)
    return sp


def collective_profile(n, dt, rng, spread=0.15):
    """Smooth random collective around hover, +/- spread * hover."""
    t = np.arange(n) * dt
    f = rng.uniform(0.3, 1.5)
    ph = rng.uniform(0, 2 * np.pi)
    return PH_C_HOVER * (1 + spread * np.sin(2 * np.pi * f * t + ph))


def sample(kind, n, dt, rng, rate_max):
    """Random setpoint sequence of the given kind (random axis and amplitude)."""
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind}")
    axis = int(rng.integers(0, 3))
    amp = float(rng.uniform(0.3, 1.0) * rate_max * rng.choice([-1, 1]))
    if kind == "step":
        sp = step(n, dt, axis, amp, t0=rng.uniform(0.05, 0.3))
    elif kind == "ramp":
        sp = ramp(n, dt, axis, amp, duration=rng.uniform(0.2, 1.0))
    elif kind == "doublet":
        sp = doublet(n, dt, axis, amp, width=rng.uniform(0.15, 0.5))
    elif kind == "sinusoid":
        sp = sinusoid(n, dt, axis, abs(amp), freq=rng.uniform(0.5, 3.0))
    elif kind == "chirp":
        sp = chirp(n, dt, axis, abs(amp))
    else:
        sp = band_limited_random(n, dt, rng, abs(amp))
    sp[:, 3] = collective_profile(n, dt, rng)
    return sp
