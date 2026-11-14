"""Dryden turbulence, mean wind and discrete gusts (plan P3.6, D1).

The model follows MIL-HDBK-1797 / MIL-F-8785C for low altitude:

* ``sigma_w = 0.1 W20`` with ``W20`` the wind speed at 20 ft (here the mean
  wind magnitude, same units);
* ``sigma_u = sigma_v = sigma_w / (0.177 + 0.000823 h_ft)^0.4``;
* ``L_w = h`` and ``L_u = L_v = h / (0.177 + 0.000823 h_ft)^1.2``;
* forming filters with time constant ``L_i / V``.

The simulator's legacy Ornstein-Uhlenbeck process is exactly the first-order
Dryden longitudinal filter (``EQ-WIND-1``); this module supersedes it for both
the longitudinal and the lateral/vertical axes.

The filters are discretised exactly (matrix exponential of the linear SDE) and
driven by white noise with intensity ``pi``, which makes the output variance
``sigma_i^2`` by construction.  Turbulence is generated in world axes
(north, west, up); the wind-axis rotation is omitted because a quadrotor's yaw
is arbitrary, and this is recorded in ``docs/sim_validation.md``.

Equations: EQ-WIND-1 (legacy OU mapping), EQ-WIND-2 (Dryden parameters),
EQ-WIND-3 (forming filters), EQ-WIND-4 (discrete Gauss gust).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = [
    "FT",
    "V_MIN",
    "DrydenSpectrum",
    "dryden_spectrum",
    "gauss_gust",
    "DrydenWind",
]

FT = 0.3048          # m per foot
V_MIN = 5.0          # m/s minimum airspeed in the L/V time constants
NOISE_INTENSITY = np.pi


@dataclass(frozen=True)
class DrydenSpectrum:
    """Low-altitude Dryden intensities and length scales (SI)."""

    altitude: float
    airspeed: float
    W20: float
    sigma_u: float
    sigma_v: float
    sigma_w: float
    L_u: float
    L_v: float
    L_w: float

    def psd(self, omega: np.ndarray, axis: str = "u") -> np.ndarray:
        """One-sided analytic PSD, ``int_0^inf S(w) dw = sigma^2``."""
        omega = np.asarray(omega, dtype=float)
        if axis == "u":
            sigma, L = self.sigma_u, self.L_u
            shape = 1.0 / (1.0 + (L * omega / self.airspeed) ** 2)
            gain = 2.0 * L / (np.pi * self.airspeed)
        elif axis in ("v", "w"):
            sigma = self.sigma_v if axis == "v" else self.sigma_w
            L = self.L_v if axis == "v" else self.L_w
            x = L * omega / self.airspeed
            shape = (1.0 + 3.0 * x**2) / (1.0 + x**2) ** 2
            gain = L / (np.pi * self.airspeed)
        else:  # pragma: no cover - argparse-style guard
            raise ValueError(f"unknown axis {axis!r}")
        return sigma**2 * gain * shape


def dryden_spectrum(
    altitude: float,
    W20: float,
    airspeed: float,
) -> DrydenSpectrum:
    """Compute the MIL-HDBK-1797 low-altitude Dryden parameters (EQ-WIND-2)."""
    h_m = max(float(altitude), 0.1)
    h_ft = h_m / FT
    V = max(float(airspeed), V_MIN)
    # Low-altitude Dryden model, MIL-F-8785C form (L_w = h, L_u = L_v = h / lam^1.2,
    # sigma_w = 0.1 W20, sigma_u/sigma_w = 1 / lam^0.4), as tabulated by MathWorks:
    # https://www.mathworks.com/help/aeroblks/drydenwindturbulencemodelcontinuous.html
    lam = 0.177 + 0.000823 * h_ft
    sigma_w = 0.1 * float(W20)
    sigma_uv = sigma_w / lam**0.4
    return DrydenSpectrum(
        altitude=h_m,
        airspeed=V,
        W20=float(W20),
        sigma_u=sigma_uv,
        sigma_v=sigma_uv,
        sigma_w=sigma_w,
        L_u=h_m / lam**1.2,
        L_v=h_m / lam**1.2,
        L_w=h_m,
    )


def gauss_gust(t: float, amplitude: float, duration: float) -> float:
    """Discrete 1-cosine gust (EQ-WIND-4); zero outside ``[0, duration]``."""
    if duration <= 0.0 or t <= 0.0 or t >= duration:
        return 0.0
    return 0.5 * amplitude * (1.0 - np.cos(2.0 * np.pi * t / duration))


class DrydenWind:
    """Stationary Dryden turbulence plus a mean wind.

    Parameters
    ----------
    altitude : float
        Height above ground in metres.
    mean_wind : (3,) array
        Mean wind in the world frame (north, west, up), m/s.
    W20 : float, optional
        Wind speed at 20 ft.  Defaults to ``||mean_wind||``.
    airspeed : float, optional
        Airspeed used in the forming filters; clamped at ``V_MIN``.
    seed : int or Generator, optional
        Explicit generator (plan convention: one RNG per environment).
    """

    def __init__(
        self,
        altitude: float,
        mean_wind: Optional[np.ndarray] = None,
        *,
        W20: Optional[float] = None,
        airspeed: float = 15.0,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.mean_wind = np.zeros(3) if mean_wind is None else np.asarray(mean_wind, dtype=float)
        if W20 is None:
            W20 = float(np.linalg.norm(self.mean_wind)) or 10.0
        self.spectrum = dryden_spectrum(altitude, W20, airspeed)
        self.rng = rng if rng is not None else np.random.default_rng(seed)
        self._x_u = 0.0
        self._x_v = np.zeros(2)
        self._x_w = np.zeros(2)

    # -- internal filter helpers -------------------------------------------

    def _u_step(self, dt: float) -> float:
        s = self.spectrum
        tau = s.L_u / s.airspeed
        a = np.exp(-dt / tau)
        q_var = NOISE_INTENSITY * (1.0 - a * a) / (2.0 * tau)
        self._x_u = a * self._x_u + np.sqrt(q_var) * self.rng.normal()
        return self._x_u

    def _second_order_step(self, dt: float, L: float) -> np.ndarray:
        """Exact discretisation of ``(1 + sqrt(3) tau s) / (1 + tau s)^2``."""
        tau = L / self.spectrum.airspeed
        alpha = dt / tau
        a = np.exp(-alpha)
        a2 = a * a
        q11 = NOISE_INTENSITY * (1.0 - a2) / (2.0 * tau)
        q12 = NOISE_INTENSITY * (1.0 - a2 * (1.0 + 2.0 * alpha)) / (4.0 * tau)
        q22 = NOISE_INTENSITY * (2.0 - a2 * (2.0 + 4.0 * alpha + 4.0 * alpha * alpha)) / (8.0 * tau)
        cov = np.array([[q11, q12], [q12, q22]])
        # Cholesky of a PSD covariance; the tiny clip guards round-off.
        chol = np.linalg.cholesky(cov + 1e-30 * np.eye(2))
        noise = chol @ self.rng.normal(size=2)
        state = self._x_v if L == self.spectrum.L_v else self._x_w
        x1_old, x2_old = state
        x1 = a * x1_old + noise[0]
        # Exact transition of the triangular system: x2 <- a*x2 + alpha*a*x1.
        x2 = a * x2_old + alpha * a * x1_old + noise[1]
        out = np.array([x1, x2])
        if L == self.spectrum.L_v:
            self._x_v = out
        else:
            self._x_w = out
        return out

    def _turbulence(self, dt: float) -> np.ndarray:
        s = self.spectrum
        x_u = self._u_step(dt)
        u = s.sigma_u * np.sqrt(2.0 * s.L_u / (np.pi * s.airspeed)) * x_u

        k = np.sqrt(s.L_v / (np.pi * s.airspeed))
        x1, x2 = self._second_order_step(dt, s.L_v)
        v = s.sigma_v * k * (np.sqrt(3.0) * x1 + (1.0 - np.sqrt(3.0)) * x2)

        k = np.sqrt(s.L_w / (np.pi * s.airspeed))
        x1, x2 = self._second_order_step(dt, s.L_w)
        w = s.sigma_w * k * (np.sqrt(3.0) * x1 + (1.0 - np.sqrt(3.0)) * x2)
        return np.array([u, v, w])

    # -- public API ---------------------------------------------------------

    def step(self, dt: float) -> np.ndarray:
        """Advance one step and return the world-frame wind vector (m/s)."""
        return self.mean_wind + self._turbulence(dt)

    def reset(self, rng: Optional[np.random.Generator] = None) -> None:
        """Zero the filter states (and optionally replace the generator)."""
        self._x_u = 0.0
        self._x_v = np.zeros(2)
        self._x_w = np.zeros(2)
        if rng is not None:
            self.rng = rng
