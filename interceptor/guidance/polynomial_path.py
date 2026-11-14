"""Closed-form polynomial target path (non-evasive target set).

Moved from ``interceptor/envs/target_generator.py`` on 2026-11-14 when the
superseded 3-stage pipeline was archived (``archive/legacy_3stage/``); the
class is unchanged. The path is ``pos(t) = start + delta * p(t / T)`` with
``p(0)=0``, ``p(1)=1`` and ``p`` a monotone shape polynomial:

    order 1:  p(tau) = tau
    order 2:  p(tau) = a*tau^2 + (1-a)*tau
    order 3:  p(tau) = a*tau^3 + b*tau^2 + (1-a-b)*tau

``delta`` is scaled down uniformly if the peak path speed would exceed the
speed cap. Polynomial paths are independent of the interceptor, so the target
is predictable from position and velocity alone.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = ["PolynomialTargetPath", "POLYNOMIAL_TARGET_TYPES"]

POLYNOMIAL_TARGET_TYPES = ("order_1", "order_2", "order_3")

# Peak-speed integration resolution for the polynomial speed cap.
_PEAK_SAMPLES = 201


class PolynomialTargetPath:
    """Closed-form target trajectory for polynomial targets.

    Parameters
    ----------
    start, end:
        World-frame positions.  ``start`` is where the target spawned;
        ``end`` is sampled at reset and clamped inside the world bounds.
    duration:
        Path duration ``T`` in seconds (the episode length).
    order:
        1, 2 or 3 -- the degree of the shape polynomial.
    rng:
        Source for the randomised shape coefficients.
    speed_cap:
        Hard cap in m/s.  ``delta`` is scaled down uniformly so that
        ``max |dp/dtau| * |delta| / T <= speed_cap``.
    shape:
        ``"random"`` samples the coefficients, ``"linear"`` forces
        ``p(tau) = tau`` regardless of ``order``.
    """

    def __init__(
        self,
        start: np.ndarray,
        end: np.ndarray,
        duration: float,
        order: int,
        rng: Optional[np.random.Generator] = None,
        speed_cap: float = 8.0,
        shape: str = "random",
    ):
        rng = rng if rng is not None else np.random.default_rng()
        self.order = int(order) if order in (1, 2, 3) else 1
        self.T = float(max(1e-6, duration))
        self.shape = shape

        start = np.asarray(start, dtype=np.float64).reshape(3)
        end = np.asarray(end, dtype=np.float64).reshape(3)
        self.start = start.copy()
        self.end = end.copy()

        self.a = 0.0
        self.b = 0.0
        if shape == "random" and self.order >= 2:
            self._sample_shape(rng)

        delta = self.end - self.start
        if speed_cap > 0.0:
            peak = self._peak_abs_pprime()
            peak_speed = peak * float(np.linalg.norm(delta)) / self.T
            if peak_speed > speed_cap and peak_speed > 0.0:
                delta = delta * (speed_cap / peak_speed)
        self.delta = delta
        self.effective_end = self.start + self.delta

    # ------------------------------------------------------------------ shape
    def _sample_shape(self, rng: np.random.Generator) -> None:
        tau = np.linspace(0.0, 1.0, _PEAK_SAMPLES)
        if self.order == 2:
            for _ in range(64):
                a = float(rng.uniform(-0.8, 0.8))
                vals = a * tau * tau + (1.0 - a) * tau
                if np.all(vals >= -1e-9) and np.all(np.diff(vals) >= -1e-9):
                    self.a = a
                    return
            self.a = 0.0
            return
        for _ in range(64):
            a = float(rng.uniform(-0.8, 0.8))
            b = float(rng.uniform(-1.0, 1.0))
            vals = a * tau ** 3 + b * tau * tau + (1.0 - a - b) * tau
            if np.all(vals >= -1e-9) and np.all(np.diff(vals) >= -1e-9):
                self.a = a
                self.b = b
                return
        self.a = 0.0
        self.b = 0.0

    def _p(self, tau: np.ndarray) -> np.ndarray:
        tau = np.asarray(tau, dtype=np.float64)
        if self.order == 1 or self.shape == "linear":
            return tau
        if self.order == 2:
            return self.a * tau * tau + (1.0 - self.a) * tau
        return self.a * tau ** 3 + self.b * tau * tau + (1.0 - self.a - self.b) * tau

    def _pprime(self, tau: np.ndarray) -> np.ndarray:
        tau = np.asarray(tau, dtype=np.float64)
        if self.order == 1 or self.shape == "linear":
            return np.ones_like(tau)
        if self.order == 2:
            return 2.0 * self.a * tau + (1.0 - self.a)
        return 3.0 * self.a * tau * tau + 2.0 * self.b * tau + (1.0 - self.a - self.b)

    def _psecond(self, tau: np.ndarray) -> np.ndarray:
        tau = np.asarray(tau, dtype=np.float64)
        if self.order == 1 or self.shape == "linear":
            return np.zeros_like(tau)
        if self.order == 2:
            return np.full_like(tau, 2.0 * self.a)
        return 6.0 * self.a * tau + 2.0 * self.b

    def _peak_abs_pprime(self) -> float:
        tau = np.linspace(0.0, 1.0, _PEAK_SAMPLES)
        return float(np.max(np.abs(self._pprime(tau))))

    # ----------------------------------------------------------------- access
    def _tau(self, t: float) -> float:
        return float(np.clip(t / self.T, 0.0, 1.0))

    def position_at(self, t: float) -> np.ndarray:
        tau = self._tau(t)
        return self.start + self.delta * float(self._p(tau))

    def velocity_at(self, t: float) -> np.ndarray:
        tau = self._tau(t)
        return self.delta * (float(self._pprime(tau)) / self.T)

    def acceleration_at(self, t: float) -> np.ndarray:
        tau = self._tau(t)
        return self.delta * (float(self._psecond(tau)) / (self.T * self.T))

    def jerk_at(self, t: float) -> np.ndarray:
        if self.order != 3 or self.shape == "linear":
            return np.zeros(3, dtype=np.float64)
        return self.delta * (6.0 * self.a / (self.T ** 3) * np.ones(3))

    @property
    def peak_speed(self) -> float:
        return self._peak_abs_pprime() * float(np.linalg.norm(self.delta)) / self.T
