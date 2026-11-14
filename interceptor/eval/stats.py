"""Evaluation statistics (plan P8.1).

Pure functions, no environment dependency.  Kill rate is reported with a
Wilson score interval (better than the normal approximation at the small
samples and extreme rates common here).  Miss distance is summarised by the
median and inter-quartile range plus the empirical distribution.  Paired
comparisons use McNemar's exact test for kills and a paired bootstrap for miss
distance, because both arms are evaluated on the same seeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np

__all__ = [
    "wilson_interval",
    "median_iqr",
    "empirical_cdf",
    "mcnemar_exact",
    "bootstrap_median_difference",
    "seed_summary",
    "KillRate",
]


def wilson_interval(kills: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (EQ-EVAL-1)."""
    if n <= 0:
        raise ValueError("n must be positive")
    p = float(kills) / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, float(centre - half)), min(1.0, float(centre + half))


@dataclass(frozen=True)
class KillRate:
    kills: int
    episodes: int
    rate: float
    lo: float
    hi: float

    @classmethod
    def from_outcomes(cls, outcomes: Iterable[bool], z: float = 1.959963984540054):
        values = np.asarray(list(outcomes), dtype=bool)
        n = int(values.size)
        k = int(values.sum())
        lo, hi = wilson_interval(k, n, z)
        return cls(k, n, k / n if n else float("nan"), lo, hi)


def median_iqr(values: Sequence[float]) -> dict:
    """Median, quartiles, IQR and mean for a miss-distance sample (EQ-EVAL-2)."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        raise ValueError("empty sample")
    q25, med, q75 = np.percentile(x, [25.0, 50.0, 75.0])
    return {"n": int(x.size), "median": float(med), "q25": float(q25),
            "q75": float(q75), "iqr": float(q75 - q25),
            "mean": float(np.mean(x))}


def empirical_cdf(values: Sequence[float], grid: Sequence[float] | None = None):
    """Return ``(grid, cdf)`` for plotting the miss-distance distribution."""
    x = np.sort(np.asarray(values, dtype=float))
    if grid is None:
        grid = x
    g = np.asarray(grid, dtype=float)
    return g, np.searchsorted(x, g, side="right") / max(x.size, 1)


def mcnemar_exact(outcomes_a: Sequence[bool], outcomes_b: Sequence[bool]) -> dict:
    """Exact McNemar test on paired kill outcomes (EQ-EVAL-3).

    ``b`` counts A-only kills, ``c`` B-only kills.  Under the null the
    discordant pairs are Binomial(b + c, 1/2); the two-sided p-value is the
    binomial test at ``min(b, c)``.
    """
    a = np.asarray(outcomes_a, dtype=bool)
    b = np.asarray(outcomes_b, dtype=bool)
    if a.size != b.size:
        raise ValueError("paired samples must have equal length")
    only_a = int(np.sum(a & ~b))
    only_b = int(np.sum(~a & b))
    n = only_a + only_b
    if n == 0:
        p = 1.0
    else:
        from scipy.stats import binomtest
        p = float(binomtest(min(only_a, only_b), n, 0.5,
                            alternative="two-sided").pvalue)
    return {"only_a": only_a, "only_b": only_b,
            "statistic": float(abs(only_a - only_b)), "p_value": p,
            "n_discordant": n}


def bootstrap_median_difference(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    statistic: Callable[[np.ndarray], float] = np.median,
) -> dict:
    """Paired bootstrap interval for ``statistic(a) - statistic(b)`` (EQ-EVAL-4)."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.size != y.size:
        raise ValueError("paired samples must have equal length")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    diffs = np.array([statistic(x[i]) - statistic(y[i]) for i in idx])
    point = float(statistic(x) - statistic(y))
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"difference": point, "lo": float(lo), "hi": float(hi),
            "alpha": alpha, "n_boot": n_boot}


def seed_summary(values: Sequence[float]) -> dict:
    """Mean and range across training seeds (never the best seed alone)."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        raise ValueError("empty sample")
    return {"mean": float(np.mean(x)), "min": float(np.min(x)),
            "max": float(np.max(x)), "range": float(np.ptp(x)),
            "n_seeds": int(x.size)}
