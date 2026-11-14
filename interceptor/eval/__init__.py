"""Evaluation package (plan P8)."""

from .stats import (  # noqa: F401
    KillRate, bootstrap_median_difference, empirical_cdf, mcnemar_exact,
    median_iqr, seed_summary, wilson_interval,
)

__all__ = [
    "KillRate", "wilson_interval", "median_iqr", "empirical_cdf",
    "mcnemar_exact", "bootstrap_median_difference", "seed_summary",
]
