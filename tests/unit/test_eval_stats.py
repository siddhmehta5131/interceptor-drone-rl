"""P8.1 tests: Wilson, median/IQR, McNemar, paired bootstrap."""

from __future__ import annotations

import numpy as np
import pytest

from interceptor.eval.stats import (
    KillRate, bootstrap_median_difference, empirical_cdf, mcnemar_exact,
    median_iqr, seed_summary, wilson_interval,
)


def test_wilson_known_values():
    lo, hi = wilson_interval(50, 100)
    assert lo == pytest.approx(0.4038, abs=0.001)
    assert hi == pytest.approx(0.5962, abs=0.001)
    # Degenerate but valid: no kills / all kills
    lo0, hi0 = wilson_interval(0, 10)
    assert lo0 == 0.0 and 0.0 < hi0 < 0.35
    lo1, hi1 = wilson_interval(10, 10)
    assert hi1 == pytest.approx(1.0) and 0.65 < lo1 < 1.0


def test_wilson_rejects_empty_sample():
    with pytest.raises(ValueError):
        wilson_interval(0, 0)


def test_kill_rate_and_interval_agree():
    kr = KillRate.from_outcomes([True] * 7 + [False] * 3)
    assert kr.kills == 7 and kr.episodes == 10
    assert kr.rate == pytest.approx(0.7)
    assert kr.lo < kr.rate < kr.hi


def test_median_iqr_matches_numpy():
    x = np.arange(1, 101, dtype=float)
    out = median_iqr(x)
    assert out["median"] == pytest.approx(50.5)
    assert out["q25"] == pytest.approx(25.75)
    assert out["iqr"] == pytest.approx(49.5)


def test_empirical_cdf_is_monotone_and_ends_at_one():
    grid, cdf = empirical_cdf([3.0, 1.0, 2.0], [0.0, 1.0, 2.0, 3.0, 4.0])
    assert np.all(np.diff(cdf) >= 0)
    assert cdf[0] == 0.0 and cdf[-1] == 1.0
    assert cdf[2] == pytest.approx(2 / 3)


def test_mcnemar_exact_known_value():
    # 1 discordant pair favouring A, 9 favouring B
    a = [True] + [False] * 9
    b = [False] * 10
    out = mcnemar_exact(a, b)
    assert out["only_a"] == 1 and out["only_b"] == 0
    # Build the intended b/c counts explicitly: 10 pairs, A wins 1, B wins 9
    a = [True] + [False] * 9
    b = [False] + [True] * 9
    out = mcnemar_exact(a, b)
    assert out["only_a"] == 1 and out["only_b"] == 9
    assert out["p_value"] == pytest.approx(0.021484375, abs=1e-9)


def test_mcnemar_no_discordance():
    out = mcnemar_exact([True, False], [True, False])
    assert out["p_value"] == 1.0 and out["n_discordant"] == 0


def test_bootstrap_identical_samples_is_zero():
    x = np.linspace(0.0, 10.0, 40)
    out = bootstrap_median_difference(x, x, n_boot=200, seed=3)
    assert out["difference"] == 0.0
    assert out["lo"] == 0.0 and out["hi"] == 0.0


def test_bootstrap_detects_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(10.0, 1.0, 200)
    b = rng.normal(9.0, 1.0, 200)
    out = bootstrap_median_difference(a, b, n_boot=1000, seed=1)
    assert out["difference"] == pytest.approx(1.0, abs=0.25)
    assert out["lo"] > 0.0


def test_seed_summary_reports_mean_not_best():
    out = seed_summary([0.2, 0.5, 0.8])
    assert out["mean"] == pytest.approx(0.5)
    assert out["range"] == pytest.approx(0.6)
    assert out["n_seeds"] == 3
