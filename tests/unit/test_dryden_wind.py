"""Dryden wind verification (plan P3.6, D1).

Variance is checked over 30 independent 10-minute realisations to keep the
sampling error of the long-correlation velocity component below 5 %.  The
spectral shape is checked on the ensemble-averaged periodogram.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import welch

from interceptor.sim.wind import (
    FT,
    V_MIN,
    DrydenWind,
    dryden_spectrum,
    gauss_gust,
)

ALT = 100.0
W20 = 15.0
AIRSPEED = 15.0
DT = 0.05
DURATION = 600.0
N_REALISATIONS = 30


@pytest.fixture(scope="module")
def ensemble():
    steps = int(round(DURATION / DT))
    spectra = None
    variances = []
    freq = None
    spectrum = None
    for seed in range(N_REALISATIONS):
        wind = DrydenWind(ALT, np.zeros(3), W20=W20, airspeed=AIRSPEED, seed=seed)
        spectrum = wind.spectrum
        data = np.array([wind._turbulence(DT) for _ in range(steps)])
        # Mean is known to be zero; the uncentred moment avoids the
        # finite-sample bias of subtracting a correlated sample mean.
        variances.append(np.mean(data**2, axis=0))
        if spectra is None:
            spectra = [[], [], []]
        for axis in range(3):
            f, p = welch(data[:, axis], fs=1.0 / DT, nperseg=2048, detrend="constant")
            spectra[axis].append(p)
        freq = f
    return spectrum, np.array(variances), freq, [np.mean(s, axis=0) for s in spectra]


def test_dryden_parameters_match_mil_hdbk_1797():
    s = dryden_spectrum(ALT, W20, AIRSPEED)
    h_ft = ALT / FT
    lam = 0.177 + 0.000823 * h_ft
    assert s.sigma_w == pytest.approx(0.1 * W20)
    assert s.sigma_u == pytest.approx(s.sigma_w / lam**0.4)
    assert s.L_w == pytest.approx(ALT)
    assert s.L_u == pytest.approx(ALT / lam**1.2)
    assert s.airspeed == AIRSPEED


def test_minimum_airspeed_clamp():
    slow = dryden_spectrum(ALT, W20, 0.5)
    assert slow.airspeed == V_MIN
    assert dryden_spectrum(ALT, W20, -3.0).airspeed == V_MIN


def test_variances_within_five_percent(ensemble):
    spectrum, variances, _, _ = ensemble
    target = np.array([spectrum.sigma_u, spectrum.sigma_v, spectrum.sigma_w])
    measured = np.sqrt(variances.mean(axis=0))
    rel = np.abs(measured / target - 1.0)
    assert np.all(rel < 0.05), f"relative variance error {rel}"


def test_spectra_match_analytic_shape(ensemble):
    spectrum, _, freq, pdfs = ensemble
    mask = (freq >= 0.05) & (freq <= 2.0)
    for axis, name in enumerate("uvw"):
        modelled = spectrum.psd(2.0 * np.pi * freq, name) * 2.0 * np.pi
        sim = pdfs[axis]
        sim_n = sim / np.trapezoid(sim[mask], freq[mask])
        mod_n = modelled / np.trapezoid(modelled[mask], freq[mask])
        rel = np.abs(sim_n[mask] / mod_n[mask] - 1.0)
        assert np.median(rel) < 0.10, (name, np.median(rel))
        assert np.max(rel) < 0.35, (name, np.max(rel))


def test_mean_wind_superposition():
    mean = np.array([5.0, -2.0, 0.5])
    wind = DrydenWind(ALT, mean, W20=0.0, airspeed=AIRSPEED, seed=0)
    samples = np.array([wind.step(0.01) for _ in range(20)])
    assert np.allclose(samples, mean, atol=1e-12)


def test_gauss_gust():
    amp, duration = 3.0, 2.0
    assert gauss_gust(-0.1, amp, duration) == 0.0
    assert gauss_gust(0.0, amp, duration) == 0.0
    assert gauss_gust(duration, amp, duration) == 0.0
    assert gauss_gust(duration / 2.0, amp, duration) == pytest.approx(amp)
    ts = np.linspace(0.0, duration, 201)
    vals = np.array([gauss_gust(t, amp, duration) for t in ts])
    assert vals.min() == pytest.approx(0.0)
    assert vals.max() == pytest.approx(amp)
    assert np.all(np.diff(vals[100:]) <= 1e-12)  # falls monotonically after the peak


def test_zero_turbulence_for_zero_intensity():
    wind = DrydenWind(ALT, np.zeros(3), W20=0.0, airspeed=AIRSPEED, seed=0)
    assert np.allclose([wind.step(0.01) for _ in range(20)], 0.0)
