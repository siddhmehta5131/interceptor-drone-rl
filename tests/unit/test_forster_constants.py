"""Quad motor constants are the values implied by Forster (2015) (BUGS #34).

Thesis mappings (doi:10.3929/ethz-b-000214143): eq. (3.3) thrust vs command, Sec. 3.3.2
rotor speed vs command, eq. (3.5) torque vs thrust, eq. (3.16) command-to-thrust pole.
"""
import numpy as np
import pytest

from interceptor.sim import constants as c


def _f(cmd):
    a, b, d = c.FORSTER_THRUST_POLY
    return a * cmd * cmd + b * cmd + d


def _w(cmd):
    k, w0 = c.FORSTER_OMEGA_LIN
    return k * cmd + w0


def test_thrust_coefficient_is_least_squares_of_forster_maps():
    cmd = np.arange(10000.0, 60001.0, 1000.0)
    kappa = float(np.sum(_f(cmd) * _w(cmd) ** 2) / np.sum(_w(cmd) ** 4))
    assert c.PH_C_L == pytest.approx(kappa, rel=1e-3)
    # one coefficient reproduces the measured thrust within 5 % from below hover (cmd 40000) to
    # full command, and within 15 % at low throttle (it overestimates there)
    err = c.PH_C_L * _w(cmd) ** 2 / _f(cmd) - 1.0
    assert np.max(np.abs(err[cmd >= 40000])) < 0.05
    assert np.max(np.abs(err)) < 0.15
    assert abs(c.PH_C_L * _w(65535.0) ** 2 / _f(65535.0) - 1.0) < 0.07


def test_torque_lag_and_speed_limits():
    assert c.PH_C_D == pytest.approx(c.FORSTER_TORQUE_PER_THRUST * c.PH_C_L, rel=1e-3)
    assert c.PH_K_MOT == pytest.approx(-c.FORSTER_TS / np.log(c.FORSTER_POLE), rel=1e-3)
    assert c.PH_OMEGA_MAX == pytest.approx(_w(65535.0), abs=0.05)
    assert float(c.esc_omega_ss(1.0)) == pytest.approx(c.PH_OMEGA_MAX, abs=0.05)


def test_esc_map_is_forster_linear_map_and_invertible():
    for cc in (0.05, 0.3, c.PH_C_HOVER, 1.0):
        assert float(c.esc_omega_ss(cc)) == pytest.approx(_w(65535.0 * cc), rel=1e-9)
        assert c.esc_cmd_for_omega(float(c.esc_omega_ss(cc))) == pytest.approx(cc, abs=1e-12)
    # hover: thrust equals weight, with headroom left below full command
    assert 4.0 * c.PH_C_L * c.PH_OMEGA_HOVER ** 2 == pytest.approx(c.PH_M * 9.81, rel=1e-12)
    assert 0.5 < c.PH_C_HOVER < 0.9
