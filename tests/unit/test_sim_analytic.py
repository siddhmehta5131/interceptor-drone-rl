"""Analytic verification of the 6-DOF model (plan P3.2).

Every case has a closed-form solution, so the tolerances are set by the
integrator rather than by a reference implementation.  All cases run at the
S8 physics rate (1000 Hz) with the validated (fourth-order) rotor handling.
"""

from __future__ import annotations

import numpy as np
import pytest

from interceptor.sim import pipeline
from interceptor.sim.constants import PH_C_HOVER, PH_G_W, PH_J, PH_M, PH_OMEGA_HOVER
from interceptor.sim.quaternion import quat2rot

DT = 1e-3
TOL = 1e-6


def _no_aero() -> pipeline.PhysicsParams:
    """Nominal inertias/rotors with the aerodynamic polynomial switched off."""
    return pipeline.PhysicsParams(
        fx_raw=np.zeros(4),
        fy_raw=np.zeros(4),
        fz_raw=np.zeros(6),
        tx_raw=np.zeros(5),
        ty_raw=np.zeros(5),
        tz_raw=np.zeros(2),
    )


def _run(state, cmd, duration, params, dt=DT):
    steps = int(round(duration / dt))
    for _ in range(steps):
        state = pipeline.rk4(*state, cmd, 4.2, params, dt=dt, predict_rotor=False)
    return state


def test_free_fall():
    """Motors off, no aero: p = 1/2 g t^2, v = g t exactly."""
    params = _no_aero()
    state = (np.array([0.0, 0.0, 10.0]), np.array([1.0, 0.0, 0.0, 0.0]),
             np.zeros(3), np.zeros(3), np.zeros(4))
    t = 0.5
    p, q, v, omega, Omega = _run(state, np.zeros(4), t, params)
    assert np.allclose(p, [0.0, 0.0, 10.0 + 0.5 * PH_G_W[2] * t**2], atol=TOL)
    assert np.allclose(v, PH_G_W * t, atol=TOL)
    assert np.allclose(omega, 0.0, atol=TOL)
    assert np.allclose(Omega, 0.0, atol=TOL)
    assert q[0] == pytest.approx(1.0)


def test_hover_equilibrium():
    """Collective at PH_C_HOVER balances weight, so the state is stationary."""
    params = _no_aero()
    state = (np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0, 0.0]),
             np.zeros(3), np.zeros(3), np.full(4, PH_OMEGA_HOVER))
    p, q, v, omega, Omega = _run(state, np.full(4, PH_C_HOVER), 1.0, params)
    assert np.allclose(p, [0.0, 0.0, 1.0], atol=TOL)
    assert np.allclose(v, 0.0, atol=TOL)
    assert np.allclose(omega, 0.0, atol=TOL)
    assert np.allclose(Omega, PH_OMEGA_HOVER, atol=1e-6)
    # Thrust balances weight to machine precision at the solved equilibrium.
    thrust = 4.0 * params.c_l * PH_OMEGA_HOVER**2
    assert thrust / (PH_M * 9.81) == pytest.approx(1.0, rel=1e-12)


def test_torque_free_rotation_conserves_momentum_and_energy():
    """No torque: world angular momentum and rotational energy are invariant."""
    params = _no_aero()
    params.c_l = 0.0          # no propeller thrust
    params.c_d = 0.0          # no propeller reaction torque
    omega0 = np.array([1.0, 2.0, 0.5])
    state = (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]),
             np.zeros(3), omega0.copy(), np.zeros(4))

    def invariants(s):
        q, omega = s[1], s[3]
        H_world = quat2rot(q) @ (PH_J @ omega)
        energy = 0.5 * omega @ (PH_J @ omega)
        return H_world, energy

    H0, E0 = invariants(state)
    s = _run(state, np.zeros(4), 0.2, params)
    H1, E1 = invariants(s)
    assert np.linalg.norm(H1 - H0) / np.linalg.norm(H0) < TOL
    assert abs(E1 - E0) / E0 < TOL
    # The body still rotates at the same body-frame rate magnitude.
    assert np.linalg.norm(s[3]) == pytest.approx(np.linalg.norm(omega0), rel=TOL)


def test_constant_torque_about_a_principal_axis():
    """tau about z with omega along z: omega_z linear, angle quadratic.

    The rotational sub-state is integrated here with the same RK4 rule on
    ``pipeline.rigid_body`` so the torque assembly itself is checked against
    the closed form.  With the spin axis a principal axis the gyroscopic term
    ``omega x J omega`` vanishes identically.
    """
    tau = 1.0e-4
    omega0 = 0.3
    Jz = PH_J[2, 2]
    duration, dt = 0.2, DT

    def f(q, omega):
        _p, q_dot, _v, omega_dot, _Om = pipeline.rigid_body(
            np.zeros(3), np.zeros(3), np.array([0.0, 0.0, tau]), np.zeros(3),
            np.zeros(3), np.zeros(3),
            q, np.zeros(3), omega, np.zeros(4), np.zeros(4),
            _no_aero(),
        )
        return q_dot, omega_dot

    q = np.array([1.0, 0.0, 0.0, 0.0])
    omega = np.array([0.0, 0.0, omega0])
    # Analytic derivative at t=0.
    _, omega_dot = f(q, omega)
    assert omega_dot[2] == pytest.approx(tau / Jz, rel=1e-12)

    for _ in range(int(round(duration / dt))):
        k1q, k1w = f(q, omega)
        k2q, k2w = f(q + 0.5 * dt * k1q, omega + 0.5 * dt * k1w)
        k3q, k3w = f(q + 0.5 * dt * k2q, omega + 0.5 * dt * k2w)
        k4q, k4w = f(q + dt * k3q, omega + dt * k3w)
        q = q + dt / 6.0 * (k1q + 2 * k2q + 2 * k3q + k4q)
        omega = omega + dt / 6.0 * (k1w + 2 * k2w + 2 * k3w + k4w)
        q = q / np.linalg.norm(q)

    alpha = tau / Jz
    assert omega[2] == pytest.approx(omega0 + alpha * duration, rel=TOL)
    assert omega[0] == pytest.approx(0.0, abs=1e-12)
    assert omega[1] == pytest.approx(0.0, abs=1e-12)

    angle_expected = omega0 * duration + 0.5 * alpha * duration**2
    angle = 2.0 * np.arctan2(q[3], q[0])
    assert angle == pytest.approx(angle_expected, rel=TOL)


def test_single_axis_thrust_acceleration():
    """Four rotors at equal speed: a_z = 4 c_l Omega^2 / m - g, no lateral motion."""
    params = _no_aero()
    cmd = np.full(4, 0.5)
    Omega_ss, _ = pipeline.esc(cmd, 4.2)
    expected_az = 4.0 * params.c_l * float(Omega_ss[0] ** 2) / params.mass + PH_G_W[2]
    state = (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]),
             np.zeros(3), np.zeros(3), np.asarray(Omega_ss, dtype=float))
    t = 0.1
    p, q, v, omega, Omega = _run(state, cmd, t, params)
    assert np.allclose(p, [0.0, 0.0, 0.5 * expected_az * t**2], atol=TOL)
    assert np.allclose(v, [0.0, 0.0, expected_az * t], atol=TOL)
    assert np.allclose(omega, 0.0, atol=TOL)
    assert np.allclose(Omega, Omega_ss, atol=1e-6)
