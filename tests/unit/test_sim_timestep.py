"""Timestep parameterisation and sub-stepping (plan P3.1, defect F13).

The historical 100 Hz replay is preserved by the default ``dt=PH_DT`` and the
legacy half-step rotor prediction (covered by ``test_fixture_parity_sim``).
The validated path at the S8 rates evaluates the rotor speed at the RK4 stage
state, which restores fourth-order convergence.
"""

from __future__ import annotations

import numpy as np
import pytest

from interceptor.sim import pipeline
from interceptor.sim.constants import PH_C_HOVER, PH_DT, PH_OMEGA_HOVER

CMD = np.array([0.45, 0.15, -0.08, 0.04])
U_BAT = 4.2


def _init(omega=None):
    return (
        np.zeros(3),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.zeros(3),
        np.zeros(3) if omega is None else np.asarray(omega, dtype=float),
        np.full(4, PH_OMEGA_HOVER),
    )


def _run(dt: float, duration: float = 0.1, predict_rotor: bool = False):
    state = _init()
    steps = int(round(duration / dt))
    assert abs(steps * dt - duration) < 1e-12
    for _ in range(steps):
        state = pipeline.rk4(*state, CMD, U_BAT, pipeline.PhysicsParams(),
                             dt=dt, predict_rotor=predict_rotor)
    return np.concatenate(state)


# ---------------------------------------------------------------------------
# Rate helpers
# ---------------------------------------------------------------------------

def test_substeps_exact_integer_ratio():
    assert pipeline.substeps(0.002, 0.001) == 2
    assert pipeline.substeps(0.02, 0.001) == 20
    assert pipeline.substeps(PH_DT, PH_DT) == 1


@pytest.mark.parametrize("control_dt,physics_dt", [(0.0025, 0.001), (0.001, 0.002), (0.0, 0.001)])
def test_substeps_rejects_bad_ratio(control_dt, physics_dt):
    with pytest.raises(ValueError):
        pipeline.substeps(control_dt, physics_dt)


def test_control_step_equals_two_physics_steps():
    start = _init()
    expected = start
    for _ in range(2):
        expected = pipeline.rk4(*expected, CMD, U_BAT, pipeline.PhysicsParams(),
                                dt=0.001, predict_rotor=False)
    got = pipeline.control_step(start, CMD, U_BAT, pipeline.PhysicsParams())
    for a, b in zip(got, expected):
        assert np.array_equal(a, b)


def test_control_step_holds_command_over_the_interval():
    start = _init()
    one = pipeline.control_step(start, CMD, U_BAT, pipeline.PhysicsParams(),
                                control_dt=0.004, physics_dt=0.001)
    manual = start
    for _ in range(4):
        manual = pipeline.rk4(*manual, CMD, U_BAT, pipeline.PhysicsParams(),
                              dt=0.001, predict_rotor=False)
    for a, b in zip(one, manual):
        assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------

def test_legacy_half_step_is_first_order():
    """Document the legacy correction: it is retained only for fixture parity."""
    reference = _run(2e-5)
    coarse = _run(1e-3, predict_rotor=True)
    finer = _run(5e-4, predict_rotor=True)
    e_coarse = np.linalg.norm(coarse - reference)
    e_finer = np.linalg.norm(finer - reference)
    order = np.log2(e_coarse / e_finer)
    assert 0.8 < order < 1.3


def test_fourth_order_convergence():
    """RK4 error decays as dt^4 over the plan's rate ladder (P3.1).

    The reference is computed at 50 kHz on the same (clip-free) state, so the
    observed slope is the integrator order, not model noise.
    """
    rates = np.array([100, 250, 500, 1000, 2000], dtype=float)
    reference = _run(1.0 / 50_000.0)
    errors = np.array([
        np.linalg.norm(_run(1.0 / hz) - reference) for hz in rates
    ])
    slope = np.polyfit(np.log(1.0 / rates), np.log(errors), 1)[0]
    assert 3.7 < slope < 4.5, f"observed order {slope:.2f}, errors {errors}"


def test_default_dt_is_the_legacy_rate():
    """No caller loses the 100 Hz behaviour by accident."""
    a = pipeline.rk4(*_init(), CMD, U_BAT, pipeline.PhysicsParams())
    b = pipeline.rk4(*_init(), CMD, U_BAT, pipeline.PhysicsParams(), dt=PH_DT)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


def test_hover_command_is_still_hover_at_all_rates():
    """Changing dt must not move the equilibrium."""
    for hz in (100, 500, 1000, 2000):
        state = (
            np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0, 0.0]),
            np.zeros(3), np.zeros(3), np.full(4, PH_OMEGA_HOVER),
        )
        cmd = np.full(4, PH_C_HOVER)
        for _ in range(int(round(1.0 / hz))):
            state = pipeline.rk4(*state, cmd, U_BAT, pipeline.PhysicsParams(),
                                 dt=1.0 / hz, predict_rotor=False)
        assert abs(state[0][2] - 1.0) < 1e-6
        assert np.linalg.norm(state[2]) < 1e-6
