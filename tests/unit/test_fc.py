"""P4 tests: environment, setpoints, metrics, baselines, bridge."""
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from interceptor.bridge.setpoint import AccelBridge
from interceptor.fc import metrics, rollout, setpoints
from interceptor.fc.env import ACTOR_DIM, PRIV_DIM, FlightControllerEnv, load_fc_config, reward_terms
from interceptor.fc.firmware_pid import firmware_rate_pid
from interceptor.fc.pid import RatePID
from interceptor.sim.constants import PH_C_HOVER


def test_env_api_and_dims():
    env = FlightControllerEnv(seed=0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (ACTOR_DIM + PRIV_DIM,) == (49,)
    for _ in range(5):
        obs, r, term, trunc, _ = env.step(env.action_space.sample())
        assert np.isfinite(r) and obs.shape == (49,)


def test_check_env():
    check_env(FlightControllerEnv(seed=1), skip_render_check=True)


def test_env_is_seeded():
    a, b = FlightControllerEnv(seed=3), FlightControllerEnv(seed=3)
    oa, _ = a.reset(seed=3)
    ob, _ = b.reset(seed=3)
    assert np.allclose(oa, ob)


def test_reward_terms_sign_and_zero():
    cfg = load_fc_config()
    assert reward_terms(np.zeros(3), 8.0, 0.0, np.zeros(4), 0.0, cfg) == 0.0
    assert reward_terms(np.ones(3), 8.0, 0.0, np.zeros(4), 0.0, cfg) < 0
    assert reward_terms(np.zeros(3), 8.0, 0.5, np.zeros(4), 0.0, cfg) < 0


@pytest.mark.parametrize("kind", setpoints.KINDS)
def test_setpoint_generators(kind):
    rng = np.random.default_rng(0)
    sp = setpoints.sample(kind, 500, 0.002, rng, 8.0)
    assert sp.shape == (500, 4)
    assert np.all(np.abs(sp[:, :3]) <= 8.0 + 1e-9)
    assert np.all(sp[:, 3] > 0)


def test_step_and_doublet_shapes():
    s = setpoints.step(100, 0.01, 1, 2.0, t0=0.2)
    assert s[19, 1] == 0 and s[20, 1] == 2.0
    d = setpoints.doublet(200, 0.01, 0, 1.0, t0=0.1, width=0.2)
    assert d[15, 0] == 1.0 and d[35, 0] == -1.0 and d[60, 0] == 0.0


def test_metrics_on_synthetic_second_order():
    t = np.linspace(0, 5, 5001)
    zeta, wn = 0.5, 10.0
    wd = wn * np.sqrt(1 - zeta ** 2)
    y = 1 - np.exp(-zeta * wn * t) * (np.cos(wd * t) + zeta / np.sqrt(1 - zeta ** 2) * np.sin(wd * t))
    assert metrics.overshoot(y, 1.0) == pytest.approx(np.exp(-np.pi * zeta / np.sqrt(1 - zeta ** 2)), abs=1e-3)
    assert metrics.rise_time(t, y, 1.0) == pytest.approx(0.1 * 1.0 * 1.6, abs=0.03)  # ~1.6/wn
    assert metrics.settling_time(t, y, 1.0) == pytest.approx(0.6, abs=0.15)
    assert metrics.saturation_fraction(np.array([[0, 0.5, 1, 0.3]])) == 0.5
    assert metrics.rms(np.array([3.0, -3.0])) == 3.0


def test_firmware_pid_formula():
    counts, integ = firmware_rate_pid(np.array([10.0, 0, 0]), np.zeros(3), np.zeros(3), dt=0.002)
    assert counts[0] == pytest.approx(250 * 10 + 500 * (10 * 0.002))
    assert integ[0] == pytest.approx(0.02)


def test_b1_step_response_meets_targets():
    cfg = load_fc_config()
    log = rollout.rollout(RatePID.tuned(), setpoints.step(300, 0.002, 0, 3.0))
    m = rollout.step_response_metrics(log, 0)
    assert m["rise"] < cfg["b1_targets"]["rise_time_max"]
    assert m["overshoot"] < cfg["b1_targets"]["overshoot_max"]


def test_bridge_hover_and_tilt_direction():
    b = AccelBridge()
    q = np.array([1.0, 0, 0, 0])
    sp = b(q, np.zeros(3))
    assert sp.collective == pytest.approx(PH_C_HOVER, rel=1e-2)
    assert np.allclose(sp.omega_cmd, 0, atol=1e-9)
    # +x acceleration -> thrust tilts toward +x -> nose-down pitch about +y (positive)
    sp = b(q, np.array([3.0, 0, 0]))
    assert sp.omega_cmd[1] > 0 and abs(sp.omega_cmd[0]) < 1e-9
    sp = b(q, np.array([0, 3.0, 0]))
    assert sp.omega_cmd[0] < 0


def test_rotor_drag_dissipative_and_linear():
    import numpy as np
    from interceptor.sim import aero
    rng = np.random.default_rng(0)
    for _ in range(20):
        v = rng.normal(size=3) * 5
        f, t = aero.rotor_drag_force_torque(v, mass=0.04, drag_xyz=np.array([0.5, 0.5, 0.7]))
        assert v @ f <= 0 and not t.any()
        f2, _ = aero.rotor_drag_force_torque(2 * v, mass=0.04, drag_xyz=np.array([0.5, 0.5, 0.7]))
        assert np.allclose(f2, 2 * f)
