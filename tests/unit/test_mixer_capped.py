"""Firmware thrust cap (crazyflie-firmware powerDistributionCap), BUGS #37."""
import numpy as np

from interceptor.sim import pipeline
from interceptor.sim.constants import PH_MIX


def test_equal_to_legacy_mixer_without_saturation():
    rng = np.random.default_rng(0)
    for _ in range(200):
        u = rng.uniform(-0.05, 0.05, 3)
        c = rng.uniform(0.2, 0.8)
        np.testing.assert_allclose(pipeline.mixer_capped(u, c), pipeline.mixer(u, c), atol=1e-12)


def test_excess_is_subtracted_from_all_motors():
    u, c = np.array([0.1, -0.05, 0.02]), 1.0
    raw = c + PH_MIX @ u
    out = pipeline.mixer_capped(u, c)
    assert np.isclose(out.max(), 1.0)
    np.testing.assert_allclose(out, np.clip(raw - (raw.max() - 1.0), 0.0, 1.0))


def test_torque_demand_survives_full_collective():
    """At full collective the legacy mixer returns equal commands (no torque);
    the firmware cap keeps the motor differences."""
    u = np.array([0.1, 0.0, 0.0])
    assert np.ptp(pipeline.mixer(u, 1.0)) < 1e-9
    capped = pipeline.mixer_capped(u, 1.0)
    np.testing.assert_allclose(capped - capped.mean(), (PH_MIX @ u) - (PH_MIX @ u).mean(), atol=1e-12)
