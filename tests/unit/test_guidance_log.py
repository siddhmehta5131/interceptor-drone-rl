"""P5.12: every guidance step exports a_des, v_des and omega_des."""

from __future__ import annotations

import numpy as np

from interceptor.guidance.env import UnifiedInterceptEnv
from interceptor.guidance.maneuvers import PlatformType


def test_step_logs_desired_command_vectors():
    env = UnifiedInterceptEnv(platform=PlatformType.QUADCOPTER, mode="apn_filtered",
                              target_set="static", seed=4)
    env.reset(seed=4)
    for _ in range(12):
        _, _, term, trunc, info = env.step(np.zeros(3))
        for key in ("a_des", "v_des", "omega_des"):
            assert key in info
            assert np.asarray(info[key]).shape == (3,)
            assert np.all(np.isfinite(info[key]))
        if term or trunc:
            break
    log = env.get_trajectory_log()
    n = len(env.traj_a_des)
    assert n > 0
    for key in ("a_des", "v_des", "omega_des"):
        assert log[key].shape == (n, 3)
    # The log is a copy of info, not a view into the live state.
    assert np.allclose(log["a_des"][-1], env.last_info["a_des"])
