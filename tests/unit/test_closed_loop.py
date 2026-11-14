"""P8.5: closed loop runs guidance through the bridge and physics."""

from __future__ import annotations

import numpy as np

from interceptor.bridge.closed_loop import (
    CONTROL_DT, GUIDANCE_DT, PHYSICS_DT, run_closed_loop,
)
from interceptor.guidance.maneuvers import ManeuverType


def test_time_scales_are_nested():
    assert GUIDANCE_DT == 0.02
    assert CONTROL_DT == 0.002
    assert PHYSICS_DT == 0.001
    assert round(GUIDANCE_DT / CONTROL_DT) == 10
    assert round(CONTROL_DT / PHYSICS_DT) == 2


def test_one_episode_end_to_end():
    r = run_closed_loop(
        seed=1, fixed_maneuver=ManeuverType.CONSTANT_G, wind=False,
        max_time=0.4)
    assert r.steps == 20
    assert r.pos.shape == (21, 3)
    assert r.a_des.shape == (20, 3)
    assert r.target.shape == (21, 3)
    assert np.all(np.isfinite(r.pos))
    assert np.all(np.isfinite(r.a_des))
    assert np.isfinite(r.miss_distance)
    assert r.diagnostics["inner"] == "b1"
