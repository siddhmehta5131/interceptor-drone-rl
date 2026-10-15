"""Decisive sign probe: stick -> body rate -> acceleration direction."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0] / "interceptor-training"))

import numpy as np

from src.envs.base_env import InterceptorBaseEnv
from src.envs.stage_config import STAGES
from src.physics.constants import PH_C_HOVER

for label, stick in (("roll +0.15", 0.15), ("roll -0.15", -0.15),
                     ("pitch +0.15", 0.15), ("pitch -0.15", -0.15),
                     ("yaw +0.15", 0.15), ("yaw -0.15", -0.15)):
    axis = label.split()[0]
    env = InterceptorBaseEnv(STAGES["stage_1"], seed=7)
    env.reset(seed=7)
    act = np.zeros(4)
    act[0] = PH_C_HOVER
    act[{"roll": 1, "pitch": 2, "yaw": 3}[axis]] = stick
    v0 = np.zeros(3)
    for _ in range(10):            # 0.1 s: small tilt, rate loop dominates
        env.step(act)
    tel = env.telemetry()
    R = np.asarray(tel["R_WB"])
    w = np.asarray(tel["omega_B"])
    vB = R.T @ (np.asarray(tel["v_WB"]) - v0)
    z_body = R[:, 2]               # thrust direction in world
    print(f"{label:>11}: omega={np.round(w, 3)}  a_B={np.round(vB / 0.1, 3)}  "
          f"tilt={np.degrees(tel['tilt']):5.2f}  thrust_dir={np.round(z_body, 3)}")