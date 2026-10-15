"""Ad-hoc sign probe: which way does a positive roll / pitch stick push the drone?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0] / "interceptor-training"))

import numpy as np

from src.envs.base_env import InterceptorBaseEnv
from src.envs.stage_config import STAGES
from src.physics.constants import PH_C_HOVER

env = InterceptorBaseEnv(STAGES["stage_1"], seed=7)
obs, _ = env.reset(seed=7)
print("obs dim", obs.shape)

for label, action in (
    ("roll +0.30", [PH_C_HOVER, 0.30, 0.0, 0.0]),
    ("roll -0.30", [PH_C_HOVER, -0.30, 0.0, 0.0]),
    ("pitch +0.30", [PH_C_HOVER, 0.0, 0.30, 0.0]),
    ("pitch -0.30", [PH_C_HOVER, 0.0, -0.30, 0.0]),
):
    env.reset(seed=7)
    a = np.array(action, dtype=np.float64)
    for _ in range(30):
        env.step(a)
    tel = env.telemetry()
    v = np.asarray(tel["v_WB"])
    vB = np.asarray(tel["R_WB"]).T @ v
    print(f"{label:>12}: v_W={np.round(v, 3)}  v_B={np.round(vB, 3)}  "
          f"tilt={np.degrees(tel['tilt']):.1f}  z={tel['p_WB'][2]:.2f}")

# hover trim check
env.reset(seed=7)
for _ in range(200):
    env.step(np.array([PH_C_HOVER, 0.0, 0.0, 0.0]))
tel = env.telemetry()
print("hover trim:", np.round(tel["p_WB"], 3), "v=", np.round(tel["v_WB"], 4),
      "tilt=", round(float(np.degrees(tel["tilt"])), 2),
      "cmd=", np.round(tel["cmd"], 3))