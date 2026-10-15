"""Fine-grained hover trace: what excites the rate loop in the first second?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0] / "interceptor-training"))

import numpy as np

from src.envs.base_env import InterceptorBaseEnv
from src.envs.stage_config import STAGES
from src.physics.constants import PH_C_HOVER, PH_KI, PH_KP

print("Kp/Ki/Kd:", PH_KP, PH_KI)
env = InterceptorBaseEnv(STAGES["stage_1"], seed=7)
obs, _ = env.reset(seed=7)
print("reset tilt deg:", round(float(np.degrees(env.telemetry()["tilt"])), 3))
print("omega_reset:", np.round(env.telemetry()["omega_B"], 5))
a = np.array([PH_C_HOVER, 0.0, 0.0, 0.0])
for i in range(1, 121):
    env.step(a)
    if i % 5 == 0:
        tel = env.telemetry()
        print(f"t={tel['episode_time_s']:5.2f} z={tel['p_WB'][2]:6.3f} "
              f"v={np.round(tel['v_WB'], 3)} w={np.round(tel['omega_B'], 3)} "
              f"tilt={np.degrees(tel['tilt']):5.2f} cmd={np.round(tel['cmd'], 3)}")