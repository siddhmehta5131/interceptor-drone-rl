"""Hover trim diagnostic: thrust vs weight at PH_C_HOVER."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0] / "interceptor-training"))

import numpy as np

from src.envs.base_env import InterceptorBaseEnv
from src.envs.stage_config import STAGES
from src.physics.constants import (PH_BAT, PH_C_HOVER, PH_C_L, PH_DT, PH_M,
                                   PH_OMEGA_HOVER)
from src.physics.pipeline import esc

print(f"PH_M={PH_M}  m*g={PH_M * 9.81:.6f} N")
print(f"PH_C_L={PH_C_L!r}  PH_C_HOVER={PH_C_HOVER!r}  PH_OMEGA_HOVER={PH_OMEGA_HOVER!r}")
oss, _ = esc(np.array([PH_C_HOVER] * 4), 4.2)
print("Omega_ss at hover cmd:", oss)
print("thrust = 4*c_l*Omega^2 =", 4 * PH_C_L * float(oss[0]) ** 2)

env = InterceptorBaseEnv(STAGES["stage_1"], seed=7)
env.reset(seed=7)
a = np.array([PH_C_HOVER, 0.0, 0.0, 0.0])
for i in range(300):
    env.step(a)
    if i % 50 == 0 or i == 299:
        tel = env.telemetry()
        print(f"  t={tel['episode_time_s']:5.2f} z={tel['p_WB'][2]:6.3f} "
              f"v_z={tel['v_WB'][2]:+6.3f} tilt={np.degrees(tel['tilt']):5.2f} "
              f"cmd={np.round(tel['cmd'], 3)} Omega={np.round(tel['Omega'], 1)}")

# what collective actually holds altitude?
for c in (PH_C_HOVER, 0.25, 0.26, 0.27, 0.30):
    env.reset(seed=7)
    for _ in range(400):
        env.step(np.array([c, 0.0, 0.0, 0.0]))
    tel = env.telemetry()
    oss, _ = esc(np.array([c] * 4), 4.2)
    print(f"c={c:.4f}: z={tel['p_WB'][2]:6.3f} v_z={tel['v_WB'][2]:+6.3f} "
          f"thrust={4 * PH_C_L * float(oss[0]) ** 2:.5f} N "
          f"(need {PH_M * 9.81:.5f})")