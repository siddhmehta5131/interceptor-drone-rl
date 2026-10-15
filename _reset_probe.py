"""Reset-state probe: is the airframe actually at rest?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0] / "interceptor-training"))

import numpy as np

from src.envs.base_env import InterceptorBaseEnv
from src.envs.stage_config import STAGES
from src.physics.constants import PH_OMEGA_HOVER, PH_R_P

env = InterceptorBaseEnv(STAGES["stage_1"], seed=7)
obs, _ = env.reset(seed=7)
tel = env.telemetry()
print("p   =", np.round(tel["p_WB"], 4))
print("v   =", np.round(tel["v_WB"], 4))
print("w   =", np.round(tel["omega_B"], 4))
print("q   =", np.round(tel["q_WB"], 4))
print("Omega=", np.round(tel["Omega"], 3), " hover=", PH_OMEGA_HOVER)
print("cmd =", np.round(tel["cmd"], 4))
print("Ubat=", tel["U_bat"])
print("r_P =\n", np.round(PH_R_P, 4))
print("tilt deg =", np.degrees(tel["tilt"]))
# first-order differences
for i in range(6):
    env.step(np.array([tel["cmd"].mean(), 0.0, 0.0, 0.0]))
    t2 = env.telemetry()
    print(f"  step {i}: w={np.round(t2['omega_B'], 4)} "
          f"Omega={np.round(t2['Omega'], 1)} cmd={np.round(t2['cmd'], 4)} "
          f"v={np.round(t2['v_WB'], 3)}")