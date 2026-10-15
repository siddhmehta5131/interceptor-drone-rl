import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "interceptor-training"))

import numpy as np

from src.physics import pipeline as PL
from src.physics.constants import PH_C_HOVER

PL.PH_KD = np.array([0.001, 0.001, 0.0])

sys.path.insert(0, str(ROOT / "stage-runtime"))
from stage_runtime.stage_config import load_stage_config
from stage_runtime.runner import build_env

cfg = load_stage_config(ROOT / "stage-runtime" / "configs" / "stage" / "maneuver_demo.json")


def heading_of(tel):
    R = np.asarray(tel["R_WB"])
    return float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))


env = build_env(cfg, seed=7)
obs, _ = env.reset(seed=7)
tel = env.telemetry()
h0 = heading_of(tel)
for i in range(120):
    obs, r, term, trunc, info = env.step([float(PH_C_HOVER), 0, 0, 0.1])
    tel = env.telemetry()
print(f"yaw +0.1: heading {h0:.1f} -> {heading_of(tel):.1f} deg "
      f"(d={heading_of(tel)-h0:+.1f})  R[0,:]={np.round(np.asarray(tel['R_WB'])[0],3)}")

env = build_env(cfg, seed=7)
obs, _ = env.reset(seed=7)
tel = env.telemetry()
h0 = heading_of(tel)
for i in range(120):
    obs, r, term, trunc, info = env.step([float(PH_C_HOVER), 0, 0, -0.1])
    tel = env.telemetry()
print(f"yaw -0.1: heading {h0:.1f} -> {heading_of(tel):.1f} deg "
      f"(d={heading_of(tel)-h0:+.1f})")
