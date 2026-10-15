import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "interceptor-training"))

import numpy as np

from src.physics import pipeline as PL
from src.physics.constants import PH_C_HOVER, PH_W_SCALE

PL.PH_KD = np.array([0.001, 0.001, 0.0])  # provisional, see _man_probe

sys.path.insert(0, str(ROOT / "stage-runtime"))
from stage_runtime.stage_config import load_stage_config
from stage_runtime.runner import build_env

cfg = load_stage_config(ROOT / "stage-runtime" / "configs" / "stage" / "maneuver_demo.json")


def tilt_direction(label, action, steps=120):
    env = build_env(cfg, seed=7)
    obs, _ = env.reset(seed=7)
    tel = env.telemetry()
    a = np.array(action, dtype=float)
    R0 = None
    for i in range(steps):
        obs, r, term, trunc, info = env.step(a)
        tel = env.telemetry()
        if R0 is None:
            R0 = np.asarray(tel["R_WB"]).copy()
    R = np.asarray(tel["R_WB"])
    z_b_world = R @ np.array([0.0, 0.0, 1.0])   # body-z expressed in world
    x_b_world = R @ np.array([1.0, 0.0, 0.0])
    print(f"{label:22s} steps={i+1:4d} tilt={np.degrees(tel['tilt']):5.1f} "
          f"bodyz_in_world={np.round(z_b_world, 3)}  pos={np.round(tel['p_WB'], 2)} "
          f"vel={np.round(tel['v_WB'], 2)}")


hover = float(PH_C_HOVER)
tilt_direction("roll +0.1", [hover, +0.1, 0, 0])
tilt_direction("roll -0.1", [hover, -0.1, 0, 0])
tilt_direction("pitch +0.1", [hover, 0, +0.1, 0])
tilt_direction("pitch -0.1", [hover, 0, -0.1, 0])
tilt_direction("yaw +0.1", [hover, 0, 0, +0.1])
