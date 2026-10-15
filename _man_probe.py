import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "interceptor-training"))

import numpy as np

import hover_env as he
from src.physics import constants as C
from src.physics.constants import PH_DT, PH_C_HOVER, PH_KD, PH_BAT

print("PH_KD   base/hover:", C.PH_KD, he.PH_KD, C.PH_KD is not None and
      np.array_equal(C.PH_KD, he.PH_KD))
print("PH_BAT  identical :", np.array_equal(C.PH_BAT, he.PH_BAT))
print("PH_C_HOVER        :", repr(float(C.PH_C_HOVER)), C.PH_C_HOVER == he.PH_C_HOVER)

sys.path.insert(0, str(ROOT / "stage-runtime"))
from stage_runtime.stage_config import load_stage_config
from stage_runtime.runner import build_env
from stage_runtime.instructions import build_instruction_controller

cfg = load_stage_config(ROOT / "stage-runtime" / "configs" / "stage" / "maneuver_demo.json")


def run_open(tag, action, steps, seed=7):
    env = build_env(cfg, seed=seed)
    obs, _ = env.reset(seed=seed)
    tel = env.telemetry()
    a = np.asarray(action, dtype=float)
    z0 = float(tel["p_WB"][2])
    for i in range(1, steps + 1):
        obs, r, term, trunc, info = env.step(a)
        tel = env.telemetry()
        if term or trunc:
            print(f"{tag:26s} ENDED step {i:4d} z={tel['p_WB'][2]:6.3f} "
                  f"tilt={np.degrees(tel['tilt']):5.1f}")
            return i
    print(f"{tag:26s} OK   {i:4d} z {z0:.2f}->{tel['p_WB'][2]:.2f} "
          f"tilt={np.degrees(tel['tilt']):5.1f} v={np.round(tel['v_WB'], 2)} "
          f"cmd={np.round(tel['cmd'], 3)}")
    return i


def run_rate(tag, roll, steps=300):
    env = build_env(cfg, seed=7)
    obs, _ = env.reset(seed=7)
    tel = env.telemetry()
    a = np.array([float(PH_C_HOVER), roll, 0.0, 0.0])
    ws = []
    for i in range(steps):
        obs, r, term, trunc, info = env.step(a)
        tel = env.telemetry()
        ws.append(float(tel["omega_B"][0]))
        if term or trunc:
            break
    tgt = roll * 2.0
    err = np.abs(np.array(ws) - tgt)
    print(f"{tag:26s} steps={len(ws):4d} w[:6]={np.round(ws[:6], 3)} "
          f"w[-3:]={np.round(ws[-3:], 3)} mean|err|={err.mean():.4f}")
    return len(ws)


def run_guidance(tag, steps=1500):
    env = build_env(cfg, seed=cfg.seed)
    obs, _ = env.reset(seed=cfg.seed)
    ctrl = build_instruction_controller(cfg)
    ctrl.reset()
    tel = env.telemetry()
    zs, tilts, xs, ys = [], [], [], []
    for i in range(1, steps + 1):
        a = ctrl.act(tel, float(PH_DT), obs)
        obs, r, term, trunc, info = env.step(a)
        tel = env.telemetry()
        zs.append(float(tel["p_WB"][2]))
        xs.append(float(tel["p_WB"][0]))
        ys.append(float(tel["p_WB"][1]))
        tilts.append(float(np.degrees(tel["tilt"])))
        if term or trunc:
            print(f"{tag:26s} ENDED step {i:4d} p=({xs[-1]:+.2f},{ys[-1]:+.2f},"
                  f"{zs[-1]:.2f}) max_tilt={max(tilts):.1f}")
            return i
    print(f"{tag:26s} OK   {steps} z {zs[0]:.2f}->{zs[-1]:.2f} "
          f"xy drift=({xs[-1]:+.3f},{ys[-1]:+.3f}) max_tilt={max(tilts):.1f}")
    return steps


run_rate("rate roll +0.10", +0.10)
run_rate("rate roll -0.10", -0.10)

# pitch rate uses action index 2; reuse run_open for it
env = build_env(cfg, seed=7)
obs, _ = env.reset(seed=7)
tel = env.telemetry()
a = np.array([float(PH_C_HOVER), 0.0, 0.10, 0.0])
wp = []
for i in range(300):
    obs, r, term, trunc, info = env.step(a)
    tel = env.telemetry()
    wp.append(float(tel["omega_B"][1]))
    if term or trunc:
        break
err = np.abs(np.array(wp) - 0.2)
print(f"{'rate pitch +0.10':26s} steps={len(wp):4d} w[:6]={np.round(wp[:6], 3)} "
      f"w[-3:]={np.round(wp[-3:], 3)} mean|err|={err.mean():.4f}")

run_open("hover c=PH_C_HOVER", [float(PH_C_HOVER), 0, 0, 0], 1500)
run_open("climb c=0.35", [0.35, 0, 0, 0], 300)
run_guidance("guidance hover")
print("ALL PROBES DONE")
