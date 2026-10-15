"""Guidance tuning harness: no rendering, prints a telemetry trace."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "interceptor-training"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "stage-runtime"))

import numpy as np

from src.envs.base_env import InterceptorBaseEnv
from src.envs.stage_config import STAGES
from src.physics.constants import PH_DT
from stage_runtime.instructions import GuidanceLaw, DEFAULT_GAINS
from stage_runtime.stage_config import load_stage_config

ap = argparse.ArgumentParser()
ap.add_argument("--stage", type=int, default=1)
ap.add_argument("--seconds", type=float, default=6.0)
ap.add_argument("--stick", type=float, default=0.08)
ap.add_argument("--every", type=int, default=25)
ap.add_argument("--kpos", type=float, default=0.55)
ap.add_argument("--kvel", type=float, default=2.2)
ap.add_argument("--krate", type=float, default=0.42)
ap.add_argument("--ktz", type=float, default=0.075)
ap.add_argument("--ktvz", type=float, default=0.085)
args = ap.parse_args()

cfg = load_stage_config(Path(__file__).resolve().parent / "stage-runtime"
                        / "configs" / "stage" / f"stage_{args.stage:02d}.json")
gains = dict(DEFAULT_GAINS)
gains.update(k_pos=args.kpos, k_vel=args.kvel, k_rate=args.krate,
             k_thrust_z=args.ktz, k_thrust_vz=args.ktvz)
law = GuidanceLaw(gains=gains, max_tilt_deg=cfg.instructions.max_tilt_deg,
                  max_speed=cfg.instructions.max_speed,
                  target=cfg.instructions.target,
                  lead_time_s=cfg.instructions.lead_time_s,
                  max_stick=args.stick)

env = InterceptorBaseEnv(cfg.stage_config(), seed=cfg.seed, **cfg.env_kwargs())
obs, _ = env.reset(seed=cfg.seed)
n = int(args.seconds / PH_DT)
tmin, killed = 1e9, False
for i in range(1, n + 1):
    tel = env.telemetry()
    action = law.action(tel)
    obs, reward, term, trunc, info = env.step(action)
    tel = env.telemetry()
    killed = killed or bool(info.get("killed", False))
    if tel.get("include_target"):
        tmin = min(tmin, tel["distance"])
    if i % args.every == 0 or term or trunc or i == 1:
        print(f"t={tel['episode_time_s']:5.2f} z={tel['p_WB'][2]:6.2f} "
              f"v={np.round(tel['v_WB'], 2)} w={np.round(tel['omega_B'], 2)} "
              f"tilt={np.degrees(tel['tilt']):5.1f} cmd={np.round(tel['cmd'], 2)} "
              f"act={np.round(action, 3)} d={tel['distance']:5.2f} "
              f"{'KILL' if killed else ''}{' TERM' if term or trunc else ''}")
    if term or trunc:
        break
print(f"min_dist={tmin if tmin < 1e9 else float('nan'):.2f} killed={killed}")