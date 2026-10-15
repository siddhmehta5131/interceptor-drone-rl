"""Local diagnostic: step a stage env with a trained checkpoint, dump obs/action/tilt.

Usage: python diag_local.py [stage_number] [steps]
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "interceptor-training"))
sys.path.insert(0, str(ROOT / "stage-runtime"))

import numpy as np

from src.envs.base_env import InterceptorBaseEnv
from src.physics.constants import PH_DT, PH_C_HOVER
from stage_runtime.stage_config import load_stage_config
from stage_runtime.controllers import build_controller, resolve_observation


def main():
    stage = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    cfg_path = ROOT / "stage-runtime" / "configs" / "stage" / f"stage_{stage:02d}.json"
    cfg = load_stage_config(cfg_path)
    from dataclasses import replace
    from stage_runtime.stage_config import ModelRef
    ckpt_file = ROOT / "stage-runtime" / "models" / f"PPO_stage_{stage}_final.zip"
    cfg = replace(cfg, control="model", model=ModelRef(path=ckpt_file, algo="PPO"))
    model_path = cfg.model.resolve(extra_roots=[ROOT / "stage-runtime" / "models"])
    cfg, ckpt = resolve_observation(cfg, model_path)
    print("obs kwargs:", cfg.observation.env_kwargs())
    print("ckpt:", ckpt)
    ctl = build_controller(cfg, model_path=model_path,
                           data_roots=[ROOT / "stage-runtime" / "models"])
    env = InterceptorBaseEnv(cfg.stage_config(), seed=cfg.seed, **cfg.env_kwargs())
    obs, _ = env.reset(seed=cfg.seed)
    tel = env.telemetry()
    print("step 0 obs:", np.array2string(np.asarray(obs, dtype=np.float64), precision=4))
    for i in range(1, steps + 1):
        a = ctl.act(tel, float(PH_DT), obs)
        obs, r, term, trunc, info = env.step(a)
        tel = env.telemetry()
        tilt = np.degrees(tel["tilt"])
        print(f"i={i:3d} a={np.array2string(a, precision=3)} "
              f"tilt={tilt:6.2f} cmd={np.array2string(np.asarray(tel['cmd']), precision=3)} "
              f"om={np.array2string(np.asarray(tel['Omega']), precision=2)} "
              f"v={np.array2string(np.asarray(tel['v_WB']), precision=2)} "
              f"z={tel['p_WB'][2]:.3f} r={r:+.3f} term={term}{trunc} "
              f"reason={info.get('termination_reason', '')}")
        print(f"      obs={np.array2string(np.asarray(obs, dtype=np.float64), precision=3)}")
        if term or trunc:
            print("TERMINATED:", info.get("termination_reason"),
                  "max_tilt", np.degrees(tel["tilt"]))
            break
    env.close()


if __name__ == "__main__":
    main()
