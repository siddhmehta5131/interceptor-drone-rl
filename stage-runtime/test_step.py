from pathlib import Path
from stage_runtime.stage_config import load_stage_config
from stage_runtime.runner import build_env
from stage_runtime.controllers import build_controller, resolve_observation

cfg = load_stage_config(Path('/app/stage-runtime/configs/stage/stage_01.json'))
cfg, ckpt = resolve_observation(cfg, Path('/app/stage-runtime/models/PPO_stage_1_final.zip'))
env = build_env(cfg, seed=1001)
obs, info = env.reset(seed=1001)
controller = build_controller(cfg, model_path=Path('/app/stage-runtime/models/PPO_stage_1_final.zip'))

import numpy as np
with np.printoptions(precision=4, suppress=True):
    for step in range(15):
        action = controller.act(env.telemetry(), 0.01, obs)
        obs, reward, term, trunc, info = env.step(action)
        print(f'Step {step}: action={action}')
        print(f'         tilt={info.get("tilt", 0):.4f}, term={term}, trunc={trunc}')
