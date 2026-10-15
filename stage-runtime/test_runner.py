from pathlib import Path
from stage_runtime.stage_config import load_stage_config
from stage_runtime.runner import run_stage
from stage_runtime.controllers import resolve_observation

cfg = load_stage_config(Path('/app/stage-runtime/configs/stage/stage_01.json'))
cfg, ckpt = resolve_observation(cfg, Path('/app/stage-runtime/models/PPO_stage_1_final.zip'))

import numpy as np
with np.printoptions(precision=4, suppress=True):
    report = run_stage(cfg, out_dir=Path('/app/stage-runtime/scratch'), render=False, verbose=True)
    print("Episodes:", report["episodes"])
