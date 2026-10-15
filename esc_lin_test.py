import numpy as np
from src.utils import load_config, load_models
from src.envs.base_env import make_env_factory
import src.physics.pipeline as p
import src.envs.base_env as be
from src.physics.constants import PH_OMEGA_HOVER, PH_C_HOVER, PH_CUT_THR
m = load_models("/data/configs/model.yaml"); c = load_config("/data/configs/config.yaml")
b = c.bind_models(m); c = b if b is not None else c
f = make_env_factory(c.stage(1), history_frames=3, history_skip=2, target_alt=5.0,
                     future_samples=3, future_skip=2, predictor='const_vel',
                     future_source='true', privileged_fields=[], seed=1)
def setslope(S):
    def esc_new(cmd, U_bat):
        cc = np.clip(cmd, 0.0, 1.0)
        oss = np.maximum(PH_OMEGA_HOVER + S*(cc - PH_C_HOVER), 0.0)
        return np.where(cc < PH_CUT_THR, 0.0, oss), U_bat
    p.esc = esc_new; be.esc = esc_new
def fresh(kick, ep=0):
    env = f(); env.reset(seed=ep); u = env.unwrapped
    if kick:
        u._omega_B[:] = 0; u._omega_B_2[:] = 0; u._v_WB[:] = 0; u._q_WB[:] = [1,0,0,0]; u._omega_B[0] = kick
    return env
for S in (300, 600, 900, 1200, 1500, 1850):
    setslope(S); out = []
    for kick in (0.0, 0.05, 0.5):
        env = fresh(kick); mx = 0
        for t in range(1, 1001):
            o, r, te, tr, i = env.step(np.array([0.2325,0,0,0], dtype=np.float32))
            mx = max(mx, i["angvel_norm"])
            if te or tr: break
        out.append(f"kick{kick}: {t} steps, max w {mx:.2f}")
    env = fresh(0.0); u = env.unwrapped; w = []
    for t in range(6):
        env.step(np.array([0.2325,0.5,0,0], dtype=np.float32)); w.append(round(float(u._omega_B[0]),2))
    print(f"slope {S}: " + " | ".join(out) + f" | roll resp {w}", flush=True)
