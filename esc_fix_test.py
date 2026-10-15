import numpy as np
from src.utils import load_config, load_models
from src.envs.base_env import make_env_factory
import src.physics.pipeline as p
from src.physics.constants import PH_OMEGA_HOVER, PH_C_HOVER, PH_CUT_THR
def esc_new(cmd, U_bat):
    c = np.clip(cmd, 0.0, 1.0)
    oss = PH_OMEGA_HOVER * np.sqrt(c / PH_C_HOVER)
    return np.where(c < PH_CUT_THR, 0.0, oss), U_bat
p.esc = esc_new
import src.envs.base_env as be; be.esc = esc_new
m = load_models("/data/configs/model.yaml"); c = load_config("/data/configs/config.yaml")
b = c.bind_models(m); c = b if b is not None else c
f = make_env_factory(c.stage(1), history_frames=3, history_skip=2, target_alt=5.0,
                     future_samples=3, future_skip=2, predictor='const_vel',
                     future_source='true', privileged_fields=[], seed=1)
def fresh(kick=0.0, ep=0):
    env = f(); env.reset(seed=ep); u = env.unwrapped
    if kick:
        u._omega_B[:] = 0; u._omega_B_2[:] = 0; u._v_WB[:] = 0; u._q_WB[:] = [1,0,0,0]; u._omega_B[0] = kick
    return env
for label, kick in [("as reset", 0.0), ("kick 0.05", 0.05), ("kick 0.5", 0.5)]:
    env = fresh(kick); mx = 0
    for t in range(1, 1001):
        o, r, te, tr, i = env.step(np.array([0.2325,0,0,0], dtype=np.float32))
        mx = max(mx, i["angvel_norm"])
        if te or tr: break
    print(f"{label}: steps {t} max angvel {mx:.3f} final angvel {i['angvel_norm']:.4f} alt_err {i['alt_err']:.3f} reward-last {r:.3f}")
# roll response sign
env = fresh(0.0); u = env.unwrapped
for t in range(1, 7):
    env.step(np.array([0.2325,0.5,0,0], dtype=np.float32)); print(t, "omega_x %.3f" % u._omega_B[0])
# thrust sweep: steps survived should now vary sensibly with thrust
for th in (0.1, 0.2, 0.2325, 0.3, 0.5):
    env = fresh(); 
    for t in range(1, 1001):
        o, r, te, tr, i = env.step(np.array([th,0,0,0], dtype=np.float32))
        if te or tr: break
    print("thrust", th, "steps", t)
