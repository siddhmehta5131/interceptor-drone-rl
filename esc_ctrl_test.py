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
KP0, KI0, KD0 = p.PH_KP.copy(), p.PH_KI.copy(), p.PH_KD.copy()
def setup(S, g):
    def esc_new(cmd, U):
        cc = np.clip(cmd, 0, 1)
        o = np.maximum(PH_OMEGA_HOVER + S*(cc - PH_C_HOVER), 0)
        return np.where(cc < PH_CUT_THR, 0.0, o), U
    p.esc = esc_new; be.esc = esc_new
    p.PH_KP = KP0*g; p.PH_KI = KI0*g; p.PH_KD = KD0*g
def run(sr, sp, k, kz=0.05, kv=0.1, n=3):
    res = []
    for ep in range(n):
        env = f(); o, _ = env.reset(seed=ep); R = 0; t = 0
        while True:
            o = np.asarray(o).ravel()
            a = np.clip([0.2325 - kz*o[12] - kv*o[13], sr*k*o[5], sp*k*o[4], 0.0], -1, 1).astype(np.float32)
            o, r, te, tr, _ = env.step(a); R += r; t += 1
            if te or tr: break
        res.append((t, R))
    return np.mean([x[0] for x in res]), np.mean([x[1] for x in res])
for S, g in ((900, 1.0), (1850, 0.5)):
    setup(S, g)
    for sr in (1, -1):
        for sp in (1, -1):
            for k in (2.0, 5.0, 10.0):
                s, r = run(sr, sp, k)
                print(f"slope {S} gain x{g} sr={sr:+d} sp={sp:+d} k={k}: steps {s:.0f} reward {r:.1f}", flush=True)
