"""make_fixtures.py -- record golden behaviour of the three original codebases.

Plan tasks P1.1 - P1.3.  The fixtures written here are the oracle that the
consolidated ``interceptor`` package is tested against: ported code must
reproduce these arrays (see ``tests/unit/test_fixture_parity_*.py``).

Three sources are recorded, each from its *original, unmodified* code:

1. ``sim_*.npz``       old Crazyflie physics pipeline (z-up frame)
                       <old_repo>/interceptor-training/src/physics
2. ``maneuver_*.npz``  archive/path_planning_updated_project/evasive_maneuvers.py
                       (NED frame)
3. ``nbenv_*.npz``     v2 Colab notebook environment, zero RL action (NED frame)
                       archive/.../unified_noisy_apn_rl_interceptor_v2.ipynb

Run once, from the repository root::

    python scripts/make_fixtures.py --old-repo <path to rl-drone-flight-simulator>

The fixtures are committed; this script only needs re-running if a new
scenario is added.  It never needs the consolidated package.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = REPO / "archive" / "path_planning_updated_project"
OUT = REPO / "tests" / "fixtures"

DEFAULT_OLD_REPO = Path(
    r"C:\Users\ADMIN\Desktop\projects\github_repos\rl-drone-flight-simulator")


# ---------------------------------------------------------------------------
# 1. Old physics pipeline
# ---------------------------------------------------------------------------

SIM_STEPS = 300          # 3 s at the original 100 Hz step


def _sim_scenarios(c_hover: float) -> dict:
    """Motor-command (open loop) and action (closed loop) sequences."""
    n = SIM_STEPS
    t = np.arange(n) * 0.01
    hover = np.full((n, 4), c_hover)

    climb = hover.copy()
    climb[50:] = 0.40

    def diff(pattern, amp=0.02, start=50, stop=110):
        cmd = hover.copy()
        cmd[start:stop] += amp * np.asarray(pattern, dtype=float)
        cmd[stop:2 * stop - start] -= amp * np.asarray(pattern, dtype=float)
        return np.clip(cmd, 0.0, 1.0)

    # Motor layout (constants.PH_R_P): M1(+x,+y) M2(-x,+y) M3(-x,-y) M4(+x,-y)
    roll = diff([+1, +1, -1, -1])
    pitch = diff([-1, +1, +1, -1])
    yaw = diff([+1, -1, +1, -1], amp=0.05)

    rng = np.random.default_rng(20261112)
    mixed = hover + 0.03 * np.cumsum(rng.normal(0, 0.15, size=(n, 4)), axis=0)
    mixed = np.clip(mixed, 0.05, 0.95)

    open_loop = {"hover": hover, "climb": climb, "roll_step": roll,
                 "pitch_step": pitch, "yaw_step": yaw, "mixed": mixed}

    # Closed-loop: action = [thrust, roll, pitch, yaw] in [-1, 1].
    act_rate = np.zeros((n, 4))
    act_rate[:, 0] = c_hover
    act_rate[40:140, 1] = 0.5
    act_rate[140:220, 2] = -0.4
    act_rate[220:, 3] = 0.6
    act_mixed = np.zeros((n, 4))
    act_mixed[:, 0] = np.clip(c_hover + 0.08 * np.sin(2 * np.pi * 0.7 * t), 0, 1)
    act_mixed[:, 1] = 0.6 * np.sin(2 * np.pi * 1.1 * t)
    act_mixed[:, 2] = 0.5 * np.sin(2 * np.pi * 0.9 * t + 0.4)
    act_mixed[:, 3] = 0.3 * np.sin(2 * np.pi * 0.5 * t + 1.0)
    closed_loop = {"pid_rate_steps": act_rate, "pid_mixed": act_mixed}
    return {"open": open_loop, "closed": closed_loop}


def make_sim_fixtures(old_repo: Path) -> int:
    src = old_repo / "interceptor-training" / "src"
    if not (src / "physics" / "pipeline.py").exists():
        raise FileNotFoundError(f"old physics not found under {src}")
    sys.path.insert(0, str(src))
    try:
        pipeline = importlib.import_module("physics.pipeline")
        constants = importlib.import_module("physics.constants")
        aero = importlib.import_module("physics.aero")
    finally:
        sys.path.pop(0)

    c_hover = float(constants.PH_C_HOVER)
    om_hover = float(constants.PH_OMEGA_HOVER)
    scen = _sim_scenarios(c_hover)
    count = 0

    def init_state():
        return (np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0, 0.0]),
                np.zeros(3), np.zeros(3), np.full(4, om_hover))

    def save(name, **arrays):
        nonlocal count
        np.savez_compressed(OUT / f"sim_{name}.npz", **arrays)
        count += 1

    # --- open loop: motor commands straight into rk4 ----------------------
    for name, cmds in scen["open"].items():
        params = pipeline.PhysicsParams()
        p, q, v, om, Om = init_state()
        hist = {k: [] for k in "p q v omega Omega".split()}
        for cmd in cmds:
            p, q, v, om, Om = pipeline.rk4(p, q, v, om, Om, cmd, 4.2, params)
            for k, val in zip(hist, (p, q, v, om, Om)):
                hist[k].append(np.array(val, dtype=np.float64))
        save(name, kind="open_loop", dt=0.01, cmd=cmds,
             **{k: np.array(vv) for k, vv in hist.items()})

    # --- open loop with constant wind and ground effect ---------------------
    cmds = scen["open"]["mixed"]
    wind = np.array([2.0, -1.0, 0.5])
    params = pipeline.PhysicsParams()
    p, q, v, om, Om = init_state()
    p = np.array([0.0, 0.0, 0.08])            # inside ground-effect height
    hist = {k: [] for k in "p q v omega Omega".split()}
    ge_hist = []
    for cmd in cmds:
        ge = aero.ground_effect_gain(p[2])
        ge_hist.append(ge)
        p, q, v, om, Om = pipeline.rk4(p, q, v, om, Om, cmd, 4.2, params,
                                       wind_W=wind, ge_gain=ge)
        for k, val in zip(hist, (p, q, v, om, Om)):
            hist[k].append(np.array(val, dtype=np.float64))
    save("mixed_wind_ge", kind="open_loop_wind_ge", dt=0.01, cmd=cmds,
         wind_W=wind, p0=np.array([0.0, 0.0, 0.08]), ge_gain=np.array(ge_hist),
         **{k: np.array(vv) for k, vv in hist.items()})

    # --- closed loop: action -> rate PID -> mixer -> rk4 --------------------
    for name, actions in scen["closed"].items():
        params = pipeline.PhysicsParams()
        p, q, v, om, Om = init_state()
        om_prev2 = np.zeros(3)
        I = np.zeros(3)
        hist = {k: [] for k in "p q v omega Omega".split()}
        cmd_hist = []
        for a in actions:
            c_cmd, om_cmd, cut = pipeline.command_from_action(a)
            u, I = pipeline.pid(om_cmd, om, om_prev2, I, cut, 0.01)
            cmd = pipeline.mixer(u, c_cmd)
            if cut:
                cmd = np.zeros(4)
            cmd_hist.append(cmd.copy())
            p_n, q_n, v_n, om_n, Om_n = pipeline.rk4(p, q, v, om, Om, cmd, 4.2, params)
            om_prev2 = om.copy()
            p, q, v, om, Om = p_n, q_n, v_n, om_n, Om_n
            for k, val in zip(hist, (p, q, v, om, Om)):
                hist[k].append(np.array(val, dtype=np.float64))
        save(name, kind="closed_loop", dt=0.01, action=actions,
             cmd=np.array(cmd_hist),
             **{k: np.array(vv) for k, vv in hist.items()})

    # --- domain-randomisation sampler (RNG draw order) ----------------------
    rng = np.random.default_rng(7)
    rows = []
    for _ in range(5):
        pp = pipeline.sample_physics_params(rng)
        rows.append(np.concatenate([
            [pp.mass], np.diag(pp.J), [pp.c_l, pp.c_d],
            pp.fx_raw, pp.fy_raw, pp.fz_raw, pp.tx_raw, pp.ty_raw, pp.tz_raw]))
    save("dr_samples", kind="domain_randomisation", seed=7, samples=np.array(rows))

    # --- scalar constants the port must reproduce ---------------------------
    save("constants", kind="constants",
         c_hover=c_hover, omega_hover=om_hover,
         mass=float(constants.PH_M), J=np.asarray(constants.PH_J),
         c_l=float(constants.PH_C_L), c_d=float(constants.PH_C_D),
         k_mot=float(constants.PH_K_MOT), omega_max=float(constants.PH_OMEGA_MAX),
         bat=np.asarray(constants.PH_BAT), mix=np.asarray(constants.PH_MIX),
         r_p=np.asarray(constants.PH_R_P),
         kp=np.asarray(constants.PH_KP), ki=np.asarray(constants.PH_KI),
         kd=np.asarray(constants.PH_KD))
    return count


# ---------------------------------------------------------------------------
# 2. Evasive manoeuvre module (NED)
# ---------------------------------------------------------------------------

MAN_STEPS = 600          # 6 s at dt = 0.01
MAN_DT = 0.01


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Initial conditions per platform (NED): copied from
# evasive_maneuvers.simulate_engagement so every law's trigger is exercised.
_MAN_IC = {
    "quadcopter": dict(p_T0=[1200.0, 100.0, -150.0], v_T0=[-15.0, 5.0, 0.0],
                       p_M0=[0.0, 0.0, -150.0], v_M0=[180.0, 15.0, 0.0]),
    "fixed_wing": dict(p_T0=[4500.0, 500.0, -3000.0], v_T0=[-180.0, 20.0, 0.0],
                       p_M0=[0.0, 0.0, -3000.0], v_M0=[550.0, 60.0, 0.0]),
    "missile": dict(p_T0=[8000.0, 800.0, -5000.0], v_T0=[-600.0, 40.0, 0.0],
                    p_M0=[0.0, 0.0, -5000.0], v_M0=[900.0, 90.0, 0.0]),
}


def make_maneuver_fixtures() -> int:
    em = _load_module(ARCHIVE / "evasive_maneuvers.py", "_orig_evasive_maneuvers")
    count = 0
    for platform in em.PlatformType:
        ic = _MAN_IC[platform.value]
        for man in em.ManeuverType:
            tgt = em.EvasiveTargetGenerator(
                platform, np.array(ic["p_T0"]), np.array(ic["v_T0"]), man)
            p_M = np.array(ic["p_M0"], dtype=float)
            v_M = np.array(ic["v_M0"], dtype=float)   # straight, constant velocity
            P, V, A, PM, E, W = [], [], [], [], [], []
            t = 0.0
            for _ in range(MAN_STEPS):
                p, v, a, _kin = tgt.step(MAN_DT, t, p_M, v_M)
                P.append(p); V.append(v); A.append(a); PM.append(p_M.copy())
                E.append(tgt.euler.copy()); W.append(tgt.omega_B.copy())
                p_M = p_M + v_M * MAN_DT
                t += MAN_DT
            np.savez_compressed(
                OUT / f"maneuver_{platform.value}_{man.value}.npz",
                frame="NED", dt=MAN_DT, platform=platform.value, maneuver=man.value,
                p_T0=np.array(ic["p_T0"]), v_T0=np.array(ic["v_T0"]),
                p_M0=np.array(ic["p_M0"]), v_M=np.array(ic["v_M0"]),
                p=np.array(P), v=np.array(V), a=np.array(A), p_M=np.array(PM),
                euler=np.array(E), omega_B=np.array(W))
            count += 1
    return count


# ---------------------------------------------------------------------------
# 3. v2 notebook environment (NED), zero RL action
# ---------------------------------------------------------------------------

NB_SEEDS = (101, 202, 303)
NB_MODES = {            # name -> (use_apn, pure_apn_mode)
    "apn_raw": (True, True),
    "apn_filtered": (True, False),
}


def make_notebook_fixtures() -> int:
    nb = json.loads((ARCHIVE / "unified_noisy_apn_rl_interceptor_v2.ipynb")
                    .read_text(encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    # cells: 0 pip, 1 imports+limits, 2 noise+oracle, 3 kinematics+target,
    #        4 policy+obs, 5 env, 6 training (runs!), 7 eval, 8 plots
    # dataclasses resolve annotations through sys.modules[cls.__module__], so
    # the notebook code must live in a registered module, not a bare dict.
    import types
    module = types.ModuleType("_orig_notebook_v2")
    sys.modules[module.__name__] = module
    ns = module.__dict__
    for cell in code_cells[1:6]:
        exec(compile("".join(cell["source"]), "<notebook v2>", "exec"), ns)

    Env = ns["NoisyInterceptorPathPlanningEnv"]
    PlatformType = ns["PlatformType"]
    ManeuverType = ns["ManeuverType"]
    # A fixed, varied manoeuvre per seed so triggers are exercised.
    fixed = {101: ManeuverType.NONE, 202: ManeuverType.BARREL_ROLL,
             303: ManeuverType.TERMINAL_BREAK}
    count = 0
    for platform in PlatformType:
        for mode, (use_apn, pure) in NB_MODES.items():
            for seed in NB_SEEDS:
                env = Env(platform=platform, use_apn=use_apn, pure_apn_mode=pure,
                          noise_scale=1.0, fixed_maneuver=fixed[seed], seed=seed)
                obs, info0 = env.reset()
                obs_hist, rew, vout = [obs.copy()], [], []
                done = False
                while not done:
                    obs, r, term, trunc, info = env.step(np.zeros(3, dtype=np.float32))
                    obs_hist.append(obs.copy()); rew.append(r)
                    vout.append(np.asarray(info["velocity_output"], dtype=float))
                    done = term or trunc
                np.savez_compressed(
                    OUT / f"nbenv_{platform.value}_{mode}_s{seed}.npz",
                    frame="NED", platform=platform.value, mode=mode, seed=seed,
                    maneuver=fixed[seed].value, noise_scale=1.0,
                    traj_M=np.array(env.traj_M), traj_T=np.array(env.traj_T),
                    traj_T_noisy=np.array(env.traj_T_noisy),
                    obs=np.array(obs_hist, dtype=np.float32),
                    reward=np.array(rew), velocity_output=np.array(vout),
                    killed=bool(info["killed"]), crashed=bool(info["crashed"]),
                    miss_distance=float(info["miss_distance"]),
                    steps=int(info["steps"]), terminated=bool(term),
                    truncated=bool(trunc))
                count += 1
    return count


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--old-repo", type=Path, default=DEFAULT_OLD_REPO)
    ap.add_argument("--only", choices=["sim", "maneuver", "notebook"], default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.only in (None, "sim"):
        print(f"sim fixtures:      {make_sim_fixtures(args.old_repo)}")
    if args.only in (None, "maneuver"):
        print(f"maneuver fixtures: {make_maneuver_fixtures()}")
    if args.only in (None, "notebook"):
        print(f"notebook fixtures: {make_notebook_fixtures()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
