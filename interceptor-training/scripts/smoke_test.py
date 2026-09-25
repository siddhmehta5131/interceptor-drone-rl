#!/usr/bin/env python3
"""smoke_test.py -- no-SB3 verification suite (decision §6 / plan §12).

Runnable on a plain Python install (numpy, scipy, gymnasium, pyyaml only --
NO torch, NO stable_baselines3).  Verifies:

  1. Stage-1 physics/observation/reward parity vs ``hover_env.AltitudeHoldEnv``
  2. All 8 stage environments: obs dims, dynamics, info, episode_stats
  3. Observation history stacking (obs_builder)
  4. Target generator kinematics + containment
  5. Reward computer terms (per-stage weights, terminal bonuses)
  6. Curriculum scheduler: advance / cap / rollback / serialisation
  7. Checkpoint manager: path layout + generic save/load roundtrip
  8. Config loader validation + config hashing

Usage:  python scripts/smoke_test.py [--quick]
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
for p in (PROJECT_ROOT, REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# tiny harness
# ---------------------------------------------------------------------------
_TESTS = []


def test(fn):
    _TESTS.append(fn)
    return fn


def _run_all():
    passed, failed = [], []
    for fn in _TESTS:
        try:
            fn()
            passed.append(fn.__name__)
            print(f"PASS  {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed.append(fn.__name__)
            print(f"FAIL  {fn.__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{len(passed)} passed, {len(failed)} failed")
    return 0 if not failed else 1


# ---------------------------------------------------------------------------
# 1. Stage-1 parity vs hover_env
# ---------------------------------------------------------------------------
@test
def parity_stage1_vs_hover():
    from hover_env import AltitudeHoldEnv
    from src.envs import STAGES
    from src.envs.base_env import InterceptorBaseEnv

    seed = 12345
    a = AltitudeHoldEnv(seed=seed)
    b = InterceptorBaseEnv(STAGES["stage_1"], seed=seed)

    obs_a, info_a = a.reset()
    obs_b, info_b = b.reset()
    assert np.array_equal(obs_a, obs_b), "reset obs mismatch"
    assert info_a["target_alt"] == info_b["target_alt"] == 5.0

    rng = np.random.default_rng(777)
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated) and steps < 2000:
        steps += 1
        thrust = float(rng.normal(2 * 0.23253743635354834 - 1.0, 0.05))
        action = np.array([thrust] + list(rng.normal(0.0, 0.02, 3)), dtype=np.float32)
        obs_a, rew_a, term_a, trunc_a, info_a = a.step(action)
        obs_b, rew_b, term_b, trunc_b, info_b = b.step(action)
        assert np.array_equal(obs_a, obs_b), f"obs mismatch at step {steps}"
        assert rew_a == rew_b, f"reward mismatch at step {steps}: {rew_a} != {rew_b}"
        assert term_a == term_b and trunc_a == trunc_b, "term mismatch"
        for k in ("target_alt", "thrust_dev", "r_alive", "r_alt", "r_tilt",
                  "r_angvel", "r_thrust", "r_smooth", "crashed", "out_of_bounds"):
            assert info_a[k] == info_b[k], f"info[{k}] mismatch at step {steps}"
        terminated, truncated = term_a, trunc_a
    assert steps > 50, "parity episode ended too early"
    print(f"   hover parity: {steps} steps bit-identical (obs/reward/term/info)")


# ---------------------------------------------------------------------------
# 2. All 8 stage environments
# ---------------------------------------------------------------------------
@test
def all_stage_envs_run():
    from src.envs import STAGES
    from src.envs.base_env import InterceptorBaseEnv

    for sid, stage in STAGES.items():
        rng = np.random.default_rng(11 + int(sid.split("_")[1]))
        env = InterceptorBaseEnv(stage, history_frames=3, history_skip=2, seed=42)
        obs, info = env.reset()
        exp_dim = (76,) if stage.obs.include_target else (14,)
        assert obs.shape == exp_dim, f"{sid} reset obs dim {obs.shape} != {exp_dim}"
        assert env.observation_space.shape == obs.shape
        assert info.get("target_alt") is not None, f"{sid} reset must carry target_alt"
        total = 0.0
        done = False
        n = 0
        while not done and n < 200:
            n += 1
            action = env.action_space.sample() if hasattr(env.action_space, "sample") \
                else np.clip(rng.normal(0.0, 0.2, 4), -1, 1)
            obs, rew, term, trunc, info = env.step(np.asarray(action, dtype=np.float32))
            total += float(rew)
            done = bool(term or trunc)
            assert obs.shape == env.observation_space.shape
            if n == 1 and stage.obs.include_target:
                assert "distance" in info, f"{sid} step info must carry distance"
                assert "target_visible" in info, f"{sid} step info must carry target_visible"
        assert done, f"{sid} should terminate within 200 steps"
        stats = info["episode_stats"]
        for key in ("episode_reward", "reward_total", "reward_mean", "final_distance",
                    "kill", "steps", "mean_alt_err", "mean_tilt"):
            assert key in stats, f"{sid} missing episode_stats.{key}"
        assert stats["steps"] == n
        if stage.target_type not in ("none", "static_waypoint"):
            assert "miss_distance" in stats and "time_to_intercept" in stats
    print("   all 8 stage envs: obs dims, 200-step rollouts, episode_stats OK")


# ---------------------------------------------------------------------------
# 3. Observation builder history stacking
# ---------------------------------------------------------------------------
@test
def obs_history_stacking():
    from src.envs.obs_builder import ObsBuilder, TARGET_FRAME_DIM
    from src.envs.stage_config import ObsStageConfig

    m, s = 3, 2
    builder = ObsBuilder(ObsStageConfig(include_target=True, obs_noise_std=0.0),
                         history_frames=m, history_skip=s,
                         rng=np.random.default_rng(3))
    builder.reset()
    base = np.zeros(TARGET_FRAME_DIM, dtype=np.float64)
    base[-1] = 1.0  # visible flag
    obs = builder.update(base)
    assert obs.shape == (TARGET_FRAME_DIM * (m + 1),) == (76,)
    # current-first stacking: the single real frame is the NEWEST slot (k=0)
    newest = obs[0:TARGET_FRAME_DIM]
    assert newest[-1] == 1.0, "newest (k=0) slot must carry visible flag 1.0"
    for k in range(1, m + 1):
        assert obs[k * 19:(k + 1) * 19][-1] == 0.0, "zero-padded frames flag 0.0"
    # stacking must use t, t-s, t-2s, t-3s frames -> distinct frames
    builder2 = ObsBuilder(ObsStageConfig(include_target=True, obs_noise_std=0.0),
                          history_frames=2, history_skip=3,
                          rng=np.random.default_rng(4))
    builder2.reset()
    for i in range(10):
        fr = np.zeros(TARGET_FRAME_DIM, dtype=np.float64)
        fr[6] = float(i)  # mark velocity x
        fr[-1] = 1.0
        builder2.update(fr)
    stacked = builder2._stack() if hasattr(builder2, "_stack") else np.zeros(0)
    assert stacked.size, "stack must produce a non-empty obs"
    # deque(len 7) holds frames i=3..9; current-first idxs k=0 -> 9, k=1 -> 6, k=2 -> 3
    newest = stacked[0:19]
    oldest = stacked[19 * 2:19 * 3]
    assert newest[6] == 9.0 and oldest[6] == 3.0, "stride t,t-s,t-2s,.. with current first"

    # a NOT-visible frame must keep target fields exactly zero even with noise
    # (plan §5.4: a hidden target is unobservable, its slots must stay 0)
    builder3 = ObsBuilder(ObsStageConfig(include_target=True, obs_noise_std=5.0),
                          history_frames=1, history_skip=1,
                          rng=np.random.default_rng(7))
    builder3.reset()
    hidden = np.zeros(TARGET_FRAME_DIM, dtype=np.float64)  # visible flag stays 0
    o3 = builder3.update(hidden)
    assert o3[12:19].sum() == 0.0, "hidden-target slots must stay exactly zero"
    print("   obs history stacking: current-first, dims, stride, hidden-noise OK")


# ---------------------------------------------------------------------------
# 4. Target generator kinematics
# ---------------------------------------------------------------------------
@test
def target_generator_kinematics():
    from src.envs import STAGES
    from src.envs.target_generator import TargetGenerator

    for sid, stage in STAGES.items():
        tg = TargetGenerator(stage, rng=np.random.default_rng(5))
        pos = np.array([0.0, 0.0, 5.0])
        tg.reset(pos)
        assert tg.pos.shape == (3,)
        sd = tg.state_dict()
        if stage.target_type in ("order_2", "order_3", "evasive"):
            assert sd["acc"] is not None
        # cap speed at 30 m/s, containment at world_radius
        for _ in range(5000):
            tg.step(0.01, np.array([0.0, 0.0, 5.0]))
            if np.hypot(tg.pos[0], tg.pos[1]) > stage.world_radius:
                # containment should pull it back after a few steps
                for _ in range(200):
                    tg.step(0.01, np.array([0.0, 0.0, 5.0]))
                assert np.hypot(tg.pos[0], tg.pos[1]) <= stage.world_radius + 1.0, \
                    f"{sid} not contained"
                break

    # evasive targets must flee the drone: jerk points AWAY (plan §6.2)
    from copy import deepcopy
    s8 = deepcopy(STAGES["stage_8"])
    s8.spawn.evasive_probability = 1.0
    tg8 = TargetGenerator(s8, rng=np.random.default_rng(6))
    drone = np.array([0.0, 0.0, 5.0])
    tg8.reset(drone)
    aligned = 0.0
    for _ in range(200):
        tg8.step(0.01, drone)
        fl = tg8.pos - drone
        fl /= max(float(np.linalg.norm(fl)), 1e-9)
        aligned = float(np.dot(tg8._jerk, fl))
    # the last evasive resample (at most ~1 s ago) points along flee; with
    # j_max = 8 m/s^3 the dominant component keeps the dot well above 4
    assert aligned > 4.0, "evasive jerk must point away from the drone"
    print("   target generator: step/containment/flee-direction OK for all 8 stages")


# ---------------------------------------------------------------------------
# 5. Reward computer terms
# ---------------------------------------------------------------------------
@test
def reward_terms_sanity():
    from src.envs import STAGES
    from src.envs.reward import RewardComputer

    R = np.eye(3)
    v = np.zeros(3)
    om = np.zeros(3)
    p = np.zeros(3)

    # stage 3: kill bonus scaled by miss distance (plan §5.5)
    stage3 = STAGES["stage_3"]
    rc = RewardComputer(stage3.reward, kill_radius=stage3.intercept.kill_radius,
                        max_episode_steps=stage3.max_episode_steps)
    rew, term, comp, met = rc.compute(
        R_WB=R, v_WB=v, omega_B=om, p_WB=p,
        action=np.array([1.0, 0, 0, 0]), prev_action=np.array([1.0, 0, 0, 0]),
        c_cmd=1.0, alt_err=0.0, step_count=100, crashed_flip=False, crashed_ground=False,
        out_of_bounds=False, killed=True, distance=0.1, prev_distance=5.0,
        target_visible=True, los_world=np.array([1.0, 0, 0]))
    assert term is True
    assert abs(comp["r_kill_bonus"] - 500.0 * (1 - 0.1 / 0.5)) < 1e-9, \
        "kill bonus must scale by max(0, 1 - distance/kill_radius)"

    # stage 4: kill bonus additionally scaled by time-to-kill
    stage4 = STAGES["stage_4"]
    rc4 = RewardComputer(stage4.reward, kill_radius=stage4.intercept.kill_radius,
                         max_episode_steps=stage4.max_episode_steps)
    kw4 = dict(action=np.zeros(4), prev_action=np.zeros(4), c_cmd=0.5,
               alt_err=0.0, crashed_flip=False, crashed_ground=False,
               out_of_bounds=False, killed=True, distance=0.0, prev_distance=5.0,
               target_visible=True, los_world=np.array([1.0, 0, 0]))
    _, _, comp4, _ = rc4.compute(R_WB=R, v_WB=v, omega_B=om, p_WB=p,
                                 step_count=0, **kw4)
    assert abs(comp4["r_kill_bonus"] - 500.0) < 1e-9  # t=0, d=0 -> full bonus
    _, _, comp4b, _ = rc4.compute(R_WB=R, v_WB=v, omega_B=om, p_WB=p,
                                  step_count=1000, **kw4)
    assert abs(comp4b["r_kill_bonus"] - 500.0 * (1 - 1000 / 2000)) < 1e-9, \
        "kill bonus must scale by (1 - steps/max_episode_steps)"

    # stage 1: relative weights (alive/alt/tilt/thrust/smooth/angvel)
    stage1 = STAGES["stage_1"]
    rc1 = RewardComputer(stage1.reward, kill_radius=0.5, max_episode_steps=1000)
    rew, term, comp1, _ = rc1.compute(
        R_WB=R, v_WB=v, omega_B=om, p_WB=p,
        action=np.zeros(4), prev_action=np.zeros(4), c_cmd=0.5,
        alt_err=0.0, step_count=1, crashed_flip=False, crashed_ground=False,
        out_of_bounds=False, killed=False)
    assert not term
    assert comp1["r_alive"] == 0.10
    assert comp1["r_tilt"] == 0.0 and comp1["r_alt"] == 0.0, "zero tilt/alt_err -> zero terms"
    # dense weights apply per-unit: 10 deg tilt -> r_tilt = -0.40 * 0.1745 rad
    th = 0.3
    Rx = np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
    _, _, comp1b, _ = rc1.compute(
        R_WB=Rx, v_WB=v, omega_B=om, p_WB=p,
        action=np.zeros(4), prev_action=np.zeros(4), c_cmd=0.5,
        alt_err=0.0, step_count=1, crashed_flip=False, crashed_ground=False,
        out_of_bounds=False, killed=False)
    assert abs(comp1b["r_tilt"] - (-stage1.reward.k_tilt * th)) < 1e-9
    print("   reward terms: kill_bonus, time_bonus, stage1 dense weights OK")


# ---------------------------------------------------------------------------
# 6. Curriculum scheduler
# ---------------------------------------------------------------------------
@test
def curriculum_advance_cap_rollback():
    from src.envs import STAGES
    from src.training import ADVANCE, CAPPED, ROLLBACK, RUNNING, CurriculumScheduler

    s1 = STAGES["stage_1"]
    s = CurriculumScheduler(s1)
    for _ in range(100):
        r = s.on_episode_end({"episode_reward": 80.0, "steps": 1000})
    assert r == ADVANCE and s.advance_triggered
    # terminal result is sticky
    assert s.on_episode_end({"episode_reward": 0.0, "steps": 1000}) == ADVANCE

    s = CurriculumScheduler(s1)
    for _ in range(50):
        r = s.on_episode_end({"episode_reward": 5.0, "steps": 200000})
    assert r == CAPPED and s.capped

    s3 = STAGES["stage_3"]
    s = CurriculumScheduler(s3, rollback_min_steps=0)
    for _ in range(100):
        r = s.on_episode_end({"kill": 0.0, "steps": 1000, "final_distance": 3.0})
    assert r == ROLLBACK
    assert s.rollback_threshold == pytest_approx(0.5 * s3.success_rate)

    # rollback disabled when threshold 0
    s = CurriculumScheduler(s3, rollback_threshold=0.0)
    for _ in range(100):
        r = s.on_episode_end({"kill": 0.0, "steps": 1000})
    assert r == RUNNING

    # serialisation roundtrip
    d = s.state_dict()
    s2 = CurriculumScheduler(s3, rollback_threshold=0.0)
    s2.load_state_dict(d)
    assert s2.total_steps == s.total_steps and s2.rate() == s.rate()
    print("   curriculum: advance / cap / rollback / disable / roundtrip OK")


def pytest_approx(x, eps=1e-9):
    class _A:
        def __init__(self, v):
            self.v = v
        def __eq__(self, o):
            return abs(self.v - o) < eps
        def __hash__(self):
            return hash(self.v)
    return _A(x)


# ---------------------------------------------------------------------------
# 7. Checkpoint manager
# ---------------------------------------------------------------------------
@test
def checkpoint_manager_roundtrip():
    from src.training import CheckpointManager

    with tempfile.TemporaryDirectory() as tmp:
        cm = CheckpointManager(tmp, "ppo_baseline", "PPO")
        cm.ensure()
        assert str(cm.stage_dir("stage_2")).endswith(tmp.replace("\\", "/") + "/checkpoints/ppo_baseline/stage_2") or \
            "checkpoints" in str(cm.stage_dir("stage_2"))
        assert cm.monitor_dir("stage_3").name == "monitor"
        assert cm.config_final_path().parent.name == "results"
        # generic payload roundtrip (used for run_state)
        payload = {"run_id": "x", "episode_buffer": [1.0, 0.0] * 40, "retries": {"3": 1}}
        cm.save_run_state(payload)
        loaded = cm.load_run_state()
        assert loaded["episode_buffer"] == payload["episode_buffer"]
        assert loaded["retries"] == payload["retries"]
        # periodic discovery
        cm.stage_dir("stage_2").mkdir(parents=True, exist_ok=True)
        (cm.stage_dir("stage_2") / "PPO_100_steps.zip").touch()
        (cm.stage_dir("stage_2") / "PPO_50000_steps.zip").touch()
        (cm.stage_dir("stage_2") / "PPO_stage_2_final.zip").touch()
        latest = cm.latest_periodic("stage_2")
        assert latest is not None and latest.name == "PPO_50000_steps.zip"
        assert cm.latest_final("stage_2").name == "PPO_stage_2_final.zip"
    print("   checkpoint manager: layout, run_state roundtrip, discovery OK")


# ---------------------------------------------------------------------------
# 8. Config loader
# ---------------------------------------------------------------------------
@test
def config_loader_validation():
    from src.utils import config_hash, load_run_config
    from src.utils.config_loader import ConfigError

    cfg = load_run_config(os.path.join(PROJECT_ROOT, "configs", "default_run.yaml"))
    assert cfg.seed == 42 and cfg.n_parallel_envs == 8
    assert config_hash(cfg) == config_hash(cfg)
    assert cfg.stage(3).intercept.kill_radius == 0.5
    assert cfg.config_stages("ppo_baseline") == [1, 2, 3, 4, 5, 6, 7]
    assert cfg.config_obs_history("ppo_5layer_deep") == {"frames": 5, "skip": 3}

    bad = {"global": {"seed": 1, "device": "auto", "data_dir": "/x",
                      "n_parallel_envs": 8, "checkpoint_interval_steps": 50000,
                      "tensorboard": True, "execution_mode": "sequential"},
           "configs": {"x": {"algo": "PPO", "policy": "MlpPolicy",
                             "net_arch": {"pi": [1], "vf": [1]},
                             "activation": "ReLU", "obs_history": {"frames": 1, "skip": 1},
                             "stages": [1], "hyperparameters": {}}}}
    # invalid: missing data_dir
    bad2 = json.loads(json.dumps(bad))
    del bad2["global"]["data_dir"]
    try:
        from src.utils.config_loader import RunConfig
        RunConfig(bad2)
        raise AssertionError("missing data_dir should fail validation")
    except ConfigError:
        pass

    # plan-schema YAML (algorithm / policy_kwargs / obs_history) canonicalizes
    # to the SB3-shaped internal config (loader accepts both spellings)
    plan = {
        "global": cfg.global_cfg,
        "configs": {
            "ps": {
                "algorithm": "PPO", "policy": "MlpPolicy",
                "policy_kwargs": {"net_arch": [64, 64], "activation_fn": "tanh"},
                "obs_history": {"m": 3, "s": 2},
                "lr": 3e-4, "buffer_size": 100000, "batch_size": 64,
                "n_steps": 512, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.0,
                "vf_coef": 0.5, "max_grad_norm": 0.5, "clip_range": 0.2, "n_epochs": 10,
                "total_timesteps": 100000, "episode_interval": 500,
                "success_rate": 0.6, "stages": [1, 2, 3, 4, 5, 6, 7],
                "rollback": {"enabled": True},
            }
        },
    }
    pcfg = RunConfig(plan)
    assert pcfg.config("ps")["algo"] == "PPO"
    assert pcfg.config("ps")["net_arch"] == {"pi": [64, 64], "vf": [64, 64]}, \
        "shared net_arch list must expand to {pi, vf}"
    assert pcfg.config("ps")["activation"] == "Tanh"
    assert pcfg.config_obs_history("ps") == {"frames": 3, "skip": 2}
    print("   config loader: validation, hashing, plan-schema canonicalize OK")


if __name__ == "__main__":
    sys.exit(_run_all())