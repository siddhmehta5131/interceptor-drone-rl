#!/usr/bin/env python3
"""smoke_test.py -- no-SB3 verification suite (plan decision §6 / §12).

Runnable on a plain Python install (numpy, scipy, gymnasium, pyyaml only --
NO torch, NO stable_baselines3).  Verifies:

  1. Stage-1 physics/observation/reward parity vs ``hover_env.AltitudeHoldEnv``
  2. All 8 stage environments: obs dims, dynamics, info, episode_stats
  3. Observation layout: history stacking + future samples + privileged fields
  4. Target generator: polynomial path (stages 5-7), containment, evasive flee
  5. Predictors: const-velocity exactness and ridge regression sanity
  6. Reward computer terms (per-stage weights, kill bonus, facing)
  7. Curriculum scheduler: advance / cap / rollback / valid_predecessors
  8. Result objects + continuation eligibility (denied / skipped / eligible)
  9. Checkpoint manager: path layout, run_state and training-meta roundtrips
 10. Cross-stage weight transfer planning
 11. Config + model loader validation, hashing, stage overrides
 12. ``future_source: true`` vs ``pred`` equivalence classes

Usage:  python scripts/smoke_test.py
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

SMOKE_MODEL = os.path.join(PROJECT_ROOT, "configs", "smoke_model.yaml")
SMOKE_CONFIG = os.path.join(PROJECT_ROOT, "configs", "smoke_config.yaml")

# observation layout used by the stage-2+ tests (mirrors the smoke configs)
M, S, N, Q = 3, 2, 3, 5
PRIV_FIELDS = ("time_remaining", "facing_error", "target_true_pos", "target_true_vel")


def _obs_kwargs(**over):
    kw = dict(
        history_frames=M,
        history_skip=S,
        future_samples=N,
        future_skip=Q,
        predictor="const_vel",
        privileged_fields=PRIV_FIELDS,
    )
    kw.update(over)
    return kw


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


def _rnd_actions(rng, n):
    """Smooth-ish random actions (hover-ish thrust + small attitude)."""
    out = []
    for _ in range(n):
        out.append(np.array([
            float(rng.uniform(-0.25, 0.25)),
            float(rng.uniform(-0.05, 0.05)),
            float(rng.uniform(-0.05, 0.05)),
            float(rng.uniform(-0.05, 0.05)),
        ], dtype=np.float32))
    return out


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
    from src.envs.obs_builder import observation_dim

    for sid, stage in STAGES.items():
        rng = np.random.default_rng(11 + int(sid.split("_")[1]))
        # Stage 8 (evasive) has no closed-form path -> pred only (plan H-2)
        env = InterceptorBaseEnv(stage, seed=42, future_source="pred",
                                 **_obs_kwargs())
        obs, info = env.reset()
        exp = observation_dim(
            include_target=stage.obs.include_target,
            history_frames=0 if not stage.obs.include_target else M,
            future_samples=0 if not stage.obs.include_target else N,
            privileged_fields=() if not stage.obs.include_target else PRIV_FIELDS,
        )
        assert obs.shape == (exp,), f"{sid} reset obs dim {obs.shape} != ({exp},)"
        assert env.observation_space.shape == obs.shape
        assert obs.dtype == np.float32
        assert info.get("target_alt") is not None, f"{sid} reset must carry target_alt"
        n, done, total = 0, False, 0.0
        while not done and n < 200:
            n += 1
            obs, rew, term, trunc, info = env.step(rng.uniform(-0.3, 0.3, 4))
            total += float(rew)
            done = bool(term or trunc)
            assert obs.shape == env.observation_space.shape
            if n == 1 and stage.obs.include_target:
                assert "distance" in info, f"{sid} step info must carry distance"
                assert "target_visible" in info, f"{sid} needs target_visible"
                assert "facing_error" in info, f"{sid} needs facing_error"
                assert "desired_yaw" in info, f"{sid} needs desired_yaw"
        assert done, f"{sid} should terminate within 200 steps"
        stats = info["episode_stats"]
        for key in ("episode_reward", "reward_total", "reward_mean", "final_distance",
                    "kill", "steps", "mean_alt_err", "mean_tilt", "episode_length_s"):
            assert key in stats, f"{sid} missing episode_stats.{key}"
        assert stats["steps"] == n
        lo, hi = stage.episode_length_range
        assert lo - 1e-9 <= stats["episode_length_s"] <= hi + 1e-9, \
            f"{sid} episode_length_s {stats['episode_length_s']} outside {lo}..{hi}"
        if stage.target_type not in ("none", "static_waypoint"):
            assert "miss_distance" in stats and "time_to_intercept" in stats
    print(f"   all 8 stage envs: obs dims ({observation_dim(include_target=True, history_frames=M, future_samples=N, privileged_fields=PRIV_FIELDS)}"
          f" target / 14 hover), rollouts, episode_stats OK")


# ---------------------------------------------------------------------------
# 3. Observation layout: history + future + privileged
# ---------------------------------------------------------------------------
@test
def obs_history_and_future():
    from src.envs.obs_builder import (
        FUTURE_SAMPLE_DIM,
        ObsBuilder,
        TARGET_FRAME_DIM,
        make_future_block,
        make_privileged_block,
        observation_dim,
        privileged_dim,
    )
    from src.envs.stage_config import ObsStageConfig

    cfg = ObsStageConfig(include_target=True, obs_noise_std=0.0)
    b = ObsBuilder(cfg, history_frames=M, history_skip=S,
                   rng=np.random.default_rng(3), future_samples=N, future_skip=Q,
                   privileged_fields=PRIV_FIELDS)
    b.reset()
    frame = np.zeros(TARGET_FRAME_DIM, dtype=np.float64)
    frame[-1] = 1.0  # visible flag
    fut = make_future_block(np.arange(N * 3, dtype=np.float64).reshape(N, 3),
                            np.zeros((N, 3)), True)
    priv = make_privileged_block(PRIV_FIELDS, time_remaining=0.5, facing_error=0.25,
                                 target_true_pos_body=np.zeros(3),
                                 target_true_vel_body=np.zeros(3))
    obs = b.update(frame, fut, priv)

    hist_dim = TARGET_FRAME_DIM * (M + 1)
    fut_dim = FUTURE_SAMPLE_DIM * N
    pd = privileged_dim(PRIV_FIELDS)
    assert pd == 8, f"privileged dim must be 8, got {pd}"
    assert hist_dim + fut_dim + pd == observation_dim(
        include_target=True, history_frames=M, future_samples=N,
        privileged_fields=PRIV_FIELDS)
    assert obs.shape == (hist_dim + fut_dim + pd,), obs.shape

    # current-first stacking: the newest slot (k=0) carries the visible flag
    assert obs[TARGET_FRAME_DIM - 1] == 1.0, "newest slot must be the live frame"
    for k in range(1, M + 1):
        assert obs[k * TARGET_FRAME_DIM + TARGET_FRAME_DIM - 1] == 0.0, \
            "zero-padded history slots must stay zero"

    # future rows: 7 fields each, strictly future, visible flag column == 1.0
    for i in range(N):
        row = obs[hist_dim + i * 7: hist_dim + (i + 1) * 7]
        assert row[6] == 1.0, "future rows must carry the visible flag"
        assert row[0] == float(i * 3), "future rows keep caller order"

    # privileged fields are appended in the declared order and are never noised
    tail = obs[hist_dim + fut_dim:]
    assert tail[0] == 0.5 and tail[1] == 0.25, "privileged scalars must be verbatim"

    # stride check: history is t, t-s, t-2s, ... newest first
    b2 = ObsBuilder(cfg, history_frames=2, history_skip=3,
                    rng=np.random.default_rng(4), future_samples=0)
    b2.reset()
    for i in range(10):
        fr = np.zeros(TARGET_FRAME_DIM, dtype=np.float64)
        fr[6] = float(i)
        fr[-1] = 1.0
        b2.update(fr)
    stacked = b2._stack()
    assert stacked[6] == 9.0, "k=0 must be the newest frame"
    assert stacked[19 * 2 + 6] == 3.0, "k=2 must be the frame 2*skip steps back"

    # a hidden target keeps target fields (and the future block) exactly zero
    b3 = ObsBuilder(ObsStageConfig(include_target=True, obs_noise_std=5.0),
                    history_frames=1, history_skip=1, rng=np.random.default_rng(7),
                    future_samples=N, future_skip=Q, privileged_fields=PRIV_FIELDS)
    b3.reset()
    hidden = np.zeros(TARGET_FRAME_DIM, dtype=np.float64)  # visible flag stays 0
    o3 = b3.update(hidden)
    assert o3[12:19].sum() == 0.0, "hidden-target history slots must stay zero"
    assert o3[TARGET_FRAME_DIM + hist_dim - TARGET_FRAME_DIM:].sum() == 0.0, \
        "hidden-target future block must stay zero"
    print("   obs layout: history stride, future rows, privileged order, hidden-zero OK")


# ---------------------------------------------------------------------------
# 4. Target generator: polynomial path, containment, evasive flee
# ---------------------------------------------------------------------------
@test
def target_generator_kinematics():
    from copy import deepcopy

    from src.envs import STAGES
    from src.envs.target_generator import TargetGenerator

    # -- stages 5-7 follow a fixed, drone-independent polynomial path (D-2)
    for sid in ("stage_5", "stage_6", "stage_7"):
        stage = STAGES[sid]
        rng = np.random.default_rng(5)
        tg = TargetGenerator(stage, rng=rng)
        start_pos = np.zeros(3)
        tg.reset(start_pos, episode_length_s=float(stage.episode_length_range[1]))
        assert tg.polynomial and tg.path is not None, f"{sid} must use a path"
        path = tg.path
        assert path.position_at(0.0).shape == (3,)

        # the drone spawn determines where the path STARTS (D-2 samples the
        # spawn/end relative to the drone) but the flight itself is a fixed
        # closed form: moving the drone afterwards must not change the path
        ts = np.arange(0.0, 2.0, 0.01)
        ref = np.array([path.position_at(t) for t in ts]) - path.start
        tg2 = TargetGenerator(stage, rng=np.random.default_rng(5))
        tg2.reset(np.array([40.0, -40.0, 3.0]),
                  episode_length_s=float(stage.episode_length_range[1]))
        p2 = tg2.path
        other = np.array([p2.position_at(t) for t in ts]) - p2.start
        assert np.allclose(ref, other, atol=1e-9), \
            f"{sid} path shape must not depend on the drone"
        for _ in range(200):
            tg.step(0.01, np.array([999.0, -999.0, 1.0]))
            tg2.step(0.01, np.array([999.0, -999.0, 1.0]))
        assert np.allclose(tg.pos, tg2.pos), \
            f"{sid} trajectory must be closed-form (no drone steering)"

        # monotone in tau, bounded by the sampled speed cap, starts at spawn
        cap = max(stage.spawn.path_speed_cap)
        assert path.position_at(0.0).shape == (3,)
        assert np.allclose(path.position_at(0.0), tg.pos)
        prev = path.position_at(0.0)
        for t in np.arange(0.0, float(stage.episode_length_range[1]), 0.01):
            pos = path.position_at(t)
            step_len = float(np.linalg.norm(pos - prev)) / 0.01
            assert step_len <= cap + 1e-6, f"{sid} speed {step_len} exceeds cap {cap}"
            prev = pos
        # the analytic derivative agrees with finite differences
        t = 1.234
        fd = (path.position_at(t + 1e-5) - path.position_at(t - 1e-5)) / 2e-5
        assert np.allclose(fd, path.velocity_at(t), atol=1e-4), \
            f"{sid} velocity_at must match d/dt position_at"

    # -- reactive stages stay inside the world and evasive targets flee
    for sid in ("stage_2", "stage_3", "stage_4", "stage_8"):
        stage = STAGES[sid]
        tg = TargetGenerator(stage, rng=np.random.default_rng(5))
        tg.reset(np.array([0.0, 0.0, 5.0]))
        assert not tg.polynomial, f"{sid} must not use a polynomial path"
        sd = tg.state_dict()
        for key in ("pos", "vel", "acc", "jerk", "visible", "kills_enabled"):
            assert key in sd, f"{sid} state_dict missing {key}"
        for _ in range(3000):
            tg.step(0.01, np.array([0.0, 0.0, 5.0]))
            assert np.hypot(tg.pos[0], tg.pos[1]) <= stage.world_radius + 1e-6, \
                f"{sid} target escaped the world"

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
    assert aligned > 4.0, "evasive jerk must point away from the drone"
    print("   target generator: polynomial paths (5-7) fixed/monotone/capped, "
          "containment + flee (8) OK")


# ---------------------------------------------------------------------------
# 5. Predictors
# ---------------------------------------------------------------------------
@test
def prediction_suite():
    from src.prediction import get_predictor

    dt, n, q = 0.01, 4, 5
    p = np.array([1.0, 2.0, 3.0])
    v = np.array([0.4, -0.2, 0.1])

    cv = get_predictor("const_vel")
    pp, pv = cv.predict(p, v, future_samples=n, future_skip=q, dt=dt)
    assert pp.shape == (n + 1, 3) and pv.shape == (n + 1, 3), "predictor shape contract"
    assert np.allclose(pp[0], p), "row 0 must be the current position"
    assert np.allclose(pv[0], v), "row 0 must be the current velocity"
    for i in range(1, n + 1):
        assert np.allclose(pp[i], p + v * (i * q * dt)), "const_vel extrapolation"

    rid = get_predictor("linear_ridge")
    curve = 0.05
    hist = [p - v * (k * dt) + np.array([0.0, 0.0, -curve * (k * dt) ** 2])
            for k in range(8, 0, -1)]
    rp, rv = rid.predict(p, v, history=hist, future_samples=n, future_skip=q, dt=dt)
    assert np.allclose(rp[0], p) and np.allclose(rv[0], v), \
        "ridge row 0 must be the measurement"
    # a constant-curvature history must pull the forecast below the
    # constant-velocity line (constant acceleration is NOT observable from
    # position alone, so only the curvature component is recovered)
    t = q * dt
    assert rp[1][2] < p[2] + v[2] * t, "ridge must recover the history curvature"
    assert abs((rp[1][2] - (p[2] + v[2] * t)) - (-curve * t * t)) < 5e-3, \
        "ridge curvature term should match the generating polynomial"

    # too little history -> graceful constant-velocity fallback
    fpp, fpv = rid.predict(p, v, history=[p - v * dt], future_samples=2,
                           future_skip=1, dt=dt)
    assert np.allclose(fpp[0], p) and fpp.shape == (3, 3)

    try:
        get_predictor("nope")
        raise AssertionError("unknown predictor must raise")
    except ValueError:
        pass
    print("   predictors: const_vel exact, ridge curvature + fallback, registry OK")


# ---------------------------------------------------------------------------
# 6. Reward computer terms
# ---------------------------------------------------------------------------
@test
def reward_terms_sanity():
    from src.envs import STAGES
    from src.envs.reward import RewardComputer

    R = np.eye(3)
    v, om, p = np.zeros(3), np.zeros(3), np.zeros(3)

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

    stage4 = STAGES["stage_4"]
    rc4 = RewardComputer(stage4.reward, kill_radius=stage4.intercept.kill_radius,
                         max_episode_steps=stage4.max_episode_steps)
    kw4 = dict(action=np.zeros(4), prev_action=np.zeros(4), c_cmd=0.5,
               alt_err=0.0, crashed_flip=False, crashed_ground=False,
               out_of_bounds=False, killed=True, distance=0.0, prev_distance=5.0,
               target_visible=True, los_world=np.array([1.0, 0, 0]))
    _, _, comp4, _ = rc4.compute(R_WB=R, v_WB=v, omega_B=om, p_WB=p, step_count=0, **kw4)
    assert abs(comp4["r_kill_bonus"] - 500.0) < 1e-9
    _, _, comp4b, _ = rc4.compute(R_WB=R, v_WB=v, omega_B=om, p_WB=p,
                                 step_count=1000, **kw4)
    assert abs(comp4b["r_kill_bonus"] - 500.0 * (1 - 1000 / 2000)) < 1e-9

    # stage 1 dense weights (hover parity reference)
    stage1 = STAGES["stage_1"]
    rc1 = RewardComputer(stage1.reward, kill_radius=0.5, max_episode_steps=1000)
    _, term1, comp1, _ = rc1.compute(
        R_WB=R, v_WB=v, omega_B=om, p_WB=p,
        action=np.zeros(4), prev_action=np.zeros(4), c_cmd=0.5,
        alt_err=0.0, step_count=1, crashed_flip=False, crashed_ground=False,
        out_of_bounds=False, killed=False)
    assert not term1
    assert comp1["r_alive"] == 0.10
    assert comp1["r_tilt"] == 0.0 and comp1["r_alt"] == 0.0
    th = 0.3
    Rx = np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
    _, _, comp1b, _ = rc1.compute(
        R_WB=Rx, v_WB=v, omega_B=om, p_WB=p,
        action=np.zeros(4), prev_action=np.zeros(4), c_cmd=0.5,
        alt_err=0.0, step_count=1, crashed_flip=False, crashed_ground=False,
        out_of_bounds=False, killed=False)
    assert abs(comp1b["r_tilt"] - (-stage1.reward.k_tilt * th)) < 1e-9

    # facing term: only inside the cutoff, scaled by |facing_error|
    assert stage3.reward.k_facing == 0.15, "stage 3 must carry the facing weight"
    base = dict(action=np.zeros(4), prev_action=np.zeros(4), c_cmd=0.5,
                alt_err=0.0, step_count=10, crashed_flip=False, crashed_ground=False,
                out_of_bounds=False, killed=False, distance=0.3, prev_distance=0.5,
                target_visible=True, los_world=np.array([1.0, 0, 0]))
    _, _, c0, m0 = rc4.compute(R_WB=R, v_WB=v, omega_B=om, p_WB=p,
                               facing_error=0.0, **base)
    assert c0["r_facing"] == 0.0, "zero facing error -> zero penalty"
    assert m0["facing_error"] == 0.0
    _, _, c1, m1 = rc4.compute(R_WB=R, v_WB=v, omega_B=om, p_WB=p,
                               facing_error=0.5, **base)
    assert abs(c1["r_facing"] - (-0.15 * 0.5)) < 1e-9, "r_facing = -k * |error|"
    assert m1["facing_error"] == 0.5
    # outside the cutoff the term is inactive
    far = dict(base)
    far["distance"] = 5.0
    _, _, c2, _ = rc4.compute(R_WB=R, v_WB=v, omega_B=om, p_WB=p,
                              facing_error=0.5, **far)
    assert c2["r_facing"] == 0.0, "facing reward must be gated by FACING_CUTOFF_M"
    print("   reward terms: kill/time bonus, stage1 dense weights, r_facing gating OK")


# ---------------------------------------------------------------------------
# 7. Curriculum scheduler
# ---------------------------------------------------------------------------
@test
def curriculum_advance_cap_rollback():
    from src.envs import STAGES
    from src.training import ADVANCE, CAPPED, ROLLBACK, RUNNING, CurriculumScheduler
    from src.training.curriculum import valid_predecessors

    s1 = STAGES["stage_1"]
    s = CurriculumScheduler(s1)
    for _ in range(100):
        r = s.on_episode_end({"episode_reward": 80.0, "steps": 1000})
    assert r == ADVANCE and s.advance_triggered
    assert s.on_episode_end({"episode_reward": 0.0, "steps": 1000}) == ADVANCE, \
        "terminal result must be sticky"

    s = CurriculumScheduler(s1)
    for _ in range(50):
        r = s.on_episode_end({"episode_reward": 5.0, "steps": 200000})
    assert r == CAPPED and s.capped

    s3 = STAGES["stage_3"]
    s = CurriculumScheduler(s3, rollback_min_steps=0)
    for _ in range(100):
        r = s.on_episode_end({"kill": 0.0, "steps": 1000, "final_distance": 3.0})
    assert r == ROLLBACK
    assert abs(s.rollback_threshold - 0.5 * s3.success_rate) < 1e-12, \
        "default rollback threshold is 50% of the stage success rate"

    s = CurriculumScheduler(s3, rollback_threshold=0.0)
    for _ in range(100):
        r = s.on_episode_end({"kill": 0.0, "steps": 1000})
    assert r == RUNNING, "threshold 0 disables rollback"

    d = s.state_dict()
    s2 = CurriculumScheduler(s3, rollback_threshold=0.0)
    s2.load_state_dict(d)
    assert s2.total_steps == s.total_steps and s2.rate() == s.rate()

    assert valid_predecessors(1) == (1,)
    assert valid_predecessors(4) == (3, 4)
    assert valid_predecessors(8) == (7, 8)
    print("   curriculum: advance / cap / rollback / disable / roundtrip / "
          "valid_predecessors OK")


# ---------------------------------------------------------------------------
# 8. Result objects + continuation eligibility
# ---------------------------------------------------------------------------
@test
def results_and_eligibility():
    from src.results import (
        STATUS_CAPPED,
        STATUS_COMPLETED,
        STATUS_DENIED,
        STATUS_SKIPPED,
        ModelResult,
        StageOutcome,
        TrainResult,
        check_continuation_eligibility,
    )

    stages = {1: StageOutcome(1, "advance", 1000, 0.9)}
    good = ModelResult("ppo_baseline", algo="PPO", stages=stages,
                       status=STATUS_COMPLETED, final_path="nope.zip")
    capped = ModelResult("ppo_baseline", algo="PPO",
                         stages={1: StageOutcome(1, "capped", 10, 0.0)},
                         status=STATUS_CAPPED)
    empty = ModelResult("ppo_baseline", algo="PPO", stages={})

    # attribute + item access by model id / stage number (D-26)
    tr = TrainResult({"ppo_baseline": good}, name="exp", run_dir="r",
                     config_path="c.yaml", model_path="m.yaml", stages=[1],
                     config_hash="h", seed=42)
    assert tr.ppo_baseline is good and tr["ppo_baseline"] is good
    assert tr.ppo_baseline[1].result == "advance"
    assert tr.completed and not tr.capped_models
    assert "ppo_baseline" in tr and len(tr) == 1

    # round trips
    assert TrainResult.from_dict(tr.to_dict()).ppo_baseline.status == STATUS_COMPLETED
    with tempfile.TemporaryDirectory() as tmp:
        tr.save(tmp)
        assert (Path(tmp) / "run_summary.json").exists()
        rt = TrainResult.from_dict(json.loads(
            (Path(tmp) / "run_summary.json").read_text(encoding="utf-8")))
        assert rt.ppo_baseline.stages[1].success_rate == 0.9

    # eligibility (D-27, D-35, D-55, D-56)
    assert check_continuation_eligibility("m", good, [2]) is None, \
        "stage 1 -> 2 must be allowed"
    assert check_continuation_eligibility("m", good, [2, 3]) is None
    status, reason = check_continuation_eligibility("m", capped, [2])
    assert status == STATUS_DENIED and "capped" in reason
    status, reason = check_continuation_eligibility("m", good, [5])
    assert status == STATUS_SKIPPED and "stage" in reason
    status, reason = check_continuation_eligibility("m", empty, [2])
    assert status == STATUS_SKIPPED, "no completed stage cannot continue"
    print("   results: dict/attr access, round trips, denied/skipped/eligible OK")


# ---------------------------------------------------------------------------
# 9. Checkpoint manager
# ---------------------------------------------------------------------------
@test
def checkpoint_manager_roundtrip():
    from src.training import CheckpointManager

    with tempfile.TemporaryDirectory() as tmp:
        cm = CheckpointManager(tmp, "ppo_baseline", "PPO")
        cm.ensure()
        sd = str(cm.stage_dir("stage_2")).replace("\\", "/")
        assert sd.endswith("checkpoints/ppo_baseline/stage_2"), sd
        assert cm.monitor_dir("stage_3").name == "monitor"
        assert cm.config_final_path().parent.name == "results"
        assert str(cm.results_dir()).replace("\\", "/").endswith("results")
        assert str(cm.tb_dir("stage_2")).replace("\\", "/").endswith(
            "tb_logs/ppo_baseline/stage_2")

        payload = {"run_id": "x", "episode_buffer": [1.0, 0.0] * 40, "retries": {"3": 1}}
        cm.save_run_state(payload)
        loaded = cm.load_run_state()
        assert loaded["episode_buffer"] == payload["episode_buffer"]
        assert loaded["retries"] == payload["retries"]

        # C-3 training metadata sidecar
        fake_zip = cm.final_path("stage_2")
        fake_zip.parent.mkdir(parents=True, exist_ok=True)
        fake_zip.touch()
        meta = {"model_id": "ppo_baseline", "algo": "PPO", "hyperparameters": {},
                "net_arch": {"pi": [1], "vf": [1]}, "config_hash": "h"}
        cm.save_training_meta(fake_zip, meta)
        assert CheckpointManager.training_meta_path(fake_zip).name == \
            "PPO_stage_2_final.training_meta.json"
        assert cm.load_training_meta(fake_zip) == meta
        assert cm.load_training_meta(cm.final_path("stage_9")) is None

        cm.stage_dir("stage_2").mkdir(parents=True, exist_ok=True)
        (cm.stage_dir("stage_2") / "PPO_100_steps.zip").touch()
        (cm.stage_dir("stage_2") / "PPO_50000_steps.zip").touch()
        latest = cm.latest_periodic("stage_2")
        assert latest is not None and latest.name == "PPO_50000_steps.zip"
        assert cm.latest_final("stage_2").name == "PPO_stage_2_final.zip"
    print("   checkpoint manager: layout, run_state, training meta, discovery OK")


# ---------------------------------------------------------------------------
# 10. Cross-stage weight transfer
# ---------------------------------------------------------------------------
@test
def weight_transfer_suite():
    from src.training.weight_transfer import (
        SHARED_STAGE1_PREFIX,
        plan_transfer,
        stage1_prefix_for,
        transfer_state_dicts,
    )

    assert SHARED_STAGE1_PREFIX == 12, \
        "rot6 + v3 + omega3 are the only shared Stage-1 observation columns"
    assert stage1_prefix_for(14, 112) == 12
    assert stage1_prefix_for(14, 76) == 12
    assert stage1_prefix_for(112, 150) is None, "no prefix when the old obs already has history"
    assert stage1_prefix_for(150, 112) is None, "no shrinking transfers"

    old = {
        "mlp_extractor.policy_net.0.weight": np.ones((8, 14), dtype=np.float32),
        "mlp_extractor.policy_net.0.bias": np.zeros(8, dtype=np.float32),
        "mlp_extractor.value_net.0.weight": np.full((8, 14), 2.0, dtype=np.float32),
    }
    new = {
        "mlp_extractor.policy_net.0.weight": np.zeros((8, 112), dtype=np.float32),
        "mlp_extractor.policy_net.0.bias": np.zeros(8, dtype=np.float32),
        "mlp_extractor.value_net.0.weight": np.zeros((8, 112), dtype=np.float32),
    }
    plan = plan_transfer(old, new, shared_prefix=12, obs_width=112)
    first = plan["mlp_extractor.policy_net.0.weight"]
    assert first["action"] == "pad" and first["axis"] == "cols" and first["keep"] == 12, first

    report = transfer_state_dicts(old, new, shared_prefix=12, obs_width=112,
                                  scale=1e-3, seed=0)
    assert len(report["padded"]) == 2, report
    assert set(report["padded"]) == {
        "mlp_extractor.policy_net.0.weight",
        "mlp_extractor.value_net.0.weight",
    }
    assert "mlp_extractor.policy_net.0.bias" in report["copied"]
    w = new["mlp_extractor.policy_net.0.weight"]
    assert np.allclose(w[:, :12], 1.0), "shared Stage-1 columns must be copied"
    assert np.allclose(new["mlp_extractor.value_net.0.weight"][:, :12], 2.0)
    assert np.allclose(w[:, 12:], 0.0, atol=1e-3), "new columns get small random values"
    assert np.allclose(new["mlp_extractor.policy_net.0.bias"], 0.0), "biases are copied"
    print("   weight transfer: 12-column prefix copy, small-random tail, plan OK")


# ---------------------------------------------------------------------------
# 11. Config + model loader validation
# ---------------------------------------------------------------------------
@test
def config_and_model_loader_validation():
    from src.utils import load_config, load_models
    from src.utils.config_loader import ConfigError

    models = load_models(SMOKE_MODEL)
    assert set(models) == {"ppo_baseline", "ppo_5layer_deep"}, sorted(models)
    assert models["ppo_5layer_deep"].obs_history == {"frames": 5, "skip": 3}
    assert models["ppo_baseline"].has_privileged
    assert models["ppo_5layer_deep"].privileged_dim == 0
    assert models["ppo_baseline"].net_arch["pi"] == [256, 256, 128]

    cfg = load_config(SMOKE_CONFIG)
    assert cfg.seed == 42 and cfg.n_parallel_envs == 2
    assert cfg.on_capped == "continue"
    assert cfg.observation.future_source == "pred", \
        "smoke config must use predictor futures (stage 8 has no path)"
    assert cfg.observation.m == 3 and cfg.observation.n == 3
    assert cfg.stage_numbers() == [1, 2, 3, 4, 5, 6, 7, 8]
    assert cfg.stage(3).intercept.kill_radius == 0.5
    assert cfg.target_alt_range == (5.0, 5.0), "smoke keeps a fixed altitude"

    # binding the model file completes the config (obs history is per model)
    before = cfg.config_hash
    cfg.bind_models(models, model_source=SMOKE_MODEL)
    assert cfg.obs_history("ppo_5layer_deep") == {"frames": 5, "skip": 3}
    assert cfg.obs_history("ppo_baseline") == {"frames": 3, "skip": 2}
    assert len(cfg.config_hash) == 64
    assert cfg.config_hash != before, "binding models must change the hash (D-20)"
    again = cfg.config_hash
    cfg.recompute_hash()
    assert cfg.config_hash == again, "hashing must be stable"

    # stage overrides: scalar keys, sub-blocks and the spawn path aliases
    doc = {
        "global": {
            "seed": 1, "device": "cpu", "data_dir": "/tmp/x", "n_parallel_envs": 1,
            "checkpoint_interval_steps": 1000, "tensorboard": False,
            "execution_mode": "sequential",
        },
        "on_capped": "stop",
        "observation": {"history_frames": 2, "history_skip": 1, "future_samples": 2,
                        "future_skip": 4, "future_source": True, "predictor": "const_vel"},
        "target_alt": {"min": 3.0, "max": 9.0},
        "stages": {
            "stage_5": {
                "max_training_steps": 1234,
                "success_threshold": 0.42,
                "path_end_distance": [11.0, 22.0],
                "path_speed_cap": [3.0, 9.0],
                "path_shape": "linear",
                "reward": {"k_facing": 0.4},
            },
            "stage_2": {"episode_length_s": {"min": 5.0, "max": 7.0}},
        },
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "c.yaml"
        path.write_text(json.dumps(doc), encoding="utf-8")
        c2 = load_config(path)
    s5 = c2.stage(5)
    assert s5.max_training_steps == 1234
    assert s5.threshold == 0.42
    assert s5.spawn.path_end_distance == (11.0, 22.0), s5.spawn.path_end_distance
    assert s5.spawn.path_speed_cap == (3.0, 9.0), s5.spawn.path_speed_cap
    assert s5.spawn.path_shape == "linear"
    assert s5.reward.k_facing == 0.4
    assert c2.stage(2).episode_length_range == (5.0, 7.0)
    assert c2.on_capped == "stop"
    assert c2.target_alt_range == (3.0, 9.0)
    assert c2.observation.future_source is True

    bad = json.loads(json.dumps(doc))
    del bad["global"]["data_dir"]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad.yaml"
        path.write_text(json.dumps(bad), encoding="utf-8")
        try:
            load_config(path)
            raise AssertionError("missing data_dir must fail validation")
        except ConfigError:
            pass
    bad2 = json.loads(json.dumps(doc))
    bad2["on_capped"] = "pause"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad2.yaml"
        path.write_text(json.dumps(bad2), encoding="utf-8")
        try:
            load_config(path)
            raise AssertionError("invalid on_capped must fail validation")
        except ConfigError:
            pass

    # plan-schema model file canonicalises to the SB3 spelling
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "m.yaml"
        path.write_text(json.dumps({
            "models": {
                "ps": {
                    "algorithm": "PPO", "policy": "MlpPolicy",
                    "policy_kwargs": {"net_arch": [64, 64], "activation_fn": "tanh"},
                    "obs_history": {"m": 3, "s": 2},
                    "hyperparameters": {"n_steps": 512},
                }
            }
        }), encoding="utf-8")
        pm = load_models(path)["ps"]
    assert pm.algo == "PPO"
    assert pm.net_arch == {"pi": [64, 64], "vf": [64, 64]}, \
        "a shared net_arch list must expand to {pi, vf}"
    assert pm.activation == "Tanh"
    assert pm.obs_history == {"frames": 3, "skip": 2}
    print("   loaders: model + config validation, hashing, overrides, aliases, "
          "plan-schema canonicalisation OK")


# ---------------------------------------------------------------------------
# 12. future_source: true vs pred
# ---------------------------------------------------------------------------
@test
def future_source_true_vs_pred():
    from src.envs import STAGES
    from src.envs.base_env import InterceptorBaseEnv
    from src.physics.quaternion import quat2rot

    diffs = {}
    for sid in ("stage_5", "stage_6", "stage_7", "stage_8", "stage_3"):
        stage = STAGES[sid]
        envs = {}
        for src in ("true", "pred"):
            envs[src] = InterceptorBaseEnv(stage, seed=99, future_source=src,
                                           **_obs_kwargs())
            envs[src].reset()
        rng = np.random.default_rng(5)
        for action in _rnd_actions(rng, 60):
            envs["true"].step(action)
            envs["pred"].step(action)
        blocks = {}
        for src, env in envs.items():
            R = quat2rot(env._q_WB)
            blocks[src] = env._future_block(R, True)
        assert blocks["true"].shape == blocks["pred"].shape == (N, 7)
        diffs[sid] = float(np.max(np.abs(blocks["true"] - blocks["pred"])))

    assert diffs["stage_5"] < 1e-9, \
        "order_1 is a straight line: const-velocity extrapolation is exact"
    assert diffs["stage_6"] > 0.0 and diffs["stage_7"] > 0.0, \
        "curved paths must differ from the const-velocity extrapolation"
    assert diffs["stage_8"] == 0.0 and diffs["stage_3"] == 0.0, \
        "targets without a closed-form path must fall back to the predictor"
    print(f"   future_source true-vs-pred max |diff|: {diffs}")


if __name__ == "__main__":
    sys.exit(_run_all())
