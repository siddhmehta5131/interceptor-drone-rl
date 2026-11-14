"""Integration tests (plan §16.2).

1. APN-only interception -- zero RL action intercepts a static target.
2. RL training smoke -- a short PPO run completes.
3. Visualiser -- a 10-frame MP4 is produced.
4. End-to-end -- short train -> evaluate -> render.

These are slower than the unit tests; the heaviest are marked ``slow``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from interceptor.envs.intercept_env import (
    APNConfig,
    DEFAULT_PRIVILEGED_FIELDS,
    InterceptEnv,
    InterceptorConfig,
)
from interceptor.utils.config_loader import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_env(stage: int = 1, seed: int = 0) -> InterceptEnv:
    cfg = load_config(str(REPO_ROOT / "configs" / "config.yaml"),
                      model_source=str(REPO_ROOT / "configs" / "model.yaml"))
    stage_cfg = cfg.stage(stage)
    ic = cfg.interceptor
    return InterceptEnv(
        stage_cfg,
        obs_config=stage_cfg.obs,
        apn_config=APNConfig(N_prime=stage_cfg.apn_N_prime,
                             t_go_min=stage_cfg.apn_t_go_min,
                             t_go_max=cfg.apn.t_go_max),
        interceptor_config=InterceptorConfig(
            v_max=ic.v_max, a_max=ic.a_max, omega_max=ic.omega_max,
            speed_ratio=ic.speed_ratio, dt=ic.dt,
        ),
    )


# ---------------------------------------------------------------- §16.2.1
def test_apn_only_intercepts_static_target():
    """With zero RL residual the pure APN law kills a static target."""
    kills = 0
    for seed in range(5):
        env = _make_env(1, seed)
        obs, info = env.reset(seed=seed)
        done = False
        while not done:
            obs, _r, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
            done = bool(terminated or truncated)
        if info.get("killed"):
            kills += 1
        env.close()
    assert kills >= 4, f"APN-only kills {kills}/5 (< 4)"


# ---------------------------------------------------------------- §16.2.2
def test_rl_training_smoke():
    """A short PPO run over the real env completes without error."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    def _env_fn():
        return _make_env(1, seed=1)

    vec_env = DummyVecEnv([_env_fn])
    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=256,
        batch_size=128,
        n_epochs=3,
        learning_rate=3e-4,
        verbose=0,
        device="cpu",
    )
    model.learn(total_timesteps=1024)
    obs = vec_env.reset()
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape == (1, 3)
    vec_env.close()


# ---------------------------------------------------------------- §16.2.3
def test_visualiser_renders_short_mp4(tmp_path):
    pytest.importorskip("pyvista")

    from interceptor.visualiser.animator import InterceptAnimator
    from interceptor.visualiser.cli import generate_trajectory
    from interceptor.visualiser.scene_builder import load_visualiser_config

    rollout = generate_trajectory(1, model_path=None, seed=0)
    trajectory = rollout["trajectory"][:10]
    assert len(trajectory) >= 2

    cfg = load_visualiser_config(str(REPO_ROOT / "configs" / "visualiser.yaml"))
    cfg["resolution"] = [320, 240]
    cfg["offscreen"] = True

    out = tmp_path / "smoke.mp4"
    animator = InterceptAnimator(
        model_name="apn",
        stage=1,
        trajectory_data=trajectory,
        config=cfg,
        kill_radius=rollout.get("kill_radius"),
    )
    animator.render(str(out), fps=5, quality=5)
    assert out.exists() and out.stat().st_size > 0


# ---------------------------------------------------------------- §16.2.4
@pytest.mark.slow
def test_end_to_end(tmp_path):
    """Short train -> evaluate -> (optionally) render."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from interceptor.training.orchestrator import evaluate_policy

    def _env_fn():
        return _make_env(1, seed=2)

    vec_env = DummyVecEnv([_env_fn])
    model = PPO("MlpPolicy", vec_env, n_steps=256, batch_size=128,
                n_epochs=3, learning_rate=3e-4, verbose=0, device="cpu")
    model.learn(total_timesteps=2048)

    model_path = tmp_path / "final.zip"
    model.save(str(model_path))
    assert model_path.exists()

    cfg = load_config(str(REPO_ROOT / "configs" / "config.yaml"),
                      model_source=str(REPO_ROOT / "configs" / "model.yaml"))
    stage_cfg = cfg.stage(1)
    loaded = PPO.load(str(model_path), device="cpu")
    metrics = evaluate_policy(loaded, stage_cfg, cfg,
                              list(DEFAULT_PRIVILEGED_FIELDS),
                              episodes=5, seed=0, deterministic=True)
    assert "kill_rate" in metrics
    assert 0.0 <= metrics["kill_rate"] <= 1.0
    vec_env.close()
