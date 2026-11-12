"""orchestrator.py -- build environments, train PPO, run the curriculum.

The orchestrator owns the training loop for ONE model definition:

* builds a vectorised :class:`~src.envs.intercept_env.InterceptEnv` factory,
* constructs a PPO policy (asymmetric actor-critic when privileged fields are
  declared) sized to the concatenated actor + privileged observation,
* trains each curriculum stage until its kill-rate threshold is met or its
  budget is exhausted, saving per-stage checkpoints, and
* records a :class:`~src.results.ModelResult`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from ..envs.intercept_env import APNConfig, InterceptEnv, InterceptorConfig
from ..envs.obs_builder import PRIVILEGED_FIELDS
from ..envs.stage_config import StageConfig
from ..results import (
    ModelResult, StageOutcome, STATUS_CAPPED, STATUS_COMPLETE, STATUS_FAILED,
)
from ..utils.config_loader import TrainConfig
from ..utils.model_loader import ModelDef
from .asymmetric_policy import policy_kwargs_for
from .callbacks import InterceptMetricsCallback

__all__ = ["Orchestrator", "resolve_device", "activation_fn",
           "obs_dim_for", "evaluate_policy"]


def resolve_device(device: str) -> str:
    if device in (None, "", "auto"):
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return str(device)


def activation_fn(name: str):
    import torch.nn as nn
    return {
        "ReLU": nn.ReLU, "Tanh": nn.Tanh, "ELU": nn.ELU, "GELU": nn.GELU,
    }.get(name, nn.ReLU)


def obs_dim_for(obs_config, privileged_fields: Sequence[str]) -> int:
    """Concatenated actor + privileged observation width."""
    priv = sum(PRIVILEGED_FIELDS[f] for f in privileged_fields)
    return int(obs_config.actor_dim) + int(priv)


def _stage_number(stage_config: StageConfig) -> int:
    tail = stage_config.id.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else 0


class Orchestrator:
    """Trains one model across the curriculum."""

    def __init__(
        self,
        cfg: TrainConfig,
        model_def: ModelDef,
        run_dir: str,
        *,
        device: Optional[str] = None,
        verbose: int = 1,
    ):
        self.cfg = cfg
        self.model_def = model_def
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.device = resolve_device(device or cfg.device)
        self.verbose = int(verbose)
        self.privileged_fields = tuple(model_def.privileged_critic)

    # ------------------------------------------------------------- factories
    def _make_env_fn(self, stage_config: StageConfig, seed: int):
        cfg = self.cfg
        privileged = self.privileged_fields

        def _init():
            env = InterceptEnv(
                stage_config,
                obs_config=stage_config.obs,
                apn_config=APNConfig(cfg.apn.N_prime, cfg.apn.t_go_min,
                                     cfg.apn.t_go_max),
                interceptor_config=InterceptorConfig(
                    v_max=cfg.interceptor.v_max, a_max=cfg.interceptor.a_max,
                    omega_max=cfg.interceptor.omega_max,
                    speed_ratio=cfg.interceptor.speed_ratio,
                    dt=cfg.interceptor.dt),
                privileged_fields=privileged,
                seed=seed,
            )
            return Monitor(env)

        return _init

    def build_vec_env(self, stage_config: StageConfig, seed: int) -> VecEnv:
        n = max(1, int(self.cfg.n_parallel_envs))
        return DummyVecEnv([self._make_env_fn(stage_config, seed + i)
                            for i in range(n)])

    def _policy_kwargs(self, obs_dim: int):
        pk = {
            "net_arch": dict(self.model_def.net_arch),
            "activation_fn": activation_fn(self.model_def.activation_name()),
            "log_std_init": 0.0,
        }
        return policy_kwargs_for(self.model_def, obs_dim,
                                 policy_kwargs=pk,
                                 privileged_fields=self.privileged_fields)

    def build_model(self, env: VecEnv, seed: int) -> PPO:
        obs_dim = int(np.prod(env.observation_space.shape))
        policy_cls, policy_kwargs = self._policy_kwargs(obs_dim)
        tb_log = str(self.run_dir / "tb") if self.cfg.tensorboard else None
        hp = dict(self.model_def.hyperparameters)
        n_envs = env.num_envs
        if "n_steps" in hp and n_envs > 1:
            n_steps = int(hp["n_steps"])
            if n_steps % n_envs != 0:
                hp["n_steps"] = int(np.ceil(n_steps / n_envs)) * n_envs
        return PPO(
            policy_cls, env, device=self.device, seed=seed,
            policy_kwargs=policy_kwargs, tensorboard_log=tb_log,
            verbose=0, **hp,
        )

    # -------------------------------------------------------------- training
    def train_stage(
        self,
        model: PPO,
        stage_config: StageConfig,
        env: VecEnv,
        *,
        budget: int,
        seed: int,
    ) -> StageOutcome:
        n = _stage_number(stage_config)
        stage_dir = self.run_dir / self.model_def.model_id
        stage_dir.mkdir(parents=True, exist_ok=True)

        metrics_cb = InterceptMetricsCallback(
            stage_config, window=int(stage_config.window), max_steps=budget,
            log_tensorboard=self.cfg.tensorboard, verbose=self.verbose)
        ckpt_cb = CheckpointCallback(
            save_freq=max(int(self.cfg.checkpoint_interval_steps), env.num_envs),
            save_path=str(stage_dir), name_prefix=f"stage_{n}", verbose=0)

        model.learn(
            total_timesteps=int(budget),
            callback=CallbackList([metrics_cb, ckpt_cb]),
            reset_num_timesteps=False,
            tb_log_name=f"{self.model_def.model_id}",
            progress_bar=False,
        )

        m = metrics_cb.metrics()
        passed = bool(metrics_cb.advanced) or m.recent_kill_rate >= stage_config.threshold
        ckpt_path = stage_dir / f"stage_{n}_final"
        model.save(str(ckpt_path))

        outcome = StageOutcome(
            stage=n,
            status=STATUS_COMPLETE if passed else STATUS_CAPPED,
            steps_trained=int(m.steps),
            kill_rate=float(m.recent_kill_rate),
            threshold=float(stage_config.threshold),
            advanced=bool(passed),
            mean_final_distance=float(m.mean_final_distance),
            mean_episode_reward=float(m.mean_episode_reward),
            mean_episode_length=float(m.mean_episode_length),
            checkpoint_path=str(ckpt_path) + ".zip",
            metrics={"episodes": float(m.episodes)},
        )
        return outcome

    # ------------------------------------------------------------------- run
    def run(
        self,
        stages: Sequence[int],
        *,
        max_steps_per_stage: Optional[int] = None,
        save_final: bool = True,
    ) -> ModelResult:
        started = time.time()
        seed = self.cfg.seed
        priv_dim = sum(PRIVILEGED_FIELDS[f] for f in self.privileged_fields)
        result = ModelResult(
            model_id=self.model_def.model_id, run_dir=str(self.run_dir),
            obs_dim=obs_dim_for(self.cfg.observation, self.privileged_fields),
            actor_dim=int(self.cfg.observation.actor_dim),
            privileged_dim=int(priv_dim),
            algorithm=self.model_def.algo,
        )

        model: Optional[PPO] = None
        for stage_no in stages:
            stage_config = self.cfg.stage(stage_no)
            env = self.build_vec_env(stage_config, seed)
            if model is None:
                model = self.build_model(env, seed)
            else:
                model.set_env(env)

            budget = int(max_steps_per_stage or stage_config.max_training_steps)
            if self.verbose:
                print(f"[orchestrator] training {self.model_def.model_id} "
                      f"stage {stage_no} ({stage_config.target_type}) "
                      f"budget={budget}")

            outcome = self.train_stage(model, stage_config, env, budget=budget,
                                       seed=seed)
            result.stages.append(outcome)
            env.close()

            if outcome.passed:
                if self.verbose:
                    print(f"[orchestrator] stage {stage_no} PASSED "
                          f"(kill_rate={outcome.kill_rate:.3f})")
                continue

            if self.cfg.on_capped == "stop":
                result.status = STATUS_CAPPED
                if self.verbose:
                    print(f"[orchestrator] stage {stage_no} capped -> stopping "
                          f"(on_capped=stop)")
                break
            if self.verbose:
                print(f"[orchestrator] stage {stage_no} capped -> continuing "
                      f"(on_capped=continue)")

        result.duration_s = time.time() - started
        if model is not None:
            result.model = model
            if save_final:
                final = self.run_dir / self.model_def.model_id / "final"
                model.save(str(final))
                result.final_path = str(final) + ".zip"
        if result.status not in (STATUS_CAPPED,):
            result.status = STATUS_COMPLETE
        return result


def evaluate_policy(
    model,
    stage_config: StageConfig,
    cfg: TrainConfig,
    privileged_fields: Sequence[str],
    *,
    episodes: int = 50,
    seed: int = 123,
    deterministic: bool = True,
    collect_trajectory: bool = False,
) -> Dict[str, object]:
    """Deterministic (or stochastic) rollout metrics for one stage (plan §15)."""
    env = InterceptEnv(
        stage_config, obs_config=stage_config.obs,
        apn_config=APNConfig(cfg.apn.N_prime, cfg.apn.t_go_min, cfg.apn.t_go_max),
        interceptor_config=InterceptorConfig(
            v_max=cfg.interceptor.v_max, a_max=cfg.interceptor.a_max,
            omega_max=cfg.interceptor.omega_max,
            speed_ratio=cfg.interceptor.speed_ratio, dt=cfg.interceptor.dt),
        privileged_fields=tuple(privileged_fields), seed=seed)

    kills = 0
    times: List[float] = []
    misses: List[float] = []
    rewards: List[float] = []
    lengths: List[int] = []
    energies: List[float] = []
    rl_fractions: List[float] = []
    zem_terminal: List[float] = []
    efficiency: List[float] = []
    trajectories: List[dict] = []

    for ep in range(int(episodes)):
        obs, info = env.reset(seed=seed + ep)
        done = False
        total_r = 0.0
        energy = 0.0
        rl_l1 = 0.0
        cmd_l1 = 0.0
        path_len = 0.0
        prev_pos = env.state.position.copy()
        start_dist = float(np.linalg.norm(env.target.pos - env.state.position))
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, r, term, trunc, info = env.step(action)
            total_r += float(r)
            a_total = np.asarray(info["a_total"])
            a_rl = np.asarray(info["a_rl"])
            a_apn = np.asarray(info["a_apn"])
            energy += float(np.sum(a_total ** 2)) * env.dt
            rl_l1 += float(np.linalg.norm(a_rl))
            cmd_l1 += float(np.linalg.norm(a_apn) + np.linalg.norm(a_rl))
            path_len += float(np.linalg.norm(env.state.position - prev_pos))
            prev_pos = env.state.position.copy()
            steps += 1
            done = bool(term or trunc)

        kills += int(info.get("killed", False))
        times.append(float(info.get("step_count", steps)) * env.dt)
        misses.append(float(info.get("distance", 0.0)))
        rewards.append(total_r)
        lengths.append(steps)
        energies.append(energy)
        rl_fractions.append(rl_l1 / max(cmd_l1, 1e-6))
        zem_terminal.append(float(info.get("ZEM", 0.0)))
        efficiency.append(start_dist / max(path_len, 1e-6))
        if collect_trajectory:
            trajectories.append({
                "log": env.get_trajectory_log(),
                "killed": bool(info.get("killed", False)),
                "stage": _stage_number(stage_config),
            })

    env.close()
    n = max(1, int(episodes))
    out: Dict[str, object] = {
        "stage": _stage_number(stage_config),
        "episodes": int(episodes),
        "kill_rate": kills / n,
        "mean_time_to_kill": float(np.mean(times)),
        "mean_miss_distance": float(np.mean(misses)),
        "mean_reward": float(np.mean(rewards)),
        "mean_episode_length": float(np.mean(lengths)),
        "mean_energy": float(np.mean(energies)),
        "mean_rl_fraction": float(np.mean(rl_fractions)),
        "mean_terminal_zem": float(np.mean(zem_terminal)),
        "mean_efficiency": float(np.mean(efficiency)),
    }
    if collect_trajectory:
        out["trajectories"] = trajectories
    return out
