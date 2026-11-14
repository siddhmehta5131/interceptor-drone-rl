"""Resumable training runner (plan P7.1).

``train(run_id, config)`` is idempotent: a run folder with a ``DONE`` marker is
returned untouched.  Otherwise the latest checkpoint is loaded and training
continues to the configured total number of environment steps.  Every
checkpoint interval writes the model, the SAC/TD3 replay buffer, the state and
a row of the learning curve to the run folder, so killing the process loses at
most one interval.

Run folder layout::

    runs/<run_id>/
      config.json          resolved configuration
      config.sha256        hash of the resolved configuration
      git.txt              commit hash (dirty marker included)
      pip_freeze.txt       Python package versions
      seed.txt             training seed
      curves.csv           step, mean_reward, mean_length
      checkpoint.zip       latest SB3 model
      replay_buffer.pkl    latest off-policy buffer (SAC/TD3)
      state.json           num_timesteps and checkpoint count
      DONE                 present only when the budget is complete
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .algos import ALGOS, build_model, load_training_config

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = ROOT / "runs"

__all__ = ["train", "load_run_model", "run_dir_for", "MODEL_CLASSES"]


def _json_dump(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str),
                   encoding="utf-8")
    os.replace(tmp, path)


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                               capture_output=True, text=True, timeout=10)
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        return "unknown"


def _pip_freeze() -> str:
    try:
        out = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                             capture_output=True, text=True, timeout=60)
        return out.stdout
    except Exception:
        return "unavailable\n"


def run_dir_for(run_id: str, run_root: Optional[Path] = None) -> Path:
    return Path(run_root or DEFAULT_RUN_ROOT) / str(run_id)


def _resolve_config(config: dict) -> dict:
    cfg = dict(config)
    defaults = load_training_config()
    cfg.setdefault("algo", "ppo")
    cfg.setdefault("task", "guidance")
    cfg.setdefault("seed", 0)
    cfg.setdefault("checkpoint_every", 10000)
    cfg.setdefault("run_root", str(DEFAULT_RUN_ROOT))
    cfg.setdefault("device", "cpu")
    cfg.setdefault("guidance", {})
    cfg.setdefault("fc", {})
    cfg.setdefault("algo_overrides", {})
    cfg.setdefault("asymmetric", False)
    cfg.setdefault("actor_obs_dim", None)
    # Network shape: configs/training.yaml unless overridden (ablations.yaml `network`
    # group). Stored in config.json so every run records the network it trained.
    net = dict(defaults["network"])
    net.update(cfg.get("network") or {})
    cfg["network"] = net
    if "total_steps" not in cfg:
        cfg["total_steps"] = int(defaults["budgets"]["algorithm_selection_steps"])
    if str(cfg["algo"]).lower() not in ALGOS:
        raise ValueError(f"unknown algo {cfg['algo']!r}")
    return cfg


def _make_env(task: str, cfg: dict):
    if task == "fc":
        from interceptor.fc.env import FlightControllerEnv
        opts = dict(cfg.get("fc") or {})
        return FlightControllerEnv(**opts)
    if task == "guidance":
        from interceptor.guidance.env import UnifiedInterceptEnv
        from interceptor.guidance.maneuvers import PlatformType
        opts = dict(cfg.get("guidance") or {})
        if "platform" in opts and isinstance(opts["platform"], str):
            opts["platform"] = PlatformType(opts["platform"])
        return UnifiedInterceptEnv(**opts)
    raise ValueError(f"unknown task {task!r}")


def _model_classes():
    from stable_baselines3 import PPO, SAC, TD3
    return {"ppo": PPO, "sac": SAC, "td3": TD3}


MODEL_CLASSES = _model_classes()


def load_run_model(run_dir: Path, env=None):
    """Load a run's checkpoint (used by notebooks and P8.7)."""
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    cls = MODEL_CLASSES[str(cfg["algo"]).lower()]
    return cls.load(str(run_dir / "checkpoint.zip"), env=env, device="cpu")


class _CheckpointCallback:
    """Duck-typed SB3 callback (avoids importing BaseCallback at module load)."""

    def __new__(cls, save_dir: Path, every: int, total_steps: int):
        from stable_baselines3.common.callbacks import BaseCallback

        class _CB(BaseCallback):
            def __init__(self):
                super().__init__(verbose=0)
                self.save_dir = Path(save_dir)
                self.every = max(int(every), 1)
                self.total_steps = int(total_steps)
                self.saved = 0
                self._curve_path = self.save_dir / "curves.csv"
                if not self._curve_path.exists():
                    with self._curve_path.open("w", newline="", encoding="utf-8") as fh:
                        csv.writer(fh).writerow(["step", "mean_reward", "mean_length"])

            def _save(self):
                self.model.save(str(self.save_dir / "checkpoint.zip"))
                if hasattr(self.model, "save_replay_buffer"):
                    try:
                        self.model.save_replay_buffer(
                            str(self.save_dir / "replay_buffer.pkl"))
                    except Exception:
                        pass
                state = {
                    "num_timesteps": int(self.model.num_timesteps),
                    "checkpoints": self.saved,
                    "total_steps": self.total_steps,
                }
                _json_dump(self.save_dir / "state.json", state)
                self.saved += 1

            def _curve(self):
                if not self.model.ep_info_buffer:
                    return
                rewards = [float(ep["r"]) for ep in self.model.ep_info_buffer]
                lengths = [float(ep["l"]) for ep in self.model.ep_info_buffer]
                with self._curve_path.open("a", newline="", encoding="utf-8") as fh:
                    csv.writer(fh).writerow([
                        int(self.model.num_timesteps),
                        float(np.mean(rewards)), float(np.mean(lengths)),
                    ])

            def _on_step(self) -> bool:
                if self.n_calls % self.every == 0:
                    self._save()
                    self._curve()
                return True

            def _on_training_end(self):
                self._save()
                self._curve()

        return _CB()


def train(run_id: str, config: dict, *, run_root: Optional[Path] = None) -> dict:
    """Train or resume ``run_id``; return a JSON-serialisable summary."""
    cfg = _resolve_config(config)
    if run_root is not None:
        cfg["run_root"] = str(run_root)
    run_dir = run_dir_for(run_id, Path(cfg["run_root"]))
    done = run_dir / "DONE"

    if done.exists():
        summary = json.loads(done.read_text(encoding="utf-8"))
        if int(summary.get("num_timesteps", 0)) >= int(cfg["total_steps"]):
            summary["resumed"] = False
            summary["already_done"] = True
            return summary
        # A larger budget was requested: resume from the existing checkpoint.
        done.unlink()

    run_dir.mkdir(parents=True, exist_ok=True)
    resolved = json.dumps(cfg, sort_keys=True, default=str)
    _json_dump(run_dir / "config.json", cfg)
    (run_dir / "config.sha256").write_text(
        hashlib.sha256(resolved.encode()).hexdigest(), encoding="utf-8")
    if not (run_dir / "git.txt").exists():
        (run_dir / "git.txt").write_text(_git_commit(), encoding="utf-8")
    if not (run_dir / "pip_freeze.txt").exists():
        (run_dir / "pip_freeze.txt").write_text(_pip_freeze(), encoding="utf-8")
    (run_dir / "seed.txt").write_text(str(cfg["seed"]), encoding="utf-8")

    env = _make_env(str(cfg["task"]), cfg)
    hp = dict(load_training_config()["algorithms"][str(cfg["algo"]).lower()])
    # The guidance reward's potential-based shaping uses env.gamma (EQ-GUIDE-8, horizon
    # gamma); Ng, Harada & Russell (1999) policy invariance needs the learner to use the
    # same discount, so it overrides the SB3 default 0.99. An explicit override still wins.
    if hasattr(env, "gamma") and "gamma" in hp:
        hp["gamma"] = float(env.gamma)
    hp.update(cfg.get("algo_overrides") or {})
    _json_dump(run_dir / "hyperparameters.json", hp)

    ckpt = run_dir / "checkpoint.zip"
    resumed = ckpt.exists()
    if resumed:
        cls = MODEL_CLASSES[str(cfg["algo"]).lower()]
        model = cls.load(str(ckpt), env=env, device=cfg["device"])
        rb = run_dir / "replay_buffer.pkl"
        if rb.exists() and hasattr(model, "load_replay_buffer"):
            try:
                model.load_replay_buffer(str(rb))
            except Exception:
                pass
        start = int(model.num_timesteps)
    else:
        model = build_model(
            str(cfg["algo"]), env, seed=int(cfg["seed"]),
            config={"network": cfg["network"],
                    "algorithms": {str(cfg["algo"]).lower(): hp}},
            device=cfg["device"], asymmetric=bool(cfg["asymmetric"]),
            actor_obs_dim=cfg["actor_obs_dim"])
        start = 0

    remaining = max(0, int(cfg["total_steps"]) - start)
    if remaining > 0:
        callback = _CheckpointCallback(run_dir, int(cfg["checkpoint_every"]),
                                       int(cfg["total_steps"]))
        model.learn(total_timesteps=remaining, callback=callback,
                    reset_num_timesteps=False)
    model.save(str(run_dir / "checkpoint.zip"))
    if hasattr(model, "save_replay_buffer"):
        try:
            model.save_replay_buffer(str(run_dir / "replay_buffer.pkl"))
        except Exception:
            pass
    summary = {
        "run_id": str(run_id),
        "task": cfg["task"], "algo": cfg["algo"], "seed": cfg["seed"],
        "total_steps": int(cfg["total_steps"]),
        "num_timesteps": int(model.num_timesteps),
        "config_sha256": (run_dir / "config.sha256").read_text(encoding="utf-8"),
        "resumed": resumed, "already_done": False,
    }
    _json_dump(done, summary)
    return summary
