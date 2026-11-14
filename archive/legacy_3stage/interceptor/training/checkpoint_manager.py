"""checkpoint_manager.py -- persistence tree under ``data_dir`` (plan §7.1).

Layout (relative to ``data_dir``, host-mounted ``/data`` inside the
container):

``checkpoints/<config>/stage_<N>/<algo>_<steps>_steps.zip``   periodic ckpt
``checkpoints/<config>/stage_<N>/<algo>_stage_<N>_final.zip`` on stage advance
``checkpoints/<config>/stage_<N>/monitor/``                   Monitor csv dir
``tb_logs/<config>/stage_<N>/``                               tensorboard logs
``results/<config>_final.zip``                                config final model
``results/run_summary.json``                                  run summary
``curriculum_state/run_state.json``                           resume state

SB3 model save/load is imported lazily so this module stays importable
without torch/stable-baselines3 (required for the local no-SB3 smoke test).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

__all__ = ["CheckpointManager"]

# algorithm short name -> SB3 class (imported lazily)
_ALGO_CLASS_NAMES = {"PPO": "PPO", "SAC": "SAC", "TD3": "TD3"}


class CheckpointManager:
    """Path builder + (de)serialisation helpers for one training config."""

    def __init__(self, data_dir: str, config_name: str, algo_name: str) -> None:
        self.data_dir = Path(data_dir)
        self.config_name = str(config_name)
        self.algo_name = str(algo_name)

    # -- path helpers -------------------------------------------------------
    def checkpoints_root(self) -> Path:
        return self.data_dir / "checkpoints" / self.config_name

    def stage_dir(self, stage_id: str) -> Path:
        return self.checkpoints_root() / stage_id

    def periodic_path(self, stage_id: str, steps: int) -> Path:
        return self.stage_dir(stage_id) / f"{self.algo_name}_{steps}_steps.zip"

    def final_path(self, stage_id: str) -> Path:
        return self.stage_dir(stage_id) / f"{self.algo_name}_stage_{stage_id.rsplit('_', 1)[1]}_final.zip"

    def monitor_dir(self, stage_id: str) -> Path:
        return self.stage_dir(stage_id) / "monitor"

    def tb_dir(self, stage_id: str) -> Path:
        return self.data_dir / "tb_logs" / self.config_name / stage_id

    def results_dir(self) -> Path:
        return self.data_dir / "results"

    def config_final_path(self) -> Path:
        return self.results_dir() / f"{self.config_name}_final.zip"

    def run_summary_path(self) -> Path:
        return self.results_dir() / "run_summary.json"

    @staticmethod
    def run_state_path(data_dir: str) -> Path:
        return Path(data_dir) / "curriculum_state" / "run_state.json"

    def log_dir(self) -> Path:
        return self.results_dir() / "logs"

    def ensure(self) -> None:
        """Create every directory used by this manager."""
        for p in (
            self.checkpoints_root(),
            self.results_dir(),
            self.data_dir / "curriculum_state",
        ):
            p.mkdir(parents=True, exist_ok=True)

    # -- SB3 model persistence (lazy SB3 import) ------------------------------
    @classmethod
    def _algo_cls(cls, algo_name: str):
        try:
            from stable_baselines3 import PPO, SAC, TD3
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "stable_baselines3/torch required for model save/load"
            ) from exc
        return {"PPO": PPO, "SAC": SAC, "TD3": TD3}[algo_name]

    def save_model(self, model: Any, stage_id: str, steps: int, *, final: bool = False) -> Path:
        """Save an SB3 model; returns the written path.

        For off-policy algos (SAC/TD3) the replay buffer is persisted next
        to the checkpoint (``<path>_buffer.pkl``) so a resumed stage does not
        restart with an empty buffer.  Silently skips empty/absent buffers.
        """
        path = self.final_path(stage_id) if final else self.periodic_path(stage_id, steps)
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(path))
        self._maybe_save_buffer(model, path)
        return path

    def save_config_final(self, model: Any) -> Path:
        path = self.config_final_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(path))
        self._maybe_save_buffer(model, path)
        return path

    @staticmethod
    def replay_buffer_path(model_path: Path) -> Path:
        """Replay-buffer pickle next to a model checkpoint."""
        return Path(str(model_path) + "_buffer.pkl")

    def _maybe_save_buffer(self, model: Any, path: Path) -> None:
        try:
            if hasattr(model, "save_replay_buffer") and model.replay_buffer is not None:
                if int(model.replay_buffer.size()) > 0:
                    model.save_replay_buffer(str(self.replay_buffer_path(path)))
        except Exception:  # pragma: no cover - buffer persistence is best-effort
            pass

    @staticmethod
    def load_replay_buffer(model: Any, model_path: Path) -> bool:
        """Restore an off-policy model's replay buffer if its pickle exists."""
        buf = CheckpointManager.replay_buffer_path(model_path)
        if not buf.exists():
            return False
        try:
            model.load_replay_buffer(str(buf))
            return True
        except Exception:  # pragma: no cover - best-effort
            return False

    # -- C-3 / D-28: `_training_meta.json` next to each checkpoint -----------
    @staticmethod
    def training_meta_path(model_path: Union[str, Path]) -> Path:
        """``<checkpoint>.training_meta.json`` beside a model zip (C-3)."""
        p = Path(model_path)
        return p.with_name(p.stem + ".training_meta.json")

    def save_training_meta(self, model_path: Union[str, Path], payload: Mapping[str, Any]) -> Path:
        """Record the architecture a checkpoint was trained with (C-3).

        The payload is the ``{model_id, algo, hyperparameters, net_arch,
        config_hash}`` block the plan requires, plus the full model definition
        so a continuation run can rebuild the exact same architecture without
        the original ``model.yaml``.
        """
        path = self.training_meta_path(model_path)
        self._atomic_write_json(path, dict(payload))
        return path

    def load_training_meta(self, model_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Read the ``_training_meta.json`` sitting next to ``model_path``."""
        return self._atomic_read_json(self.training_meta_path(model_path))

    @classmethod
    def load_model(
        cls,
        algo_name: str,
        path: str,
        env: Any = None,
        device: str = "auto",
        **kwargs: Any,
    ) -> Any:
        """Load an SB3 model (``device`` used only when env is not given)."""
        algo_cls = cls._algo_cls(algo_name)
        if env is not None:
            return algo_cls.load(str(path), env=env, device="auto", **kwargs)
        return algo_cls.load(str(path), device=device, **kwargs)

    # -- generic JSON persistence --------------------------------------------
    @staticmethod
    def _atomic_write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=str(path.parent), prefix=path.stem + "_", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, default=float)
            os.replace(tmp, str(path))
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    @staticmethod
    def _atomic_read_json(path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return None

    def save_run_state(self, state: Dict[str, Any]) -> Path:
        path = self.run_state_path(self.data_dir)
        self._atomic_write_json(path, state)
        return path

    def load_run_state(self) -> Optional[Dict[str, Any]]:
        return self._atomic_read_json(self.run_state_path(self.data_dir))

    def save_run_summary(self, summary: Dict[str, Any]) -> Path:
        path = self.run_summary_path()
        self._atomic_write_json(path, summary)
        return path

    def load_run_summary(self) -> Optional[Dict[str, Any]]:
        return self._atomic_read_json(self.run_summary_path())

    # -- discovery -----------------------------------------------------------
    def latest_periodic(self, stage_id: str) -> Optional[Path]:
        """Most recent ``<algo>_<steps>_steps.zip`` for a stage, or None."""
        d = self.stage_dir(stage_id)
        if not d.exists():
            return None
        candidates = []
        for child in d.iterdir():
            name = child.name
            if name.endswith("_steps.zip") and name.startswith(self.algo_name + "_"):
                try:
                    steps = int(name[len(self.algo_name) + 1:-len("_steps.zip")])
                except ValueError:
                    continue
                candidates.append((steps, child))
        if not candidates:
            return None
        return max(candidates, key=lambda t: t[0])[1]

    def latest_final(self, stage_id: str) -> Optional[Path]:
        p = self.final_path(stage_id)
        return p if p.exists() else None

    # -- curriculum state (deque of episode records) ---------------------------
    @staticmethod
    def pack_episode_buffer(records: list) -> Dict[str, Any]:
        return {"episode_buffer": records}

    @staticmethod
    def unpack_episode_buffer(state: Dict[str, Any], maxlen: int) -> list:
        buf = state.get("episode_buffer", [])
        if len(buf) > maxlen:
            buf = buf[-maxlen:]
        return list(buf)