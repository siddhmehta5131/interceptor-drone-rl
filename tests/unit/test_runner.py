"""P7.1: resumable, idempotent training runs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.runner import train


def _config(total):
    return {
        "task": "guidance", "algo": "ppo", "seed": 0,
        "total_steps": total, "checkpoint_every": 64,
        "guidance": {"platform": "quadcopter", "mode": "apn_rl",
                     "target_set": "static", "N_nav": 5.0, "seed": 0},
        "algo_overrides": {"n_steps": 64, "batch_size": 32, "n_epochs": 1},
    }


@pytest.mark.slow
def test_resume_from_checkpoint(tmp_path: Path):
    run_dir = tmp_path / "run0"
    first = train("run0", _config(128), run_root=tmp_path)
    assert first["num_timesteps"] >= 128
    assert (run_dir / "DONE").exists()
    assert (run_dir / "checkpoint.zip").exists()
    assert (run_dir / "config.sha256").exists()
    assert (run_dir / "pip_freeze.txt").exists()
    assert (run_dir / "curves.csv").exists()

    # A second call with a larger budget resumes instead of restarting.
    second = train("run0", _config(256), run_root=tmp_path)
    assert second["resumed"] is True
    assert second["num_timesteps"] > first["num_timesteps"]

    # A third call is a no-op.
    third = train("run0", _config(256), run_root=tmp_path)
    assert third["already_done"] is True
    done = json.loads((run_dir / "DONE").read_text(encoding="utf-8"))
    assert done["num_timesteps"] == second["num_timesteps"]


def test_learner_gamma_matches_shaping_gamma(tmp_path: Path):
    """Ng et al. (1999) invariance: the learner discounts with the env's shaping gamma."""
    from interceptor.guidance.reward import GuidanceReward
    train("g0", _config(64), run_root=tmp_path)
    hp = json.loads((tmp_path / "g0" / "hyperparameters.json").read_text(encoding="utf-8"))
    assert hp["gamma"] == pytest.approx(GuidanceReward().gamma)
    assert hp["gamma"] == pytest.approx(0.1 ** (1 / 1100), rel=1e-9)


def test_network_override_is_used_and_recorded(tmp_path: Path):
    """ablations.yaml `network` group: the override reaches SB3 and config.json."""
    import torch.nn as nn
    from training.runner import load_run_model
    cfg = dict(_config(64), network={"pi": [64, 64], "vf": [64, 64], "activation": "tanh"})
    train("n0", cfg, run_root=tmp_path)
    saved = json.loads((tmp_path / "n0" / "config.json").read_text(encoding="utf-8"))
    assert saved["network"] == {"pi": [64, 64], "vf": [64, 64], "activation": "tanh"}
    model = load_run_model(tmp_path / "n0")
    assert model.policy.net_arch == {"pi": [64, 64], "vf": [64, 64]}
    assert model.policy.activation_fn is nn.Tanh


def test_default_network_recorded(tmp_path: Path):
    from training.algos import load_training_config
    train("n1", _config(64), run_root=tmp_path)
    saved = json.loads((tmp_path / "n1" / "config.json").read_text(encoding="utf-8"))
    assert saved["network"] == load_training_config()["network"]
