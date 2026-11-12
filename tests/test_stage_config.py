"""Unit tests for the stage configuration table (plan §16.1.6)."""

from __future__ import annotations

import numpy as np

from src.envs.stage_config import (
    DEFAULT_STAGES,
    STAGES,
    StageConfig,
    stages_from_dict,
    stages_to_dict,
)


def test_three_stages_defined():
    assert set(STAGES.keys()) == {"stage_1", "stage_2", "stage_3"}


def test_stage_ids_and_types():
    assert STAGES["stage_1"].target_type == "static"
    assert STAGES["stage_2"].target_type in ("order_1", "order_2", "order_3")
    assert STAGES["stage_3"].target_type == "evasive"


def test_actor_dim_is_85_for_all_stages():
    for stage in STAGES.values():
        assert stage.obs.actor_dim == 85


def test_success_thresholds_descend():
    assert STAGES["stage_1"].threshold > STAGES["stage_2"].threshold
    assert STAGES["stage_2"].threshold > STAGES["stage_3"].threshold


def test_rl_action_scale_grows_with_stage():
    scales = [STAGES[f"stage_{i}"].obs.rl_action_scale for i in (1, 2, 3)]
    assert scales[0] < scales[1] < scales[2]


def test_sample_episode_length_within_range():
    rng = np.random.default_rng(0)
    for stage in STAGES.values():
        lo, hi = stage.episode_length_s
        for _ in range(20):
            v = stage.sample_episode_length(rng)
            assert lo - 1e-9 <= v <= hi + 1e-9


def test_is_polynomial_and_evasive_flags():
    assert not STAGES["stage_1"].is_polynomial
    assert STAGES["stage_2"].is_polynomial
    assert STAGES["stage_3"].is_evasive


def test_validate_passes_for_defaults():
    for stage in STAGES.values():
        stage.validate(stage.id)


def test_stages_roundtrip():
    d = stages_to_dict(STAGES)
    restored = stages_from_dict(d)
    assert set(restored.keys()) == set(STAGES.keys())
    for key in STAGES:
        assert restored[key].target_type == STAGES[key].target_type
        assert restored[key].obs.actor_dim == STAGES[key].obs.actor_dim


def test_default_stages_available():
    assert isinstance(DEFAULT_STAGES, dict)
    assert len(DEFAULT_STAGES) == 3
    assert all(isinstance(s, StageConfig) for s in DEFAULT_STAGES.values())
