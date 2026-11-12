"""P1.11: unified config, logging and seeding."""

import numpy as np

from interceptor.common import (
    TrainConfig, derive_seed, get_logger, load_config, make_rng,
    seed_everything,
)
from interceptor.common.config import ConfigError
from interceptor.utils.config_loader import TrainConfig as LegacyTrainConfig


def test_config_loader_is_single_implementation():
    cfg = load_config("configs/config.yaml")
    assert isinstance(cfg, TrainConfig)
    assert isinstance(cfg, LegacyTrainConfig)
    assert cfg.config_hash
    assert cfg.stage_numbers()


def test_make_rng_and_derived_seeds_are_reproducible():
    a = make_rng(17).standard_normal(8)
    b = make_rng(17).standard_normal(8)
    np.testing.assert_array_equal(a, b)
    assert derive_seed(17, "env", 0) == derive_seed(17, "env", 0)
    assert derive_seed(17, "env", 0) != derive_seed(17, "env", 1)
    assert 0 <= derive_seed(17, "env", 0) < 2 ** 32


def test_seed_everything_returns_seed_and_logger_hierarchy():
    assert seed_everything(5) == 5
    first = np.random.random()
    seed_everything(5)
    assert np.random.random() == first
    assert get_logger("unit").name == "interceptor.unit"
    assert isinstance(ConfigError("x"), ValueError)
