"""P1.11: logging and seeding (the 3-stage config loader is archived)."""

import numpy as np

from interceptor.common import derive_seed, get_logger, make_rng, seed_everything


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
