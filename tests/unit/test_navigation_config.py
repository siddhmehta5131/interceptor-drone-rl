"""P5.11: the sweep winners are recorded in config and resolvable."""

from __future__ import annotations

from interceptor.guidance.config import load_navigation, resolve_n_prime
from interceptor.guidance.maneuvers import PlatformType


def test_navigation_config_has_sweep_source():
    nav = load_navigation()
    assert 3.0 <= nav["N_prime"] <= 5.0
    table = nav["N_prime_by_platform"]
    for platform in PlatformType:
        assert set(table[platform.value]) == {"apn_raw", "apn_filtered"}
        for arm in table[platform.value].values():
            assert 3.0 <= float(arm) <= 5.0


def test_resolve_n_prime_per_platform_and_arm():
    assert resolve_n_prime(PlatformType.QUADCOPTER, "apn_raw") == 3.0
    assert resolve_n_prime(PlatformType.QUADCOPTER, "apn_filtered") == 5.0
    assert resolve_n_prime(PlatformType.FIXED_WING, "apn_filtered") == 3.5
    assert resolve_n_prime(PlatformType.MISSILE, "apn_filtered") == 3.5
    # APN + RL keeps the filtered navigation constant.
    assert resolve_n_prime(PlatformType.MISSILE, "apn_rl") == 3.5
