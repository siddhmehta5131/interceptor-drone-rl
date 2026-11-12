"""P2.1/P2 exit: every configured physical constant has a source."""

from pathlib import Path

import pytest

from interceptor.common.sources import (
    SOURCE_FILES, iter_source_leaves, load_yaml, unsourced_leaves,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("relative", SOURCE_FILES)
def test_config_uses_sourced_leaf_schema(relative):
    data = load_yaml(ROOT / relative)
    leaves = list(iter_source_leaves(data))
    assert leaves, f"{relative} contains no sourced leaves"
    bad = unsourced_leaves(data)
    report = "\n".join(f"  {leaf.path}: source={leaf.source!r}" for leaf in bad)
    assert not bad, f"{relative}: {len(bad)} unsourced leaves\n{report}"


def test_todo_count_is_zero_across_all_source_files():
    missing = {}
    for relative in SOURCE_FILES:
        bad = unsourced_leaves(load_yaml(ROOT / relative))
        if bad:
            missing[relative] = [leaf.path for leaf in bad]
    assert not missing, f"TODO count = {sum(map(len, missing.values()))}: {missing}"


def test_airframe_and_platform_key_values_are_present():
    quad = load_yaml(ROOT / "configs/quadrotor.yaml")
    assert quad["mass"]["value"] == pytest.approx(0.04085)
    assert quad["thrust_coefficient"]["value"] == pytest.approx(2.618e-8)
    platforms = load_yaml(ROOT / "configs/platforms.yaml")["platforms"]
    assert set(platforms) == {"quadcopter", "fixed_wing", "missile"}
    for name, cfg in platforms.items():
        for field in ("v_min", "v_max", "a_t_max", "a_t_min", "a_n_max",
                      "a_n_down_max", "tau_accel", "max_jerk", "omega_max",
                      "kill_radius", "r_init_min", "r_init_max", "alt_init",
                      "v_target_nominal", "a_target_max", "dt", "max_time"):
            assert field in cfg, f"{name} missing {field}"
    aim5 = load_yaml(ROOT / "configs/airframes/aim5.yaml")
    assert aim5["physical"]["mass_initial"]["value"] == pytest.approx(63.8)
    assert len(aim5["aerodynamics"]["cl_vs_alpha_mach"]["value"]) == 8
    assert len(aim5["propulsion"]["mass_vs_time"]["value"]) == 11
