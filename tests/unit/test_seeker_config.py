"""configs/seeker.yaml is the noise model that runs (BUGS #25).

Before the fix the env built ``SensorNoiseConfig()`` from dataclass defaults and the
YAML was read only by the source checks, so an edit to the YAML did nothing.
"""
from pathlib import Path

import pytest
import yaml

from interceptor.guidance.env import UnifiedInterceptEnv
from interceptor.guidance.seeker import (
    SEEKER_YAML,
    SEEKER_YAML_FIELDS,
    load_sensor_noise_config,
)


def _raw():
    return yaml.safe_load(Path(SEEKER_YAML).read_text(encoding="utf-8"))


def test_loader_matches_yaml_values():
    raw = _raw()
    cfg = load_sensor_noise_config()
    for key, field in SEEKER_YAML_FIELDS.items():
        want = raw[key]["value"]
        got = getattr(cfg, field)
        if isinstance(want, bool):
            assert got is want
        else:
            assert got == pytest.approx(float(want)), key


def test_env_default_uses_yaml():
    env = UnifiedInterceptEnv()
    assert env.noise_cfg == load_sensor_noise_config()
    # headline angular noise is the Echodyne 0.5 deg figure, not the old 2.5 mrad default
    assert env.noise_cfg.pos_sigma_angular_rad == pytest.approx(0.008727)


def test_yaml_edit_takes_effect(tmp_path):
    raw = _raw()
    raw["position_sigma_angular"]["value"] = 0.123
    p = tmp_path / "seeker.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert load_sensor_noise_config(p).pos_sigma_angular_rad == pytest.approx(0.123)


def test_unknown_key_is_rejected(tmp_path):
    raw = _raw()
    raw["range_sigma"] = {"value": 1.0, "unit": "m", "source": "x"}
    p = tmp_path / "seeker.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(KeyError):
        load_sensor_noise_config(p)


def test_missing_key_is_rejected(tmp_path):
    raw = _raw()
    del raw["dropout_probability"]
    p = tmp_path / "seeker.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(KeyError):
        load_sensor_noise_config(p)
