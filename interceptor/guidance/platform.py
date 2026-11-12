"""Platform kinematic envelopes for the guidance layer (plan P1.7, P2).

The values are the v2 notebook presets, copied unchanged during the
consolidation phase.  ``load_platforms`` overlays the sourced values in
``configs/platforms.yaml`` when it is available; the presets remain the
fallback so importing this module never depends on the working directory.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, Mapping, Optional

from .maneuvers import PlatformType

__all__ = ["InterceptorLimits", "PLATFORMS", "load_platforms", "G_ACCEL"]

G_ACCEL = 9.81


@dataclass(frozen=True)
class InterceptorLimits:
    """Guidance-level kinematic envelope for one interceptor class."""

    platform: PlatformType
    v_min: float
    v_max: float
    a_t_max: float
    a_t_min: float
    a_n_max: float
    a_n_down_max: float
    tau_accel: float
    max_jerk: float
    omega_max: float
    can_hover: bool
    omnidirectional: bool
    kill_radius: float
    r_init_min: float
    r_init_max: float
    alt_init: float
    v_target_nominal: float
    a_target_max: float
    dt: float
    max_time: float

    @classmethod
    def from_preset(cls, platform: PlatformType) -> "InterceptorLimits":
        if platform == PlatformType.QUADCOPTER:
            return cls(
                platform=platform, v_min=0.0, v_max=45.0,
                a_t_max=3.2 * G_ACCEL, a_t_min=-3.2 * G_ACCEL,
                a_n_max=5.2 * G_ACCEL, a_n_down_max=1.0 * G_ACCEL,
                tau_accel=0.05, max_jerk=300.0, omega_max=6.0,
                can_hover=True, omnidirectional=True, kill_radius=3.0,
                r_init_min=150.0, r_init_max=450.0, alt_init=150.0,
                v_target_nominal=18.0, a_target_max=2.5 * G_ACCEL,
                dt=0.02, max_time=18.0,
            )
        if platform == PlatformType.FIXED_WING:
            return cls(
                platform=platform, v_min=20.0, v_max=255.0,
                a_t_max=4.2 * G_ACCEL, a_t_min=-2.0 * G_ACCEL,
                a_n_max=10.5 * G_ACCEL, a_n_down_max=5.0 * G_ACCEL,
                tau_accel=0.12, max_jerk=180.0, omega_max=3.5,
                can_hover=False, omnidirectional=False, kill_radius=12.0,
                r_init_min=1200.0, r_init_max=2800.0, alt_init=2000.0,
                v_target_nominal=105.0, a_target_max=4.2 * G_ACCEL,
                dt=0.02, max_time=22.0,
            )
        if platform == PlatformType.MISSILE:
            return cls(
                platform=platform, v_min=180.0, v_max=1200.0,
                a_t_max=18.0 * G_ACCEL, a_t_min=-4.0 * G_ACCEL,
                a_n_max=38.0 * G_ACCEL, a_n_down_max=38.0 * G_ACCEL,
                tau_accel=0.07, max_jerk=1000.0, omega_max=12.0,
                can_hover=False, omnidirectional=False, kill_radius=10.0,
                r_init_min=3500.0, r_init_max=7000.0, alt_init=4500.0,
                v_target_nominal=400.0, a_target_max=11.0 * G_ACCEL,
                dt=0.02, max_time=20.0,
            )
        raise ValueError(f"unknown platform {platform!r}")

    @classmethod
    def from_mapping(cls, platform: PlatformType, data: Mapping) -> "InterceptorLimits":
        """Overlay a (possibly nested ``{value: ...}``) mapping on the preset."""
        base = cls.from_preset(platform)
        values = {}
        for f in fields(cls):
            if f.name == "platform":
                continue
            raw = data.get(f.name)
            if raw is None:
                continue
            if isinstance(raw, Mapping):
                raw = raw.get("value", raw.get("min"))
            values[f.name] = raw
        return cls(**{**{f.name: getattr(base, f.name) for f in fields(cls)},
                      "platform": platform, **values})


PLATFORMS: Dict[PlatformType, InterceptorLimits] = {
    p: InterceptorLimits.from_preset(p) for p in PlatformType
}


def _unwrap_leaf(value):
    if isinstance(value, Mapping) and "value" in value:
        return value["value"]
    return value


def load_platforms(path: "str | Path | None" = "configs/platforms.yaml") -> Dict[PlatformType, InterceptorLimits]:
    """Load ``platforms.yaml``; fall back silently to the presets."""
    if path is None:
        return dict(PLATFORMS)
    p = Path(path)
    if not p.exists():
        return dict(PLATFORMS)
    import yaml

    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    root = data.get("platforms", data)
    out = dict(PLATFORMS)
    for platform in PlatformType:
        raw = root.get(platform.value)
        if not raw:
            continue
        flat = {k: _unwrap_leaf(v) for k, v in raw.items()
                if k not in ("description", "source", "notes")}
        # A few config keys are descriptive rather than dataclass fields.
        out[platform] = InterceptorLimits.from_mapping(platform, flat)
    return out
