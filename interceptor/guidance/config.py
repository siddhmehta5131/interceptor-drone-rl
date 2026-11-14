"""Guidance configuration access (plan P5.11)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..common.sources import load_yaml
from .maneuvers import PlatformType

ROOT = Path(__file__).resolve().parents[2]

__all__ = ["load_navigation", "resolve_n_prime"]


def _unwrap(value):
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    if isinstance(value, dict):
        return {k: _unwrap(v) for k, v in value.items()}
    return value


def load_navigation(path: Optional[Path] = None) -> dict:
    raw = load_yaml(Path(path) if path else ROOT / "configs" / "guidance.yaml")
    return _unwrap(raw["navigation"])


def resolve_n_prime(platform: PlatformType | str, mode: str) -> float:
    """Return the P5.11 sweep winner for ``(platform, arm)``.

    ``mode`` is the environment mode (``apn_raw`` or ``apn_filtered``); the
    filtered value is used for APN+RL as well, because the navigation term is
    the same and only the residual is added.
    """
    nav = load_navigation()
    key = platform.value if isinstance(platform, PlatformType) else str(platform)
    arm = "apn_filtered" if mode in ("apn_filtered", "apn_rl", "pure_rl") else "apn_raw"
    table = nav.get("N_prime_by_platform", {})
    if key in table and arm in table[key]:
        return float(table[key][arm])
    return float(nav["N_prime"])
