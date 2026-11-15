"""P2.2/P2.4: equation references and airframe derivations."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from interceptor.guidance.airframes import (
    aerosonde_load_factor, aerosonde_stall_speed, aerosonde_turn_rate,
    aim5_aero_coefficients, aim5_gmax, aim5_normal_force_coefficient,
    load_airframe,
)

ROOT = Path(__file__).resolve().parents[2]


def test_equation_reference_check_passes():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_equation_refs.py")],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS" in result.stdout


def test_aim5_glimit_reproduces_cadac_formula():
    cfg = load_airframe(ROOT / "configs/airframes/aim5.yaml")
    alpha = cfg["physical"]["max_angle_of_attack"]["value"]
    mach = cfg["flight_condition_q12"]["mach"]["value"]
    rho = cfg["flight_condition_q12"]["air_density"]["value"]
    speed = cfg["flight_condition_q12"]["speed"]["value"]
    mass = cfg["physical"]["mass_burnout"]["value"]
    area = cfg["physical"]["reference_area"]["value"]
    q = 0.5 * rho * speed ** 2
    cl, cd = aim5_aero_coefficients(alpha, mach, config=cfg)
    expected_cnp = cd * np.sin(np.deg2rad(alpha)) + cl * np.cos(np.deg2rad(alpha))
    assert aim5_normal_force_coefficient(alpha, mach, config=cfg) == pytest.approx(expected_cnp)
    gmax = aim5_gmax(alpha, mach, q, mass, config=cfg)
    assert gmax == pytest.approx(expected_cnp * q * area / (mass * 9.81))
    assert gmax == pytest.approx(36.5, rel=0.05)


def test_aerosonde_limit_formulas_are_physical():
    cfg = load_airframe(ROOT / "configs/airframes/aerosonde.yaml")
    mass = cfg["physical"]["mass"]["value"]
    rho = cfg["physical"]["air_density_sea_level"]["value"]
    area = cfg["physical"]["wing_area"]["value"]
    v_stall = aerosonde_stall_speed(mass, rho, area, cl_max=1.45)
    assert v_stall == pytest.approx(cfg["derived_limits"]["v_min"]["value"], rel=0.08)
    n = aerosonde_load_factor(25.0, rho, area, cl=0.5, mass=mass)
    assert 0.0 < n < 2.0
    omega = aerosonde_turn_rate(25.0, 3.8)  # 14 CFR 23.337 normal category
    assert omega == pytest.approx(cfg["derived_limits"]["omega_max"]["value"], rel=0.02)


def test_fixed_wing_limits_follow_from_propulsion_model():
    """BUGS #32: v_max, a_t_max, a_t_min are the Beard & McLain model values (EQ-FW-3/4)."""
    from interceptor.guidance.airframes import (
        aerosonde_axial_acceleration, aerosonde_max_level_speed, aerosonde_thrust)
    from interceptor.guidance.maneuvers import PlatformType
    from interceptor.guidance.platform import PLATFORMS
    cfg = load_airframe(ROOT / "configs/airframes/aerosonde.yaml")
    lim = PLATFORMS[PlatformType.FIXED_WING]
    assert aerosonde_thrust(0.0, 1.0, cfg) == pytest.approx(84.57, abs=0.01)  # static thrust
    assert lim.v_max == pytest.approx(aerosonde_max_level_speed(cfg), abs=0.05)
    assert lim.a_t_max == pytest.approx(aerosonde_axial_acceleration(25.0, 1.0, cfg), abs=0.005)
    assert lim.a_t_min == pytest.approx(aerosonde_axial_acceleration(25.0, 0.0, cfg), abs=0.005)
    assert aerosonde_axial_acceleration(lim.v_max, 1.0, cfg) == pytest.approx(0.0, abs=0.02)
    for key in ("v_max", "a_t_max", "a_t_min"):
        assert cfg["derived_limits"][key]["value"] == getattr(lim, key)
