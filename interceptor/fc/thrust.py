"""Collective-thrust inversion (shared with the bridge, plan P8.4).

Total thrust ``T`` at hover-like conditions is ``4 c_l Omega^2`` (EQ-SIX-5);
the ESC map (EQ-SIX-7, ``constants.esc_cmd_for_omega``) is then inverted for
the common motor command.  Aerodynamic thrust loss is neglected; the rate
loop and outer loop absorb it.
"""
from __future__ import annotations

import numpy as np

from ..sim.constants import PH_C_L, PH_M, esc_cmd_for_omega


def collective_for_thrust(thrust: float, c_l: float = PH_C_L) -> float:
    """Motor command in [0, 1] that produces total thrust ``thrust`` (N)."""
    omega = np.sqrt(max(thrust, 0.0) / (4.0 * c_l))
    return esc_cmd_for_omega(omega)


def hover_thrust(mass: float = PH_M, g: float = 9.81) -> float:
    return mass * g
