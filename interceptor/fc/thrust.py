"""Collective-thrust inversion (shared with the bridge, plan P8.4).

Total thrust ``T`` at hover-like conditions is ``4 c_l Omega^2`` (EQ-SIX-5);
the ESC curve ``Omega = b0 + b2 sqrt(cmd)`` (EQ-SIX-7) is then inverted for
the common motor command.  Aerodynamic thrust loss is neglected; the rate
loop and outer loop absorb it.
"""
from __future__ import annotations

import numpy as np

from ..sim.constants import PH_BAT, PH_C_L, PH_M


def collective_for_thrust(thrust: float, c_l: float = PH_C_L) -> float:
    """Motor command in [0, 1] that produces total thrust ``thrust`` (N)."""
    omega = np.sqrt(max(thrust, 0.0) / (4.0 * c_l))
    s = (omega - PH_BAT[0]) / PH_BAT[2]
    return float(np.clip(s, 0.0, 1.0) ** 2)


def hover_thrust(mass: float = PH_M, g: float = 9.81) -> float:
    return mass * g
