"""Guidance -> flight-controller bridge (plan P8.4, mathematics section 7).

Desired world acceleration ``a_des`` becomes a thrust vector
``f = m (a_des - g)`` (EQ-BRIDGE-1); its norm gives the collective through the
inverted thrust curve (EQ-BRIDGE-2) and its direction, with a stated yaw rule,
the desired attitude (EQ-BRIDGE-3).  The quaternion error then gives the rate
setpoint (EQ-BRIDGE-4, firmware attitude gain Kp = 6 1/s).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..fc.attitude import attitude_to_rate, quat_from_thrust_yaw
from ..fc.thrust import collective_for_thrust
from ..sim.constants import PH_C_L, PH_G_W, PH_M, esc_omega_ss


def max_thrust(c_l: float = PH_C_L) -> float:
    """Total thrust at full command (N)."""
    return 4.0 * c_l * float(esc_omega_ss(1.0)) ** 2


@dataclass
class RateSetpoint:
    collective: float
    omega_cmd: np.ndarray
    q_des: np.ndarray


class AccelBridge:
    """Acceleration command -> (collective, body-rate) setpoint."""

    def __init__(self, mass: float = PH_M, max_tilt: float = np.deg2rad(60.0),
                 rate_limit: float = 12.0, yaw: float = 0.0):
        self.mass, self.max_tilt, self.rate_limit, self.yaw = mass, max_tilt, rate_limit, yaw
        self.t_max = max_thrust()

    def thrust_vector(self, a_des) -> np.ndarray:
        """EQ-BRIDGE-1 with tilt and thrust limits applied.

        Vertical component keeps priority: lateral components are scaled down
        first so the vehicle never tilts beyond ``max_tilt`` nor demands more
        than full thrust.
        """
        f = self.mass * (np.asarray(a_des, float) - PH_G_W)
        fz = max(f[2], 1e-3)
        lat = np.linalg.norm(f[:2])
        lim = fz * np.tan(self.max_tilt)
        if lat > lim:
            f[:2] *= lim / lat
        f[2] = fz
        n = np.linalg.norm(f)
        if n > self.t_max:
            lat = np.linalg.norm(f[:2])
            room = np.sqrt(max(self.t_max ** 2 - fz ** 2, 0.0)) if fz < self.t_max else 0.0
            if fz >= self.t_max:
                f = np.array([0.0, 0.0, self.t_max])
            else:
                f[:2] *= room / lat if lat > 0 else 0.0
        return f

    def __call__(self, q, a_des) -> RateSetpoint:
        f = self.thrust_vector(a_des)
        q_des = quat_from_thrust_yaw(f, self.yaw)
        return RateSetpoint(collective_for_thrust(np.linalg.norm(f)),
                            attitude_to_rate(q, q_des, limit=self.rate_limit), q_des)
