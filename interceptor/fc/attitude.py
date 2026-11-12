"""Common outer loops (plan P4.6): quaternion attitude controller and a simple
position loop.  Any inner rate controller plugs in underneath.

Attitude: ``omega_cmd = -2 Kp sign(q_e,w) q_e,xyz`` with ``q_e = q_d^-1 (x) q``
and Kp = 6 1/s (firmware roll/pitch attitude Kp, ``platform_defaults_cf2.h``).
Position: PD on position error -> desired acceleration, then thrust vector
``m (a_des - g)`` -> collective and desired attitude with fixed yaw.
"""
from __future__ import annotations

import numpy as np

from ..sim.constants import PH_G_W, PH_M
from ..sim.quaternion import quat_mult
from .thrust import collective_for_thrust

KP_ATT = np.array([6.0, 6.0, 6.0])
RATE_LIMIT = 12.0  # rad/s, bound to be replaced by the P4.9 measurement


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def attitude_to_rate(q, q_des, kp=KP_ATT, limit=RATE_LIMIT):
    """Body-rate command that drives ``q`` to ``q_des`` (both [w,x,y,z])."""
    qe = quat_mult(quat_conj(q_des), q)
    if qe[0] < 0:
        qe = -qe
    return np.clip(-2.0 * kp * qe[1:], -limit, limit)


def quat_from_thrust_yaw(thrust_dir, yaw=0.0):
    """Quaternion whose body z-axis points along ``thrust_dir`` (world)."""
    z = np.asarray(thrust_dir, float)
    z = z / np.linalg.norm(z)
    xc = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    y = np.cross(z, xc)
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    R = np.column_stack([x, y, z])
    w = np.sqrt(max(1.0 + R[0, 0] + R[1, 1] + R[2, 2], 0.0)) / 2.0
    if w > 1e-6:
        return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w),
                         (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)])
    i = int(np.argmax(np.diag(R)))
    q = np.zeros(4)
    q[1 + i] = np.sqrt(max(1 + 2 * R[i, i] - np.trace(R), 0)) / 2
    return q / np.linalg.norm(q)


class PositionLoop:
    """Position PD -> (collective, desired quaternion)."""

    def __init__(self, kp=6.0, kd=4.0, mass=PH_M, max_tilt=np.deg2rad(35.0), a_max=6.0):
        self.kp, self.kd, self.mass = kp, kd, mass
        self.max_tilt, self.a_max = max_tilt, a_max

    def __call__(self, p, v, p_des, v_des=np.zeros(3), yaw=0.0):
        a = self.kp * (p_des - p) + self.kd * (v_des - v)
        a = np.clip(a, -self.a_max, self.a_max)
        f = self.mass * (a - PH_G_W)
        fz = max(f[2], 1e-3)
        lat = np.linalg.norm(f[:2])
        if lat > fz * np.tan(self.max_tilt):
            f[:2] *= fz * np.tan(self.max_tilt) / lat
        f[2] = fz
        return collective_for_thrust(np.linalg.norm(f)), quat_from_thrust_yaw(f, yaw)
