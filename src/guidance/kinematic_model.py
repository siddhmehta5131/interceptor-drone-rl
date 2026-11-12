"""kinematic_model.py -- Simple kinematic interceptor model.

The interceptor is a point mass with:
- Position  p (3D)
- Velocity  v (3D) -- clamped to [0, v_max]
- Orientation q (4D quaternion, wxyz) -- for visualisation only
- Angular velocity omega (3D) -- for heading changes during intercept

No aerodynamics, no motor model, no rigid-body dynamics.  The velocity is
updated by the acceleration command from APN + RL, and the orientation
quaternion is slaved to the velocity direction (plus an optional yaw-rate
command) purely so the visualiser has something to draw.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = ["InterceptorState", "KinematicModel", "quat_from_vectors", "quat_multiply"]

_EPS = 1e-9


@dataclass
class InterceptorState:
    """Point-mass interceptor state (world frame)."""

    position: np.ndarray          # (3,) world frame [x, y, z]
    velocity: np.ndarray          # (3,) world frame [vx, vy, vz]
    orientation: np.ndarray       # (4,) quaternion [w, x, y, z] (for vis)
    angular_velocity: np.ndarray  # (3,) body/world [wx, wy, wz]

    def copy(self) -> "InterceptorState":
        return InterceptorState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            orientation=self.orientation.copy(),
            angular_velocity=self.angular_velocity.copy(),
        )


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two ``[w, x, y, z]`` quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=float)


def quat_from_vectors(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Shortest-arc quaternion ``[w, x, y, z]`` rotating unit ``u`` toward unit ``v``."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < _EPS or nv < _EPS:
        return np.array([1.0, 0.0, 0.0, 0.0])
    u = u / nu
    v = v / nv
    c = float(np.clip(np.dot(u, v), -1.0, 1.0))
    if c > 1.0 - 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if c < -1.0 + 1e-12:
        # 180-degree rotation: pick any perpendicular axis.
        axis = np.cross(u, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(u, np.array([0.0, 1.0, 0.0]))
        axis = axis / np.linalg.norm(axis)
        return np.array([0.0, axis[0], axis[1], axis[2]])
    axis = np.cross(u, v)
    s = np.sqrt((1.0 + c) * 2.0)
    return np.array([s / 2.0, axis[0] / s, axis[1] / s, axis[2] / s])


class KinematicModel:
    """Point-mass kinematic integrator with velocity/rotation clamps."""

    def __init__(self, v_max: float, a_max: float, omega_max: float, dt: float):
        self.v_max = float(v_max)          # maximum speed (m/s)
        self.a_max = float(a_max)          # maximum acceleration (m/s^2)
        self.omega_max = float(omega_max)  # maximum angular rate (rad/s)
        self.dt = float(dt)

    # ------------------------------------------------------------------
    def reset(self, position: np.ndarray, velocity: np.ndarray,
              orientation: Optional[np.ndarray] = None) -> InterceptorState:
        """Create a fresh state; orientation defaults to velocity-aligned."""
        pos = np.asarray(position, dtype=float).copy()
        vel = np.asarray(velocity, dtype=float).copy()
        if orientation is None:
            orientation = quat_from_vectors(np.array([1.0, 0.0, 0.0]), vel)
        return InterceptorState(
            position=pos,
            velocity=vel,
            orientation=np.asarray(orientation, dtype=float).copy(),
            angular_velocity=np.zeros(3, dtype=float),
        )

    # ------------------------------------------------------------------
    def clamp_acceleration(self, a_cmd: np.ndarray) -> np.ndarray:
        a_cmd = np.asarray(a_cmd, dtype=float)
        n = np.linalg.norm(a_cmd)
        if n > self.a_max and n > _EPS:
            a_cmd = a_cmd * (self.a_max / n)
        return a_cmd

    # ------------------------------------------------------------------
    def step(self, state: InterceptorState, a_cmd: np.ndarray,
             omega_cmd: Optional[np.ndarray] = None) -> InterceptorState:
        """Advance one timestep.

        1. Clamp acceleration to a_max
        2. Update velocity: v += a * dt, clamp to v_max
        3. Update position: p += v * dt
        4. Clamp angular velocity to omega_max
        5. Update orientation quaternion
        """
        a_cmd = self.clamp_acceleration(a_cmd)

        new_vel = state.velocity + a_cmd * self.dt
        speed = float(np.linalg.norm(new_vel))
        if speed > self.v_max and speed > _EPS:
            new_vel = new_vel * (self.v_max / speed)

        new_pos = state.position + new_vel * self.dt

        if omega_cmd is None:
            omega_cmd = np.zeros(3, dtype=float)
        omega_cmd = np.asarray(omega_cmd, dtype=float)
        wn = np.linalg.norm(omega_cmd)
        if wn > self.omega_max and wn > _EPS:
            omega_cmd = omega_cmd * (self.omega_max / wn)

        # Orientation follows the velocity direction; add a yaw spin if asked.
        if speed > _EPS:
            target_q = quat_from_vectors(np.array([1.0, 0.0, 0.0]), new_vel)
            new_orientation = _slerp(state.orientation, target_q, 0.35)
            if np.linalg.norm(omega_cmd) > _EPS:
                angle = float(np.linalg.norm(omega_cmd) * self.dt)
                axis = omega_cmd / np.linalg.norm(omega_cmd)
                half = angle / 2.0
                dq = np.array([np.cos(half), axis[0] * np.sin(half),
                               axis[1] * np.sin(half), axis[2] * np.sin(half)])
                new_orientation = quat_multiply(dq, new_orientation)
                new_orientation = new_orientation / max(np.linalg.norm(new_orientation), _EPS)
        else:
            new_orientation = state.orientation.copy()

        return InterceptorState(
            position=new_pos,
            velocity=new_vel,
            orientation=new_orientation,
            angular_velocity=omega_cmd.copy(),
        )


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two quaternions."""
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1 = -q1
        d = -d
    d = float(np.clip(d, -1.0, 1.0))
    if d > 0.9995:
        out = q0 + t * (q1 - q0)
    else:
        theta_0 = np.arccos(d)
        theta = theta_0 * t
        s0 = np.sin(theta_0 - theta) / np.sin(theta_0)
        s1 = np.sin(theta) / np.sin(theta_0)
        out = s0 * q0 + s1 * q1
    return out / max(np.linalg.norm(out), _EPS)
