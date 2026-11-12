"""Target sets for the unified guidance environment (P1.8).

``evasive`` uses the unconstrained generic point-mass target from notebook v2.
Its 15 laws are ported from NED to the project z-up frame; the conversion is
the proper rotation ``T = diag(1, -1, -1)`` (same as the manoeuvre port).
``static`` and ``polynomial`` reuse the closed-form target path from codebase A.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from interceptor.envs.target_generator import PolynomialTargetPath
from .kinematics import EngagementKinematics, compute_engagement_kinematics
from .maneuvers import ManeuverType

__all__ = ["GenericEvasiveTarget", "StaticTarget", "PolynomialTarget",
           "HARD_MANEUVERS", "make_target"]

HARD_MANEUVERS = {
    ManeuverType.SPIRAL_CORKSCREW, ManeuverType.BARREL_ROLL,
    ManeuverType.SQUARE_BANG_BANG, ManeuverType.TERMINAL_BREAK,
    ManeuverType.SPLIT_S, ManeuverType.SCISSOR, ManeuverType.ENERGY_DEPLETION,
}


class GenericEvasiveTarget:
    """Unconstrained point-mass target executing one ``ManeuverType`` law."""

    def __init__(self, p0, v0, maneuver: ManeuverType, v_nominal: float,
                 a_max: float, rng: np.random.Generator,
                 floor_altitude: float = 35.0):
        self.p_W = np.array(p0, dtype=float)
        self.v_W = np.array(v0, dtype=float)
        self.a_W = np.zeros(3)
        self.maneuver = maneuver
        self.v_nominal = float(v_nominal)
        self.a_max = float(a_max)
        self.rng = rng
        self.floor_altitude = float(floor_altitude)
        self.psi_ref = float(np.arctan2(-self.v_W[1], self.v_W[0]))
        self.scissor_sign = 1.0 if rng.random() > 0.5 else -1.0
        self.split_s_completed = False
        self.phi_phase = float(rng.uniform(0.0, 2.0 * np.pi))
        self.omega_base = float(rng.uniform(1.1, 2.0))

    @staticmethod
    def _heading(v) -> float:
        return float(np.arctan2(-v[1], v[0]))

    def _unit_tangent_and_normals(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        speed = float(np.linalg.norm(self.v_W))
        if speed < 1e-3:
            t_hat = np.array([np.cos(self.psi_ref), -np.sin(self.psi_ref), 0.0])
        else:
            t_hat = self.v_W / speed
        horiz = np.hypot(t_hat[0], t_hat[1])
        if horiz < 1e-3:
            n_h = np.array([0.0, -1.0, 0.0])
        else:
            # Rotated NED right-pointing normal: n_h = T [-t_y, t_x, 0]
            n_h = np.array([t_hat[1] / horiz, -t_hat[0] / horiz, 0.0])
        n_v = -np.cross(t_hat, n_h)
        n_v /= max(np.linalg.norm(n_v), 1e-6)
        return t_hat, n_h, n_v

    def compute_maneuver_command(self, t: float, kin: EngagementKinematics,
                                 v_M: np.ndarray) -> np.ndarray:
        t_hat, n_h, n_v = self._unit_tangent_and_normals()
        speed_T = float(np.linalg.norm(self.v_W))
        m = self.maneuver
        if m == ManeuverType.NONE:
            return np.zeros(3)
        if m == ManeuverType.CONSTANT_G:
            return self.a_max * n_h if t >= 0.25 else np.zeros(3)
        if m == ManeuverType.SINUSOIDAL_WEAVE:
            return self.a_max * np.sin(self.omega_base * t + self.phi_phase) * n_h
        if m == ManeuverType.BARREL_ROLL:
            phase = self.omega_base * t + self.phi_phase
            return self.a_max * (np.cos(phase) * n_h + np.sin(phase) * n_v)
        if m == ManeuverType.TERMINAL_BREAK:
            if kin.t_go <= 2.5:
                u = np.cross(np.cross(t_hat, kin.los_vec), t_hat)
                n = np.linalg.norm(u)
                return self.a_max * (u / n if n > 1e-6 else n_h)
            return np.zeros(3)
        if m == ManeuverType.ENERGY_DEPLETION:
            perp = kin.los_vec - np.dot(kin.los_vec, t_hat) * t_hat
            n = np.linalg.norm(perp)
            steer = perp / n if n > 1e-6 else n_h
            return (0.25 * self.a_max * t_hat + 0.7 * self.a_max * steer
                    + 0.4 * self.a_max * np.sin(self.omega_base * t) * n_h)
        if m == ManeuverType.SPLIT_S:
            dpsi = abs(float(np.arctan2(
                np.sin(self._heading(self.v_W) - self.psi_ref),
                np.cos(self._heading(self.v_W) - self.psi_ref))))
            if dpsi > 2.8 and abs(self.v_W[2]) < 0.2 * max(speed_T, 1.0):
                self.split_s_completed = True
            if self.split_s_completed:
                return 0.2 * self.a_max * t_hat
            return -self.a_max * n_v + 0.2 * self.a_max * t_hat
        if m == ManeuverType.SQUARE_BANG_BANG:
            sign = 1.0 if np.sin(self.omega_base * t + self.phi_phase) >= 0 else -1.0
            return self.a_max * sign * n_h
        if m == ManeuverType.SCISSOR:
            psi_max = np.deg2rad(65.0)
            dpsi = float(np.arctan2(
                np.sin(self._heading(self.v_W) - self.psi_ref),
                np.cos(self._heading(self.v_W) - self.psi_ref)))
            if dpsi >= psi_max:
                self.scissor_sign = -1.0
            elif dpsi <= -psi_max:
                self.scissor_sign = 1.0
            return -0.25 * self.a_max * t_hat + self.scissor_sign * self.a_max * n_h
        if m == ManeuverType.NOTCHING:
            radial_vel = float(np.dot(self.v_W, kin.los_vec))
            perp = kin.los_vec - np.dot(kin.los_vec, t_hat) * t_hat
            n = np.linalg.norm(perp)
            n_lambda = perp / n if n > 1e-6 else n_h
            a_centripetal = speed_T ** 2 / max(kin.r, 50.0)
            a_mag = np.clip(-4.0 * radial_vel + a_centripetal, -self.a_max, self.a_max)
            return a_mag * n_lambda
        if m == ManeuverType.TURN_INTO_MISSILE:
            threat = -kin.los_vec
            perp = threat - np.dot(threat, t_hat) * t_hat
            n = np.linalg.norm(perp)
            return self.a_max * (perp / n if n > 1e-6 else n_h) + 0.2 * self.a_max * t_hat
        if m == ManeuverType.CRANK:
            az = float(np.arctan2(kin.los_vec[1], kin.los_vec[0]))
            side = 1.0 if np.sin(0.35 * t + self.phi_phase) >= 0 else -1.0
            psi_des = az + side * np.deg2rad(60.0)
            err = float(np.arctan2(np.sin(psi_des - self._heading(self.v_W)),
                                   np.cos(psi_des - self._heading(self.v_W))))
            a_lat = np.clip(3.0 * max(speed_T, 5.0) * err, -self.a_max, self.a_max)
            return a_lat * n_h
        if m == ManeuverType.SPEED_CHANGE:
            sign = 1.0 if np.sin(1.2 * t + self.phi_phase) >= 0 else -1.0
            return 0.8 * self.a_max * sign * t_hat
        if m == ManeuverType.MINIMUM_RANGE_TURN:
            if kin.t_go <= 4.0:
                return -0.3 * self.a_max * t_hat + self.a_max * n_h
            return np.zeros(3)
        if m == ManeuverType.DIVE_DENSE_AIR:
            gamma_dive = np.deg2rad(-30.0) if self.p_W[2] > 100.0 else 0.0
            horiz = np.hypot(self.v_W[0], self.v_W[1])
            gamma = float(np.arctan2(self.v_W[2], max(horiz, 1e-6)))
            a_pitch = np.clip(3.0 * max(speed_T, 5.0) * (gamma_dive - gamma),
                              -self.a_max, self.a_max)
            return 0.25 * self.a_max * t_hat + a_pitch * n_v
        if m == ManeuverType.SPIRAL_CORKSCREW:
            phase = self.omega_base * t + 0.18 * t ** 2 + self.phi_phase
            return (0.2 * self.a_max * t_hat
                    + self.a_max * (np.cos(phase) * n_h + np.sin(phase) * n_v))
        return np.zeros(3)

    def state_at_future(self, dt_ahead: float):
        pos = self.p_W + self.v_W * dt_ahead + 0.5 * self.a_W * dt_ahead ** 2
        vel = self.v_W + self.a_W * dt_ahead
        return pos, vel, self.a_W.copy()

    def step(self, dt: float, t: float, p_M, v_M) -> EngagementKinematics:
        kin = compute_engagement_kinematics(self.p_W, self.v_W, self.a_W, p_M, v_M)
        a_cmd = self.compute_maneuver_command(t, kin, v_M)
        n = float(np.linalg.norm(a_cmd))
        if n > self.a_max:
            a_cmd = a_cmd * (self.a_max / n)
        self.a_W = a_cmd
        self.v_W = self.v_W + self.a_W * dt
        speed = float(np.linalg.norm(self.v_W))
        v_hi = 1.30 * self.v_nominal
        if speed > v_hi:
            self.v_W = self.v_W / speed * v_hi
        self.p_W = self.p_W + self.v_W * dt
        if self.p_W[2] < self.floor_altitude:
            self.p_W[2] = self.floor_altitude
            if self.v_W[2] < 0.0:
                self.v_W[2] = -0.5 * self.v_W[2]
        return kin


class StaticTarget:
    """Stationary point target (non-evasive set)."""

    maneuver = ManeuverType.NONE

    def __init__(self, p0):
        self.p_W = np.array(p0, dtype=float)
        self.v_W = np.zeros(3)
        self.a_W = np.zeros(3)

    def state_at_future(self, dt_ahead):
        return self.p_W.copy(), self.v_W.copy(), self.a_W.copy()

    def step(self, dt, t, p_M, v_M):
        return compute_engagement_kinematics(self.p_W, self.v_W, self.a_W, p_M, v_M)


class PolynomialTarget:
    """Thin wrapper around :class:`PolynomialTargetPath` with a common API."""

    def __init__(self, path: PolynomialTargetPath):
        self.path = path
        self.maneuver = ManeuverType.NONE
        self._t = 0.0
        self.p_W = path.position_at(0.0)
        self.v_W = path.velocity_at(0.0)
        self.a_W = path.acceleration_at(0.0)

    def state_at_future(self, dt_ahead):
        t = self._t + float(dt_ahead)
        return (self.path.position_at(t), self.path.velocity_at(t),
                self.path.acceleration_at(t))

    def step(self, dt, t, p_M, v_M):
        self._t += dt
        self.p_W = self.path.position_at(self._t)
        self.v_W = self.path.velocity_at(self._t)
        self.a_W = self.path.acceleration_at(self._t)
        return compute_engagement_kinematics(self.p_W, self.v_W, self.a_W, p_M, v_M)


def make_target(target_set: str, maneuver: Optional[ManeuverType], p0, v0,
                *, v_nominal: float, a_max: float, rng: np.random.Generator,
                floor_altitude: float = 35.0):
    if target_set == "evasive":
        man = maneuver if maneuver is not None else ManeuverType.NONE
        return GenericEvasiveTarget(p0, v0, man, v_nominal, a_max, rng,
                                    floor_altitude=floor_altitude)
    if target_set == "static":
        return StaticTarget(p0)
    raise ValueError("polynomial targets are built by the environment")
