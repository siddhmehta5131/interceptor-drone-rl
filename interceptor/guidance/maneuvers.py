"""maneuvers.py -- platform-constrained target with 14 evasive manoeuvre laws.

World frame (project convention): right-handed, x north, y west, z UP.

Ported from ``archive/path_planning_updated_project/evasive_maneuvers.py``,
which used North-East-Down.  The two frames are related by the proper rotation
``(x, y, z)_NED -> (x, -y, -z)``; because it is a rotation, every law written
with dot and cross products is unchanged.  Only the frame-specific pieces were
rewritten, and each is isolated in a helper below:

* ``_heading``     compass heading psi (clockwise from north, seen from above)
* ``_climb_angle`` flight-path angle gamma (positive climbing)
* ``_altitude``    height above ground
* the horizontal normal ``n_h`` (points to the right of the velocity)
* the quadcopter free-fall limit on downward acceleration

Parity with the original is pinned by
``tests/unit/test_fixture_parity_maneuvers.py`` (plan P1.6).  Constants are
copied unchanged; sourcing them is plan P2.8.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple
import numpy as np


G_ACCEL = 9.81  # m/s^2


def _heading(v: np.ndarray) -> float:
    """Compass heading psi [rad]: 0 = north (+x), positive toward east (-y)."""
    return float(np.arctan2(-v[1], v[0]))


def _climb_angle(v: np.ndarray) -> float:
    """Flight-path angle gamma [rad], positive when climbing (+z)."""
    return float(np.arctan2(v[2], max(np.hypot(v[0], v[1]), 1e-6)))


def _altitude(p: np.ndarray) -> float:
    """Height above ground [m]."""
    return float(p[2])


class PlatformType(Enum):
    QUADCOPTER = "quadcopter"
    FIXED_WING = "fixed_wing"
    MISSILE = "missile"


class ManeuverType(Enum):
    NONE = "none"
    CONSTANT_G = "constant_g"
    SINUSOIDAL_WEAVE = "sinusoidal_weave"
    BARREL_ROLL = "barrel_roll"
    TERMINAL_BREAK = "terminal_break"
    ENERGY_DEPLETION = "energy_depletion"
    SPLIT_S = "split_s"
    SQUARE_BANG_BANG = "square_bang_bang"
    SCISSOR = "scissor"
    NOTCHING = "notching"
    TURN_INTO_MISSILE = "turn_into_missile"
    CRANK = "crank"
    SPEED_CHANGE = "speed_change"
    MINIMUM_RANGE_TURN = "minimum_range_turn"
    DIVE_DENSE_AIR = "dive_dense_air"
    SPIRAL_CORKSCREW = "spiral_corkscrew"


@dataclass
class PlatformLimits:
    """Kinematic envelope constraints for a 3-DOF point-frame target."""
    platform: PlatformType
    v_min: float               # Minimum speed [m/s] (0 for quadcopter)
    v_max: float               # Maximum speed [m/s]
    a_t_max: float             # Max forward tangential acceleration [m/s^2]
    a_t_min: float             # Max braking tangential deceleration [m/s^2] (negative)
    a_n_max: float             # Max normal/lateral acceleration [m/s^2]
    a_n_down_max: float        # Max push-down normal acceleration [m/s^2]
    tau_accel: float           # First-order acceleration lag time constant [s]
    max_jerk: float            # Max acceleration rate of change [m/s^3]
    can_hover: bool            # Position-hold capability
    omnidirectional: bool      # True for Quadcopter, False for Fixed-Wing/Missile
    roll_slew_rate: float      # Max rotation rate of normal vector [rad/s]

    @classmethod
    def from_preset(cls, platform: PlatformType) -> "PlatformLimits":
        if platform == PlatformType.QUADCOPTER:
            return cls(
                platform=platform,
                v_min=0.0,
                v_max=25.0,
                a_t_max=1.5 * G_ACCEL,
                a_t_min=-2.5 * G_ACCEL,
                a_n_max=3.5 * G_ACCEL,
                a_n_down_max=1.0 * G_ACCEL,
                tau_accel=0.08,
                max_jerk=200.0,
                can_hover=True,
                omnidirectional=True,
                roll_slew_rate=10.0,
            )
        elif platform == PlatformType.FIXED_WING:
            return cls(
                platform=platform,
                v_min=20.0,
                v_max=220.0,
                a_t_max=1.0 * G_ACCEL,
                a_t_min=-0.8 * G_ACCEL,
                a_n_max=8.0 * G_ACCEL,
                a_n_down_max=3.0 * G_ACCEL,
                tau_accel=0.35,
                max_jerk=60.0,
                can_hover=False,
                omnidirectional=False,
                roll_slew_rate=3.0,
            )
        elif platform == PlatformType.MISSILE:
            return cls(
                platform=platform,
                v_min=180.0,
                v_max=1500.0,
                a_t_max=15.0 * G_ACCEL,
                a_t_min=-3.0 * G_ACCEL,
                a_n_max=30.0 * G_ACCEL,
                a_n_down_max=30.0 * G_ACCEL,
                tau_accel=0.15,
                max_jerk=800.0,
                can_hover=False,
                omnidirectional=False,
                roll_slew_rate=25.0,
            )
        raise ValueError(f"Unknown platform type: {platform}")


@dataclass
class EngagementKinematics:
    """Instantaneous relative geometry between Interceptor (M) and Target (T)."""
    r_vec: np.ndarray          # Range vector r = r_T - r_M [m]
    r: float                   # Scalar range ||r|| [m]
    v_cl_vec: np.ndarray       # Closing velocity vector v_M - v_T [m/s]
    v_cl: float                # Scalar closing speed -dr/dt [m/s]
    los_vec: np.ndarray        # Unit Line-of-Sight vector lambda = r_vec / r
    los_rate_vec: np.ndarray   # 3D LOS rate vector d(lambda)/dt [rad/s]
    t_go: float                # Estimated time-to-go r / max(v_cl, eps) [s]
    theta_aspect: float        # Target angle wrt LOS [rad]
    delta_lead: float          # Missile lead angle wrt LOS [rad]


def compute_engagement_kinematics(
    p_T: np.ndarray, v_T: np.ndarray, p_M: np.ndarray, v_M: np.ndarray
) -> EngagementKinematics:
    """Computes exact 3D LOS, closing velocity, aspect angle, and time-to-go."""
    r_vec = p_T - p_M
    r = float(np.linalg.norm(r_vec))
    if r < 1e-6:
        r = 1e-6
    los_vec = r_vec / r

    v_rel = v_T - v_M  # dr/dt
    v_cl_vec = -v_rel
    v_cl = float(-np.dot(v_rel, los_vec))

    # d(lambda)/dt = (v_rel - (v_rel . lambda) * lambda) / r
    los_rate_vec = (v_rel + v_cl * los_vec) / r

    t_go = r / max(v_cl, 1.0) if v_cl > 0 else 999.0

    speed_T = float(np.linalg.norm(v_T))
    speed_M = float(np.linalg.norm(v_M))

    if speed_T > 1e-3:
        cos_theta = np.clip(np.dot(v_T / speed_T, los_vec), -1.0, 1.0)
        theta_aspect = float(np.arccos(cos_theta))
    else:
        theta_aspect = 0.0

    if speed_M > 1e-3:
        cos_delta = np.clip(np.dot(v_M / speed_M, los_vec), -1.0, 1.0)
        delta_lead = float(np.arccos(cos_delta))
    else:
        delta_lead = 0.0

    return EngagementKinematics(
        r_vec=r_vec,
        r=r,
        v_cl_vec=v_cl_vec,
        v_cl=v_cl,
        los_vec=los_vec,
        los_rate_vec=los_rate_vec,
        t_go=t_go,
        theta_aspect=theta_aspect,
        delta_lead=delta_lead,
    )


class EvasiveTargetGenerator:
    """
    3-DOF Point-Frame Target Trajectory Generator with 14 Evasive Maneuver Laws.
    Maintains the 6 kinematic vectors:
      - p_W     : World position [x, y, z]^T (z-up)
      - v_W     : World velocity [vx, vy, vz]^T (z-up)
      - a_W     : World acceleration [ax, ay, az]^T (z-up)
      - euler   : Orientation [phi, theta, psi]^T (Roll, Pitch, Yaw)
      - omega_B : Body angular velocity [p, q, r]^T
      - alpha_B : Body angular acceleration [p_dot, q_dot, r_dot]^T
    """

    def __init__(
        self,
        platform: PlatformType,
        p0: np.ndarray,
        v0: np.ndarray,
        maneuver: ManeuverType,
        maneuver_params: Optional[Dict] = None,
    ):
        self.limits = PlatformLimits.from_preset(platform)
        self.maneuver = maneuver
        self.params = maneuver_params or {}

        # State vectors
        self.p_W = np.array(p0, dtype=float)
        self.v_W = np.array(v0, dtype=float)
        self.a_W = np.zeros(3, dtype=float)

        # Orientation & angular state vectors
        self.euler = self._compute_euler_from_velocity(self.v_W, self.a_W)
        self.omega_B = np.zeros(3, dtype=float)
        self.alpha_B = np.zeros(3, dtype=float)

        # Reference heading for scissor/split-S maneuvers
        self.psi_ref = _heading(self.v_W)
        self.scissor_sign = 1.0
        self.split_s_completed = False

    def _unit_tangent_and_normals(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns orthonormal path frame basis vectors (z-up world):
          - t_hat : unit velocity tangent vector
          - n_h   : horizontal right-hand unit normal in xy-plane
          - n_v   : vertical upward-pitching unit normal (t_hat x n_h)
        """
        speed = np.linalg.norm(self.v_W)
        if speed < 1e-3:
            t_hat = np.array([np.cos(self.psi_ref), -np.sin(self.psi_ref), 0.0])
        else:
            t_hat = self.v_W / speed

        horiz_norm = np.hypot(t_hat[0], t_hat[1])
        if horiz_norm < 1e-3:
            n_h = np.array([0.0, -1.0, 0.0])
        else:
            n_h = np.array([t_hat[1] / horiz_norm, -t_hat[0] / horiz_norm, 0.0])

        # t_hat x n_h points downward (-z) when level; negate for pull-up (+z)
        n_v = -np.cross(t_hat, n_h)
        n_v /= max(np.linalg.norm(n_v), 1e-6)
        return t_hat, n_h, n_v

    def _compute_euler_from_velocity(
        self, v_W: np.ndarray, a_W: np.ndarray
    ) -> np.ndarray:
        """Derives kinematic [phi, theta, psi]^T from velocity and acceleration."""
        speed = np.linalg.norm(v_W)
        if speed < 1e-3:
            return np.array([0.0, 0.0, self.psi_ref if hasattr(self, "psi_ref") else 0.0])

        psi = _heading(v_W)
        theta = _climb_angle(v_W)  # +pitch is nose-up (+z)

        if self.limits.platform == PlatformType.MISSILE:
            phi = 0.0  # Skid-to-turn (roll-stabilized)
        else:
            # Bank angle from lateral acceleration relative to gravity
            _, n_h, _ = self._unit_tangent_and_normals()
            a_lat = float(np.dot(a_W, n_h))
            phi = float(np.arctan2(a_lat, G_ACCEL))

        return np.array([phi, theta, psi])

    # =========================================================================
    # THE 14 MATHEMATICAL EVASIVE MANEUVER LAWS
    # =========================================================================

    def compute_maneuver_command(
        self, t: float, kin: EngagementKinematics, v_M: np.ndarray
    ) -> np.ndarray:
        """Calculates the raw 3D commanded acceleration vector a_cmd [m/s^2] (z-up world)."""
        t_hat, n_h, n_v = self._unit_tangent_and_normals()
        speed_T = float(np.linalg.norm(self.v_W))
        m = self.maneuver
        p = self.params

        # No evasion
        if m == ManeuverType.NONE:
            return np.zeros(3)

        # 1. Constant-g (Step) Maneuver: a_T(t) = n_T * g * u(t - t_0)
        if m == ManeuverType.CONSTANT_G:
            t_0 = p.get("t_0", 0.5)
            n_g = p.get("n_g", self.limits.a_n_max / G_ACCEL)
            if t >= t_0:
                return min(n_g * G_ACCEL, self.limits.a_n_max) * n_h
            return np.zeros(3)

        # 2a. Planar Sinusoidal Weave: a_T(t) = n_T * g * sin(omega_T * t + phi_T)
        elif m == ManeuverType.SINUSOIDAL_WEAVE:
            # Default to optimal resonant frequency omega_opt = (1/tau_M)*sqrt((N-2)/2)
            tau_M = p.get("tau_M", 0.4)
            N_nav = p.get("N_nav", 3.0)
            omega_T = p.get("omega_T", (1.0 / tau_M) * np.sqrt(max(N_nav - 2.0, 0.5) / 2.0))
            phi_T = p.get("phi_T", 0.0)
            n_g = p.get("n_g", self.limits.a_n_max / G_ACCEL)
            amp = min(n_g * G_ACCEL, self.limits.a_n_max)
            return amp * np.sin(omega_T * t + phi_T) * n_h

        # 2b. 3D Helical Barrel Roll: a_T(t) = n_T * g * [cos(w*t)*n_h + sin(w*t)*n_v]
        elif m == ManeuverType.BARREL_ROLL:
            omega_T = p.get("omega_T", 1.5)
            phi_T = p.get("phi_T", 0.0)
            n_g = p.get("n_g", self.limits.a_n_max / G_ACCEL)
            amp = min(n_g * G_ACCEL, self.limits.a_n_max)
            phase = omega_T * t + phi_T
            return amp * (np.cos(phase) * n_h + np.sin(phase) * n_v)

        # 3. Terminal Break Turn: wait until t_go <= t_break, pull max Gs normal to LOS
        elif m == ManeuverType.TERMINAL_BREAK:
            tau_M = p.get("tau_M", 0.5)
            N_nav = p.get("N_nav", 3.0)
            default_t_break = 2.0 * tau_M if N_nav <= 3.0 else 1.268 * tau_M
            t_break = p.get("t_break", default_t_break)
            if kin.t_go <= t_break:
                cross_1 = np.cross(t_hat, kin.los_vec)
                u_perp = np.cross(cross_1, t_hat)
                norm_u = np.linalg.norm(u_perp)
                u_perp = u_perp / norm_u if norm_u > 1e-6 else n_h
                return self.limits.a_n_max * u_perp
            return np.zeros(3)

        # 4. Energy Depletion (Drag) Maneuver: align velocity with LOS + weave
        elif m == ManeuverType.ENERGY_DEPLETION:
            k_align = p.get("k_align", 4.0)
            omega_w = p.get("omega_w", 1.2)
            # Project LOS normal to current velocity tangent to steer theta -> 0
            los_perp = kin.los_vec - np.dot(kin.los_vec, t_hat) * t_hat
            norm_lp = np.linalg.norm(los_perp)
            steer_dir = los_perp / norm_lp if norm_lp > 1e-6 else n_h
            a_turn = k_align * speed_T * steer_dir
            a_weave = 0.35 * self.limits.a_n_max * np.sin(omega_w * t) * n_h
            return self.limits.a_t_max * t_hat + a_turn + a_weave

        # 5. Split-S / Vertical Maneuver: invert and pull downward through vertical half-loop
        elif m == ManeuverType.SPLIT_S:
            # Check if heading has reversed by ~180 deg and level flight reached
            current_psi = _heading(self.v_W)
            delta_psi = abs(np.arctan2(np.sin(current_psi - self.psi_ref), np.cos(current_psi - self.psi_ref)))
            if delta_psi > 2.95 and abs(self.v_W[2]) < 0.15 * speed_T:
                self.split_s_completed = True
            if self.split_s_completed:
                return self.limits.a_t_max * t_hat
            # Pull downward normal vector (-n_v points toward the ground when starting level)
            downward_normal = -n_v
            return self.limits.a_n_max * downward_normal + 0.5 * self.limits.a_t_max * t_hat

        # 6. Square (Bang-Bang) Maneuver: a_T(t) = a_max * sgn(sin(omega_T * t + phi_T))
        elif m == ManeuverType.SQUARE_BANG_BANG:
            omega_T = p.get("omega_T", 1.5)
            phi_T = p.get("phi_T", 0.0)
            s = np.sin(omega_T * t + phi_T)
            sign_val = 1.0 if s >= 0.0 else -1.0
            return self.limits.a_n_max * sign_val * n_h

        # 7. Scissor Maneuver: periodic heading reversals across psi_ref + max braking
        elif m == ManeuverType.SCISSOR:
            psi_max = p.get("psi_max", np.deg2rad(65.0))
            current_psi = _heading(self.v_W)
            delta_psi = float(np.arctan2(np.sin(current_psi - self.psi_ref), np.cos(current_psi - self.psi_ref)))
            if delta_psi >= psi_max:
                self.scissor_sign = -1.0
            elif delta_psi <= -psi_max:
                self.scissor_sign = 1.0
            return self.limits.a_t_min * t_hat + self.scissor_sign * self.limits.a_n_max * n_h

        # 8. Notching (Beaming / Doppler Clutter Defeat): drive (v_T . lambda) -> 0
        elif m == ManeuverType.NOTCHING:
            k_notch = p.get("k_notch", 5.0)
            radial_vel = float(np.dot(self.v_W, kin.los_vec))
            # Unit normal in the plane of v_T and LOS pointing toward LOS
            los_perp = kin.los_vec - np.dot(kin.los_vec, t_hat) * t_hat
            norm_lp = np.linalg.norm(los_perp)
            n_lambda = los_perp / norm_lp if norm_lp > 1e-6 else n_h
            # Centripetal acceleration v_T^2 / r holds the 90-deg beam circle
            a_centripetal = (speed_T ** 2) / max(kin.r, 50.0)
            a_mag = -k_notch * radial_vel + a_centripetal
            return np.clip(a_mag, -self.limits.a_n_max, self.limits.a_n_max) * n_lambda

        # 9. Turning into the Missile: aggressive turn toward -LOS to spike v_cl and LOS rate
        elif m == ManeuverType.TURN_INTO_MISSILE:
            threat_dir = -kin.los_vec
            perp_vec = threat_dir - np.dot(threat_dir, t_hat) * t_hat
            norm_p = np.linalg.norm(perp_vec)
            turn_dir = perp_vec / norm_p if norm_p > 1e-6 else n_h
            return self.limits.a_n_max * turn_dir + self.limits.a_t_max * t_hat

        # 10. Crank Maneuver: hold aspect angle at +/- gimbal limit wrt threat LOS (-lambda)
        elif m == ManeuverType.CRANK:
            theta_gimbal = p.get("theta_gimbal", np.deg2rad(60.0))
            omega_crank = p.get("omega_crank", 0.25)
            k_psi = p.get("k_psi", 3.0)
            los_azimuth = _heading(-kin.los_vec)
            side = 1.0 if np.sin(omega_crank * t) >= 0.0 else -1.0
            psi_des = los_azimuth + side * theta_gimbal
            current_psi = _heading(self.v_W)
            psi_err = float(np.arctan2(np.sin(psi_des - current_psi), np.cos(psi_des - current_psi)))
            a_lat = np.clip(k_psi * speed_T * psi_err, -self.limits.a_n_max, self.limits.a_n_max)
            return a_lat * n_h

        # 11. Speed Change Maneuver: pure axial burst/braking along velocity vector
        elif m == ManeuverType.SPEED_CHANGE:
            freq = p.get("freq", 0.8)
            if np.sin(freq * t) >= 0.0:
                return self.limits.a_t_max * t_hat
            else:
                return self.limits.a_t_min * t_hat

        # 12. Minimum Range / Close-In Turn: out-turn inside missile's min turning circle
        elif m == ManeuverType.MINIMUM_RANGE_TURN:
            a_M_max = p.get("a_M_max", 30.0 * G_ACCEL)
            speed_M = float(np.linalg.norm(v_M))
            r_boundary = (2.0 * (speed_M ** 2) / a_M_max) * max(abs(np.sin(kin.delta_lead)), 0.25)
            r_trigger = p.get("r_trigger", max(r_boundary, 1200.0))
            if kin.r <= r_trigger:
                return self.limits.a_t_min * t_hat + self.limits.a_n_max * n_h
            return np.zeros(3)

        # 13. Dive into Dense Air: pitch steeply toward low altitude (-z)
        elif m == ManeuverType.DIVE_DENSE_AIR:
            gamma_dive = p.get("gamma_dive", np.deg2rad(-35.0))  # Negative = diving
            k_gamma = p.get("k_gamma", 3.0)
            current_gamma = _climb_angle(self.v_W)
            # Floor safety at altitude h = 150 m
            if _altitude(self.p_W) <= 150.0:
                gamma_dive = 0.0
            gamma_err = gamma_dive - current_gamma
            a_pitch = np.clip(k_gamma * speed_T * gamma_err, -self.limits.a_n_down_max, self.limits.a_n_max)
            return self.limits.a_t_max * t_hat + a_pitch * n_v

        # 14. Spiral (Corkscrew) Maneuver: chirped frequency 3D descending spiral
        elif m == ManeuverType.SPIRAL_CORKSCREW:
            omega_0 = p.get("omega_0", 1.0)
            alpha_chirp = p.get("alpha_chirp", 0.35)
            phi_s = omega_0 * t + 0.5 * alpha_chirp * (t ** 2)
            a_spiral = self.limits.a_n_max * (np.cos(phi_s) * n_h + np.sin(phi_s) * n_v)
            return 0.3 * self.limits.a_t_max * t_hat + a_spiral

        return np.zeros(3)

    # =========================================================================
    # KINEMATIC CONSTRAINT ENFORCEMENT & STATE UPDATE
    # =========================================================================

    def _apply_platform_constraints(self, a_cmd: np.ndarray) -> np.ndarray:
        """Enforces platform-specific acceleration and speed limits."""
        if self.limits.omnidirectional:
            # Quadcopter: 3D thrust sphere constraint + free-fall floor
            norm_a = float(np.linalg.norm(a_cmd))
            if norm_a > self.limits.a_n_max:
                a_cmd = a_cmd * (self.limits.a_n_max / norm_a)
            # Downward acceleration (-z) cannot exceed 1 g free-fall
            a_cmd[2] = max(a_cmd[2], -self.limits.a_n_down_max)
            return a_cmd

        # Fixed-Wing & Missile: decompose into tangential (along v) and normal (perp to v)
        t_hat, _, _ = self._unit_tangent_and_normals()
        a_t = float(np.dot(a_cmd, t_hat))
        a_n_vec = a_cmd - a_t * t_hat

        # Clamp tangential acceleration
        a_t = float(np.clip(a_t, self.limits.a_t_min, self.limits.a_t_max))

        # Enforce stall & max speed boundaries on tangential acceleration
        speed = float(np.linalg.norm(self.v_W))
        if speed <= self.limits.v_min and a_t < 0.0:
            a_t = 0.0
        elif speed >= self.limits.v_max and a_t > 0.0:
            a_t = 0.0

        # Clamp normal acceleration magnitude
        a_n_mag = float(np.linalg.norm(a_n_vec))
        if a_n_mag > self.limits.a_n_max:
            a_n_vec = a_n_vec * (self.limits.a_n_max / a_n_mag)

        return a_t * t_hat + a_n_vec

    def step(
        self, dt: float, t: float, p_M: np.ndarray, v_M: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, EngagementKinematics]:
        """Advances the target state by dt and updates all 6 kinematic vectors."""
        kin = compute_engagement_kinematics(self.p_W, self.v_W, p_M, v_M)
        a_cmd_raw = self.compute_maneuver_command(t, kin, v_M)
        a_cmd = self._apply_platform_constraints(a_cmd_raw)

        # First-order actuator lag with jerk limit: da/dt = (a_cmd - a) / tau
        da_dt = (a_cmd - self.a_W) / max(self.limits.tau_accel, 1e-3)
        jerk_mag = float(np.linalg.norm(da_dt))
        if jerk_mag > self.limits.max_jerk:
            da_dt = da_dt * (self.limits.max_jerk / jerk_mag)

        self.a_W = self.a_W + da_dt * dt
        self.v_W = self.v_W + self.a_W * dt

        # Enforce hard velocity magnitude bounds
        speed = float(np.linalg.norm(self.v_W))
        if not self.limits.can_hover and speed < self.limits.v_min:
            self.v_W = (self.v_W / max(speed, 1e-6)) * self.limits.v_min
        elif speed > self.limits.v_max:
            self.v_W = (self.v_W / speed) * self.limits.v_max

        self.p_W = self.p_W + self.v_W * dt

        # Update Orientation Euler angles and Body Angular Rates/Accelerations
        new_euler = self._compute_euler_from_velocity(self.v_W, self.a_W)
        # Unwrap angle differences
        d_euler = np.arctan2(np.sin(new_euler - self.euler), np.cos(new_euler - self.euler))
        new_omega_B = d_euler / max(dt, 1e-6)
        self.alpha_B = (new_omega_B - self.omega_B) / max(dt, 1e-6)
        self.omega_B = new_omega_B
        self.euler = new_euler

        return self.p_W.copy(), self.v_W.copy(), self.a_W.copy(), kin


class ProportionalNavigationInterceptor:
    """
    3D Proportional Navigation (PN) and Augmented PN (APN) Interceptor Model
    with first-order flight control lag tau_M, saturation limit a_M_max, and
    optional atmospheric drag decay for energy-depletion & dense-air testing.
    """

    def __init__(
        self,
        p0: np.ndarray,
        v0: np.ndarray,
        N_nav: float = 3.0,
        tau_M: float = 0.25,
        a_M_max: float = 30.0 * G_ACCEL,
        use_apn: bool = False,
        enable_drag: bool = False,
    ):
        self.p_W = np.array(p0, dtype=float)
        self.v_W = np.array(v0, dtype=float)
        self.a_W = np.zeros(3, dtype=float)
        self.N_nav = N_nav
        self.tau_M = tau_M
        self.a_M_max = a_M_max
        self.use_apn = use_apn
        self.enable_drag = enable_drag

    @staticmethod
    def atmospheric_density(altitude_m: float) -> float:
        """Exponential barometric model: rho(h) = rho_0 * exp(-h / H_s)."""
        rho_0 = 1.225
        H_s = 8500.0
        return float(rho_0 * np.exp(-max(altitude_m, 0.0) / H_s))

    def step(
        self, dt: float, p_T: np.ndarray, v_T: np.ndarray, a_T: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        kin = compute_engagement_kinematics(p_T, v_T, self.p_W, self.v_W)

        # 3D True Proportional Navigation: a_c = N * v_cl * los_rate_vec
        a_cmd = self.N_nav * max(kin.v_cl, 0.0) * kin.los_rate_vec
        if self.use_apn:
            a_T_perp = a_T - np.dot(a_T, kin.los_vec) * kin.los_vec
            a_cmd = a_cmd + 0.5 * self.N_nav * a_T_perp

        # Saturate lateral acceleration command
        a_mag = float(np.linalg.norm(a_cmd))
        if a_mag > self.a_M_max:
            a_cmd = a_cmd * (self.a_M_max / a_mag)

        # First-order flight control lag
        da_dt = (a_cmd - self.a_W) / max(self.tau_M, 1e-3)
        self.a_W = self.a_W + da_dt * dt

        # Optional parasite + induced drag decay along velocity vector
        drag_accel = np.zeros(3)
        if self.enable_drag:
            speed_M = float(np.linalg.norm(self.v_W))
            if speed_M > 1.0:
                t_hat_M = self.v_W / speed_M
                alt = _altitude(self.p_W)
                rho = self.atmospheric_density(alt)
                c_parasite = 1.8e-4
                c_induced = 2.5e-4
                a_drag_mag = c_parasite * rho * (speed_M ** 2) + c_induced * (np.linalg.norm(self.a_W) ** 2) / max(rho, 0.1)
                drag_accel = -a_drag_mag * t_hat_M

        self.v_W = self.v_W + (self.a_W + drag_accel) * dt
        self.p_W = self.p_W + self.v_W * dt
        return self.p_W.copy(), self.v_W.copy(), self.a_W.copy()


def simulate_engagement(
    platform: PlatformType,
    maneuver: ManeuverType,
    maneuver_params: Optional[Dict] = None,
    dt: float = 0.01,
    max_time: float = 20.0,
) -> Dict:
    """Runs a closed-loop 3D engagement simulation and returns miss distance & logs."""
    # Set default initial conditions based on platform speed regime
    if platform == PlatformType.QUADCOPTER:
        p_T0 = np.array([1200.0, -100.0, 150.0])
        v_T0 = np.array([-15.0, -5.0, 0.0])
        p_M0 = np.array([0.0, 0.0, 150.0])
        v_M0 = np.array([180.0, -15.0, 0.0])
        a_M_max = 15.0 * G_ACCEL
    elif platform == PlatformType.FIXED_WING:
        p_T0 = np.array([4500.0, -500.0, 3000.0])
        v_T0 = np.array([-180.0, -20.0, 0.0])
        p_M0 = np.array([0.0, 0.0, 3000.0])
        v_M0 = np.array([550.0, -60.0, 0.0])
        a_M_max = 25.0 * G_ACCEL
    else:  # MISSILE
        p_T0 = np.array([8000.0, -800.0, 5000.0])
        v_T0 = np.array([-600.0, -40.0, 0.0])
        p_M0 = np.array([0.0, 0.0, 5000.0])
        v_M0 = np.array([900.0, -90.0, 0.0])
        a_M_max = 35.0 * G_ACCEL

    target = EvasiveTargetGenerator(platform, p_T0, v_T0, maneuver, maneuver_params)
    interceptor = ProportionalNavigationInterceptor(
        p_M0, v_M0, N_nav=3.0, tau_M=0.3, a_M_max=a_M_max, enable_drag=True
    )

    t = 0.0
    min_miss = float("inf")
    t_intercept = 0.0
    traj_T, traj_M = [], []

    steps = int(max_time / dt)
    for _ in range(steps):
        p_T, v_T, a_T, kin = target.step(dt, t, interceptor.p_W, interceptor.v_W)
        p_M, v_M, a_M = interceptor.step(dt, p_T, v_T, a_T)

        traj_T.append(p_T)
        traj_M.append(p_M)

        dist = float(np.linalg.norm(p_T - p_M))
        if dist < min_miss:
            min_miss = dist
            t_intercept = t

        # Terminate shortly after point of closest approach (closing velocity turns negative)
        if t > 0.5 and kin.v_cl < 0.0 and dist > min_miss + 5.0:
            break
        t += dt

    return {
        "platform": platform.value,
        "maneuver": maneuver.value,
        "miss_distance_m": round(min_miss, 3),
        "intercept_time_s": round(t_intercept, 2),
        "traj_T": np.array(traj_T),
        "traj_M": np.array(traj_M),
    }


if __name__ == "__main__":
    print(f"{'Maneuver':<22} | {'Platform':<12} | {'Intercept Time (s)':<18} | {'Miss Distance (m)':<18}")
    print("-" * 78)
    for m_type in ManeuverType:
        plat = (
            PlatformType.QUADCOPTER
            if m_type in (ManeuverType.MINIMUM_RANGE_TURN, ManeuverType.SPEED_CHANGE)
            else PlatformType.FIXED_WING
        )
        res = simulate_engagement(plat, m_type)
        print(
            f"{res['maneuver']:<22} | {res['platform']:<12} | "
            f"{res['intercept_time_s']:<18.2f} | {res['miss_distance_m']:<18.3f}"
        )
