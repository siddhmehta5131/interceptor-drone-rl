"""Fixed instruction sets for the stage-mode runtime.

Two families, both deterministic and model-free:

``OpenLoopScript``
    Replays a fixed ``[thrust, roll, pitch, yaw]`` programme sampled from the
    stage JSON (``instructions.steps``).  Useful for hover attitude checks and
    for reproducing a manoeuvre exactly.

``GuidanceLaw``
    A fixed proportional guidance law (position -> velocity -> body-frame
    acceleration -> rate/thrust commands).  Not learned, not tuned per run --
    the same gains produce the same flight every time, which is what a demo
    needs.  Sign conventions come from ``command_from_action``:

        c_cmd  = max(0, thrust)                  (throttle cut below 0.02)
        omega_cmd = [roll, pitch, yaw] * 2.0     rad/s, body x=forward, z=up

    Pushing the right motors up banks the drone left, so a desired body-frame
    acceleration is commanded with ``rate = -gain * a_body``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.physics.constants import PH_C_HOVER, PH_W_SCALE

#: Default guidance gains.  Overridable per stage via ``instructions.gains``.
DEFAULT_GAINS: Dict[str, float] = {
    "k_pos": 0.55,        # position error -> desired velocity (1/s)
    "k_vel": 2.20,        # velocity error -> desired acceleration (1/s^2)
    "k_rate": 0.42,       # desired body acceleration -> rate stick (-)
    "k_thrust_z": 0.075,  # altitude error -> thrust offset
    "k_thrust_vz": 0.085, # vertical speed error -> thrust offset
    "k_thrust_ff": 0.085, # horizontal acceleration -> thrust feed-forward
    "k_yaw": 0.60,        # heading error -> yaw stick
}


def _wrap(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


@dataclass(frozen=True)
class OpenLoopScript:
    """A fixed action programme held piecewise-constant in time."""

    steps: Sequence[Tuple[float, Tuple[float, float, float, float]]]
    loop: bool = False

    @property
    def duration(self) -> float:
        return float(self.steps[-1][0]) if self.steps else 0.0

    def action_at(self, t: float) -> np.ndarray:
        if not self.steps:
            return np.zeros(4, dtype=np.float64)
        if t <= self.steps[0][0]:
            return np.asarray(self.steps[0][1], dtype=np.float64)
        for i in range(1, len(self.steps)):
            t0, a0 = self.steps[i - 1]
            t1, a1 = self.steps[i]
            if t < t1 or i == len(self.steps) - 1:
                # hold the previous action until the next waypoint
                return np.asarray(a0 if t < t1 else a1, dtype=np.float64)
        return np.asarray(self.steps[-1][1], dtype=np.float64)

    def reset(self) -> None:      # controller protocol
        return None


@dataclass
class GuidanceLaw:
    """Fixed proportional guidance toward a stage-appropriate goal.

    Cascade structure, all terms measured from telemetry::

        position error -> velocity setpoint -> acceleration setpoint
                       -> desired body rate -> rate error -> stick

    The last line matters: the physics rate PID is *very* stiff, and a motor
    command that dips below the ESC deadband (0.02) makes its steady-state
    speed collapse to zero, which kicks the airframe hard.  Feeding the
    measured body rate back into the stick keeps the PID error near zero and
    the four motor commands symmetric, which is the only way to fly a stable
    open-loop demo on this airframe.  ``max_stick`` bounds the stick so the
    mixer can never drive a rotor into the deadband.
    """

    gains: Dict[str, float]
    max_tilt_deg: float = 25.0
    max_speed: float = 6.0
    target: str = "waypoint"     # waypoint | target | hover
    lead_time_s: float = 0.0
    max_stick: float = 0.16
    _t: float = 0.0

    # -- goal selection -------------------------------------------------

    def goal(self, tel: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(goal_position, goal_velocity)`` from telemetry."""
        p = np.asarray(tel["p_WB"], dtype=np.float64)
        if self.target == "target" and tel.get("include_target"):
            gp = np.asarray(tel["target_pos"], dtype=np.float64)
            gv = np.asarray(tel["target_vel"], dtype=np.float64)
            if self.lead_time_s > 0.0:
                gp = gp + gv * self.lead_time_s
            return gp, gv
        if self.target == "hover":
            return np.array([0.0, 0.0, float(tel.get("target_alt", 5.0))]), np.zeros(3)
        # waypoint: stage 2 exposes a sampled waypoint through the target
        if tel.get("include_target"):
            gp = np.asarray(tel["target_pos"], dtype=np.float64).copy()
            # hold the altitude the waypoint is at so the drone climbs to meet it
            gp[2] = float(tel.get("target_alt", gp[2]))
            return gp, np.zeros(3)
        return np.array([0.0, 0.0, float(tel.get("target_alt", 5.0))]), np.zeros(3)

    # -- control law ----------------------------------------------------

    def action(self, tel: Mapping[str, Any]) -> np.ndarray:
        g = self.gains
        p = np.asarray(tel["p_WB"], dtype=np.float64)
        v = np.asarray(tel["v_WB"], dtype=np.float64)
        R = np.asarray(tel["R_WB"], dtype=np.float64)
        w = np.asarray(tel.get("omega_B", (0.0, 0.0, 0.0)), dtype=np.float64).reshape(3)
        tilt = float(tel.get("tilt", 0.0))

        goal, goal_vel = self.goal(tel)
        v_des = np.clip((goal - p) * g["k_pos"], -self.max_speed, self.max_speed)
        # track a moving goal's own velocity so the law does not lag behind it
        if np.any(goal_vel):
            v_des = np.clip(v_des + goal_vel, -self.max_speed, self.max_speed)

        a_des = (v_des - v) * g["k_vel"]
        a_des[2] = 0.0                       # vertical is handled by the collective
        a_body = R.T @ a_des                  # body frame: x forward, y right

        # limit the commanded lean to max_tilt_deg
        a_horiz = float(np.linalg.norm(a_body[:2]))
        a_max = 9.81 * float(np.tan(np.deg2rad(self.max_tilt_deg)))
        if a_max > 0.0 and a_horiz > a_max:
            a_body[:2] *= a_max / a_horiz
            a_horiz = a_max

        # desired body rates.  A positive roll stick banks the drone left
        # (-y body) and a positive pitch stick pitches it nose-up (-x body), so
        # both rate setpoints carry the opposite sign of the acceleration.
        w_des = np.array([-g["k_rate"] * a_body[1],
                          -g["k_rate"] * a_body[0],
                          0.0])

        # yaw: face the goal heading when there is one
        los = np.asarray(tel.get("target_pos", goal), dtype=np.float64) - p
        if float(np.linalg.norm(los[:2])) > 1e-3:
            heading = float(np.arctan2(los[1], los[0]))
            current = float(np.arctan2(R[1, 0], R[0, 0]))
            w_des[2] = float(np.clip(g["k_yaw"] * _wrap(heading - current),
                                     -1.0, 1.0))

        # stick = rate error / PH_W_SCALE, so the inner PID only ever sees a
        # small residual and the mixer stays symmetric (no ESC deadband cuts)
        stick = np.clip((w_des - w) / float(PH_W_SCALE), -self.max_stick, self.max_stick)

        # collective: altitude P + vertical velocity feed-forward, divided by
        # cos(tilt) because a leaning rotor only lifts along the body z axis
        z_goal = float(goal[2])
        vz_des = float(np.clip((z_goal - p[2]) * g["k_pos"],
                               -self.max_speed, self.max_speed))
        collective = (PH_C_HOVER
                      + g["k_thrust_z"] * (z_goal - p[2])
                      + g["k_thrust_vz"] * (vz_des - v[2])
                      + g["k_thrust_ff"] * a_horiz)
        collective /= max(0.4, float(np.cos(tilt)))
        thrust = float(np.clip(collective, 0.02, 1.0))
        return np.array([thrust, stick[0], stick[1], stick[2]], dtype=np.float64)

    def reset(self) -> None:
        self._t = 0.0


class ScriptedController:
    """Open-loop wrapper implementing the controller protocol."""

    def __init__(self, script: OpenLoopScript):
        self.script = script
        self._t = 0.0

    def reset(self) -> None:
        self._t = 0.0

    def act(self, tel: Mapping[str, Any], dt: float,
            obs: Optional[np.ndarray] = None) -> np.ndarray:
        action = self.script.action_at(self._t)
        self._t += dt
        return action

    def describe(self) -> str:
        return f"open_loop script, {len(self.script.steps)} steps, " \
               f"{self.script.duration:.2f} s"


class GuidanceController:
    """Guidance wrapper implementing the controller protocol."""

    def __init__(self, law: GuidanceLaw):
        self.law = law
        self._t = 0.0

    def reset(self) -> None:
        self._t = 0.0
        self.law.reset()

    def act(self, tel: Mapping[str, Any], dt: float,
            obs: Optional[np.ndarray] = None) -> np.ndarray:
        self._t += dt
        return self.law.action(tel)

    def describe(self) -> str:
        return (f"guidance ({self.law.target}), k_pos={self.law.gains['k_pos']}, "
                f"k_vel={self.law.gains['k_vel']}, max_tilt={self.law.max_tilt_deg} deg")


def build_instruction_controller(cfg) -> Any:
    """Build the instruction controller described by a ``StageDemoConfig``."""
    instr = cfg.instructions
    if instr.mode == "open_loop":
        return ScriptedController(OpenLoopScript(steps=instr.steps))
    gains = dict(DEFAULT_GAINS)
    gains.update({k: float(v) for k, v in instr.gains.items()})
    law = GuidanceLaw(
        gains=gains,
        max_tilt_deg=instr.max_tilt_deg,
        max_speed=instr.max_speed,
        target=instr.target,
        lead_time_s=instr.lead_time_s,
        max_stick=instr.max_stick,
    )
    return GuidanceController(law)


def guidance_gains(cfg) -> Dict[str, float]:
    gains = dict(DEFAULT_GAINS)
    gains.update({k: float(v) for k, v in cfg.instructions.gains.items()})
    return gains


__all__ = [
    "DEFAULT_GAINS",
    "GuidanceController",
    "GuidanceLaw",
    "OpenLoopScript",
    "ScriptedController",
    "build_instruction_controller",
    "guidance_gains",
]
