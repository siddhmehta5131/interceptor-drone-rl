"""
visualiser.py
-------------
Standalone real-time visualisation utilities for drone simulation.

Provides three pure, decoupled functions that can be dropped into any
drone simulation loop without modification:

    plot_trajectory(p_WB, q_WB)   -- live 3-D pose / trail display
    plot_input_bars(u_input)       -- live 4-channel transmitter bar chart
    read_gamepad(deadzone_val)     -- gamepad polling -> (thrust, roll, pitch, yaw)

None of these functions share state with each other or with the physics
pipeline. They communicate only through their arguments and return values.

Author  : Siddh Mehta
Project : RL-Based Autonomous Interceptor Drone
Date    : 2025

Variable convention (project-wide)
-----------------------------------
TRUE INPUT  : thrust, roll, pitch, yaw
              Polled directly from the Amkette Evo Gamepad Pro 4 (or any
              standard HID gamepad). Cannot be derived from any other
              quantity in the system.

TRUE OUTPUT : p_WB (3,) world-frame position + q_WB (4,) attitude quaternion
              The only state the human operator actually cares about —
              a fully-defined pose in 3-space.

Dependencies
------------
    numpy, matplotlib          -- always required
    pygame                     -- required only by read_gamepad()
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3-D projection)


# ---------------------------------------------------------------------------
# Internal math helper
# ---------------------------------------------------------------------------

def _quat2rotm(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to a 3×3 rotation matrix.

    The returned matrix R satisfies  v_world = R @ v_body,  i.e. its columns
    are the body-frame x/y/z axes expressed in world coordinates.

    Parameters
    ----------
    q : (4,) array-like
        Quaternion in [w, x, y, z] order.  Need not be unit-norm; the
        function normalises internally.

    Returns
    -------
    R : (3, 3) ndarray
    """
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2),  2*(x*y - w*z),       2*(x*z + w*y)      ],
        [2*(x*y + w*z),         1 - 2*(x**2 + z**2),  2*(y*z - w*x)     ],
        [2*(x*z - w*y),         2*(y*z + w*x),        1 - 2*(x**2 + y**2)],
    ])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_trajectory(p_WB: np.ndarray, q_WB: np.ndarray) -> None:
    """Stream drone pose into a persistent 3-D matplotlib figure.

    Opens a new figure on the first call; on every subsequent call updates
    only the trajectory trail and the body-frame arrow triad — no flicker,
    no figure recreation.

    Parameters
    ----------
    p_WB : (3,) array-like
        Position in the world frame [x, y, z] (metres).
    q_WB : (4,) array-like
        Attitude quaternion [w, x, y, z], body → world convention.

    Side effects
    ------------
    Draws to the "Drone trajectory (TRUE OUTPUT)" figure window.  State is
    cached on the function object to avoid global variables.
    """
    p_WB = np.asarray(p_WB, dtype=float)
    q_WB = np.asarray(q_WB, dtype=float)

    if not hasattr(plot_trajectory, "_fig") or \
            not plt.fignum_exists(plot_trajectory._fig.number):
        fig = plt.figure(num="Drone trajectory (TRUE OUTPUT)")
        ax  = fig.add_subplot(111, projection="3d")
        ax.set_xlabel("X_W (m)")
        ax.set_ylabel("Y_W (m)")
        ax.set_zlabel("Z_W (m)")
        ax.set_title("Drone Trajectory — TRUE OUTPUT")
        ax.view_init(elev=20, azim=35)

        trail_line, = ax.plot([], [], [], color=(0.1, 0.1, 0.1),
                              linewidth=1.5, label="trail")
        arrow_len = 1.0
        hx = ax.quiver(0, 0, 0, 1, 0, 0, length=arrow_len, color="r", label="x_B (front)")
        hy = ax.quiver(0, 0, 0, 0, 1, 0, length=arrow_len, color="g", label="y_B")
        hz = ax.quiver(0, 0, 0, 0, 0, 1, length=arrow_len, color="b", label="z_B (up)")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))

        plot_trajectory._fig        = fig
        plot_trajectory._ax         = ax
        plot_trajectory._trail_line = trail_line
        plot_trajectory._trail_xyz  = [[], [], []]
        plot_trajectory._hx         = hx
        plot_trajectory._hy         = hy
        plot_trajectory._hz         = hz
        plot_trajectory._arrow_len  = arrow_len

    ax = plot_trajectory._ax
    R  = _quat2rotm(q_WB)

    # Extend trail
    trail = plot_trajectory._trail_xyz
    trail[0].append(p_WB[0])
    trail[1].append(p_WB[1])
    trail[2].append(p_WB[2])
    plot_trajectory._trail_line.set_data(np.asarray(trail[0]), np.asarray(trail[1]))
    plot_trajectory._trail_line.set_3d_properties(np.asarray(trail[2]))

    # Re-draw body-frame arrows at new position / orientation
    L = plot_trajectory._arrow_len
    for h_attr, col_idx in [("_hx", 0), ("_hy", 1), ("_hz", 2)]:
        getattr(plot_trajectory, h_attr).remove()
        col  = ["r", "g", "b"][col_idx]
        axis = R[:, col_idx]
        setattr(plot_trajectory, h_attr,
                ax.quiver(*p_WB, *axis, length=L, color=col))

    plt.draw()
    plt.pause(0.001)


def plot_input_bars(u_input) -> None:
    """Stream transmitter commands into a persistent 2-D bar-chart figure.

    Opens a new figure on the first call; on every subsequent call only bar
    heights are updated.

    Parameters
    ----------
    u_input : (4,) array-like  OR  dict with keys thrust/roll/pitch/yaw
        Normalised commands in [-1, 1].

    Side effects
    ------------
    Draws to the "Transmitter input (TRUE INPUT)" figure window.
    """
    if isinstance(u_input, dict):
        vals = [u_input["thrust"], u_input["roll"], u_input["pitch"], u_input["yaw"]]
    else:
        vals = list(u_input)

    if not hasattr(plot_input_bars, "_fig") or \
            not plt.fignum_exists(plot_input_bars._fig.number):
        fig, ax = plt.subplots(num="Transmitter input (TRUE INPUT)")
        labels  = ["thrust", "roll", "pitch", "yaw"]
        bars    = ax.bar(labels, vals, color=["steelblue", "tomato", "seagreen", "orchid"])
        ax.set_ylim(-1.1, 1.1)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.grid(True, axis="y", alpha=0.4)
        ax.set_ylabel("Normalised command  [-1 … 1]")
        ax.set_title("Live Transmitter Commands — TRUE INPUT")

        plot_input_bars._fig  = fig
        plot_input_bars._ax   = ax
        plot_input_bars._bars = bars

    for bar, v in zip(plot_input_bars._bars, vals):
        bar.set_height(v)

    plt.draw()
    plt.pause(0.001)


def read_gamepad(deadzone_val: float = 0.08):
    """Poll the connected gamepad and return normalised TRPY commands.

    Connects to the first detected joystick on first call; subsequent calls
    only poll without reinitialising pygame.

    Parameters
    ----------
    deadzone_val : float, optional
        Absolute axis value below which the axis is treated as centred
        (default 0.08).

    Returns
    -------
    thrust, roll, pitch, yaw : float each, in [-1, 1]

    Raises
    ------
    RuntimeError
        If no gamepad is connected when this function is first called.

    Axis mapping (Amkette Evo Gamepad Pro 4 / standard dual-stick layout)
    ----------------------------------------------------------------------
        axis 0  left  stick X → yaw
        axis 1  left  stick Y → thrust   (sign-flipped: stick up = positive)
        axis 3  right stick X → roll
        axis 4  right stick Y → pitch    (sign-flipped: stick up = positive)
    """
    import pygame  # lazy import — not needed if using simulated input

    if not hasattr(read_gamepad, "_joy"):
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError(
                "No gamepad detected. Connect a controller and try again."
            )
        joy = pygame.joystick.Joystick(0)
        joy.init()
        print(f"[INFO] Connected to gamepad: {joy.get_name()}")
        read_gamepad._joy = joy

    joy = read_gamepad._joy
    pygame.event.pump()

    def _dz(v: float) -> float:
        """Apply symmetric deadzone."""
        return 0.0 if abs(v) < deadzone_val else v

    n = joy.get_numaxes()
    lx = _dz(joy.get_axis(0)) if n > 0 else 0.0
    ly = _dz(joy.get_axis(1)) if n > 1 else 0.0
    rx = _dz(joy.get_axis(3)) if n > 3 else 0.0
    ry = _dz(joy.get_axis(4)) if n > 4 else 0.0

    thrust =  -ly   # left stick up  → positive thrust
    yaw    =   lx   # left stick CW  → positive yaw
    roll   =   rx   # right stick R  → positive roll
    pitch  =  -ry   # right stick up → positive pitch

    return thrust, roll, pitch, yaw
