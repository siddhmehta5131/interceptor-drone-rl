"""Figure panels for the stage runtime.

Layout (one figure per stage, one frame per physics step):

* left  - 3-D world: drone with a live body-frame triad, the target and its
  predicted position, and fading trails;
* right - motor panel: **bars = mixer command** (0..1 per rotor) with a
  **twin-axis line = actual rotor RPM**;
* right - HUD block: stage, control source, controller, elapsed time,
  distance / kill radius, tilt, battery, reward and observation widths.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt                  # noqa: E402
from matplotlib import gridspec                  # noqa: E402
from matplotlib.lines import Line2D              # noqa: E402

from src.physics.constants import PH_OMEGA_MAX    # noqa: E402

RAD_S_TO_RPM = 60.0 / (2.0 * np.pi)
MOTOR_LABELS = ("M1", "M2", "M3", "M4")


@dataclass
class FrameData:
    """Everything the figure needs for one rendered frame."""

    t: float
    p: np.ndarray
    v: np.ndarray
    R: np.ndarray
    Omega: np.ndarray
    cmd: np.ndarray
    U_bat: float
    tilt: float
    distance: float
    action: np.ndarray
    stage: int
    control: str
    controller: str
    reward: float = 0.0
    episode: int = 1
    target_pos: Optional[np.ndarray] = None
    target_pred: Optional[np.ndarray] = None
    include_target: bool = False
    kills_enabled: bool = False
    kill_radius: float = 0.0
    killed: bool = False
    obs_dim: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


class StagePanel:
    """Builds and updates the stage figure."""

    def __init__(self, cfg, *, dpi: int = 100, trail: int = 220):
        self.cfg = cfg
        self.dpi = int(dpi)
        self.trail = int(trail)
        self.fig = plt.figure(figsize=(cfg.video.width_in, cfg.video.height_in),
                             dpi=self.dpi)
        gs = gridspec.GridSpec(
            3, 2, figure=self.fig, width_ratios=[1.65, 1.0],
            height_ratios=[1.0, 0.85, 0.75], hspace=0.42, wspace=0.24,
            left=0.03, right=0.97, top=0.93, bottom=0.05,
        )
        self.ax3d = self.fig.add_subplot(gs[:, 0], projection="3d")
        self.ax_motor = self.fig.add_subplot(gs[0, 1])
        self.ax_speed = self.fig.add_subplot(gs[1, 1])
        self.ax_hud = self.fig.add_subplot(gs[2, 1])
        self.ax_hud.axis("off")

        self.title = cfg.title or f"Stage {cfg.stage}"
        self.fig.suptitle(
            f"{self.title}  -  control: {cfg.control}  -  "
            f"{cfg.segments} x {cfg.segment_seconds:g} s  -  "
            f"{cfg.video.fps} fps  -  {cfg.video.fmt.upper()}",
            fontsize=13, fontweight="bold",
        )

        # -- 3-D world ---------------------------------------------------
        self.ax3d.set_xlabel("x [m]  forward")
        self.ax3d.set_ylabel("y [m]  right")
        self.ax3d.set_zlabel("z [m]  up")
        self.ax3d.set_xlim(-8, 8)
        self.ax3d.set_ylim(-8, 8)
        self.ax3d.set_zlim(0, 12)
        self.ax3d.view_init(elev=cfg.camera.elevation_deg, azim=cfg.camera.azimuth_deg)
        self.ax3d.grid(True, alpha=0.25)
        (self._body,) = self.ax3d.plot([], [], [], marker="o", color="#111111", ms=6,
                                       zorder=6)
        # Drone wireframe
        (self._arm1,) = self.ax3d.plot([], [], [], color="#444444", lw=2.0, zorder=5)
        (self._arm2,) = self.ax3d.plot([], [], [], color="#444444", lw=2.0, zorder=5)
        (self._front_dir,) = self.ax3d.plot([], [], [], color="#d62728", lw=3.0, zorder=6)
        (self._rear_rotors,) = self.ax3d.plot([], [], [], marker="o", ms=4,
                                              color="#444444", ls="none", zorder=7)
        (self._front_rotors,) = self.ax3d.plot([], [], [], marker="o", ms=5,
                                               color="#d62728", ls="none", zorder=7)
        (self._target,) = self.ax3d.plot([], [], [], marker="^", ms=9,
                                          color="#ff7f0e", ls="none", zorder=7)
        (self._target_pred,) = self.ax3d.plot([], [], [], marker="x", ms=7,
                                              color="#9467bd", ls="none", zorder=7)
        self._kill_sphere_artist = None
        self._trail_drone = deque(maxlen=self.trail)
        self._trail_target = deque(maxlen=self.trail)
        (self._line_drone,) = self.ax3d.plot([], [], [], color="#1f77b4", lw=1.4,
                                             alpha=0.75)
        (self._line_target,) = self.ax3d.plot([], [], [], color="#ff7f0e", lw=1.2,
                                              alpha=0.6, ls="--")
        self._ground_z = 0.0

        # -- motor panel: bars = command, line = rotor RPM ---------------
        self.ax_motor.set_title("Rotors: bars = mixer command, line = rotor RPM",
                                fontsize=9.5)
        self.ax_motor.set_xticks(range(4))
        self.ax_motor.set_xticklabels(MOTOR_LABELS)
        self.ax_motor.set_ylim(0.0, 1.0)
        self.ax_motor.set_ylabel("command c [0-1]", fontsize=8)
        self.ax_motor.tick_params(labelsize=7)
        self._bars = self.ax_motor.bar(range(4), [0, 0, 0, 1], width=0.55,
                                       color="#4c78a8", zorder=2)
        self.ax_rpm = self.ax_motor.twinx()
        rpm_max = PH_OMEGA_MAX * RAD_S_TO_RPM * 1.05
        self.ax_rpm.set_ylim(0.0, rpm_max)
        self.ax_rpm.set_ylabel("rotor RPM", fontsize=8, color="#d62728")
        self.ax_rpm.tick_params(labelsize=7, colors="#d62728")
        self.ax_motor.axhline(
            float(np.clip(0.23253743635354834, 0.0, 1.0)), color="#2ca02c",
            ls=":", lw=1.0, alpha=0.8,
        )
        (self._rpm_line,) = self.ax_rpm.plot(range(4), [0, 0, 0, 0],
                                            color="#d62728", marker="o", ms=4,
                                            lw=1.6, zorder=3)
        self._motor_text = self.ax_motor.text(
            0.5, 1.02, "", transform=self.ax_motor.transAxes, ha="center", va="bottom",
            fontsize=7.5, color="#333333",
        )

        # -- speed / tilt ------------------------------------------------
        self.ax_speed.set_title("Tracking", fontsize=9.5)
        self.ax_speed.tick_params(labelsize=7)
        self._speed_hist = deque(maxlen=self.trail)
        self._dist_hist = deque(maxlen=self.trail)
        self.ax_speed.set_xlabel("simulated time [s]", fontsize=8)
        (self._speed_line,) = self.ax_speed.plot([], [], color="#17becf", lw=1.4,
                                                 label="|v| [m/s]")
        (self._dist_line,) = self.ax_speed.plot([], [], color="#8c564b", lw=1.4,
                                                label="distance [m]")
        self.ax_speed.legend(fontsize=6.5, loc="upper right", framealpha=0.8)
        self._speed_text = self.ax_speed.text(
            0.02, 0.06, "", transform=self.ax_speed.transAxes, fontsize=7.5,
            va="bottom", color="#222222",
        )

        self._hud = self.ax_hud.text(
            0.0, 1.0, "", va="top", ha="left", family="monospace", fontsize=8.2,
            transform=self.ax_hud.transAxes,
        )
        self._legend = [
            Line2D([0], [0], color="#d62728", lw=2.5, label="drone front"),
            Line2D([0], [0], color="#444444", lw=2.0, label="drone arms"),
            Line2D([0], [0], marker="^", color="w", markerfacecolor="#ff7f0e",
                   ls="none", ms=8, label="target"),
            Line2D([0], [0], marker="x", color="w", markerfacecolor="#9467bd",
                   ls="none", ms=7, label="predicted"),
        ]
        self.ax3d.legend(handles=self._legend, fontsize=7, loc="upper left",
                         framealpha=0.85)

    # -- helpers ---------------------------------------------------------

    def _set_sphere(self, center: np.ndarray, radius: float) -> None:
        if getattr(self, "_kill_sphere_artist", None) is not None:
            self._kill_sphere_artist.remove()
            self._kill_sphere_artist = None
            
        if radius <= 0.0:
            return
            
        u = np.linspace(0.0, 2.0 * np.pi, 20)
        v = np.linspace(0.0, np.pi, 10)
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
        self._kill_sphere_artist = self.ax3d.plot_wireframe(x, y, z, color="#d62728", lw=0.4, alpha=0.35)

    def _camera(self, p: np.ndarray) -> None:
        """Follow / fixed / overview box around the drone and the target."""
        mode = self.cfg.camera.mode
        pts = [np.asarray(p, dtype=np.float64)]
        if mode == "follow" and self._trail_target:
            pts.append(np.asarray(self._trail_target[-1], dtype=np.float64))
        centre = np.mean(pts, axis=0)
        if mode == "overview":
            span = max(10.0, float(np.max([np.ptp(q) for q in pts])) if len(pts) > 1 else 10.0)
            span = min(span, 60.0)
        else:
            span = self.cfg.camera.distance
        span = float(np.clip(span * self.cfg.camera.zoom, 4.0, 120.0))
        self.ax3d.set_xlim(centre[0] - span, centre[0] + span)
        self.ax3d.set_ylim(centre[1] - span, centre[1] + span)
        self.ax3d.set_zlim(max(0.0, centre[2] - 0.6 * span), centre[2] + 0.8 * span)

    # -- frame ------------------------------------------------------------

    def update(self, f: FrameData) -> None:
        p = np.asarray(f.p, dtype=np.float64)
        R = np.asarray(f.R, dtype=np.float64)
        cmd = np.clip(np.asarray(f.cmd, dtype=np.float64).reshape(4), 0.0, 1.0)
        omega = np.asarray(f.Omega, dtype=np.float64).reshape(4)

        self._trail_drone.append(p.copy())
        if f.include_target and f.target_pos is not None:
            self._trail_target.append(np.asarray(f.target_pos, dtype=np.float64))

        # drone marker (a small sphere-ish dot) + body triad
        self._body.set_data([p[0]], [p[1]])
        self._body.set_3d_properties([p[2]])
        # drone model update
        arm = 0.22
        dx = arm * np.cos(np.pi / 4.0)
        dy = arm * np.sin(np.pi / 4.0)
        
        # local coordinates
        local_fr = np.array([dx, -dy, 0.0])
        local_fl = np.array([dx, dy, 0.0])
        local_rr = np.array([-dx, -dy, 0.0])
        local_rl = np.array([-dx, dy, 0.0])
        local_front = np.array([arm * 1.5, 0.0, 0.0])
        
        # world coordinates
        fr = p + R @ local_fr
        fl = p + R @ local_fl
        rr = p + R @ local_rr
        rl = p + R @ local_rl
        front_tip = p + R @ local_front
        
        self._arm1.set_data([rl[0], fr[0]], [rl[1], fr[1]])
        self._arm1.set_3d_properties([rl[2], fr[2]])
        
        self._arm2.set_data([rr[0], fl[0]], [rr[1], fl[1]])
        self._arm2.set_3d_properties([rr[2], fl[2]])
        
        self._front_dir.set_data([p[0], front_tip[0]], [p[1], front_tip[1]])
        self._front_dir.set_3d_properties([p[2], front_tip[2]])
        
        self._rear_rotors.set_data([rl[0], rr[0]], [rl[1], rr[1]])
        self._rear_rotors.set_3d_properties([rl[2], rr[2]])
        
        self._front_rotors.set_data([fl[0], fr[0]], [fl[1], fr[1]])
        self._front_rotors.set_3d_properties([fl[2], fr[2]])

        if f.include_target and f.target_pos is not None:
            tp = np.asarray(f.target_pos, dtype=np.float64)
            self._target.set_data([tp[0]], [tp[1]])
            self._target.set_3d_properties([tp[2]])
            self._target.set_visible(True)
            if len(self._trail_target) > 1:
                arr = np.asarray(self._trail_target)
                self._line_target.set_data(arr[:, 0], arr[:, 1])
                self._line_target.set_3d_properties(arr[:, 2])
            if f.target_pred is not None:
                pr = np.asarray(f.target_pred, dtype=np.float64)
                self._target_pred.set_data([pr[0]], [pr[1]])
                self._target_pred.set_3d_properties([pr[2]])
                self._target_pred.set_visible(True)
            self._set_sphere(tp, f.kill_radius if f.kill_radius > 0.0 else 0.0)
        else:
            self._target.set_visible(False)
            self._target_pred.set_visible(False)
            self._set_sphere(np.zeros(3), 0.0)

        if len(self._trail_drone) > 1:
            arr = np.asarray(self._trail_drone)
            self._line_drone.set_data(arr[:, 0], arr[:, 1])
            self._line_drone.set_3d_properties(arr[:, 2])

        self._camera(p)

        # motor panel
        for bar, value in zip(self._bars, cmd):
            bar.set_height(float(value))
            bar.set_y(0.0)
        rpm = omega * RAD_S_TO_RPM
        self._rpm_line.set_data(range(4), rpm)
        self._motor_text.set_text(
            "cmd " + " ".join(f"{c:4.2f}" for c in cmd)
            + "   |   RPM " + " ".join(f"{r:5.0f}" for r in rpm)
        )

        # tracking panel
        self._speed_hist.append(float(np.linalg.norm(f.v)))
        self._dist_hist.append(float(f.distance))
        ts = np.linspace(max(0.0, f.t - float(self.cfg.segment_seconds) * 3.0),
                         f.t, len(self._speed_hist))
        self._speed_line.set_data(ts, list(self._speed_hist))
        self._dist_line.set_data(ts, list(self._dist_hist))
        self.ax_speed.relim()
        self.ax_speed.autoscale_view()
        self.ax_speed.set_ylim(bottom=0.0)
        self._speed_text.set_text(
            f"|v| {self._speed_hist[-1]:.2f} m/s\n"
            f"tilt {np.degrees(f.tilt):.1f} deg\n"
            f"U_bat {f.U_bat:.2f} V\n"
            f"dist {f.distance:.2f} m"
        )

        self._hud.set_text(self._hud_text(f))
        self.fig.canvas.draw_idle()

    def _hud_text(self, f: FrameData) -> str:
        act = np.asarray(f.action, dtype=np.float64).reshape(4)
        lines = [
            f"stage       {f.stage}",
            f"control     {f.control}",
            f"controller  {f.controller}",
            f"time        {f.t:6.2f} s   episode {f.episode}",
            f"reward      {f.reward:+.2f}   obs {f.obs_dim}",
            f"action      [{act[0]:+.3f} {act[1]:+.3f} {act[2]:+.3f} {act[3]:+.3f}]",
            f"thr/roll/pi/yaw  ->  {act[0] * 100.0:5.1f}% "
            f"{act[1]:+.2f} {act[2]:+.2f} {act[3]:+.2f}",
        ]
        if f.include_target:
            verdict = "KILLED" if f.killed else (
                "in radius" if 0.0 < f.kill_radius <= f.distance else "tracking")
            lines.append(f"target      {f.distance:.2f} m  ({verdict})")
        else:
            lines.append("target      none (hover stage)")
        if f.extra:
            for key in sorted(f.extra):
                lines.append(f"{key:<10} {f.extra[key]}")
        return "\n".join(lines)

    def close(self) -> None:
        plt.close(self.fig)


__all__ = ["FrameData", "MOTOR_LABELS", "RAD_S_TO_RPM", "StagePanel"]