"""animator.py -- frame-by-frame animation renderer and MP4 exporter (plan §14.4).

Takes a trajectory log (the list of per-step dicts produced by
:meth:`InterceptEnv.step`; see ``env.get_trajectory_log``) and writes an MP4.

Usage::

    animator = InterceptAnimator(model_name="apn_rl_v1", stage=2,
                                 trajectory_data=traj)
    animator.render("output/stage2_intercept.mp4", fps=30, quality=10)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .renderer import InterceptRenderer
from .scene_builder import load_visualiser_config

__all__ = ["InterceptAnimator"]


class InterceptAnimator:
    """Render a trajectory log to a video file."""

    def __init__(
        self,
        model_name: str,
        stage: int,
        trajectory_data: Sequence[Dict[str, Any]],
        config_path: str = "configs/visualiser.yaml",
        config: Optional[Dict[str, Any]] = None,
        camera: Optional[str] = None,
        kill_radius: Optional[float] = None,
    ):
        self.model_name = model_name
        self.stage = int(stage)
        self.trajectory: List[Dict[str, Any]] = list(trajectory_data)
        self.config = config if config is not None else load_visualiser_config(config_path)
        self.kill_radius = kill_radius
        self.camera = camera
        self.renderer = InterceptRenderer(self.config)
        if camera:
            self.renderer.set_camera(camera)

    # -------------------------------------------------------------------

    def render(self, output_path: str, fps: int = 30, quality: int = 10) -> str:
        """Render the full trajectory to ``output_path`` and return it."""
        fps = int(fps or self.config.get("fps", 30))
        quality = int(quality or self.config.get("quality", 10))

        self.renderer.open_movie(output_path, fps=fps, quality=quality)
        self._add_stage_banner()
        if str(self.camera or "").lower() == "static":
            self._frame_trajectory()

        for frame_idx, frame in enumerate(self.trajectory):
            vecs = frame.get("a_apn"), frame.get("a_rl"), frame.get("a_total")
            self.renderer.update_frame(
                interceptor_pos=frame.get("interceptor_pos", np.zeros(3)),
                interceptor_quat=frame.get("interceptor_quat", np.array([1.0, 0.0, 0.0, 0.0])),
                target_pos=frame.get("target_pos", np.zeros(3)),
                target_quat=None,
                target_vel=frame.get("target_vel"),
                a_apn=vecs[0],
                a_rl=vecs[1],
                a_total=vecs[2],
                kill_radius=self.kill_radius,
            )
            self._update_hud(frame_idx, frame)
            self.renderer.write_frame()

        self.renderer.close()
        return output_path

    def _frame_trajectory(self) -> None:
        """Fix the camera on the whole trajectory (both drones, start→end)."""
        pts = []
        for frame in self.trajectory:
            for key in ("interceptor_pos", "target_pos"):
                p = frame.get(key)
                if p is not None:
                    pts.append(np.asarray(p, dtype=np.float64).reshape(3))
        self.renderer.frame_all(pts)

    def _add_stage_banner(self) -> None:
        try:
            self.renderer.plotter.add_text(
                f"Model: {self.model_name}  |  Stage {self.stage}  |  APN + RL Interceptor",
                position="upper_left",
                font_size=13,
                color="white",
                name="banner",
            )
        except Exception:
            pass

    def _update_hud(self, frame_idx: int, data: Dict[str, Any]) -> None:
        """Heads-up display with per-step telemetry."""
        distance = float(data.get("distance", np.nan))
        closing = float(data.get("closing_vel", np.nan))
        los = float(data.get("LOS_rate", np.nan))
        t_go = float(data.get("t_go", np.nan))
        a_apn = np.asarray(data.get("a_apn", np.zeros(3)), dtype=np.float64)
        a_rl = np.asarray(data.get("a_rl", np.zeros(3)), dtype=np.float64)
        vel_out = data.get("velocity_output")
        speed = float(np.linalg.norm(vel_out[:3])) if vel_out is not None else np.nan

        apn_n = float(np.linalg.norm(a_apn))
        rl_n = float(np.linalg.norm(a_rl))
        total = apn_n + rl_n
        rl_frac = (rl_n / total) if total > 1e-9 else 0.0

        text = (
            f"t = {frame_idx}\n"
            f"range = {distance:6.2f} m\n"
            f"closing = {closing:6.2f} m/s\n"
            f"speed = {speed:6.2f} m/s\n"
            f"LOS rate = {los:6.3f} rad/s\n"
            f"t_go = {t_go:5.2f} s\n"
            f"|a_APN| = {apn_n:5.2f}  |a_RL| = {rl_n:5.2f}\n"
            f"RL share = {rl_frac * 100:4.1f} %"
        )
        self.renderer.add_hud(text)
