"""renderer.py -- PyVista-based 3D scene renderer (plan §14.3).

Loads drone models, sets up lighting / ground / HUD and exposes
:meth:`InterceptRenderer.update_frame` to push one trajectory sample into the
scene.  Rendering is headless by default (``off_screen=True``) so it works in
CI and on machines with no display; pass ``offscreen: false`` for an
interactive window.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from .scene_builder import (
    apply_mesh_rotation,
    load_drone_mesh,
    load_visualiser_config,
    quat_from_direction,
    require_pyvista,
    rotation_matrix_from,
    trail_polydata,
    transform_matrix,
)

__all__ = ["InterceptRenderer"]


class InterceptRenderer:
    """Owns the PyVista plotter, its actors and the evolving trails."""

    def __init__(self, config: Optional[Mapping[str, Any]] = None):
        pv = require_pyvista()
        self.pv = pv
        self.config: Dict[str, Any] = dict(config) if config else load_visualiser_config()

        self.show = dict(self.config.get("show", {}))
        self.colors = dict(self.config.get("colors", {}))
        trail_cfg = dict(self.config.get("trails", {}))
        self._max_points = int(trail_cfg.get("max_points", 400))
        self._trail_width = float(trail_cfg.get("width", 2))
        self._trail_colors = {
            "interceptor": trail_cfg.get("interceptor", self.colors.get("interceptor", "steelblue")),
            "target": trail_cfg.get("target", self.colors.get("target", "crimson")),
        }

        self.plotter = pv.Plotter(
            off_screen=bool(self.config.get("offscreen", True)),
            window_size=list(self.config.get("resolution", [1920, 1080])),
        )

        self._setup_background()
        self._setup_ground()
        self._setup_drones()
        self._setup_lighting()

        self.interceptor_trail: list = []
        self.target_trail: list = []
        self._dynamic: list = []  # actor names to clear each frame

        self.set_camera(str(self.config.get("camera", {}).get("mode", "chase")))
        if self.show.get("axes", True):
            try:
                self.plotter.add_axes()
            except Exception:
                pass

    # ------------------------------------------------------------- setup

    def _setup_background(self) -> None:
        bg = self.config.get("background", "sky")
        try:
            if bg == "sky":
                self.plotter.set_background("lightskyblue", top="midnightblue")
            else:
                self.plotter.set_background(bg)
        except Exception:
            self.plotter.set_background("black")

    def _setup_ground(self) -> None:
        ground_cfg = dict(self.config.get("ground", {}))
        size = float(ground_cfg.get("size", 200.0))
        plane = self.pv.Plane(
            center=(0, 0, 0),
            direction=(0, 0, 1),
            i_size=size,
            j_size=size,
            i_resolution=20,
            j_resolution=20,
        )
        self.plotter.add_mesh(
            plane, color=ground_cfg.get("color", "darkseagreen"), opacity=0.5,
            name="ground",
        )

    def _setup_drones(self) -> None:
        fallback = float(self.config.get("fallback_size", 1.5))
        interceptor_mesh, _ = load_drone_mesh(
            self.config.get("interceptor_model_path"), self.config.get("interceptor_scale", 0.01), fallback
        )
        target_mesh, _ = load_drone_mesh(
            self.config.get("target_model_path"), self.config.get("target_scale", 0.01), fallback
        )
        # Bake each mesh's orientation correction so it loads upright / nose +X.
        interceptor_mesh = apply_mesh_rotation(
            interceptor_mesh, rotation_matrix_from(self.config.get("interceptor_rotation"))
        )
        target_mesh = apply_mesh_rotation(
            target_mesh, rotation_matrix_from(self.config.get("target_rotation"))
        )
        self.interceptor_mesh = interceptor_mesh
        self.target_mesh = target_mesh

        self.interceptor_actor = self.plotter.add_mesh(
            interceptor_mesh.copy(),
            color=self._trail_colors["interceptor"],
            smooth_shading=True,
            specular=0.7,
            name="interceptor",
        )
        self.target_actor = self.plotter.add_mesh(
            target_mesh.copy(),
            color=self._trail_colors["target"],
            smooth_shading=True,
            specular=0.7,
            name="target",
        )

    def _setup_lighting(self) -> None:
        lighting = dict(self.config.get("lighting", {}))
        try:
            self.plotter.add_light(
                self.pv.Light(
                    position=(50, 50, 100),
                    focal_point=(0, 0, 0),
                    color="white",
                    intensity=float(lighting.get("intensity", 0.8)),
                )
            )
        except Exception:
            pass
        # Shadows need a shadow-capable GL context.  On headless/GLES hosts they
        # raise shader-compile errors and slow every frame down, so they are
        # opt-in via ``lighting.shadows``.
        if lighting.get("shadows", False):
            try:
                self.plotter.enable_shadows()
            except Exception:
                pass

    # -------------------------------------------------------------- frame

    def _set_actor_matrix(self, actor, position, quat, scale: float = 1.0) -> None:
        matrix = transform_matrix(position, quat, scale)
        try:
            actor.user_matrix = matrix
        except Exception:  # pragma: no cover - very old pyvista
            import vtk

            m = vtk.vtkMatrix4x4()
            for i in range(4):
                for j in range(4):
                    m.SetElement(i, j, float(matrix[i, j]))
            actor.SetUserMatrix(m)

    def _clear_dynamic(self) -> None:
        for name in self._dynamic:
            try:
                self.plotter.remove_actor(name)
            except Exception:
                pass
        self._dynamic = []

    def _add_dynamic_mesh(self, mesh, name: str, **kwargs) -> None:
        kwargs.setdefault("reset_camera", False)
        self.plotter.add_mesh(mesh, name=name, **kwargs)
        self._dynamic.append(name)

    def _update_trails(self) -> None:
        for key, points, color in (
            ("interceptor", self.interceptor_trail, self._trail_colors["interceptor"]),
            ("target", self.target_trail, self._trail_colors["target"]),
        ):
            pts = points[-self._max_points:]
            poly = trail_polydata(pts)
            if poly is not None:
                self._add_dynamic_mesh(
                    poly, f"trail_{key}", color=color, line_width=self._trail_width,
                    render_lines_as_tubes=False,
                )

    def _update_vectors(self, interceptor_pos, a_apn, a_rl, a_total) -> None:
        specs = (
            ("apn_arrow", a_apn, "green", 0.6),
            ("rl_arrow", a_rl, "orange", 0.6),
            ("total_arrow", a_total, "white", 0.6),
        )
        for name, vec, color, arrow_scale in specs:
            if not self.show.get(name, True) or vec is None:
                continue
            v = np.asarray(vec, dtype=np.float64).reshape(3)
            n = float(np.linalg.norm(v))
            if n < 1e-6:
                continue
            try:
                # Length: ``arrow_length`` metres for |a| = 30 m/s^2, clipped to [0.1, 1.5]x,
                # so the arrow stays proportionate at any scene scale.
                length = float(self.config.get("arrow_length", 12.0)) * float(
                    np.clip(n / 30.0, 0.1, 1.5)) * arrow_scale / 0.6
                arrow = self.pv.Arrow(start=interceptor_pos, direction=v / n, scale=length)
                self._add_dynamic_mesh(arrow, name, color=color)
            except Exception:
                pass

    def _update_los(self, interceptor_pos, target_pos) -> None:
        if not self.show.get("los_line", True):
            return
        try:
            poly = trail_polydata([interceptor_pos, target_pos])
            if poly is not None:
                self._add_dynamic_mesh(poly, "los_line", color="yellow", line_width=1)
        except Exception:
            pass

    def _update_kill_zone(self, target_pos, kill_radius: Optional[float]) -> None:
        if not self.show.get("kill_zone", True) or not kill_radius:
            return
        try:
            sphere = self.pv.Sphere(radius=float(kill_radius), center=target_pos)
            self._add_dynamic_mesh(sphere, "kill_zone", color="yellow", opacity=0.15)
        except Exception:
            pass

    def update_frame(
        self,
        interceptor_pos: Sequence[float],
        interceptor_quat: Sequence[float],
        target_pos: Sequence[float],
        target_quat: Optional[Sequence[float]] = None,
        *,
        a_apn: Optional[Sequence[float]] = None,
        a_rl: Optional[Sequence[float]] = None,
        a_total: Optional[Sequence[float]] = None,
        kill_radius: Optional[float] = None,
        target_vel: Optional[Sequence[float]] = None,
    ) -> None:
        """Push one trajectory sample into the scene."""
        interceptor_pos = np.asarray(interceptor_pos, dtype=np.float64).reshape(3)
        target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)

        self._set_actor_matrix(self.interceptor_actor, interceptor_pos, interceptor_quat)
        if target_quat is None and target_vel is not None:
            target_quat = quat_from_direction(target_vel)
        if target_quat is None:
            target_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self._set_actor_matrix(self.target_actor, target_pos, target_quat)

        self.interceptor_trail.append(interceptor_pos.copy())
        self.target_trail.append(target_pos.copy())

        self._clear_dynamic()
        self._update_trails()
        self._update_los(interceptor_pos, target_pos)
        self._update_vectors(interceptor_pos, a_apn, a_rl, a_total)
        self._update_kill_zone(target_pos, kill_radius)
        self._update_camera(interceptor_pos, target_pos)

    # ------------------------------------------------------------- camera

    def set_camera(self, mode: str) -> None:
        self._camera_mode = mode

    def _update_camera(self, interceptor_pos, target_pos) -> None:
        cam = dict(self.config.get("camera", {}))
        mode = self._camera_mode
        distance = float(cam.get("distance", 15.0))
        elevation = float(cam.get("elevation", 8.0))
        center = 0.5 * (interceptor_pos + target_pos)

        try:
            if mode == "static":
                # Camera is framed once (see ``frame_all``) and never touched.
                return
            if mode == "top-down":
                self.plotter.camera_position = "xy"
                self.plotter.camera.focal_point = tuple(center)
            elif mode == "los":
                direction = target_pos - interceptor_pos
                n = float(np.linalg.norm(direction)) + 1e-9
                direction = direction / n
                eye = interceptor_pos - direction * 2.0 + np.array([0, 0, 1.0])
                self.plotter.camera.position = tuple(eye)
                self.plotter.camera.focal_point = tuple(target_pos)
            elif mode == "chase":
                heading = target_pos - interceptor_pos
                n = float(np.linalg.norm(heading)) + 1e-9
                heading = heading / n
                eye = interceptor_pos - heading * distance + np.array([0, 0, elevation])
                self.plotter.camera.position = tuple(eye)
                self.plotter.camera.focal_point = tuple(interceptor_pos)
            elif mode == "fixed":
                self.plotter.camera.position = (center[0] + 60.0, center[1] - 60.0, center[2] + 40.0)
                self.plotter.camera.focal_point = tuple(center)
            elif mode == "auto":
                # Scale-aware follow camera: sits at 1.8x the current range (never closer
                # than ``auto_min_distance``) so both vehicles stay large in the frame.
                rng = float(np.linalg.norm(target_pos - interceptor_pos))
                dist = max(1.8 * rng, float(cam.get("auto_min_distance", 40.0)))
                d = np.array([1.0, -1.0, 0.55])
                d = d / np.linalg.norm(d)
                self.plotter.camera_position = [tuple(center + d * dist), tuple(center), (0, 0, 1)]
            elif mode == "orbit":
                theta = 0.4 * len(self.interceptor_trail)
                r = distance * 3.0
                eye = center + np.array([r * np.cos(theta), r * np.sin(theta), elevation * 3.0])
                self.plotter.camera.position = tuple(eye)
                self.plotter.camera.focal_point = tuple(center)
        except Exception:
            pass

    def frame_all(self, points) -> None:
        """Fix the camera once so *every* point is inside the frame.

        Used by the ``static`` camera mode: the eye and focal point are computed
        from the full trajectory (both drones' start *and* end positions) and
        then held constant for every frame, so the camera neither moves nor
        rotates.
        """
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if pts.size == 0:
            return
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        if pts.shape[0] == 0:
            return

        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
        center = 0.5 * (lo + hi)
        radius = max(0.5 * float(np.linalg.norm(hi - lo)), 1e-6)

        cam = dict(self.config.get("camera", {}))
        eye_dir = np.asarray(cam.get("static_view_dir", [1.0, -1.0, 0.55]), dtype=np.float64)
        eye_dir = eye_dir / (float(np.linalg.norm(eye_dir)) + 1e-9)
        margin = float(cam.get("static_margin", 1.45))

        try:
            fov = float(self.plotter.camera.view_angle)
        except Exception:
            fov = 30.0
        if not np.isfinite(fov) or fov <= 1.0 or fov >= 179.0:
            fov = 30.0
        dist = radius / np.tan(np.deg2rad(0.5 * fov)) * margin

        eye = center + eye_dir * dist
        try:
            self.plotter.camera_position = [tuple(eye), tuple(center), (0.0, 0.0, 1.0)]
        except Exception:  # pragma: no cover - very old pyvista
            self.plotter.camera.position = tuple(eye)
            self.plotter.camera.focal_point = tuple(center)
            self.plotter.camera.up = (0.0, 0.0, 1.0)

    # ------------------------------------------------------------- output

    def add_hud(self, text: str) -> None:
        """Replace the dynamic HUD text actor."""
        if not self.show.get("hud", True):
            return
        try:
            self.plotter.remove_actor("hud")
        except Exception:
            pass
        try:
            self.plotter.add_text(text, position="upper_left", font_size=11, color="white", name="hud")
        except Exception:
            pass

    def open_movie(self, path: str, fps: int = 30, quality: int = 10) -> None:
        import os

        os.makedirs(str(__import__("pathlib").Path(path).parent) or ".", exist_ok=True)
        self.plotter.open_movie(path, framerate=int(fps), quality=int(quality))

    def write_frame(self) -> None:
        self.plotter.write_frame()

    def close(self) -> None:
        try:
            self.plotter.close()
        except Exception:
            pass
