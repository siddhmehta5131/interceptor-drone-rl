"""Render :class:`TrajectoryLog` objects to MP4 (P10.2 - P10.6).

* guidance / closed-loop logs -> 3D scene via the existing renderer;
* flight-controller logs -> close attitude view with commanded vs actual rate
  traces drawn alongside;
* ``side_by_side`` stacks two or three finished videos horizontally.

Default resolution is 1280x720 (weak laptop); pass ``resolution=(1920, 1080)``
in Colab.  All paths run off-screen, so a headless runtime works when VTK has an
off-screen backend (notebook 07 installs ``xvfb``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from .schema import TrajectoryLog
from .scene_builder import load_visualiser_config

__all__ = ["render_log", "render_fc", "side_by_side", "platform_models"]

DEFAULT_RESOLUTION = (1280, 720)


def platform_models(platform: str) -> dict:
    """Mesh choice per platform: the quad keeps the project mesh; others use primitives."""
    if platform == "fixed_wing":
        return {"interceptor_model_path": "primitive:fixed_wing", "interceptor_rotation": "identity",
                "interceptor_scale": 1.0}
    if platform == "missile":
        return {"interceptor_model_path": "primitive:missile", "interceptor_rotation": "identity",
                "interceptor_scale": 1.0}
    return {}


def _config(resolution, overrides=None, config_path="configs/visualiser.yaml"):
    cfg = load_visualiser_config(config_path)
    cfg["resolution"] = list(resolution)
    cfg["offscreen"] = True
    cfg.update(overrides or {})
    return cfg


def render_log(log: TrajectoryLog, path: str | Path, *, fps: int = 30,
               resolution: Tuple[int, int] = DEFAULT_RESOLUTION, camera: str = "auto",
               title: str = "", stride: Optional[int] = None) -> str:
    """Render a guidance or closed-loop log in the 3D scene.  Returns ``path``."""
    from .animator import InterceptAnimator
    log.validate()
    if log.kind == "fc":
        return render_fc(log, path, fps=fps, resolution=resolution, title=title)
    stride = stride or max(1, int(round(1.0 / (fps * log.dt))))
    frames = log.to_frames()[::stride]
    cfg = _config(resolution, platform_models(str(log.meta.get("platform", "quadcopter"))))
    # Vehicles must stay visible: scale meshes with the scene extent (400 m scene -> x~7).
    pts = np.concatenate([log.interceptor_pos, log.target_pos])
    factor = max(1.0, float(np.ptp(pts, axis=0).max()) / 60.0)
    for key in ("interceptor_scale", "target_scale"):
        cfg[key] = float(cfg.get(key, 0.01)) * factor
    cfg.setdefault("camera", {})["auto_min_distance"] = 12.0 * factor
    cfg["arrow_length"] = 6.0 * factor
    cfg["fallback_size"] = float(cfg.get("fallback_size", 1.5)) * factor
    anim = InterceptAnimator(model_name=title or log.meta.get("label", log.kind), stage=0,
                             trajectory_data=frames, config=cfg, camera=camera,
                             kill_radius=log.kill_radius)
    return anim.render(str(path), fps=fps, quality=8)


def render_fc(log: TrajectoryLog, path: str | Path, *, fps: int = 30,
              resolution: Tuple[int, int] = DEFAULT_RESOLUTION, title: str = "") -> str:
    """Quad attitude close view (left) and commanded/actual rate traces (right)."""
    import imageio.v2 as imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pyvista as pv
    from .scene_builder import placeholder_drone, transform_matrix

    W, H = resolution
    half = W // 2
    stride = max(1, int(round(1.0 / (fps * log.dt))))
    idx = np.arange(0, len(log), stride)
    t = log.t

    plotter = pv.Plotter(off_screen=True, window_size=[half, H])
    plotter.set_background("lightskyblue", top="midnightblue")
    actor = plotter.add_mesh(placeholder_drone(1.0), color="steelblue", smooth_shading=True)
    plotter.camera_position = [(2.2, -2.2, 1.2), (0, 0, 0), (0, 0, 1)]
    plotter.add_axes()

    fig, axes = plt.subplots(3, 1, figsize=(half / 100, H / 100), dpi=100, sharex=True)
    om_cmd, om = log._channel("omega_cmd"), log._channel("omega")
    cursors = []
    for i, name in enumerate(("roll rate p", "pitch rate q", "yaw rate r")):
        axes[i].plot(t, om_cmd[:, i], "--", color="tab:orange", lw=1.2, label="commanded")
        axes[i].plot(t, om[:, i], color="tab:blue", lw=1.2, label="actual")
        axes[i].set_ylabel(f"{name} [rad/s]", fontsize=8)
        cursors.append(axes[i].axvline(0, color="k", lw=0.8))
    axes[0].legend(loc="upper right", fontsize=8)
    axes[2].set_xlabel("time [s]")
    fig.suptitle(title or log.meta.get("label", "flight controller"), fontsize=10)
    fig.tight_layout()

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8,
                                macro_block_size=1)
    try:
        for k in idx:
            q = log.interceptor_quat[k] if log.interceptor_quat is not None else [1, 0, 0, 0]
            actor.user_matrix = transform_matrix(np.zeros(3), q)
            left = plotter.screenshot(return_img=True)
            for c in cursors:
                c.set_xdata([t[k], t[k]])
            fig.canvas.draw()
            right = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            hgt = min(left.shape[0], right.shape[0])
            frame = np.concatenate([left[:hgt], right[:hgt]], axis=1)
            frame = frame[: frame.shape[0] // 2 * 2, : frame.shape[1] // 2 * 2]
            writer.append_data(frame)
    finally:
        writer.close()
        plotter.close()
        plt.close(fig)
    return str(path)


def side_by_side(paths: Sequence[str | Path], out: str | Path, labels: Sequence[str] = (),
                 fps: int = 30) -> str:
    """Stack two or three videos horizontally (P10.4).  Shorter clips hold their last frame."""
    import imageio.v2 as imageio
    if not 2 <= len(paths) <= 3:
        raise ValueError("side_by_side takes two or three videos")
    readers = [imageio.get_reader(str(p)) for p in paths]
    clips = [[f for f in r] for r in readers]
    for r in readers:
        r.close()
    n = max(len(c) for c in clips)
    height = min(c[0].shape[0] for c in clips)
    height -= height % 2
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out), fps=fps, codec="libx264", quality=8, macro_block_size=1)
    try:
        for k in range(n):
            row = []
            for i, c in enumerate(clips):
                f = c[min(k, len(c) - 1)][:height]
                if labels and i < len(labels):
                    f = _stamp(f, labels[i])
                row.append(f)
            frame = np.concatenate(row, axis=1)
            writer.append_data(frame[:, : frame.shape[1] // 2 * 2])
    finally:
        writer.close()
    return str(out)


def _stamp(frame: np.ndarray, text: str) -> np.ndarray:
    """Draw a caption strip along the bottom edge (Pillow; fast)."""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.fromarray(frame).convert("RGB")
    d = ImageDraw.Draw(img, "RGBA")
    try:
        font = ImageFont.load_default(size=max(12, frame.shape[0] // 28))
    except TypeError:  # old Pillow
        font = ImageFont.load_default()
    box = d.textbbox((0, 0), text, font=font)
    w, h = box[2] - box[0], box[3] - box[1]
    x, y = (img.width - w) // 2, img.height - h - 14
    d.rectangle([x - 8, y - 6, x + w + 8, y + h + 8], fill=(0, 0, 0, 150))
    d.text((x, y), text, fill=(255, 255, 255, 255), font=font)
    return np.asarray(img)
