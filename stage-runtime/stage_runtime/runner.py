"""Stage runner: fly one stage, render it, write the video + a JSON sidecar.

One invocation == one stage JSON == one MP4 whose segments are concatenated.
The physics runs at 100 Hz (``PH_DT``); frames are grabbed every
``frame_stride`` steps so a ``segment_seconds`` slice of simulation becomes
exactly ``segment_seconds`` of video at the configured fps.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.envs.base_env import InterceptorBaseEnv
from src.physics.constants import PH_DT

from .controllers import build_controller, resolve_observation
from .panels import FrameData, StagePanel
from .recorder import FrameRecorder, RecorderError, ffmpeg_available, probe_video
from .stage_config import StageDemoConfig, StageDemoError, load_stage_config


class RunError(RuntimeError):
    pass


def frame_stride(fps: int, dt: float = float(PH_DT)) -> int:
    """Physics steps per rendered frame (``1`` = every step, slow motion)."""
    return max(1, int(round(1.0 / (int(fps) * float(dt)))))


def build_env(cfg: StageDemoConfig, *, seed: int) -> InterceptorBaseEnv:
    """One env instance for a stage, built from the JSON + curriculum table."""
    kwargs = cfg.env_kwargs()
    try:
        return InterceptorBaseEnv(cfg.stage_config(), seed=seed, **kwargs)
    except TypeError as exc:                     # unexpected kwarg -> explain
        raise RunError(f"{cfg.path.name}: env rejected the JSON options: {exc}") from exc


def _frame_from(cfg, tel: Dict[str, Any], action: np.ndarray, reward: float,
                t: float, episode: int, obs_dim: int, killed: bool,
                controller_desc: str) -> FrameData:
    extra = {
        "throttle_cut": float(np.min(np.asarray(tel["cmd"]))) <= 0.0,
    }
    return FrameData(
        t=float(t),
        p=np.asarray(tel["p_WB"], dtype=np.float64),
        v=np.asarray(tel["v_WB"], dtype=np.float64),
        R=np.asarray(tel["R_WB"], dtype=np.float64),
        Omega=np.asarray(tel["Omega"], dtype=np.float64),
        cmd=np.asarray(tel["cmd"], dtype=np.float64),
        U_bat=float(tel["U_bat"]),
        tilt=float(tel["tilt"]),
        distance=float(tel["distance"]),
        action=np.asarray(action, dtype=np.float64),
        stage=cfg.stage,
        control=cfg.control,
        controller=controller_desc,
        reward=float(reward),
        episode=int(episode),
        target_pos=(np.asarray(tel["target_pos"], dtype=np.float64)
                    if tel.get("include_target") else None),
        include_target=bool(tel.get("include_target")),
        kills_enabled=bool(tel.get("kills_enabled")),
        kill_radius=float(tel.get("kill_radius", 0.0)),
        killed=bool(killed),
        obs_dim=int(obs_dim),
        extra=extra,
    )


def run_stage(cfg: StageDemoConfig, *, out_dir: Path, data_roots=(),
              render: bool = True, write_sidecar: bool = True,
              verbose: bool = True) -> Dict[str, Any]:
    """Fly one stage and return its report dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- control source -------------------------------------------------
    model_path = None
    if cfg.control == "model":
        model_path = cfg.model.resolve(extra_roots=data_roots)
        cfg, ckpt = resolve_observation(cfg, model_path)
    else:
        ckpt = None

    controller = build_controller(cfg, model_path=model_path, data_roots=data_roots)
    controller_desc = (controller.describe()
                       if hasattr(controller, "describe") else cfg.control)

    # -- video sink -----------------------------------------------------
    video_path = cfg.video_path(out_dir)
    stride = frame_stride(cfg.video.fps)
    steps_per_segment = cfg.steps_per_segment
    recorder = None
    panel = None
    if render:
        if cfg.video.fmt == "mp4" and not ffmpeg_available():
            raise RunError(
                "MP4 output requires imageio-ffmpeg. Install it "
                "(pip install imageio imageio-ffmpeg) or render with "
                "\"format\": \"gif\"/\"png\"."
            )
        panel = StagePanel(cfg, dpi=cfg.video.dpi, trail=cfg.video.trail)
        recorder = FrameRecorder(video_path, fmt=cfg.video.fmt, fps=cfg.video.fps,
                                 dpi=cfg.video.dpi, bitrate=cfg.video.bitrate)

    episodes: List[Dict[str, Any]] = []
    frames = 0
    t0 = time.time()
    env = None
    try:
        for seg in range(cfg.segments):
            seed = cfg.seed + 1000 * seg
            if env is None or cfg.reset_each_segment:
                env = build_env(cfg, seed=seed)
                obs, _info = env.reset(seed=seed)
            else:
                obs, _info = env.reset(seed=seed)
            controller.reset()
            tel = env.telemetry()
            obs_dim = int(np.asarray(obs).shape[-1])

            seg_reward = 0.0
            seg_min_dist = tel["distance"] if tel.get("include_target") else float("nan")
            seg_max_tilt = 0.0
            killed = False
            for step in range(1, steps_per_segment + 1):
                action = controller.act(tel, float(PH_DT), obs)
                obs, reward, terminated, truncated, info = env.step(action)
                tel = env.telemetry()
                killed = killed or bool(info.get("killed", False))
                seg_reward += float(reward)
                seg_max_tilt = max(seg_max_tilt, float(tel["tilt"]))
                if tel.get("include_target"):
                    seg_min_dist = min(seg_min_dist, float(tel["distance"]))
                if recorder is not None and panel is not None and (
                        step % stride == 0 or step == steps_per_segment):
                    panel.update(_frame_from(cfg, tel, action, reward,
                                             step * float(PH_DT), seg + 1,
                                             obs_dim, killed, controller_desc))
                    recorder.grab(panel.fig)
                    frames += 1
                if terminated or truncated:
                    break

            episodes.append({
                "segment": seg + 1,
                "seed": seed,
                "steps": step,
                "reward_sum": round(seg_reward, 6),
                "min_distance_m": (None if seg_min_dist != seg_min_dist
                                   else round(float(seg_min_dist), 4)),
                "max_tilt_deg": round(float(np.degrees(seg_max_tilt)), 3),
                "killed": bool(killed),
            })
            if verbose:
                print(f"[stage {cfg.stage}] segment {seg + 1}/{cfg.segments}: "
                      f"{step} steps, reward {seg_reward:+.1f}, "
                      f"tilt {np.degrees(seg_max_tilt):.1f} deg, "
                      f"killed={killed}")
            if terminated or truncated:
                if verbose:
                    print(f"[stage {cfg.stage}] episode ended early at step {step}")
    finally:
        if recorder is not None:
            recorder.close()
        if panel is not None:
            panel.close()
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    report: Dict[str, Any] = {
        "stage": cfg.stage,
        "title": cfg.title,
        "control": cfg.control,
        "controller": controller_desc,
        "config_file": str(cfg.path),
        "seed": cfg.seed,
        "segments": cfg.segments,
        "segment_seconds": cfg.segment_seconds,
        "total_sim_seconds": round(cfg.total_seconds, 3),
        "video": {
            "path": str(video_path),
            "format": cfg.video.fmt,
            "fps": cfg.video.fps,
            "frame_stride": stride,
            "frames": frames,
            "duration_s": round(frames / float(cfg.video.fps), 3) if frames else 0.0,
        },
        "checkpoint": (str(model_path) if model_path else None),
        "checkpoint_meta": (None if ckpt is None else {
            "algo": ckpt.algo,
            "from_sidecar": ckpt.from_sidecar,
            "history_frames": ckpt.history_frames,
            "history_skip": ckpt.history_skip,
            "privileged_fields": list(ckpt.privileged_fields),
        }),
        "observation": cfg.observation.env_kwargs(),
        "episodes": episodes,
        "wall_seconds": round(time.time() - t0, 3),
        "python": sys.version.split()[0],
        "platform": platform.platform(terse=True),
    }
    if render and cfg.video.fmt != "png":
            probed = probe_video(video_path)
            report["video"].update({k: probed[k] for k in
                                    ("bytes", "codec", "width", "height", "frames")
                                    if k in probed})
    if write_sidecar and cfg.video.sidecar:
        side = cfg.sidecar_path(out_dir)
        side.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["sidecar"] = str(side)
    return report


def render_config(path: Path, out_dir: Path, *, data_roots=(), render: bool = True,
                  verbose: bool = True) -> Dict[str, Any]:
    cfg = load_stage_config(path)
    try:
        return run_stage(cfg, out_dir=out_dir, data_roots=data_roots,
                         render=render, verbose=verbose)
    except (StageDemoError, RecorderError, RunError) as exc:
        raise RunError(f"{Path(path).name}: {exc}") from exc


__all__ = ["RunError", "build_env", "frame_stride", "render_config", "run_stage"]