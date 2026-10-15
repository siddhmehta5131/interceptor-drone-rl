"""Scripted physics-simulator manoeuvre demo (RK4 env + fixed guidance law).

Generates ``final_videos/physics_simulator_maneuver.mp4``: one continuous
flight through a fixed script

    up -> left -> right -> front -> back -> spiral out -> circle -> spiral in -> down

Physics: the training environment (``InterceptorBaseEnv``), i.e. the same
``command_from_action`` -> PID rate control -> RK4 integration pipeline the RL
stages use.  Control: the model-free ``GuidanceLaw`` from the stage runtime
with a time-parameterised goal schedule (position + velocity feed-forward);
the goal velocity also supplies the heading for the spiral/circle legs so the
nose leads the turn.

Usage:  python maneuver_video.py            render the mp4
        python maneuver_video.py --dry-run  simulate only, report tracking error
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

DRY = "--dry-run" in sys.argv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "interceptor-training"))
sys.path.insert(0, str(ROOT / "stage-runtime"))

from src.physics.constants import PH_DT                                    # noqa: E402
from stage_runtime.instructions import DEFAULT_GAINS, GuidanceLaw          # noqa: E402
from stage_runtime.panels import StagePanel                                # noqa: E402
from stage_runtime.recorder import FrameRecorder, ffmpeg_available, probe_video  # noqa: E402
from stage_runtime.runner import _frame_from, build_env, frame_stride      # noqa: E402
from stage_runtime.stage_config import load_stage_config                   # noqa: E402

CFG_PATH = ROOT / "stage-runtime" / "configs" / "stage" / "maneuver_demo.json"
OUT_DIR = ROOT / "final_videos"
VIDEO_NAME = "physics_simulator_maneuver"

SPAWN_ALT = 5.0          # stage 1 spawn altitude [m]
HUB_Z = 8.0              # cruising altitude for the manoeuvres [m]
CARD = 3.0               # cardinal translation leg length [m]
SPIRAL_R0, SPIRAL_R1 = 1.0, 3.0     # spiral radii [m]
OMEGA = 0.8              # spiral / circle yaw rate [rad/s]
TURNS_OUT = 1.5
TURNS_CIRCLE = 1.0
TURNS_IN = 1.5
LEG_S = 2.5              # seconds per cardinal out/back leg
CLIMB_S = 3.0            # seconds for climb / descent


# --------------------------------------------------------------------- goals
def _smooth(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def _smooth_dot(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return 6.0 * u * (1.0 - u)


@dataclass
class Phase:
    name: str
    dur: float
    fn: Callable[[float], Tuple[np.ndarray, np.ndarray, Optional[float]]]
    t0: float = 0.0
    t1: float = 0.0


def move(label: str, p0, p1, dur: float) -> Phase:
    """Smooth-step translation from ``p0`` to ``p1`` (zero speed at both ends)."""
    a = np.asarray(p0, dtype=np.float64)
    d = np.asarray(p1, dtype=np.float64) - a

    def fn(tau: float):
        u = tau / dur if dur > 0 else 1.0
        gp = a + d * _smooth(u)
        gv = d * (_smooth_dot(u) / dur if dur > 0 else 0.0)
        return gp, gv, None                       # heading = toward the goal

    return Phase(label, dur, fn)


def spiral(label: str, r0: float, r1: float, z: float, turns: float,
           phase0: float) -> Phase:
    """Archimedean spiral at constant angular rate; nose points along travel."""
    dur = turns * 2.0 * np.pi / OMEGA
    dr = (r1 - r0) / dur

    def fn(tau: float):
        th = phase0 + OMEGA * tau
        r = r0 + dr * tau
        c, s = np.cos(th), np.sin(th)
        gp = np.array([r * c, r * s, z])
        gv = np.array([dr * c - r * OMEGA * s,
                       dr * s + r * OMEGA * c, 0.0])
        heading = float(np.arctan2(gv[1], gv[0])) if np.hypot(*gv[:2]) > 1e-3 else None
        return gp, gv, heading

    return Phase(label, dur, fn)


def build_schedule() -> List[Phase]:
    hub = np.array([0.0, 0.0, HUB_Z])
    start = np.array([0.0, 0.0, SPAWN_ALT])
    phases: List[Phase] = [move("up", start, hub, CLIMB_S)]
    for label, off in (("left", [0.0, -CARD, 0.0]),
                       ("right", [0.0, CARD, 0.0]),
                       ("front", [CARD, 0.0, 0.0]),
                       ("back", [-CARD, 0.0, 0.0])):
        phases.append(move(label, hub, hub + np.asarray(off), LEG_S))
        phases.append(move(label, hub + np.asarray(off), hub, LEG_S))

    # angles stay continuous: 0 -> 3pi -> 5pi -> 8pi (== 0 mod 2pi)
    a0 = 0.0
    phases.append(spiral("spiral out", SPIRAL_R0, SPIRAL_R1, HUB_Z,
                         TURNS_OUT, a0))
    a0 += TURNS_OUT * 2.0 * np.pi
    phases.append(spiral("circle", SPIRAL_R1, SPIRAL_R1, HUB_Z,
                         TURNS_CIRCLE, a0))
    a0 += TURNS_CIRCLE * 2.0 * np.pi
    pin = spiral("spiral in", SPIRAL_R1, SPIRAL_R0, HUB_Z, TURNS_IN, a0)
    phases.append(pin)
    a0 += TURNS_IN * 2.0 * np.pi

    end_pt = np.array([SPIRAL_R0 * np.cos(a0), SPIRAL_R0 * np.sin(a0), HUB_Z])
    phases.append(move("down", end_pt, start, CLIMB_S))

    t = 0.0
    for ph in phases:
        ph.t0, ph.t1 = t, t + ph.dur
        t += ph.dur
    return phases


class ScheduledGuidance(GuidanceLaw):
    """``GuidanceLaw`` whose goal comes from the manoeuvre schedule."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self._goal = (np.zeros(3), np.zeros(3))

    def set_goal(self, gp, gv) -> None:
        self._goal = (np.asarray(gp, dtype=np.float64),
                      np.asarray(gv, dtype=np.float64))

    def goal(self, tel):
        return self._goal


def main() -> int:
    if not ffmpeg_available():
        print("imageio-ffmpeg is required for mp4 output")
        return 1

    phases = build_schedule()
    total = phases[-1].t1
    print(f"schedule: {len(phases)} legs, {total:.2f} s")

    cfg = load_stage_config(CFG_PATH)
    cfg = replace(cfg, segment_seconds=round(total, 2), segments=1)

    seed = int(cfg.seed)
    env = build_env(cfg, seed=seed)
    obs, _info = env.reset(seed=seed)
    tel = env.telemetry()
    obs_dim = int(np.asarray(obs).shape[-1])

    gains = dict(DEFAULT_GAINS)
    gains.update({k: float(v) for k, v in cfg.instructions.gains.items()})
    law = ScheduledGuidance(
        gains=gains,
        max_tilt_deg=float(cfg.instructions.max_tilt_deg),
        max_speed=float(cfg.instructions.max_speed),
        target="hover",
        lead_time_s=float(cfg.instructions.lead_time_s),
        max_stick=float(cfg.instructions.max_stick),
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    video_path = OUT_DIR / f"{VIDEO_NAME}.{cfg.video.fmt}"
    stride = frame_stride(cfg.video.fps)
    panel = StagePanel(cfg, dpi=cfg.video.dpi, trail=cfg.video.trail) \
        if not DRY else None
    rec = FrameRecorder(video_path, fmt=cfg.video.fmt, fps=cfg.video.fps,
                        dpi=cfg.video.dpi, bitrate=cfg.video.bitrate) \
        if not DRY else None

    steps = int(round(total / float(PH_DT)))
    frames = 0
    max_tilt = 0.0
    terminated = truncated = False
    reason = ""
    phase_seen: List[dict] = []
    track: dict = {}            # phase -> [n, sum_err, max_err]
    current = None
    t0 = time.time()
    try:
        for step in range(1, steps + 1):
            t = step * float(PH_DT)
            ph = next((p for p in phases if p.t0 < t <= p.t1), phases[-1])
            if ph is not current:
                current = ph
                phase_seen.append({"phase": ph.name, "t0": round(ph.t0, 3),
                                   "t1": round(ph.t1, 3)})
                print(f"  t={ph.t0:6.2f}s  {ph.name}")
            gp, gv, heading = ph.fn(min(t, ph.t1) - ph.t0)
            law.set_goal(gp, gv)

            tel_c = dict(tel)
            if heading is not None:      # steer the nose along the travel vector
                p = np.asarray(tel["p_WB"], dtype=np.float64)
                tel_c["target_pos"] = p + 0.05 * np.array(
                    [np.cos(heading), np.sin(heading), 0.0])
            action = law.action(tel_c)

            obs, reward, terminated, truncated, info = env.step(action)
            tel = env.telemetry()
            tilt = float(tel["tilt"])
            max_tilt = max(max_tilt, tilt)

            err = float(np.linalg.norm(np.asarray(tel["p_WB"], dtype=np.float64)
                                       - gp))
            s = track.setdefault(ph.name, [0, 0.0, 0.0])
            s[0] += 1
            s[1] += err
            s[2] = max(s[2], err)

            if not DRY and (step % stride == 0 or terminated or truncated):
                fr = _frame_from(cfg, tel, action, float(reward), t, 1,
                                 obs_dim, False,
                                 f"guidance | {ph.name}")
                fr.extra["phase"] = ph.name
                panel.update(fr)
                rec.grab(panel.fig)
                frames += 1

            if terminated or truncated:
                reason = ",".join(k for k, v in dict(info).items() if v) \
                    if isinstance(info, dict) else ""
                print(f"  !! episode ended at step {step} "
                      f"(t={t:.2f}s) terminated={terminated} "
                      f"truncated={truncated} info={reason}")
                break
    finally:
        if rec is not None:
            rec.close()
        if panel is not None:
            panel.close()
        try:
            env.close()
        except Exception:
            pass

    wall = time.time() - t0
    if DRY:
        print(f"\ndry-run: steps={step} wall={wall:.1f}s "
              f"max_tilt={np.degrees(max_tilt):.1f}deg ended="
              f"{bool(terminated or truncated)}")
        for name, (n, s, mx) in track.items():
            print(f"  {name:12s} mean={s / n:.3f} m  max={mx:.3f} m")
        return 0 if not (terminated or truncated) else 2

    wall = time.time() - t0
    duration = frames / float(cfg.video.fps)
    report = {
        "title": cfg.title,
        "control": "instructions (scheduled guidance, model-free)",
        "physics": "interceptor-training InterceptorBaseEnv (command_from_action "
                   "-> PID rate control -> RK4 step, 100 Hz)",
        "config_file": str(CFG_PATH),
        "seed": seed,
        "phases": phase_seen,
        "scheduled_seconds": round(total, 3),
        "steps": min(steps, step),
        "terminated_early": bool(terminated or truncated),
        "termination_reason": reason,
        "max_tilt_deg": round(float(np.degrees(max_tilt)), 3),
        "final_position_m": [round(float(x), 4) for x in tel["p_WB"]],
        "video": {
            "path": str(video_path),
            "format": cfg.video.fmt,
            "fps": cfg.video.fps,
            "frame_stride": stride,
            "frames": frames,
            "duration_s": round(duration, 3),
        },
        "wall_seconds": round(wall, 3),
        "python": sys.version.split()[0],
    }
    report["video"].update({k: v for k, v in probe_video(video_path).items()
                            if k in ("bytes", "codec", "width", "height", "frames")})
    side = OUT_DIR / f"{VIDEO_NAME}.json"
    side.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report["video"], indent=2))
    print(f"max tilt {report['max_tilt_deg']} deg | "
          f"ended early: {report['terminated_early']} | "
          f"final pos {report['final_position_m']} | wall {wall:.1f}s")
    return 0 if not report["terminated_early"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
