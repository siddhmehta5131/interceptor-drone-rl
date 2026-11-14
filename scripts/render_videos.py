"""Render the delivered video set (P10.2 - P10.7).

    python scripts/render_videos.py --out results/videos            # 1280x720
    python scripts/render_videos.py --resolution 1920x1080          # Colab

Per platform: one hard manoeuvre, A2 (apn_filtered) beside A3 (apn_rl, needs a
checkpoint, otherwise A2 beside A1/raw APN and the label says so); one
flight-controller clip (B1 beside B2, RL if a checkpoint is given); one
closed-loop clip.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from interceptor.guidance.maneuvers import ManeuverType, PlatformType
from interceptor.visualiser import render, schema


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/videos")
    ap.add_argument("--resolution", default="1280x720")
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--maneuver", default=None, help="ManeuverType value; default: first")
    ap.add_argument("--policy", action="append", default=[],
                    help="platform=path to an apn_rl checkpoint (arm A3)")
    ap.add_argument("--fc-policy", default=None, help="RL rate controller checkpoint")
    ap.add_argument("--only", choices=["guidance", "fc", "closed_loop"], default=None)
    args = ap.parse_args(argv)
    res = tuple(int(v) for v in args.resolution.lower().split("x"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    policies = dict(p.split("=", 1) for p in args.policy)
    maneuver = ManeuverType(args.maneuver) if args.maneuver else list(ManeuverType)[1]

    if args.only in (None, "guidance"):
        for platform in PlatformType:
            clips, labels = [], []
            for label, mode, pol in (("A2 filtered APN", "apn_filtered", None),
                                     ("A3 APN+RL", "apn_rl", policies.get(platform.value))):
                if mode == "apn_rl" and pol is None:
                    label, mode = "A1 raw APN (no A3 checkpoint)", "apn_raw"
                model = None
                if pol:
                    from stable_baselines3 import PPO
                    model = PPO.load(pol, device="cpu")
                lg = schema.rollout_guidance(platform, mode, maneuver, args.seed, policy=model)
                lg.save(out / f"{platform.value}_{mode}.npz")
                p = out / f"_{platform.value}_{mode}.mp4"
                render.render_log(lg, p, resolution=res, title=label)
                clips.append(p)
                labels.append(f"{label}  kill={lg.meta['killed']}  miss={lg.meta['miss_distance']:.2f} m")
            render.side_by_side(clips, out / f"guidance_{platform.value}.mp4", labels)
            for c in clips:
                c.unlink()
            print("wrote", out / f"guidance_{platform.value}.mp4")

    if args.only in (None, "fc"):
        from interceptor.fc import setpoints
        from interceptor.fc.firmware_pid import FirmwarePID
        from interceptor.fc.pid import RatePID
        from interceptor.fc.rollout import rollout
        sp = setpoints.doublet(1500, 0.002, 0, 3.0)
        clips = []
        for label, ctl in (("B1 PID", RatePID.tuned(dt=0.002)), ("B2 firmware PID", FirmwarePID(dt=0.002))):
            lg = schema.from_fc_rollout(rollout(ctl, sp), label)
            p = out / f"_fc_{label[:2]}.mp4"
            render.render_fc(lg, p, resolution=(res[0] // 2, res[1]), title=label)
            clips.append(p)
        render.side_by_side(clips, out / "fc_tracking.mp4", ["B1 PID", "B2 firmware PID"])
        for c in clips:
            c.unlink()
        print("wrote", out / "fc_tracking.mp4")

    if args.only in (None, "closed_loop"):
        from interceptor.bridge.closed_loop import run_closed_loop
        res_cl = run_closed_loop(mode="apn_filtered", seed=args.seed, inner="b1",
                                 fixed_maneuver=maneuver)
        lg = schema.from_closed_loop(res_cl, label="closed loop A2 + B1")
        lg.meta["platform"] = "quadcopter"
        lg.save(out / "closed_loop.npz")
        render.render_log(lg, out / "closed_loop.mp4", resolution=res, title="closed loop A2 + B1")
        print("wrote", out / "closed_loop.mp4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
