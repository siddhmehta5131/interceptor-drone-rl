"""P9.4 residual analysis for one hard manoeuvre per platform (needs an A3 checkpoint).

    python scripts/residual_analysis.py --policy quadcopter=runs/.../checkpoint.zip --maneuver <value>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from interceptor.eval.sweeps import residual_components
from interceptor.guidance.maneuvers import ManeuverType, PlatformType
from interceptor.visualiser import schema

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", action="append", required=True, help="platform=path (arm A3)")
    ap.add_argument("--maneuver", default=None)
    ap.add_argument("--seed", type=int, default=10000)
    args = ap.parse_args(argv)
    from stable_baselines3 import PPO
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    maneuver = ManeuverType(args.maneuver) if args.maneuver else list(ManeuverType)[-1]
    out = ROOT / "results" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for spec in args.policy:
        plat, path = spec.split("=", 1)
        lg = schema.rollout_guidance(PlatformType(plat), "apn_rl", maneuver, args.seed,
                                     policy=PPO.load(path, device="cpu"))
        par, perp = residual_components(lg.interceptor_pos, lg.target_pos, lg.a_apn, lg.a_rl)
        t = lg.t
        fig, ax = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
        ax[0].plot(t, np.linalg.norm(lg.a_apn, axis=1), label="|a_APN|")
        ax[0].plot(t, np.linalg.norm(lg.a_rl, axis=1), label="|a_RL|")
        tv = np.gradient(np.gradient(lg.target_pos, lg.dt, axis=0), lg.dt, axis=0)
        ax[0].plot(t, np.linalg.norm(tv, axis=1), ":", label="|a_target| (finite diff.)")
        ax[0].set_ylabel("acceleration [m/s^2]")
        ax[0].legend()
        ax[1].plot(t, par, label="along LOS")
        ax[1].plot(t, perp, label="across LOS")
        ax[1].set_ylabel("residual component [m/s^2]")
        ax[1].legend()
        ax[2].plot(t, lg.distance())
        ax[2].set(xlabel="time [s]", ylabel="range [m]")
        ax[0].set_title(f"{plat}: {maneuver.value}, seed {args.seed}")
        fig.tight_layout()
        fig.savefig(out / f"residual_{plat}.png", dpi=130)
        plt.close(fig)
        print("wrote", out / f"residual_{plat}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
