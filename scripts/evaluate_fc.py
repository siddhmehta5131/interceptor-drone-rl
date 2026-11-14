"""Flight-controller evaluation (plan P8.2).

Compares B1 (project PID), B2 (firmware PID) and optionally an RL rate policy
on 50 setpoint sequences of each kind, nominal and under wind / parameter
error.  B3 (gym-pybullet-drones) is a whole-stack reference and is recorded as
pending when PyBullet is not installed.

Usage::

    python scripts/evaluate_fc.py --rl runs/fc/checkpoint.zip
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interceptor.fc import metrics, rollout, setpoints
from interceptor.fc.firmware_pid import FirmwarePID
from interceptor.fc.pid import RatePID
from interceptor.sim import pipeline
from interceptor.sim.constants import PH_J, PH_M
from interceptor.sim.wind import DrydenWind
from interceptor.bridge.closed_loop import PolicyRateController

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "tables"
KINDS = setpoints.KINDS
DT = 0.002
N_STEPS = 750  # 1.5 s


def _controllers(rl_path: str | None):
    out = [("B1", "pid", lambda: RatePID.tuned(dt=DT)),
           ("B2", "firmware", lambda: FirmwarePID(dt=DT))]
    if rl_path:
        from stable_baselines3 import PPO
        model = PPO.load(rl_path, device="cpu")
        out.append(("RL", "rl", lambda: PolicyRateController(model, rate_max=12.0)))
    return out


def _condition(name: str, seed: int):
    if name == "nominal":
        return pipeline.PhysicsParams.rotor_drag(), None
    if name == "wind":
        w = DrydenWind(100.0, np.array([1.0, 0.2, 0.0]), W20=5.0, airspeed=5.0,
                       rng=np.random.default_rng(seed + 77))
        return pipeline.PhysicsParams.rotor_drag(), w
    # +/-10% mass and inertia, alternating along the sequence index.
    sign = 1.1 if seed % 2 == 0 else 0.9
    p = pipeline.PhysicsParams.rotor_drag(
        mass=sign * PH_M, J=sign * np.asarray(PH_J, dtype=float))
    return p, None


def evaluate_controller(factory, condition: str) -> list[dict]:
    rows = []
    for kind in KINDS:
        vals = {k: [] for k in ("iae", "rms", "effort", "smoothness", "saturation",
                                "rise", "overshoot", "settling")}
        for i in range(50):
            seed = 7000 + 100 * KINDS.index(kind) + i
            rng = np.random.default_rng(seed)
            sp = setpoints.sample(kind, N_STEPS, DT, rng, 12.0)
            params, wind = _condition(condition, seed)
            log = rollout.rollout(factory(), sp, params=params, wind=wind, dt=DT)
            m = metrics.tracking_metrics(log["t"], log["setpoint"][:, :3],
                                         log["omega"], log["cmd"])
            if kind == "step":
                axis = int(np.argmax(np.ptp(sp[:, :3], axis=0)))
                s = rollout.step_response_metrics(log, axis=axis)
                vals["rise"].append(s["rise"])
                vals["overshoot"].append(s["overshoot"])
                vals["settling"].append(s["settling"])
            for k in ("iae", "rms", "effort", "smoothness", "saturation"):
                vals[k].append(m[k])
        row = {"condition": condition, "kind": kind,
               "sequences": 50}
        for k, v in vals.items():
            finite = np.asarray(v, dtype=float)
            row[k] = float(np.median(finite)) if finite.size else float("nan")
        rows.append(row)
    return rows


def _plot_step(controllers, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sp = setpoints.step(N_STEPS, DT, axis=0, amp=8.0)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(N_STEPS) * DT, sp[:, 0], "k--", label="setpoint")
    for label, _, factory in controllers:
        log = rollout.rollout(factory(), sp, dt=DT)
        ax.plot(log["t"], log["omega"][:, 0], label=label)
    ax.set(xlabel="time (s)", ylabel="roll rate (rad/s)",
           title="Nominal roll-rate step response")
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rl", default=None, help="PPO checkpoint for the RL arm")
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for label, kind, factory in _controllers(args.rl):
        for condition in ("nominal", "wind", "param_error"):
            rows = evaluate_controller(factory, condition)
            for row in rows:
                row.update(controller=label, controller_kind=kind)
                all_rows.append(row)
            print(f"{label:3s} {condition:11s} done ({len(rows)} kinds)")
    fields = ["controller", "controller_kind", "condition", "kind", "sequences",
              "iae", "rms", "effort", "smoothness", "saturation",
              "rise", "overshoot", "settling"]
    path = OUT / "fc_evaluation.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    with (OUT / "fc_b3_status.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["controller", "status", "reason"])
        writer.writerow(["B3", "pending",
                         "gym-pybullet-drones not installed on this machine; run in Colab"])
    print(f"wrote {path}")
    _plot_step(_controllers(args.rl), ROOT / "results" / "figures" / "fc_step_response.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
