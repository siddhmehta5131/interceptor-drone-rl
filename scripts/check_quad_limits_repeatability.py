"""Repeatability of the P4.9 quad envelope (docs/tuned_values_plan.md, family F).

The quad platform limits in ``configs/platforms.yaml`` are measured by
``scripts/measure_quad_limits.py``.  This check reruns the measurement

* with two other sets of procedure settings (step size, smoothing window, hold
  times), to show the envelope is a property of the vehicle model and not of the
  measurement procedure, and
* with the two assumed physics constants varied (motor lag 0.5x / 2x, vertical
  rotor drag 0 / 0.7 1/s), to show how far the envelope moves if those
  assumptions are wrong (BUGS #27).

Writes ``results/tables/quad_limits_repeatability.csv``; the last column is the
largest relative change of any envelope quantity against the baseline.

    python scripts/check_quad_limits_repeatability.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from interceptor.sim import pipeline  # noqa: E402
from interceptor.sim.constants import PH_K_MOT, ROTOR_DRAG_XYZ  # noqa: E402
from measure_quad_limits import measure  # noqa: E402

#: envelope quantities that feed configs/platforms.yaml
KEYS = ("tau_accel_s", "lateral_a_max_measured", "vertical_up_a_max_measured",
        "vertical_down_a_max_measured", "v_max_measured", "peak_rate_hard_attitude_step")

#: name -> (kind, measure() keyword arguments)
PROCEDURE = {
    "baseline": {},
    "settings_A": dict(a_step=3.0, smooth_w=15, lat_T=1.0, lat_step=0.5, v_hold_T=30.0,
                       v_hold_frac=0.85, peak_window=150),
    "settings_B": dict(a_step=5.0, smooth_w=35, lat_T=1.5, lat_step=2.0, v_hold_T=50.0,
                       v_hold_frac=0.95, peak_window=250),
}


def _drag(kz: float):
    d = np.array(ROTOR_DRAG_XYZ, dtype=np.float64)
    d[2] = kz
    return pipeline.PhysicsParams.rotor_drag(drag_xyz=d)


def _with_motor_lag(scale: float):
    """Measure with the plant's motor lag scaled; the controller design is unchanged."""
    old = pipeline.PH_K_MOT
    pipeline.PH_K_MOT = PH_K_MOT * scale
    try:
        return measure()
    finally:
        pipeline.PH_K_MOT = old


def main() -> int:
    rows = []
    for name, kw in PROCEDURE.items():
        rows.append(("procedure", name, measure(**kw)))
    for scale in (0.5, 2.0):
        rows.append(("physics", f"motor_lag_x{scale:g}", _with_motor_lag(scale)))
    for kz in (0.0, 0.7):
        rows.append(("physics", f"kz_{kz:g}", measure(params=_drag(kz))))

    base = rows[0][2]
    out = ROOT / "results" / "tables" / "quad_limits_repeatability.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["kind", "case", *KEYS, "max_rel_change_pct"])
        for kind, name, r in rows:
            rel = max(abs(r[k] - base[k]) / abs(base[k]) for k in KEYS) * 100.0
            w.writerow([kind, name, *(f"{r[k]:.4f}" for k in KEYS), f"{rel:.1f}"])
            print(f"{kind:9s} {name:16s} " + " ".join(f"{r[k]:8.3f}" for k in KEYS) + f"  max change {rel:5.1f} %")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
