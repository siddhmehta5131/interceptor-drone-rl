"""esc_diagnostic.py  --  characterise the fitted ESC / thrust model (R-1).

The shipped ESC steady-state polynomial

    Omega_ss(c) = b0 + b1*U_bat + b2*sqrt(c) + b3*c + b4*U_bat*sqrt(c)

is monotonically **decreasing** over the whole command range, i.e. more
throttle produces *less* motor speed (see ``BUGS.md`` item 13).  Training inside
the simulator is unaffected, but the mapping will not transfer to real
hardware, and ``PH_C_HOVER`` depends on it.

This script is the promised diagnostic: it prints the speed/thrust curves, the
slope at the hover command and the command that would hover under a corrected
(monotone increasing) fit, so the discrepancy is quantified before anyone
touches ``PH_BAT``.  It is read-only, needs nothing beyond NumPy and never
mutates the constants.

Usage
-----
    python scripts/esc_diagnostic.py                 # text table
    python scripts/esc_diagnostic.py --csv esc.csv   # machine readable
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.physics.constants import (  # noqa: E402
    PH_BAT,
    PH_C_HOVER,
    PH_CUT_THR,
    PH_M,
    PH_OMEGA_HOVER,
)
from src.physics.pipeline import PhysicsParams, esc, prop_ft  # noqa: E402

U_BAT = 4.2


def esc_speed(cmd: np.ndarray, u_bat: float = U_BAT) -> np.ndarray:
    """Steady-state motor speed for a command vector (rad/s)."""
    return esc(np.asarray(cmd, dtype=np.float64), u_bat)[0]


def thrust_n(cmd, u_bat: float = U_BAT) -> np.ndarray:
    """Total vertical thrust (N) for a command vector (all four props)."""
    arr = np.atleast_1d(np.asarray(cmd, dtype=np.float64))
    flat = arr.reshape(-1)
    params = PhysicsParams()
    out = np.empty(flat.shape[0], dtype=np.float64)
    for i, c in enumerate(flat):
        om = esc(np.full(4, float(c)), u_bat)[0]
        f, _ = prop_ft(om, params)
        out[i] = float(f[:, 2].sum())
    return out.reshape(arr.shape)


def slope_at(cmd: float, u_bat: float = U_BAT, h: float = 1e-6) -> float:
    """dOmega_ss/dc at ``cmd`` (rad/s per unit command)."""
    lo = max(0.0, cmd - h)
    hi = cmd + h
    d = esc_speed(np.array([hi]), u_bat)[0] - esc_speed(np.array([lo]), u_bat)[0]
    return float(d / (hi - lo))


def report(rows: int = 21) -> None:
    cmd = np.linspace(0.0, 1.0, rows)
    speed = esc_speed(cmd)
    thrust = thrust_n(cmd)
    weight = PH_M * 9.81

    print("ESC diagnostic (R-1) -- read-only, PH_BAT is NOT modified")
    print(f"  U_bat                       : {U_BAT:.2f} V")
    print(f"  PH_BAT                      : {[float(b) for b in PH_BAT]}")
    print(f"  PH_CUT_THR                  : {PH_CUT_THR}")
    print(f"  mass x g                    : {weight:.6f} N  (PH_M = {PH_M} kg)")
    print(f"  PH_C_HOVER (command)        : {PH_C_HOVER!r}")
    print(f"  PH_OMEGA_HOVER (rad/s)      : {PH_OMEGA_HOVER!r}")
    print()
    print("    cmd     Omega_ss    4-prop thrust    thrust/weight")
    for c, om, f in zip(cmd, speed, thrust):
        print(f"  {c:6.3f}  {om:10.3f}  {f:14.6f}  {f / weight:14.4f}")
    print()
    print(f"  Omega_ss(0.02)              : {esc_speed(np.array([PH_CUT_THR]))[0]:.3f} rad/s")
    print(f"  Omega_ss(PH_C_HOVER)        : {esc_speed(np.array([PH_C_HOVER]))[0]:.3f} rad/s")
    print(f"  Omega_ss(1.00)              : {esc_speed(np.array([1.0]))[0]:.3f} rad/s")
    print(f"  dOmega_ss/dc at hover cmd   : {slope_at(PH_C_HOVER):.3f} rad/s per command")
    print(f"  dOmega_ss/dc at cmd = 1.0   : {slope_at(1.0):.3f} rad/s per command")
    print(f"  hover thrust error          : "
          f"{thrust_n(np.array([PH_C_HOVER]))[0] - weight:+.6e} N")
    print()
    monotone = bool(np.all(np.diff(speed[cmd >= PH_CUT_THR]) <= 0.0))
    print(f"  monotonically decreasing over [cut, 1] : {monotone}")
    print(f"  decreasing as ESC map demands        : "
          f"{'no -- thrust falls with command' if monotone else 'yes'}")
    print()
    print("  Interpretation: a real ESC maps command -> speed monotonically")
    print("  increasing; the hover equilibrium sits at a crossing point instead")
    print(f"  of a rising curve. Any change to PH_BAT moves PH_C_HOVER and")
    print("  therefore breaks the bit-exact Stage-1 parity with hover_env.py --")
    print("  re-identify the ESC on hardware before touching it (BUGS.md item 13).")


def write_csv(path: Path, rows: int = 201) -> None:
    cmd = np.linspace(0.0, 1.0, rows)
    speed = esc_speed(cmd)
    thrust = thrust_n(cmd)
    weight = PH_M * 9.81
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["cmd", "omega_ss_rad_s", "thrust_n", "thrust_over_weight"])
        for c, om, f in zip(cmd, speed, thrust):
            w.writerow([f"{c:.6f}", f"{om:.6f}", f"{f:.9f}", f"{f / weight:.9f}"])
    print(f"wrote {path}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", metavar="PATH", help="also write the sweep as CSV")
    ap.add_argument("--rows", type=int, default=21, help="table resolution")
    args = ap.parse_args(argv)
    report(rows=max(3, args.rows))
    if args.csv:
        write_csv(Path(args.csv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())