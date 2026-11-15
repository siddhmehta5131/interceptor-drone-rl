"""Build docs/decision_register.csv: every sourced config value plus every
modelling decision, with a status saying whether an external citation exists.

Run: python scripts/build_decision_register.py
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "decision_register.csv"

EXTERNAL = re.compile(
    r"Forster|NanoBench|Bitcraze|Beard|McLain|Zipfel|CADAC|MIL-HDBK|MIL-F|Schulman|Raffin|"
    r"Wilson|Zarchan|Fujimoto|Haarnoja|mavsim|Crazyflie|Raymer|Anderson|Stevens|"
    r"Bar-Shalom|Shneydor|Siouris|Stable-Baselines", re.I)
INTERNAL = re.compile(
    r"Q5 baseline|notebook v2|estimate|analytic|decision S|decision Q|"
    r"implementation_plan|legacy|hand-set|unsourced|measured P4\.9|replaced", re.I)


def classify(src: str) -> str:
    ext, internal = bool(EXTERNAL.search(src)), bool(INTERNAL.search(src))
    if ext and not internal:
        return "cited"
    if ext and internal:
        return "partly_cited"
    if internal or not src:
        return "NEEDS_CITATION"
    return "derived_or_internal"  # e.g. an EQ-xxx derivation; check mathematics.md


def walk(node, path, rows, fname):
    if isinstance(node, dict):
        if "value" in node and "source" in node:
            rows.append((fname, ".".join(path), node["value"], node.get("unit", ""), str(node["source"])))
            return
        for k, v in node.items():
            walk(v, path + [str(k)], rows, fname)


# category, item, choice, current_source, status, needs_external_data, ref_key, notes
DECISIONS = [
    ("simulator", "Quadrotor reference vehicle", "Crazyflie 2.x (40.85 g NanoBench set)", "S2/Q2 plan; Bitcraze, NanoBench", "partly_cited", "Crazyflie datasheet; NanoBench paper", "bitcraze_cf21; nanobench", "Confirm mass and thrust limits from primary datasheet"),
    ("simulator", "6-DOF rigid-body dynamics (Newton-Euler)", "body-frame rotation, quaternion attitude", "mathematics.md s6", "derived_or_internal", "Textbook citation", "stevens_lewis; beard_mclain", "Cite standard derivation"),
    ("simulator", "Integrator", "RK4 fixed step; 1000 Hz physics, 500 Hz FC, 50 Hz guidance", "S8", "NEEDS_CITATION", "Numerical-methods reference; own convergence plot exists", "butcher2016", "Convergence ladder is own evidence"),
    ("simulator", "Attitude representation", "Unit quaternion with renormalisation", "mathematics.md s6.3", "derived_or_internal", "Quaternion kinematics reference", "sola2017", ""),
    ("simulator", "Rotor thrust/torque model", "T=c_l w^2 plus aero polynomial; linear rotor-drag fallback", "constants.py; BUGS 11", "NEEDS_CITATION", "Rotor aerodynamics references", "forster2015; powers2013", "Aero polynomial signs hand flipped: unvalidated"),
    ("simulator", "Motor first-order lag", "tau = 20 ms", "PROJECT_CHOICE (assumed; not in NanoBench, BUGS #27)", "derived_or_internal", "Motor step-response measurement", "", "Defended by the motor-lag x0.5 / x2 rows of results/tables/quad_limits_repeatability.csv (4.1 % / 5.9 % largest envelope change)"),
    ("simulator", "ESC throttle-to-speed map", "two-point sqrt fit", "BUGS R-1", "NEEDS_CITATION", "NanoBench thrust-stand data", "nanobench", "Pending data fit P3.4"),
    ("simulator", "Wind model", "Dryden turbulence", "mathematics.md 5A.3", "partly_cited", "MIL-HDBK-1797 / MIL-F-8785C", "milhdbk1797", "Verify scale lengths and sigmas"),
    ("simulator", "Validation reference simulator", "gym-pybullet-drones", "P3.7", "NEEDS_CITATION", "Panerati et al. 2021", "panerati2021", "Not yet run"),
    ("guidance", "Law", "Augmented proportional navigation (APN)", "mathematics.md s1", "derived_or_internal", "Zarchan; Shneydor", "zarchan; shneydor1998", ""),
    ("guidance", "Navigation constant N", "swept in P5.11 (nav_sweep.csv)", "scripts/sweep_nav.py", "derived_or_internal", "Literature range 3-5", "zarchan", "Empirical sweep plus literature range"),
    ("guidance", "Seeker filter", "Constant-acceleration Kalman tracker", "P5.4, Q7", "NEEDS_CITATION", "Kalman 1960; Bar-Shalom", "kalman1960; barshalom2001", ""),
    ("guidance", "Seeker noise levels", "configs/seeker.yaml: angular 8.727 mrad (EchoShield <0.5 deg used as 1-sigma); other terms PROJECT_CHOICE", "seeker.yaml; Echodyne statement hosted by US ITA", "partly_cited", "A sensor datasheet stating a 1-sigma error model", "echodyne_ita", "Swept x0-4 in P9.1"),
    ("guidance", "Kill radius per platform", "1.30 m / 3.00 m / 4.98 m (hit-to-kill contact rule)", "platforms.yaml, Q6", "NEEDS_CITATION", "Warhead lethal radius data", "", ""),
    ("guidance", "Platform limits (v, a, g, tau)", "per platform", "platforms.yaml", "NEEDS_CITATION", "Real vehicle specs: see data CSVs", "data/*.csv", "Core task: replace Q5 baseline by real vehicles"),
    ("guidance", "Target speed and acceleration ratios", "1.51/1.06/1.5 speed; 2.08/1/4.06 accel", "quad 280/185 (Sting product page / Shahed-136 IRGC poster via Janes; T11, I11); fixed-wing 32.4/30.6 (Aerosonde model v_max / Orlan-10 cruise, T16); missile accel ratio from 9 g fighter limit (T02-T04); missile speed ratio and quad accel ratio are Q5 project choices", "partly_cited", "Manufacturer datasheets for Orlan-10 and Shahed-136; g limit of one-way attack drones", "data/*.csv", ""),
    ("guidance", "Evasive manoeuvre library (16 laws)", "maneuvers.yaml", "notebook v2", "NEEDS_CITATION", "Evasion literature", "shaw1985", ""),
    ("guidance", "Target max g by platform", "0.73 g / 3.8 g / 9 g", "Q5; missile-class target 9 g from F-16C/J-20 limits (data/threat_platforms.csv T03 T04, secondary)", "NEEDS_CITATION", "Drone g limits not found; primary source for fighter g", "data/threat_platforms.csv", ""),
    ("guidance", "Fixed-wing reference", "Aerosonde UAV", "S9; Beard & McLain App. E", "cited", "", "beard_mclain", "Check against real attack-drone data"),
    ("guidance", "Missile reference", "AIM5 generic (Zipfel CADAC)", "S9", "cited", "", "zipfel2014", "Generic, not a fielded missile"),
    ("guidance", "Fixed-wing structural load factor", "+3.8 g / -1.52 g (14 CFR 23.337 normal category)", "implementation_plan Q11", "NEEDS_CITATION", "UAS load-factor standard", "", ""),
    ("guidance", "Missile flight condition", "Mach 2 at 10 km, post burnout", "Q12", "NEEDS_CITATION", "Missile performance references", "", ""),
    ("rl", "Algorithm choice", "PPO (provisional) from PPO/SAC/TD3 comparison", "docs/algorithm_selection.md", "NEEDS_CITATION", "Schulman 2017; Haarnoja 2018; Fujimoto 2018", "schulman2017; haarnoja2018; fujimoto2018", "Final decision pending 18-run experiment"),
    ("rl", "Hyperparameters", "SB3 2.9.0 library defaults = PPO / SAC / TD3 paper tables; TD3 lr 3e-4 and network [256, 256, 128] are project choices", "configs/training.yaml: Schulman 2017 App. A Table 3; Haarnoja 2018 App. D Table 1; Fujimoto 2018 Sec. 6.1; sfujim/TD3; SB3 2.9.0 source", "cited", "", "ppo; sac; td3; sb3", "Each value cites its table/section and SB3 file:line; tests/unit/test_algos.py checks them against the installed SB3"),
    ("rl", "Discount gamma", "0.99791", "P5.9", "derived_or_internal", "Sutton and Barto", "sutton_barto2018", ""),
    ("rl", "Potential-based reward shaping", "progress term", "P5.9, EQ-GUIDE-1", "NEEDS_CITATION", "Ng, Harada, Russell 1999", "ng1999", ""),
    ("rl", "Reward weights", "guidance.yaml reward block", "P5.9", "NEEDS_CITATION", "Document sweeps", "", ""),
    ("rl", "Asymmetric actor-critic", "privileged critic", "P7", "NEEDS_CITATION", "Pinto et al. 2018", "pinto2018", ""),
    ("rl", "Residual RL authority", "0.35 of a_n_max", "F5/P9.3", "NEEDS_CITATION", "Johannink 2019", "johannink2019", "Ablated in P9.3"),
    ("rl", "Seeds and episodes per cell", "training.yaml", "P6.3", "derived_or_internal", "Henderson 2018; Agarwal 2021", "henderson2018; agarwal2021", ""),
    ("fc", "Flight controller interface", "thrust + body rates -> 4 motor commands", "S1", "NEEDS_CITATION", "PX4 / Crazyflie firmware / Kaufmann", "kaufmann2022", ""),
    ("fc", "Baselines", "B1 PID, B2 firmware PID, B3 gym-pybullet-drones", "P4", "NEEDS_CITATION", "Panerati 2021; Bitcraze firmware", "panerati2021", ""),
    ("eval", "Statistics", "Wilson CI, McNemar, paired bootstrap", "mathematics.md s11", "NEEDS_CITATION", "Wilson 1927; McNemar 1947; Efron 1979", "wilson1927; mcnemar1947; efron1979", ""),
    ("data", "Threat set: fighters, missiles, attack drones near India", "to be collected", "professor request", "NEEDS_CITATION", "data/threat_platforms.csv", "", ""),
    ("data", "Indian interceptors, SAMs, counter-drone", "to be collected", "professor request", "NEEDS_CITATION", "data/interceptor_platforms.csv", "", ""),
]


def main() -> None:
    rows = []
    for f in sorted((ROOT / "configs").rglob("*.yaml")):
        data = yaml.safe_load(f.read_text(encoding="utf8"))
        walk(data, [], rows, f.relative_to(ROOT).as_posix())
    with OUT.open("w", newline="", encoding="utf8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "category", "item", "value", "unit", "current_source", "status",
                    "needs_external_data", "proposed_reference_key", "notes"])
        for i, (fname, p, v, u, s) in enumerate(rows, 1):
            w.writerow([f"C{i:04d}", "config_value", f"{fname}::{p}", v, u, s, classify(s), "", "", ""])
        for j, d in enumerate(DECISIONS, 1):
            w.writerow([f"D{j:03d}", d[0], d[1], d[2], "", d[3], d[4], d[5], d[6], d[7]])
    print(f"wrote {OUT}: {len(rows)} config rows, {len(DECISIONS)} decisions")


if __name__ == "__main__":
    main()
