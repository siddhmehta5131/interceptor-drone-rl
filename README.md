# RL-Based Autonomous Interceptor Drone — Flight Simulator

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![NumPy](https://img.shields.io/badge/NumPy-Simulation-013243?logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualisation-11557c)
![License](https://img.shields.io/badge/License-MIT-yellow)

A **real-time physics-based quadrotor flight simulator** controlled via a USB gamepad. The simulator implements a full 9-stage rigid-body dynamics pipeline calibrated to the Crazyflie 2.0/2.1 nano-quadrotor, with aerodynamic coefficients fitted from real PX4 flight-log data.

Built as the simulation backbone for a reinforcement-learning based autonomous interceptor drone project (BTP).

---

## 🧠 Architecture

The simulation pipeline follows a paper-derived, modular architecture where each stage is a pure function with documented inputs and outputs:

```
Gamepad input (thrust, roll, pitch, yaw)
    │
    ▼
Stage A ─── Rate PID Controller ─────────→ torque command u
Stage B ─── Mixer ────────────────────────→ per-motor commands
Stage 2 ─── ESC / Battery Model ──────────→ steady-state motor speeds
Stage 3 ─── Motor Dynamics (1st-order) ───→ actual motor speeds + derivatives
Stage 4 ─── Propeller Force/Torque ───────→ per-prop thrust + drag
Stage 5 ─── Force Aggregation ────────────→ total body wrench
Stage 6 ─── Gyroscopic Torques ───────────→ reaction + inertial coupling
Stage 7 ─── Aerodynamic Drag Model ───────→ polynomial drag (fitted from PX4)
Stage 8 ─── Newton-Euler Dynamics ────────→ state derivatives
Stage 9 ─── Euler Integration ────────────→ new position + attitude
    │
    ▼
Live 3D matplotlib visualisation (position trail + body-frame arrows)
```

---

## ✨ Features

- 🎮 **Live gamepad control** — fly the drone with any USB controller (tested on Amkette Evo Gamepad Pro 4)
- 📐 **Paper-derived physics** — all equations follow standard rigid-body + propeller aerodynamics theory
- 📊 **Real-world calibration** — mass, inertia, arm length from Crazyflie 2.0 published data
- 🌊 **Fitted aero model** — drag coefficients from least-squares regression on 15 PX4 flight logs
- 🔋 **ESC polynomial** — battery/motor model fitted to real ESC telemetry
- 🏠 **Room boundary simulation** — floor collision + wall/ceiling detection
- 📈 **Dual live visualisation** — 3D trajectory + 4-channel input bar chart

---

## 🗂️ Project Structure

```
rl-drone-flight-simulator/
├── src/
│   ├── simulation.py       # Full 9-stage physics pipeline + live loop
│   └── visualiser.py       # Standalone visualisation utilities
├── docs/
│   └── coefficient_fitting.md
├── requirements.txt
├── .gitignore
└── README.md
```

### Module Breakdown

| File | Purpose |
|------|---------|
| `src/visualiser.py` | 3 standalone functions: `plot_trajectory()`, `plot_input_bars()`, `read_gamepad()` — fully decoupled from physics |
| `src/simulation.py` | 10 stage functions (PID → Mixer → ESC → Motors → Props → Forces → Gyro → Aero → Dynamics → Integration) + calibration constants + live loop |

---

## ⚙️ Prerequisites

- Python 3.10+
- A USB gamepad / joystick (any standard dual-stick layout)
- Display capable of running matplotlib interactively

---

## 🔨 Setup

```bash
# 1. Clone the repo
git clone https://github.com/siddhmehta5131/rl-drone-flight-simulator
cd rl-drone-flight-simulator

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run the full simulation
```bash
cd src
python simulation.py
```

Two windows open:
1. **3D Trajectory** — drone position trail + body-frame coordinate arrows
2. **Input Bar Chart** — live thrust/roll/pitch/yaw from the gamepad

Push the left stick up for thrust. Right stick controls roll/pitch. Close either window to exit.

### Use the visualiser standalone
```python
from visualiser import plot_trajectory, plot_input_bars, read_gamepad

# In your own simulation loop:
thrust, roll, pitch, yaw = read_gamepad()
plot_input_bars({"thrust": thrust, "roll": roll, "pitch": pitch, "yaw": yaw})
plot_trajectory(my_position, my_quaternion)
```

---

## 🔧 Physical Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Mass | 27 g | Crazyflie 2.0 (Forster thesis) |
| Inertia Ixx/Iyy | 1.4 × 10⁻⁵ kg·m² | Published |
| Inertia Izz | 2.17 × 10⁻⁵ kg·m² | Published |
| Arm length | 39.7 mm | Published |
| Lift coefficient c_l | 5.0 × 10⁻⁸ N·(rad/s)⁻² | Tuned for hover at 50% stick |
| Drag coefficient c_d | 1.25 × 10⁻⁹ N·m·(rad/s)⁻² | Forster ratio |
| Motor time constant | 20 ms | Best estimate |
| PID gains (Kp) | [0.15, 0.15, 0.20] | Median across 15 PX4 logs |
| Aero drag coefficients | 6 polynomials | Fitted via least-squares |
| ESC polynomial | 5 coefficients | Fitted to ESC telemetry |

---

## 🐛 Bugs Fixed (from development versions)

1. **Division by zero in `mixer()`** — when all motors command the same value (`cmd_max_val == c_cmd`), the scaling denominator is zero. Added `abs(denom) > 1e-9` guard.
2. **Double-step motor integration** — `rigid_body_dynamics` was receiving `Omega_new` (already integrated in Stage 3) and then Euler-integrating it again in Stage 9. Fixed to pass old `Omega`.
3. **Reverse-spinning motors** — `Omega` could go negative during fast transients. Clamped with `np.maximum(Omega, 0.0)`.
4. **ESC command inversion** — the fitted ESC polynomial maps cmd=0 → max speed (firmware convention). Added `cmd_esc = 1 - cmd` to align with the simulation's convention.
5. **Throttle-cut ESC bypass** — the ESC polynomial has a non-zero intercept at cmd=0, causing phantom lift. When throttle is cut, the entire ESC model is now bypassed.
6. **Duplicate `n_axes` call** — `joy.get_numaxes()` was called twice in `read_gamepad()`. Removed duplicate.
7. **Global aero coefficients** — `aerodynamic_force_torque()` accepted an `aero_coeffs` parameter but used module globals. Rewritten to accept all 6 coefficient arrays explicitly.

---

## 📄 License

MIT © 2025 **Siddh Mehta**

