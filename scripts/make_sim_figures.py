"""Generate the P3 simulator figures (docs/figures/*.png).

Figures
-------
esc_fit.png          ESC steady-state curve with the two measured anchors.
convergence.png      RK4 error against step size on the P3.1 rate ladder.
dryden_psd.png       Simulated Dryden spectra against the analytic model.
wind_drift.png       Open-loop position drift under mean wind and turbulence.

Run from the repository root: ``python scripts/make_sim_figures.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.signal import welch  # noqa: E402

from interceptor.sim import constants, pipeline  # noqa: E402
from interceptor.sim.wind import DrydenWind, dryden_spectrum  # noqa: E402

FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

CMD = np.array([0.45, 0.15, -0.08, 0.04])
U_BAT = 4.2


def esc_fit() -> None:
    c = np.linspace(0.0, 1.0, 400)
    omega, _ = pipeline.esc(c, U_BAT)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(c, omega, label=r"$\Omega_{ss}(c)$ adopted fit")
    ax.axhline(0.0, color="0.8", lw=0.6)
    ax.scatter([constants.PH_C_HOVER], [constants.PH_OMEGA_HOVER],
               color="tab:red", zorder=3, label="hover anchor (measured)")
    ax.scatter([1.0], [constants.PH_OMEGA_MAX], color="tab:green", zorder=3,
               label="full-throttle anchor")
    ax.set_xlabel("normalised collective command c")
    ax.set_ylabel("steady-state motor speed (rad/s)")
    ax.set_title("ESC fit: two-anchor monotone curve (P3.4, F9)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "esc_fit.png", dpi=150)
    plt.close(fig)


def _rollout(dt: float, duration: float = 0.1) -> np.ndarray:
    state = (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3),
             np.zeros(3), np.full(4, constants.PH_OMEGA_HOVER))
    for _ in range(int(round(duration / dt))):
        state = pipeline.rk4(*state, CMD, U_BAT, pipeline.PhysicsParams(),
                             dt=dt, predict_rotor=False)
    return np.concatenate(state)


def convergence() -> None:
    rates = np.array([100, 250, 500, 1000, 2000], dtype=float)
    reference = _rollout(1.0 / 50_000.0)
    errors = np.array([np.linalg.norm(_rollout(1.0 / hz) - reference) for hz in rates])
    slope = np.polyfit(np.log(1.0 / rates), np.log(errors), 1)[0]

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.loglog(1.0 / rates, errors, "o-", label=f"RK4 (measured slope {slope:.2f})")
    ax.loglog(1.0 / rates, errors[0] * (rates[0] / rates) ** 4, "--",
              label=r"reference slope 4")
    ax.set_xlabel("physics step dt (s)")
    ax.set_ylabel("state error against 50 kHz reference")
    ax.set_title("Timestep convergence (P3.1 / P3.2)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "convergence.png", dpi=150)
    plt.close(fig)


def dryden_psd() -> None:
    alt, W20, V, dt = 100.0, 15.0, 15.0, 0.01
    wind = DrydenWind(alt, np.zeros(3), W20=W20, airspeed=V, seed=1)
    data = np.array([wind._turbulence(dt) for _ in range(60_000)])
    spectrum = wind.spectrum
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), sharey=True)
    for ax, (name, idx) in zip(axes, (("u", 0), ("v", 1), ("w", 2))):
        f, p = welch(data[:, idx], fs=1.0 / dt, nperseg=4096, detrend="constant")
        omega = 2.0 * np.pi * f
        model = spectrum.psd(omega, name) * 2.0 * np.pi
        ax.loglog(f[1:], p[1:], label="simulated")
        ax.loglog(f[1:], model[1:], "--", label="analytic")
        ax.set_title(f"{name} axis")
        ax.set_xlabel("frequency (Hz)")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("PSD (m$^2$/s$^2$/Hz)")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Dryden spectra at 100 m, W20 = 15 m/s (P3.6)")
    fig.tight_layout()
    fig.savefig(FIG / "dryden_psd.png", dpi=150)
    plt.close(fig)


def wind_drift() -> None:
    dt = 0.002
    wind = DrydenWind(100.0, np.array([5.0, 0.0, 0.0]), W20=15.0, airspeed=5.0, seed=2)
    params = pipeline.PhysicsParams()
    state = (np.array([0.0, 0.0, 100.0]), np.array([1.0, 0.0, 0.0, 0.0]),
             np.zeros(3), np.zeros(3), np.full(4, constants.PH_OMEGA_HOVER))
    cmd = np.full(4, constants.PH_C_HOVER)
    times = np.arange(0, 20.0, dt)
    pos = []
    for _ in times:
        w = wind.step(dt)
        state = pipeline.control_step(state, cmd, U_BAT, params,
                                      control_dt=dt, physics_dt=dt, wind_W=w)
        pos.append(state[0].copy())
    pos = np.array(pos)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(times, pos[:, 0], label="north")
    ax.plot(times, pos[:, 1], label="west")
    ax.plot(times, pos[:, 2], label="up")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("position (m)")
    ax.set_title("Open-loop drift under 5 m/s mean wind + Dryden (P3.9)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "wind_drift.png", dpi=150)
    plt.close(fig)


def main() -> None:
    esc_fit()
    convergence()
    dryden_psd()
    wind_drift()
    print(f"wrote figures to {FIG}")


if __name__ == "__main__":
    main()
