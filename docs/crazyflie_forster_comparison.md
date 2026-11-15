# Crazyflie motor constants compared with Förster (2015)

Item 5 of the 2026-11-15 sourcing pass. Our quadrotor model uses two motor
constants that no published source gives (BUGS #27): the thrust coefficient
`PH_C_L = 2.618e-8 N/(rad/s)^2` (our own fit to NanoBench hover data, with rotor
speed taken from an ESC map that was later rejected, R-1) and the motor
first-order time constant `PH_K_MOT = 0.02 s` (a "best guess" from the
predecessor simulator). This note compares both with the published Crazyflie 2.0
identification.

**Source.** Förster, J. (2015). *System Identification of the Crazyflie 2.0 Nano
Quadrocopter.* Bachelor thesis, ETH Zürich (supervisors M. Hamer, R. D'Andrea).
doi:10.3929/ethz-b-000214143. PDF downloaded from the ETH Research Collection
on 2026-11-15 (item 9973b834-69c3-45a0-bb2c-240da4314818, 38.6 MB) and read.
The vehicle is a 28.0 g Crazyflie 2.0; ours is a 40.85 g Crazyflie 2.1 (NanoBench).

## What the thesis gives

| Quantity | Förster (2015) | Where |
|---|---|---|
| thrust per motor vs. command `cmd` in [0, 65535] | `f = 2.130295e-11 cmd^2 + 1.032633e-6 cmd + 5.484560e-4` N | Sec. 3.3.1, eq. (3.3); summary table p. 57 |
| rotor speed vs. command | `omega = 0.04076521 cmd + 380.8359` rad/s (not valid for cmd < 1000) | Sec. 3.3.2, Fig. 3.10; p. 57 |
| torque vs. thrust | `tau = 0.005964552 f + 1.563383e-5` N m | Sec. 3.3.3, eq. (3.5); p. 57 |
| command-to-thrust transfer function | `H1(z) = 6.0705967e-8 / (1 - 0.9745210 z^-1)`, `H2(z) = 7.2345374e-8 / (1 - 0.9695404 z^-1)` | Sec. 3.4.4, eqs. (3.15), (3.16), p. 34; H2 in the summary table p. 57 |
| input rate of that experiment | 500 Hz (`Ts = 0.002 s`); fit ignores frequencies above 12.5 Hz | Sec. 3.4.2, p. 31; Sec. 3.4.3 |

The thesis states the model `f = kappa * sum(omega_i^2)` (eq. 3.1) but does not
report a single value of `kappa`; the numbers below are computed by us from its
two mappings.

## Comparison

| Quantity | Ours | From Förster | Ratio ours / Förster |
|---|---|---|---|
| thrust coefficient `f / omega^2` | 2.618e-8 | 1.74e-8 to 2.12e-8 over cmd 10,000-60,000; least squares 1.82e-8; 1.81e-8 at the 40.85 g hover point (cmd 48,320, omega 2,351 rad/s) | 1.44 |
| maximum thrust per motor | 0.205 N (`PH_C_L * 2800^2`) | 0.160 N at cmd 65,535 (omega 3,052 rad/s) | 1.28 |
| thrust-to-weight at 40.85 g | 2.05 | 1.59 | 1.29 |
| torque / thrust | 1/480 = 0.0021 m (`PH_C_D / PH_C_L`, assumed) | 0.0060 m (eq. 3.5 slope) | 0.35 |
| motor time constant | 0.020 s (assumed) | `-Ts / ln(pole)` = 0.065 s (H2), 0.078 s (H1) | 0.26-0.31 |

Calculations: `kappa(cmd) = f(cmd) / omega(cmd)^2`; least squares over cmd =
10,000, 11,000, ..., 60,000; hover thrust per motor `m g / 4`; continuous
pole of `1 / (1 - a z^-1)` sampled at `Ts`: `tau = -Ts / ln a`. For a
first-order rotor-speed lag the linearised thrust has the same pole, so the
time constants are comparable.

## Reading of the comparison

* **Thrust coefficient.** Ours is 44 % above the value implied by Förster. The
  likely cause is the rotor speed in our fit, which came from the rejected ESC
  map (R-1), not a measurement. The thrust ceiling (thrust-to-weight 2.05 vs.
  1.59) and therefore the quad acceleration limits inherit this.
* **Torque ratio.** Our assumed `c_d = c_l / 480` gives about a third of the
  measured yaw torque per newton of thrust. It affects yaw authority only.
* **Motor lag.** Förster's identified command-to-thrust pole is 3-4 times slower
  than our 20 ms. It includes everything between the radio command and the load
  cell, and the fit is limited to 12.5 Hz, so it is an upper bound on the motor
  lag rather than a clean motor constant. Our repeatability study varies the lag
  only x0.5 / x2 (10-40 ms), which does not reach 65 ms.
* Differences in airframe (2.0 vs. 2.1, 28 g vs. 40.85 g) do not explain a 44 %
  difference in `f / omega^2`, which is a propeller property; both use the stock
  Crazyflie propellers (not verified for the NanoBench vehicle).

**No value was changed.** These are platform values, so whether to adopt the
Förster-derived constants is the project owner's decision (BUGS #34). If they
are adopted, every quadrotor result, the P4.9 limit measurement and the
flight-controller training become stale.
