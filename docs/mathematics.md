# Mathematics

This document is the authoritative derivation record for the implementation.
Equation IDs are stable strings of the form `EQ-<TOPIC>-<n>`; every ID is
checked by `scripts/check_equation_refs.py`, which requires each code/config
reference to exist here and each ID here to be unique.

World frame: **x north, y west, z up**, gravity `g = (0, 0, -9.81) m/s^2`.
Units are SI and angles are radians unless a symbol says otherwise.

---

## 1. Engagement geometry and augmented proportional navigation

### 1.1 Geometry

Let the interceptor state be `(p_M, v_M, a_M)` and the true target state be
`(p_T, v_T, a_T)`. Define

\[
\mathbf r = \mathbf p_T-\mathbf p_M,\qquad
\hat{\mathbf r}=\mathbf r/\|\mathbf r\|,
\tag{EQ-GEO-1}
\]

\[
\mathbf v_{\rm rel}=\mathbf v_T-\mathbf v_M,\qquad
v_{\rm cl}=-\mathbf v_{\rm rel}\cdot\hat{\mathbf r}.
\tag{EQ-GEO-2}
\]

The inertial line-of-sight rate is the component of relative velocity
perpendicular to the LOS:

\[
\dot{\hat{\mathbf r}}=
\frac{\mathbf v_{\rm rel}+v_{\rm cl}\hat{\mathbf r}}{r_{\rm eff}},
\qquad r_{\rm eff}=\max(r, r_{\rm min}).
\tag{EQ-GEO-3}
\]

`r_min = 15 m` keeps the rate finite before closest approach; the raw-APN arm
uses `r_min = 2 m` to reproduce the benchmark. These guards are numerical, not
physical, and are recorded in the code next to the call.

### 1.2 Time to go and zero-effort miss

For a closing engagement,

\[
t_{\rm go}=\operatorname{clip}\!\left(\frac{r}{v_{\rm cl}},0.05,20\right),
\qquad v_{\rm cl}>0,
\tag{EQ-APN-4}
\]

and `t_go = 20 s` otherwise. The zero-effort miss (ZEM) is the miss that would
result if neither vehicle accelerated from now on:

\[
\mathbf Z=\mathbf v_{\rm rel}t_{\rm go}+\tfrac12\mathbf a_T t_{\rm go}^2,
\qquad
\mathbf Z_\perp=\mathbf Z-(\mathbf Z\cdot\hat{\mathbf r})\hat{\mathbf r}.
\tag{EQ-APN-5}
\]

### 1.3 APN law

Perpendicular proportional navigation commands

\[
\mathbf a_{\rm PN}=N'\,\frac{\mathbf v_{\rm rel,\perp}}{t_{\rm go}}
=N'\,v_{\rm cl}\,\dot{\hat{\mathbf r}}_\perp .
\tag{EQ-APN-1}
\]

Augmented PN adds half the target's perpendicular acceleration:

\[
\mathbf a_{\rm aug}=\tfrac12N'\,\mathbf a_{T,\perp},\qquad
\mathbf a_{\rm APN}=\mathbf a_{\rm PN}+\mathbf a_{\rm aug}.
\tag{EQ-APN-2}
\]

The separate axial speed-management term is

\[
\mathbf a_{\rm axial}=
\begin{cases}
0.8\,a_{t,\max}\hat{\mathbf r}, & \text{omnidirectional and speed management active},\\
a_{t,\max}\hat{\mathbf v}_M, & \text{non-holonomic and speed management active},\\
0, & \text{otherwise},
\end{cases}
\tag{EQ-APN-3}
\]

with “active” meaning `||v_M|| < 0.96 v_max` **or** `v_cl < 0.45 v_max`.
The term is implemented alone in `compute_axial_acceleration()`
(`interceptor/guidance/apn.py`, re-exported by `kinematics.py`) and is only
added by `compute_3d_apn_command(..., include_axial_boost=True)`.
The normalised command is then `a_APN + a_axial`; the two arms A1/A2 differ in
whether the measurement is raw or filtered and whether this term is present.

### 1.4 ZEM form

Using the identity `v_rel,perp = Z_perp / t_go` for a non-manoeuvring target,
the command can be written

\[
\mathbf a_{\rm PN}=N'\frac{\mathbf Z_\perp}{t_{\rm go}^2}.
\tag{EQ-APN-6}
\]

The implementation stores `zem_perp` and `zem_norm` for logging, which is why
the two forms are tested to be algebraically equivalent.

### 1.5 Navigation constant

For a zero-lag interceptor with perfect information the linear-quadratic optimal
PN law has effective navigation ratio `N' = 3`, and the optimal APN law is PN
plus `N' a_T / 2` with the same ratio (Palumbo, Blauwkamp & Lloyd 2010, Johns
Hopkins APL Technical Digest 29(1), Eq. 22 p. 49 and Eq. 27 p. 51; the same
paper is the source of EQ-APN-2). Against a hard-turning target PN with gain 3
needs three times the target acceleration and APN half of that; at gain 4 the
two need the same and above 4 APN needs more, and "higher gains may lead to
excessive noise throughput" (p. 52, Fig. 7). The lower bound 3 is therefore
cited; the upper bound 5 is a project choice that brackets the noise trade-off
(no published upper bound was found; Zarchan's text is a library item, see
`docs/library_pending.md`). The swept range is

\[
N'_{\min}=3,\qquad N'_{\max}=5.
\tag{EQ-APN-7}
\]

P5.11 sweeps `N'` over this range separately for A1/A2 and each platform and
selects by kill rate then median miss. The result is written back to
`configs/guidance.yaml` (`navigation.N_prime_by_platform`) with
`source: sweep P5.11`. In the last sweep (stale since the 2026-11-14 value
changes, rerun in Colab) the quadcopter A2 pick `N' = 5` lies on the upper
grid edge, but its kill rate was 0.93 at every grid point and the median miss
varied by at most 0.13 m, so the pick is a tie-break on a flat response, not a
trend that the grid cuts off.

### 1.6 Kill radius and engagement geometry

The kill radius is a physical rule, not a tuned threshold:

\[
r_{\rm kill} = \tfrac12 b_{\rm target} + \tfrac12 b_{\rm interceptor}.
\tag{EQ-GEO-4}
\]

This is a hit-to-kill contact rule: the interceptor must touch the target's
span. No warhead is assumed, because no credible lethal-radius figure was found
(`data/lethality_and_gload.csv`); the rule is therefore conservative.
Quad against a Shahed-136 (span 2.5 m, Janes) with a Crazyflie (0.092 m, Bitcraze
datasheet): 1.30 m. Aerosonde (2.8956 m) against an Orlan-10 (3.1 m, Airforce
Technology profile, trade press): 3.00 m. AIM5 (0.15 m) against an F-16C (9.8 m, USAF fact sheet): 4.98 m. P9
reports the full miss-distance distribution, so no conclusion depends on the
radius. Initial range follows a stated time-to-intercept window:

\[
r_{0,\min}=v_{\rm cl,0}t_{\min},\qquad
r_{0,\max}=v_{\rm cl,0}t_{\max}.
\tag{EQ-GEO-5}
\]

Altitude is set so the target's 35 m floor is never reached during the vertical
manoeuvre budget:

\[
h_0 = 35 + \Delta h_{\rm budget}.
\tag{EQ-GEO-6}
\]

The per-platform numbers are in `configs/guidance.yaml` and mirrored in
`configs/platforms.yaml`; the source table generates both.

### 1.7 Target speed and acceleration ratios (decision Q5)

For each platform the baseline target is

\[
v_{T,0}=\rho_v v_{M,\max},\qquad
a_{T,\max}=\rho_a a_{M,n,\max},
\tag{EQ-TGT-1}
\]

with `rho_v = 0.66 (quad), 0.765 (fixed wing), 0.33 (missile)` and
`rho_a = 0.48, 1.0, 0.25`. The quad envelope is the closed-loop Crazyflie
measurement of P4.9 (27.7 m/s, 15.4 m/s^2, 9.5 m/s^2 downward, jerk 88.3 m/s^3;
section 6.1 states how far these depend on the assumed rotor drag); its target
keeps the real speed ratio of a fielded pair, Sting 280 km/h (manufacturer
product page) over Shahed-136 185 km/h (IRGC poster via Janes) = 1.51, giving
18.3 m/s and, with the acceleration ratio 2.08, 7.404 m/s^2. Spawn range is
92-277 m (the 10-25 s window of `EQ-GEO-5`).
The fixed-wing target flies the Orlan-10 cruise speed, 110 km/h = 30.6 m/s
(Airforce Technology profile, 24 Mar 2023; trade press, no manufacturer
datasheet read), so the 32.4 m/s Aerosonde (EQ-FW-3) has a speed ratio of only
1.06, and a target drawn at the top of its 0.75-1.15 speed band (35.2 m/s) is
faster than the interceptor. The earlier
choice, the Bayraktar TB2 at 46.3 m/s, was faster than the interceptor and is now
covered by the P9.1 speed-ratio sweep instead. The missile-class target limit is the 9 g fighter
limit (F-16C, J-20; JF-17 Block III is 8 g; `data/threat_platforms.csv` T02-T04,
secondary sources), giving `rho_a = 88.29/358.1 = 0.25`. The quad acceleration ratio (2.08) remains a design choice
because no g data exists for one-way attack drones. Every value's reference is in
its `source` string in `configs/platforms.yaml` and in `docs/references.md`.
P9 varies them in sensitivity sweeps and reports the effect on kill rate.

---

## 2. Unified kinematic model

Each interceptor is a point mass with a first-order acceleration lag,
tangential/normal acceleration limits, speed limits, jerk limit and turn-rate
limit. The 200 Hz integration holds the 50 Hz command constant (`S8`); the
guidance environment uses a 20 ms decision step and integrates with the same
first-order update at that step size.

Let `s = ||v_M||`, `t_hat = v_M/s`, and let `a_cmd` be the constrained
command. The tangential and normal components are

\[
a_t=\mathbf a_{\rm cmd}\cdot\hat{\mathbf v}_M,\qquad
\mathbf a_n=\mathbf a_{\rm cmd}-a_t\hat{\mathbf v}_M.
\tag{EQ-KIN-1}
\]

The envelope is

\[
a_{t,\min}\le a_t\le a_{t,\max},\qquad
\|\mathbf a_n\|\le a_{n,\max},
\tag{EQ-KIN-2}
\]

with a non-holonomic platform additionally forbidden to drop below `v_min`.
The acceleration state follows

\[
\dot{\mathbf a}_M=\frac{\mathbf a_{\rm cmd}-\mathbf a_M}{\tau_a},
\qquad
\|\dot{\mathbf a}_M\|\le j_{\max},
\tag{EQ-KIN-3}
\]

and the body turn rate is

\[
\boldsymbol\omega_M=
\frac{\hat{\mathbf v}_M\times\mathbf a_M}{\max(s,1)}
,\qquad \|\boldsymbol\omega_M\|\le\omega_{\max}.
\tag{EQ-KIN-4}
\]

The speed update is `v_M += a_M dt`, followed by the speed clamp. The position
update is `p_M += v_M dt`. A non-holonomic vehicle below `v_min` is projected
back onto the `v_min` sphere rather than allowed to hover.

The implemented functions are `apply_interceptor_constraints`,
`heading_rate_from_acceleration` and `closest_approach_distance` in
`interceptor/guidance/kinematics.py`.

### 2.1 Closest approach

Between two samples the relative motion is linear, so the continuous
closest-approach distance is exact for the sampled endpoints:

\[
\mathbf r(\lambda)=\mathbf r_0+\lambda\Delta\mathbf r,\qquad
\lambda^*=\operatorname{clip}\!\left(
-\frac{\mathbf r_0\cdot\Delta\mathbf r}{\|\Delta\mathbf r\|^2},0,1\right),
\tag{EQ-GEO-7}
\]

and `d_CPA = ||r(lambda*)||`. The environment records the episode minimum
independently of the kill radius.

---

## 3. Fixed-wing airframe: Aerosonde

The point-mass equations are

\[
\dot V=\frac{T\cos\alpha-D}{m}-g\sin\gamma,\qquad
\dot\gamma=\frac{(L+T\sin\alpha)\cos\phi}{mV}-\frac{g\cos\gamma}{V}.
\tag{EQ-FW-1}
\]

With `L = 0.5 rho V^2 S C_L`, `D = q S C_D` and the parabolic polar
`C_D = C_D0 + C_L^2/(pi e AR)`, stall speed at `C_L,max` is

\[
V_{\rm stall}=\sqrt{\frac{2mg}{\rho S C_{L,\max}}}.
\tag{EQ-FW-2}
\]

The propeller model gives thrust and torque from advance ratio
`J = V/(nD)`, `T = rho n^2 D^4 C_T(J)`, `Q = rho n^2 D^5 C_Q(J)`, and the
motor/ESC relation. Level-flight maximum speed is the root of

\[
T(V)=D(V)
\tag{EQ-FW-3}
\]

at sea level. With the book propulsion model (Beard & McLain uavbook draft
2025-10-02, eqs. 4.17-4.22: operating speed from the motor/propeller torque
balance, `V_in = V_max delta_t`) and the mavsim Aerosonde parameters, the
full-throttle root with `C_D = C_Dp + C_L^2/(pi e AR)` is 32.43 m/s, so
`v_max = 32.4 m/s` (`airframes.py::aerosonde_max_level_speed`; until 2026-11-15
the config held a v1-notebook 40 m/s, BUGS #32). Excess thrust gives the axial
acceleration envelope

\[
a_{t,\max}=\frac{T_{\max}(V)-D(V)}{m},\qquad
a_{t,\min}=\frac{T_{\rm idle}(V)-D(V)}{m}.
\tag{EQ-FW-4}
\]

Evaluated at the 25 m/s trim speed with the same model
(`airframes.py::aerosonde_axial_acceleration`): `a_t,max = +2.47 m/s^2` (full
throttle) and `a_t,min = -3.02 m/s^2` (zero throttle, windmilling propeller).
Both depend on speed (`+4.9 m/s^2` at 15 m/s, zero at `v_max`; static thrust
84.6 N); the kinematic model uses the constants, and the `axial_limit_scale`
sweep tests that simplification. Until 2026-11-15 the config held v1-notebook
values of `+4.2 g / -2 g` (BUGS #32).

The load factor and negative-load limits are

\[
n=\frac{L}{mg},\qquad -0.4\,n_{\max}\le n\le n_{\max}=3.8,
\tag{EQ-FW-5}
\]

where 3.8 is the normal-category positive limit manoeuvring load factor and the
negative limit is 0.4 times it (14 CFR 23.337, 2011 ed.). The aerodynamic limit
`0.5 rho V^2 S C_Lmax / (m g)` is below 3.8 under 28.5 m/s, so the constant
kinematic limit is optimistic at low speed. The coordinated-turn rate is

\[
\omega=\frac{g\sqrt{n^2-1}}{V}.
\tag{EQ-FW-6}
\]

The acceleration lag is the roll/pitch first-order response,

\[
\tau_a=\tau_{\rm roll}\approx 0.12\,{\rm s},
\tag{EQ-FW-7}
\]

where 0.12 s is a v1-notebook value: no such roll/pitch constant was found in
Beard & McLain (BUGS #32), so it is a project choice. With `a_n,max = 3.8 g` the
bound `a_n,max / tau_a = 311 m/s^3` sits above the configured jerk limit of
180 m/s^3, which is a project choice. The values are evaluated at sea level
and the 25 m/s trim condition; the derivation is reproduced by
`interceptor/guidance/airframes.py` and the source table points at
`configs/airframes/aerosonde.yaml`.

---

## 4. Missile airframe: AIM5

AIM5 is a boost-coast missile. Its mass and thrust decks are functions of time:

\[
m(t)=\text{deck}(t),\qquad T(t)=\text{deck}(t)+(p_{\rm SL}-p)a_e.
\tag{EQ-MS-2}
\]

The axial acceleration follows

\[
a_t=\frac{T-D}{m},\qquad
D=q S C_D(\alpha,M),
\tag{EQ-MS-3}
\]

and the speed envelope is obtained by integrating `dot V = a_t - g sin gamma`
through the boost and coast phases. The axial limit adopted is

\[
a_{t,\max}=18g,\qquad a_{t,\min}=-4g.
\tag{EQ-MS-4}
\]

The normal-force limit is the CADAC++ calculation. With

\[
C_{N,\max}=C_D(\alpha_{\max},M)\sin\alpha_{\max}
          + C_L(\alpha_{\max},M)\cos\alpha_{\max},
\tag{EQ-MS-1}
\]

the permissible load factor is

\[
n_{\max}=\frac{C_{N,\max}\,q S}{mg},\qquad
a_{n,\max}=n_{\max}g.
\tag{EQ-MS-1}
\]

At the Q12 condition (10 km, Mach 2, post-burnout mass 38.6 kg) the deck gives
`C_N,max = 10.865`, `q = 71.97 kPa`, `S = 0.01767 m^2`, hence
`n_max = 36.5` and `a_n,max = 358.1 m/s^2`. The same formula is reproduced by
`interceptor/guidance/airframes.py::aim5_gmax` and by the unit test.

The turn-rate cap follows from the normal limit at the Q12 speed:

\[
\omega_{\max}=a_{n,\max}/V,
\tag{EQ-MS-5}
\]

and the autopilot lag is the CADAC rate-loop constant

\[
\tau_a=t_r=0.1\,{\rm s},\qquad j_{\max}=a_{n,\max}/(4\tau_a).
\tag{EQ-MS-6}
\]

The numerical decks, including the powered and unpowered drag tables, are in
`configs/airframes/aim5.yaml` with their CADAC citation and MIT licence note.

---

## 5. Evasive manoeuvre laws

All laws output a commanded acceleration in the target path frame. Let
`t_hat` be the unit velocity, `n_h` the horizontal normal, and
`n_v = -t_hat x n_h` the vertical normal. The common saturation is
`||a|| <= a_max`, and the hard set is oversampled 2.5x during training
(`EQ-MAN-1`).

The 15 laws are:

1. **NONE** (`EQ-MAN-2`): `a=0`.
2. **CONSTANT_G** (`EQ-MAN-3`): `a = a_max n_h` after `0.25 s`.
3. **SINUSOIDAL_WEAVE** (`EQ-MAN-4`):
   `a = a_max sin(omega t + phi) n_h`.
4. **BARREL_ROLL** (`EQ-MAN-5`):
   `a = a_max[cos(omega t+phi)n_h + sin(omega t+phi)n_v]`.
5. **TERMINAL_BREAK** (`EQ-MAN-6`): for `t_go <= 2.5 s`, steer along the
   component of `n_h` perpendicular to the LOS; otherwise zero.
6. **ENERGY_DEPLETION** (`EQ-MAN-7`):
   `a = 0.25 a_max t_hat + 0.7 a_max n_LOS + 0.4 a_max sin(omega t)n_h`.
7. **SPLIT_S** (`EQ-MAN-8`): roll inverted with `-a_max n_v + 0.2 a_max t_hat`
   until the heading change exceeds 2.8 rad, then `0.2 a_max t_hat`.
8. **SQUARE_BANG_BANG** (`EQ-MAN-9`):
   `a = a_max sign(sin(omega t+phi)) n_h`.
9. **SCISSOR** (`EQ-MAN-10`): reverse the horizontal sign at `|psi-psi_ref| =
   65 deg`; command `-0.25 a_max t_hat +/- a_max n_h`.
10. **NOTCHING** (`EQ-MAN-11`):
    `a = clip(-4 v_radial + V^2/r, -a_max, a_max) n_LOS_perp`.
11. **TURN_INTO_MISSILE** (`EQ-MAN-12`):
    `a = a_max n_threat + 0.2 a_max t_hat`.
12. **CRANK** (`EQ-MAN-13`): `psi_des = psi_LOS +/- 60 deg`, with a 0.35 rad/s
    side switch and `a = clip(3 V e_psi, -a_max, a_max) n_h`.
13. **SPEED_CHANGE** (`EQ-MAN-14`):
    `a = 0.8 a_max sign(sin(1.2 t + phi)) t_hat`.
14. **MINIMUM_RANGE_TURN** (`EQ-MAN-15`): for `t_go <= 4 s`,
    `a = -0.3 a_max t_hat + a_max n_h`; otherwise zero.
15. **DIVE_DENSE_AIR** (`EQ-MAN-16`): pitch toward `-30 deg` when above 100 m,
    plus `0.25 a_max t_hat`; `a_pitch = clip(3V(gamma_d - gamma),...) n_v`.
16. **SPIRAL_CORKSCREW** (`EQ-MAN-17`):
    `a = 0.2 a_max t_hat + a_max[cos(phase)n_h + sin(phase)n_v]`,
    `phase = omega t + 0.18 t^2 + phi`.

The notebook contains 15 non-NONE laws; the implementation's `ManeuverType`
enum includes `NONE`, so the complete set is 16. The constants in
`configs/maneuvers.yaml` are the non-dimensional versions of the code constants
and are the reference for the P5.13 behaviour matrix.

---

## 5A. Measurement, aerodynamic and control auxiliary equations

### 5A.1 Seeker measurement and filter

The range-dependent position standard deviation is

\[
\sigma_p=\kappa\left(\sigma_{p,0}+\alpha_r r\right),
\tag{EQ-SEEK-1}
\]

with `sigma_p0 = 0.5 m`, `alpha_r = 8.727 mrad` (0.5 deg), and `kappa` the
evaluation noise scale (headline 1.0). Velocity and acceleration noise are
proportional to the platform's nominal target speed and manoeuvre limit:

\[
\sigma_v=\kappa\,0.03\,v_{T,{\rm nom}},\qquad
\sigma_a=\kappa\,0.18\,a_{T,\max}.
\tag{EQ-SEEK-2}
\]

Each step a Bernoulli draw with probability `0.04 kappa` drops the measurement.
The filter then coasts on its constant-acceleration prediction and holds the
acceleration estimate unchanged:

\[
\hat p^+=\hat p+\hat v\,\Delta t+\tfrac12\hat a\,\Delta t^2,\qquad
\hat v^+=\hat v+\hat a\,\Delta t,\qquad
\hat a^+=\hat a .
\tag{EQ-SEEK-3}
\]

Otherwise position, velocity and acceleration are each measured (`y_p, y_v, y_a`
with the noise above) and each channel is blended with its own prediction, using
the same `p^-`, `v^-` as `EQ-SEEK-3`:

\[
\begin{aligned}
\hat p^+ &= \alpha_p\,y_p+(1-\alpha_p)\,\hat p^-,\
\hat v^+ &= \alpha_v\,y_v+(1-\alpha_v)\,\hat v^-,\
\hat a^+ &= \alpha_a\,y_a+(1-\alpha_a)\,\hat a,
\end{aligned}
\tag{EQ-SEEK-4}
\]

with `(alpha_p, alpha_v, alpha_a) = (0.65, 0.55, 0.35)` at 50 Hz. This is a
fixed-gain complementary (exponential-smoothing) filter on three measured
channels, not the residual-driven alpha-beta-gamma form, in which velocity and
acceleration are inferred from the position residual alone. The same gains are
used for every platform so the noise schedule is comparable. Implemented by
`SeekerSensorFilter.observe` in `interceptor/guidance/seeker.py`; the values are
read from `configs/seeker.yaml`.

**Status of these numbers.** `sigma_p0`, the 3 % and 18 % fractions, the 4 %
dropout and the fixed gains are project choices; no datasheet gives this model.
(Earlier source text attributing them to motion-capture residuals and measured
radio packet loss was wrong and was removed on 2026-11-14.) The one sourced
value is the angular term: a manufacturer statement for counter-UAS radars
(Echodyne, hosted by the US International Trade Administration, 2024) gives
EchoShield angular tracking accuracy below 0.5° (8.727 mrad) and EchoGuard about
1° azimuth (17.5 mrad), range accuracy 1–2 m. The statistic (1σ or bound) is not
stated; we take the 0.5° bound as 1σ, the pessimistic reading. The P9.1 sweep
scales all sigmas and the dropout probability by `kappa ∈ {0, 0.25, 0.5, 1, 2, 4}`:
`kappa = 0.25` is close to the earlier unsourced 2.5 mrad, `kappa = 1` is
EchoShield and `kappa = 2` is EchoGuard, so the sweep shows how results change
from near-ideal to twice radar-grade noise.

### 5A.1a Kalman tracker (P5.4)

Option `filter_type="kalman"` replaces the fixed-gain filter. Per axis the state
is `x=[p,v,a]`, with the white-noise-jerk constant-acceleration model
(Bar-Shalom, Li and Kirubarajan 2001, section 6.2.3):

\[
x_{k+1}=Fx_k+w_k,\quad
F=\begin{bmatrix}1&\Delta t&\Delta t^2/2\0&1&\Delta t\0&0&1\end{bmatrix}.
\tag{EQ-KF-1}
\]

\[
Q=q_j\begin{bmatrix}\Delta t^5/20&\Delta t^4/8&\Delta t^3/6\
\Delta t^4/8&\Delta t^3/3&\Delta t^2/2\
\Delta t^3/6&\Delta t^2/2&\Delta t\end{bmatrix}.
\tag{EQ-KF-2}
\]

The seeker measures all three components, so `H=I` and

\[
y_k=x_k+v_k,\quad R=\mathrm{diag}(\sigma_p^2,\sigma_v^2,\sigma_a^2),
\tag{EQ-KF-3}
\]

with the sigmas of EQ-SEEK-1/2 evaluated at the current range. The jerk
spectral density is

\[
q_j=(a_{T,\max}/\tau_{\rm man})^2,
\tag{EQ-KF-4}
\]

with manoeuvre time scale `tau_man = 1 s` (a design parameter, not a measured
value). On dropout only the prediction runs. The update uses the Joseph form,
`S=P+R`, `K=PS^{-1}`:

\[
P^+=(I-K)P(I-K)^\top+KRK^\top.
\tag{EQ-KF-5}
\]

### 5A.2 Rotor aerodynamic polynomial

The NanoBench fit is

\[
f_i=\sum_k c_{i,k}\,m\,b_k(v_B,\bar\Omega^2),
\tag{EQ-AERO-8}
\]

where `b_k` are the polynomial basis terms listed in
`configs/quadrotor.yaml` and `m` is the (possibly randomised) mass. Yaw damping
uses

\[
\tau_z=J_z\left(c_{z,0}v_x+c_{z,1}v_y\right).
\tag{EQ-AERO-9}
\]

The coefficients are constrained to be dissipative (force opposite velocity);
the fit residual and the decision to retain the polynomial are documented in
`docs/sim_validation.md`.

### 5A.3 Dryden wind mapping

The Ornstein-Uhlenbeck process used by the legacy simulator is the first-order
Dryden longitudinal filter. Its continuous form is

\[
\dot u=-\theta u+\sigma\sqrt{2\theta}\,\xi(t),
\tag{EQ-WIND-1}
\]

and the legacy discrete update is
`u += theta(mean-u)dt + sigma sqrt(dt) N(0,1)`. It is retained as the mapping
of the old `wind_ou_step`; the P3.6 generator uses the exact Dryden
discretisation below.

The MIL-HDBK-1797 low-altitude parameters, with `h` in metres and
`h_ft = h/0.3048`, are

\[
\sigma_w=0.1W_{20},\quad
\sigma_u=\sigma_v=\frac{\sigma_w}{(0.177+0.000823h_{\rm ft})^{0.4}},
\tag{EQ-WIND-2}
\]

\[
L_w=h,\qquad L_u=L_v=\frac{h}{(0.177+0.000823h_{\rm ft})^{1.2}},
\]

with the wind speed at 20 ft `W20` in the same units as `sigma_w`. A minimum
airspeed `V = max(V_air, 5 m/s)` keeps the forming-filter time constants
finite.

The forming filters are

\[
\begin{aligned}
H_u(s)&=\sigma_u\sqrt{\frac{2L_u}{\pi V}}\frac{1}{1+(L_u/V)s},\\
H_v(s)&=\sigma_v\sqrt{\frac{L_v}{\pi V}}\frac{1+\sqrt3\,(L_v/V)s}{(1+(L_v/V)s)^2},
\end{aligned}
\tag{EQ-WIND-3}
\]

and `H_w` is `H_v` with `(sigma_w, L_w)`. With white-noise intensity `pi`
their output variances are `sigma_u^2`, `sigma_v^2` and `sigma_w^2`
respectively. `interceptor/sim/wind.py` discretises each filter exactly with
the matrix exponential and realises the second-order noise with its exact
covariance, so the variance is preserved at any step size.

The discrete gust uses the 1-cosine shape

\[
U(t)=
\begin{cases}
\frac{U_{\rm ds}}{2}\left(1-\cos\frac{2\pi t}{T_g}\right), & 0\le t\le T_g,\\
0, & \text{otherwise}.
\end{cases}
\tag{EQ-WIND-4}
\]

The `mean_std`, `theta_range` and `sigma_range` entries in
`configs/quadrotor.yaml` belong to a separate, simpler gust model: an
Ornstein-Uhlenbeck process used only to randomise wind while training the flight
controller. They are project choices, not Dryden parameters; the stationary gust
standard deviation `sigma / sqrt(2 theta)` spans 0.21-3.16 m/s, against Dryden
`sigma_w = 0.1 W20` = 0.77 / 1.54 / 2.31 m/s for the light / moderate / severe
`W20` of 15 / 30 / 45 knots (MathWorks Dryden block documentation, implementing
MIL-F-8785C; the specification itself was not read). `tests/unit/test_dryden_wind.py`
checks the parameters, the variances over 30 ten-minute realisations (5 %),
the spectral shape against `EQ-WIND-3`, and the gust shape.

### 5A.4 Rate-loop PID

The project baseline B1 is

\[
u=K_p e_\omega+K_i\int e_\omega dt-K_d\dot\omega,
\tag{EQ-CTRL-3}
\]

where `u` is a normalised motor-command increment and `e_omega` is the body-rate
error. The discrete derivative uses the previous sample, and the integral is
clamped by the mixer's saturation logic. The gains in
`configs/quadrotor.yaml` are the 500 Hz loop-shaping result; B2 instead uses the
Crazyflie firmware gains verbatim.

---

## 6. Quadrotor 6-DOF dynamics

The simulator state is
`x = (p, q, v, omega, Omega)` with position `p` and velocity `v` in the world
frame, body rates `omega` and quaternion `q = [w, x, y, z]` (body to world),
and four rotor speeds `Omega_j`. The implementation is
`interceptor/sim/pipeline.py`; the rotor forces and aerodynamic polynomial are
in `interceptor/sim/aero.py`.

### 6.1 Translation

With body-frame propeller force `f_prop`, aerodynamic force `f_aero`, and
gravity `g = (0, 0, -9.81) m/s^2`,

\[
\dot{\mathbf p}=\mathbf v,\qquad
\dot{\mathbf v}=\frac{1}{m}R(\mathbf q)\left(\mathbf f_{\rm prop}+\mathbf f_{\rm aero}\right)+\mathbf g .
\tag{EQ-SIX-1}
\]

Wind enters through the air-relative body velocity
`v_rel = R(q)^T (v - v_wind)` (`EQ-WIND-1`, `EQ-AERO-8`). Ground effect scales
the collective axis of `f_prop` by

\[
\kappa_{\rm ge}(z)=1+\kappa_{\max}\max\!\left(0,1-\frac{z}{h_{\rm ge}}\right),
\tag{EQ-SIX-10}
\]

**Rotor-drag model (P3.5 fallback).** The polynomial above has a `v Omega^2`
term worth about 1.5 1/s of drag at hover speed (finding F10), which caps the
quad at about 3 m/s. The NanoBench data needed for the constrained refit is not
available, so the plan's decision rule (keep the polynomial only if it beats a
linear rotor-drag model) cannot be evaluated; the simpler model is adopted for
all interceptor work:

\[
\mathbf F_{drag}=-m\,D\,\mathbf v_{rel},\qquad D=\mathrm{diag}(k_x,k_y,k_z),\quad k_i>0 .
\tag{EQ-SIX-11}
\]

It is dissipative by construction (`v . F = -m sum k_i v_i^2 <= 0`) and gives a
terminal speed `a_max / k`. The values `k = (0.5, 0.5, 0.5) 1/s` are an
**assumption** at the order of magnitude of published rotor-drag identifications
(Faessler et al., RA-L 2018: `d_x` 0.49-0.54, `d_y` 0.24-0.39 1/s on a 610 g
quadrotor, `d_z` not identified), not a Crazyflie fit. `k_z` was 0.7 until
2026-11-14 and was set equal to `k_x, k_y` because no published value supports a
different vertical coefficient.

**The quad top speed is drag-limited, and the drag is assumed.** The repeatability
study (`scripts/check_quad_limits_repeatability.py`,
`results/tables/quad_limits_repeatability.csv`) re-measures the P4.9 envelope
under changed settings. Largest change of any limit against the baseline
(27.7 m/s, 15.4 m/s^2): measurement-procedure settings A 5.1 %, B 5.4 % (top speed
moves with the hold fraction, slightly over the 5 % target); motor lag x0.5
4.1 %, x2 5.9 %; `k_z = 0.7` gives 22.0 m/s (20.5 %); `k_z = 0` gives 513 m/s,
that is, no bound. The acceleration limits, set by thrust-to-weight, move by at
most 11 % across the drag cases; the top speed is only as good as the drag
value. For that reason the guidance study keeps the interceptor/target speed
ratio, not the absolute speed, tied to real vehicles (section 1.7), and P9.1
sweeps that ratio. The polynomial remains selectable
(`aero_model="polynomial"`) and is what the fixture-parity tests use.
Implemented by `aero.rotor_drag_force_torque`.

with `h_ge = 0.15 m` and `kappa_max = 0.20`. This linear ramp is a project
choice. It is not the classical hover result of Cheeseman & Bennett (ARC R&M
3021, 1955, eq. 2), `T_g/T_inf = 1/(1 - R^2/(16 Z^2))`, which is cited only as
the reason a thrust increase near the ground exists. The ramp is inactive above
0.15 m, so no interception result depends on it.

### 6.2 Rotation

The body-frame angular momentum equation with a diagonal inertia `J` is

\[
J\dot{\boldsymbol\omega}=
\boldsymbol\tau_{\rm prop}+\boldsymbol\tau_{\rm mot}
+\boldsymbol\tau_{\rm aero}-\boldsymbol\omega\times J\boldsymbol\omega ,
\tag{EQ-SIX-2}
\]

where the last term is the inertial (gyroscopic) torque and the motor reaction
torque is

\[
\boldsymbol\tau_{\rm mot}=J_{\rm mp}\sum_{j=1}^{4}\boldsymbol\zeta_j\dot\Omega_j ,
\qquad \boldsymbol\zeta_j=\sigma_j\hat{\mathbf z},
\tag{EQ-SIX-8}
\]

with `J_mp = 2.0e-9 kg m^2`, `sigma_j = +1, -1, +1, -1`, and `Omega_dot_j` the
motor-lag derivative below.

### 6.3 Quaternion kinematics

\[
\dot{\mathbf q}=\tfrac12\,\mathbf q\otimes
\begin{bmatrix}0\\ \boldsymbol\omega\end{bmatrix},
\qquad \|\mathbf q\|=1 .
\tag{EQ-SIX-3}
\]

The integrator renormalises after every step, so the norm drift never
accumulates.

### 6.4 Rotor forces, torque assembly and motor lag

Each rotor produces thrust along the body z axis and a reaction torque about
the same axis:

\[
\mathbf f_j=c_l\Omega_j^2\hat{\mathbf z},\qquad
\boldsymbol\tau_j=\sigma_j c_d\Omega_j^2\hat{\mathbf z},
\tag{EQ-SIX-5}
\]

and the propeller contributions are assembled at the arm positions
`r_j = (+/-a, +/-a, 0)`:

\[
\mathbf f_{\rm prop}=\sum_j\mathbf f_j,\qquad
\boldsymbol\tau_{\rm prop}=\sum_j\left(\boldsymbol\tau_j+\mathbf r_j\times\mathbf f_j\right).
\tag{EQ-SIX-4}
\]

The ESC maps a normalised command `c` and battery voltage `U` to the
steady-state rotor speed

\[
\Omega_{\rm ss}(c,U)=b_0+b_1U+b_2\sqrt c+b_3c+b_4U\sqrt c,
\qquad \Omega_{\rm ss}=0\ \text{for}\ c<c_{\rm cut},
\tag{EQ-SIX-7}
\]

and each rotor follows a first-order lag

\[
\dot\Omega_j=\frac{\Omega_{{\rm ss},j}-\Omega_j}{\tau_m},
\qquad \tau_m=0.02\,{\rm s}.
\tag{EQ-SIX-6}
\]

The maximum motor speed (2800 rad/s) and efficiency (0.7) are labelled "best guess" in the predecessor simulator and have no source; they set the thrust ceiling, hence thrust-to-weight 2.05 and the acceleration limits.

**Status of the rotor constants.** `c_l = 2.618e-8` is our own fit (median of
`m g / (4 Omega^2)` over 9,179 hover samples from 10 NanoBench trajectories), not
a value published by NanoBench; rotor speed was not logged and came from an ESC
map that was later rejected (BUGS R-1), so the fit has to be repeated. `c_d` is
the assumed ratio `c_l / 480`. `tau_m = 0.02 s` is an assumed value with no
step-response identification behind it; halving or doubling it changes the
measured flight envelope by at most 4.1 % and 5.9 %
(`results/tables/quad_limits_repeatability.csv`).

The legacy simulator evaluated thrust at the half-step speed
`Omega + dt/2 * Omega_dot`; that correction is retained for fixture parity but
is first-order, so the validated path evaluates `EQ-SIX-5` at the RK4 stage
state instead.

### 6.5 Integration and rates

The state is advanced with classical RK4 at step `h`,

\[
\begin{aligned}
k_1&=f(x_n), & k_2&=f(x_n+\tfrac h2 k_1),\\
k_3&=f(x_n+\tfrac h2 k_2), & k_4&=f(x_n+h k_3),\\
x_{n+1}&=x_n+\tfrac h6(k_1+2k_2+2k_3+k_4),
\end{aligned}
\tag{EQ-SIX-9}
\]

with `q` renormalised and `Omega >= 0` after every step. The rates follow
decision S8 in `configs/quadrotor.yaml`: physics 1000 Hz, flight controller
500 Hz (two RK4 steps per control update), guidance 50 Hz, kinematic
integration 200 Hz. `pipeline.control_step` enforces the integer sub-step
ratio and holds the command constant across it. `pipeline.rk4` keeps the
historical `dt = 0.01 s` default for the P1 fixtures.

---

## 7. Guidance-to-flight-controller bridge

Desired world acceleration `a_des` (z up, gravity `g = (0,0,-9.81)`) becomes a
thrust vector

\[
\mathbf f=m(\mathbf a_{des}-\mathbf g),
\tag{EQ-BRIDGE-1}
\]

limited so that the tilt from vertical is at most `theta_max` (lateral part
scaled first, vertical part has priority) and `|f| <= T_max = 4 c_l Omega_max^2`.
The collective is found by inverting the thrust curve of 6.4 and the ESC curve:

\[
c=\left(\frac{\sqrt{|\mathbf f|/(4c_l)}-b_0}{b_2}\right)^2 .
\tag{EQ-BRIDGE-2}
\]

The desired attitude has body z along `f` and yaw fixed at `psi_d` (stated rule:
yaw held at zero), `R_d = [x_d, y_d, z_d]` with `z_d = f/|f|`,
`y_d = z_d x x_c / |z_d x x_c|`, `x_d = y_d x z_d`, `x_c = (cos psi_d, sin psi_d, 0)`:

\[
\mathbf q_d=\mathrm{quat}(R_d).
\tag{EQ-BRIDGE-3}
\]

The rate setpoint is the quaternion-error law with the firmware attitude gain
`K_p = 6 1/s`, `q_e = q_d^{-1} (x) q`, taking the shortest rotation:

\[
\boldsymbol\omega_{cmd}=-2K_p\,\mathrm{sgn}(q_{e,w})\,\mathbf q_{e,xyz}.
\tag{EQ-BRIDGE-4}
\]

Small-angle check: for `q_e ~ (1, theta/2)`, `omega_cmd = -K_p theta`, a first
order attitude loop with time constant `1/K_p`.

## 8. Flight-controller design and reward

### 8.1 Rate-loop shaping (baseline B1)

About one body axis, with the mixer output `u` as input, the plant is

\[
G(s)=\frac{K}{s(\tau s+1)},
\]

`K = d(omega_dot)/du` (measured from the model at hover) and `tau = 0.02 s`
the motor lag. A PI controller `C(s) = k_p (1 + w_i/s)` with crossover
`w_c = 1/(3 tau)` and zero `w_i = w_c/5` gives, near `w_c`, a phase
`-90 deg - atan(w_c tau) - atan(w_i/w_c) = -90 - 18.4 - 11.3 = -119.7 deg`, i.e.
a phase margin of about 60 deg. Gain at crossover equals one when

\[
k_p=\frac{w_c}{K},\qquad k_i=k_p\,w_i,\qquad w_c=\frac1{3\tau}.
\tag{EQ-FC-1}
\]

Targets (verified by `tests/unit/test_fc.py`): rise time below 0.12 s,
overshoot below 20 %, on roll/pitch.

### 8.2 Reward

With `e = omega_cmd - omega`, `r_max` the rate range, `T_err` the relative
thrust error (actual minus commanded total thrust over hover thrust),
`da` the change of action and `s` the fraction of motors at 0 or 1:

\[
r=-w_\omega\left\|\frac{\mathbf e}{r_{max}}\right\|^2-w_T T_{err}^2-w_s\|\Delta\mathbf a\|^2-w_{sat}\,s .
\tag{EQ-FC-2}
\]

Each term is normalised to be of order one at its worst expected value, so the
weights `w` express relative priority; `w_omega` dominates because rate tracking
is the task, `w_s` penalises motor chatter (action smoothness), `w_sat` keeps
headroom, `w_T` keeps the collective honest. Weights are in `configs/fc.yaml`
and are subject to the P9.3 sensitivity run; they are design choices, not
measurements.

## 9. Guidance observation and reward

### 9.1 Observation

One frame is the 16-value vector

\[
\mathbf x_t=\Big[
\frac{\Delta\mathbf p}{s_{pos}},\,
\frac{\Delta\mathbf v}{s_{vel}},\,
\frac{\mathbf v_M}{s_{vel}},\,
\frac{R}{s_{range}},\,
\frac{v_c}{s_{closing}},\,
\frac{\boldsymbol\omega_{LOS}}{s_{los}},\,
\frac{t_{go}}{s_{tgo}},\,
\mathbb 1_{\text{visible}}
\Big],
\tag{EQ-OBS-1}
\]

where `dp = est_p_T - p_M`, `dv = est_v_T - v_M`, `R = |dp|`,
`vc = -dv.dp_hat`, `omega_LOS = dp x dv / max(R,15)^2` and
`t_go = clip(R / max(vc,0.1), 0, max_time)`.  The divisors are the
per-platform `ObservationScales` (`interceptor/guidance/observation.py`); they
default to the platform envelope and are overridden by the sourced
`observation_scales` block in `configs/platforms.yaml` (P5.8, fix F18).  Every
component is clipped to `[-10, 10]`.

The actor observation concatenates the current frame with `m` history frames
sampled every `p` decisions, and `n` future samples of
`(rel_pos(3), rel_vel(3), visible(1))` produced by the constant-acceleration
prediction at horizons `q, 2q, ..., nq` steps:

\[
\mathbf o^{actor}=
\big[\mathbf x_{t},\mathbf x_{t-p},\ldots,\mathbf x_{t-mp}\big]
\;\Vert\;
\big[\mathbf z_{t+q},\ldots,\mathbf z_{t+nq}\big],
\qquad
\dim = 16(1+m)+7n .
\tag{EQ-OBS-2}
\]

With `m=3, p=2, n=3, q=5` this gives 16x4 + 7x3 = **85** actor values.  The
privileged critic block is a separate **16** values (time remaining, optimal
acceleration and velocity from the oracle, and the true target future position,
velocity and acceleration), so the total is **101**.  Implemented by
`AsymmetricObsBuilder` in `interceptor/guidance/observation.py`.

### 9.2 Reward

Let `d` be the true range, `r_init_max` the spawn-range scale and
`w_progress` the progress weight.  The progress potential is

\[
\Phi(s)=-\frac{w_{progress}\,d}{r_{init,max}},
\tag{EQ-GUIDE-1}
\]

and the progress reward is its discounted difference,

\[
r_{progress}=\gamma\,\Phi(s')-\Phi(s),
\tag{EQ-GUIDE-2}
\]

which is potential-based shaping and therefore leaves the optimal policy
unchanged (Ng, Harada & Russell, 1999).  The remaining shaping terms are

\[
r_{aux}=w_{align}\,\hat v_M\!\cdot\!\hat R
      -w_{los}\|\boldsymbol\omega_{LOS}\|
      -w_{zem}\frac{\mathrm{ZEM}}{\max(d,10)},
\tag{EQ-GUIDE-3}
\]

with `v_hat_M` the unit interceptor velocity, `R_hat` the LOS direction and
`ZEM` the zero-effort-miss norm.  Action penalties are

\[
r_{act}=-w_{res}\|\mathbf u\|^2-w_{smooth}\|\mathbf u_t-\mathbf u_{t-1}\|^2,
\tag{EQ-GUIDE-4}
\]

where `u` is the normalised residual in `[-1,1]^3` and `w_res` is 0.04 in
`apn_rl`, 0.015 in `pure_rl` and zero for the pure-APN arms.  The time cost is

\[
r_{time}=-c_{time},
\tag{EQ-GUIDE-5}
\]

and the terminal terms are

\[
r_{term}=
\begin{cases}
B_{kill}+B_{speed}\left(1-\dfrac{k}{K}\right), & \text{killed},\\
-B_{crash}, & \text{crashed},\\
-B_{div}\dfrac{d_{miss}}{r_{init,max}}, & \text{diverged},
\end{cases}
\tag{EQ-GUIDE-6}
\]

with `k` the step index and `K` the episode step budget.  The total reward is

\[
r=r_{progress}+r_{aux}+r_{act}+r_{time}+r_{term}.
\tag{EQ-GUIDE-7}
\]

The discount is chosen from the episode horizon `H` so that a terminal bonus
at the first decision retains a fraction `rho`:

\[
\gamma=\rho^{1/H}.
\tag{EQ-GUIDE-8}
\]

The notebook horizon is `H = 22 s / 0.02 s = 1100` steps.  With `rho = 0.1`,
`gamma = 0.1^(1/1100) = 0.997909`, so a 150-point kill is worth 15.0 at step 0
and `gamma^1100 = 0.1`.  The old `gamma = 0.995` discounted the same bonus to
0.6 (fix F16).  The weights and the discount are in `configs/guidance.yaml`
(`reward:`) and mirrored for PPO in `configs/model.yaml`; the implementation is
`interceptor/guidance/reward.py::GuidanceReward`.  The exact v2 notebook reward
is retained in `notebook_reward()` for golden-fixture replay only.

## 10. Algorithms

The algorithm derivations, criteria table and the pre-registered decision rule
are written in `docs/algorithm_selection.md` (P6.1/P6.2).  PPO, SAC and TD3 are
constructed by `training/algos.py`; the selected algorithm is recorded in
`configs/training.yaml::selected_algo`.  Reserved equation IDs `EQ-ALG-*` are
not needed because no algorithm equation is evaluated in code.

## 11. Evaluation statistics

The protocol reports a binomial kill rate, a miss-distance median/IQR, a paired
McNemar test on kill outcomes and a paired bootstrap on miss distance.  The
implementation is `interceptor/eval/stats.py`.

**Evaluation protocol.** Each reported cell (platform, arm, target set) uses 100
episodes on fixed seeds starting at 10000, shared across arms so that
comparisons are paired. The count follows from the precision wanted: the Wilson
95 % half-width (`EQ-EVAL-1`) is largest at `p_hat = 0.5`, where 100 episodes
give at most ±9.6 percentage points (30 episodes would give ±16.9, 400 would
give ±4.9). Differences between arms smaller than that are not claimed from the
intervals; they are tested with the paired McNemar test (11.3). Sweeps and
smoke tests use fewer episodes and are labelled as such. Training budgets
(`configs/training.yaml`: 1,000,000 flight-controller steps, 5,000,000 guidance
steps, 200,000 per run for algorithm selection, 3 training seeds) were set from
the throughput benchmark (`results/tables/training_throughput.json`: 281 and
2,583 steps/s on the laptop, about 1.0 h and 0.5 h per run at those rates; the
Colab figures are still to be measured).

### 11.1 Wilson score interval

For `k` kills in `n` episodes with `p_hat = k/n` and `z = 1.959964` (95 %),

\[
\frac{\hat p+\frac{z^2}{2n}
      \pm z\sqrt{\frac{\hat p(1-\hat p)}{n}+\frac{z^2}{4n^2}}}
     {1+\frac{z^2}{n}}.
\tag{EQ-EVAL-1}
\]

Unlike the normal approximation it stays inside `[0,1]` and is reliable for
the small per-cell counts in the protocol.

### 11.2 Median and IQR

For a sample `x_(1) <= ... <= x_(n)`, the reported location and spread are

\[
\tilde x=\operatorname{median}(x),\qquad
\mathrm{IQR}=Q_{0.75}-Q_{0.25},
\tag{EQ-EVAL-2}
\]

where the quartiles use linear interpolation between order statistics.  The
median is robust to the heavy right tail of miss distances.

### 11.3 Exact McNemar test

For paired binary outcomes let `b` be the number of seeds/episodes where A
kills and B does not, and `c` the reverse.  Under the null `b` and `c` are
exchangeable, so

\[
p=2\sum_{k=0}^{\min(b,c)}\binom{b+c}{k}2^{-(b+c)},
\qquad p\leftarrow\min(p,1).
\tag{EQ-EVAL-3}
\]

This is the exact two-sided binomial test on the discordant pairs (the
implementation uses `scipy.stats.binomtest`).

### 11.4 Paired bootstrap

For paired samples `(x_i, y_i)`, draw `B` index sets with replacement and form

\[
\Delta_b = \operatorname{stat}(\{x_{i}:i\in I_b\})
          - \operatorname{stat}(\{y_{i}:i\in I_b\}),
\qquad
\text{CI}=[\Delta_{(0.025)},\Delta_{(0.975)}].
\tag{EQ-EVAL-4}
\]

The same indices are used for both arms, which preserves the pairing.  The
default is `B = 2000` with the caller's explicit seed.

