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

Classical guidance accepts `3 <= N' <= 5`; below 3 the miss distance grows
with the initial heading error, above 5 the loop amplifies noise and target
acceleration estimates. The admissible range is

\[
N'_{\min}=3,\qquad N'_{\max}=5.
\tag{EQ-APN-7}
\]

P5.11 sweeps `N'` over this range separately for A1/A2 and each platform and
selects by kill rate then median miss. The result is written back to
`configs/platforms.yaml` with `source: sweep P5.11`.

### 1.6 Kill radius and engagement geometry

The kill radius is a physical rule, not a tuned threshold:

\[
r_{\rm kill} = \max(2\,\ell_{\rm span}, r_{\rm lethal}).
\tag{EQ-GEO-4}
\]

Quad: `2 * 0.0397 m` rotor span plus a 1.5 m debris margin gives 3 m. Aerosonde:
2.9 m span plus 9 m fragment radius gives 12 m. AIM5: the warhead lethal radius
is 10 m. Initial range follows a stated time-to-intercept window:

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

with `rho_v = 0.40 (quad), 0.625 (fixed wing), 0.33 (missile)` and
`rho_a = 0.48, 1.0, 0.30`. The ratios are design choices, not measurements;
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

at sea level, which gives `v_max = 40 m/s` for the Aerosonde configuration.
Excess thrust gives the axial acceleration envelope

\[
a_{t,\max}=\frac{T_{\max}(V)-D(V)}{m},\qquad
a_{t,\min}=\frac{T_{\rm idle}(V)-D(V)}{m}.
\tag{EQ-FW-4}
\]

The load factor and negative-load limits are

\[
n=\frac{L}{mg},\qquad |n|\le n_{\max}=10.5,
\tag{EQ-FW-5}
\]

and the coordinated-turn rate is

\[
\omega=\frac{g\sqrt{n^2-1}}{V}.
\tag{EQ-FW-6}
\]

The acceleration lag is the roll/pitch first-order response,

\[
\tau_a=\tau_{\rm roll}\approx 0.12\,{\rm s},
\tag{EQ-FW-7}
\]

and the jerk limit is `a_n,max / tau_a = 858 m/s^3` analytically, capped at
180 m/s^3 by the actuator-rate budget. The values are evaluated at sea level
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

with `sigma_p0 = 0.5 m`, `alpha_r = 0.0025 rad`, and `kappa` the evaluation
noise scale (headline 1.0). Velocity and acceleration noise are proportional
to the platform's nominal target speed and manoeuvre limit:

\[
\sigma_v=\kappa\,0.03\,v_{T,{\rm nom}},\qquad
\sigma_a=\kappa\,0.18\,a_{T,\max}.
\tag{EQ-SEEK-2}
\]

A Bernoulli draw with probability 0.04 drops the measurement; the filter
coasts with the constant-acceleration prediction and decays acceleration by
0.98 per 20 ms step (`EQ-SEEK-3`). Otherwise the alpha-beta-gamma update is

\[
\begin{aligned}
\hat p^+ &= \hat p^-+\alpha_p(y-\hat p^-),\\
\hat v^+ &= \hat v^-+\alpha_v(y-\hat p^-)/\Delta t,\\
\hat a^+ &= \hat a^-+\alpha_a(y-\hat p^-)/\Delta t^2,
\end{aligned}
\tag{EQ-SEEK-4}
\]

with `(alpha_p, alpha_v, alpha_a) = (0.65, 0.55, 0.35)` at 50 Hz. The same
gains are used for every platform so the noise schedule is comparable.

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

and the discrete update is `u += theta(mean-u)dt + sigma sqrt(dt) N(0,1)`.
The mapping used by P3.6 is

\[
\theta=V/(\sqrt{3}L_u),\qquad
\sigma=\sigma_u\sqrt{2\theta},
\tag{EQ-WIND-2}
\]

with low-altitude `L_u` and `sigma_u` from MIL-HDBK-1797. The simulator uses a
minimum airspeed of 5 m/s in the Dryden denominator.

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

## 6-10. Forward references

Sections 6 (6-DOF quadrotor), 7 (tracker), 8 (flight-control reward), 9
(guidance reward and discount), and 10 (closed-loop bridge) are completed by
the corresponding phases (P3, P5, P4, P5, P8). This file is generated as one
document so the equation-reference check sees the final IDs.

Equation IDs already reserved by later phases: `EQ-SIX-*`, `EQ-TRK-*`,
`EQ-FC-*`, `EQ-GUIDE-*`, `EQ-BRIDGE-*`.
