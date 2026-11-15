# Equations reference

Every formula evaluated in the code base, written from the source files, with inputs, outputs and the implementing file.
Derivations and sources are in `docs/mathematics.md` (same equation IDs where one exists), `docs/algorithm_selection.md` and `docs/source_table.md`.
Where the code and `docs/mathematics.md` differ, this document follows the code and Part G lists the difference.

**Conventions.** World frame: x north, y west, z up. Gravity $\mathbf g = (0, 0, -9.81)\ \mathrm{m/s^2}$. SI units, radians. Subscript $M$ = interceptor, $T$ = target. Quaternion $\mathbf q = [w, x, y, z]$, body to world. $\operatorname{sat}(\mathbf x, L) = \mathbf x\,\min(1, L/\lVert\mathbf x\rVert)$ limits a vector's length to $L$.

**Rates.** Physics 1000 Hz, flight controller 500 Hz, guidance decision 50 Hz ($\Delta t = 0.02$ s).

**Signal chain.**

```
target truth -> seeker noise -> tracker -> engagement geometry -> APN (+ RL residual)
   -> platform constraints -> point-mass kinematics                  (all three platforms)
   -> bridge -> rate controller -> motors -> 6-DOF rigid body        (quad closed loop only)
```

**Parts.** A guidance and path planning. B quadrotor simulator. C flight controller, bridge and closed loop. D learning algorithms. E evaluation statistics. F older code path still in the package. G differences between code and maths document, and coverage.

---

## Part A. Guidance (path planning)

### A1. Engagement geometry

Code: `interceptor/guidance/kinematics.py::compute_engagement_kinematics`

| | |
|---|---|
| **Inputs** | $\mathbf p_T, \mathbf v_T, \mathbf a_T$ (target, estimated or true), $\mathbf p_M, \mathbf v_M$ (interceptor), $r_{\min}$ |
| **Outputs** | $\mathbf r$, $r$, $\hat{\mathbf r}$, $\mathbf v_{rel}$, $v_{cl}$, $\dot{\hat{\mathbf r}}$, $t_{go}$, $\mathbf Z_\perp$, $\lVert\mathbf Z_\perp\rVert$ |

Relative position and line of sight (EQ-GEO-1), with $r$ floored at $10^{-6}$ m:

$$\mathbf r = \mathbf p_T - \mathbf p_M, \qquad r = \lVert\mathbf r\rVert, \qquad \hat{\mathbf r} = \mathbf r / r$$

Relative velocity and closing speed (EQ-GEO-2):

$$\mathbf v_{rel} = \mathbf v_T - \mathbf v_M, \qquad v_{cl} = -\mathbf v_{rel}\cdot\hat{\mathbf r}$$

Line-of-sight rate (EQ-GEO-3), with $r_{\min} = 15$ m (2 m for the raw-APN arm):

$$\dot{\hat{\mathbf r}} = \frac{\mathbf v_{rel} + v_{cl}\,\hat{\mathbf r}}{\max(r, r_{\min})}$$

Time to go (EQ-APN-4):

$$t_{go} = \begin{cases}\operatorname{clip}\!\left(\dfrac{r}{\max(v_{cl}, 1)},\ 0.05,\ 20\right) & v_{cl} > 0\\[2mm] 20 & v_{cl}\le 0\end{cases}$$

Zero-effort miss (EQ-APN-5): the miss if neither vehicle accelerates further.

$$\mathbf Z = \mathbf v_{rel}\,t_{go} + \tfrac12\,\mathbf a_T\,t_{go}^2, \qquad \mathbf Z_\perp = \mathbf Z - (\mathbf Z\cdot\hat{\mathbf r})\,\hat{\mathbf r}$$

### A2. Augmented proportional navigation

Code: `interceptor/guidance/kinematics.py::compute_3d_apn_command`, `interceptor/guidance/apn.py::compute_axial_acceleration`

| | |
|---|---|
| **Inputs** | geometry from A1, target acceleration estimate $\hat{\mathbf a}_T$, $\mathbf v_M$, platform limits, navigation constant $N'$ |
| **Output** | commanded acceleration $\mathbf a_{APN}$ (world frame, m/s²) |

Perpendicular projection: $\mathbf x_\perp = \mathbf x - (\mathbf x\cdot\hat{\mathbf r})\,\hat{\mathbf r}$.

Proportional navigation (EQ-APN-1):

$$\mathbf a_{PN} = N'\,\frac{\mathbf v_{rel,\perp}}{\max(t_{go}, 0.06)}$$

Equivalent zero-effort-miss form for a non-manoeuvring target (EQ-APN-6):

$$\mathbf a_{PN} = N'\,\frac{\mathbf Z_\perp}{t_{go}^2}$$

Augmentation for target acceleration (EQ-APN-2):

$$\mathbf a_{aug} = \tfrac12 N'\,\hat{\mathbf a}_{T,\perp}, \qquad \mathbf a_{APN} = \mathbf a_{PN} + \mathbf a_{aug}$$

Axial speed-management term (EQ-APN-3). It is zero only when both $\lVert\mathbf v_M\rVert \ge 0.96\,v_{\max}$ and $v_{cl} \ge 0.45\,v_{\max}$; otherwise

$$\mathbf a_{axial} = \begin{cases}0.8\,a_{t,\max}\,\hat{\mathbf r} & \text{omnidirectional}\\ a_{t,\max}\,\hat{\mathbf v}_M & \text{non-holonomic}\end{cases}$$

Navigation constant range (EQ-APN-7): $3 \le N' \le 5$, selected per platform and arm by sweep (`scripts/sweep_nav.py`).

### A3. Command composition (the four arms)

Code: `interceptor/guidance/env.py::step`

| | |
|---|---|
| **Inputs** | $\mathbf a_{APN}$, $\mathbf a_{axial}$, policy action $\mathbf u\in[-1,1]^3$, residual authority $\beta = 0.35$ |
| **Output** | raw acceleration command $\mathbf a_{raw}$ |

| Arm | Measurements | Command |
|---|---|---|
| A1 raw APN | raw seeker output, $r_{\min} = 2$ m | $\mathbf a_{raw} = \mathbf a_{APN}$ |
| A2 filtered APN | filtered estimate, $r_{\min} = 15$ m | $\mathbf a_{raw} = \mathbf a_{APN} + \mathbf a_{axial}$ |
| A3 APN + RL | filtered estimate | $\mathbf a_{raw} = \mathbf a_{APN} + \mathbf a_{axial} + \beta\,a_{n,\max}\,\mathbf u$ |
| A4 pure RL, omnidirectional | none used by the law | $\mathbf a_{raw} = a_{n,\max}\,\mathbf u$ |
| A4 pure RL, non-holonomic | none used by the law | $\mathbf a_{raw} = a_{n,\max}\,\mathbf u + 0.5\,a_{t,\max}\,\hat{\mathbf v}_M$ |

### A4. Generic interceptor kinematics (point mass)

Code: `interceptor/guidance/kinematics.py::apply_interceptor_constraints`, `heading_rate_from_acceleration`; integration in `interceptor/guidance/env.py::step`

| | |
|---|---|
| **State** | $\mathbf p_M$, $\mathbf v_M$, $\mathbf a_M$ |
| **Inputs** | $\mathbf a_{raw}$, platform limits ($v_{\min}, v_{\max}, a_{t,\min}, a_{t,\max}, a_{n,\max}, \tau_a, j_{\max}, \omega_{\max}$), $\Delta t$ |
| **Outputs** | new $\mathbf p_M, \mathbf v_M, \mathbf a_M$, body turn rate $\boldsymbol\omega_M$; logged velocity command $[\mathbf v_M, \boldsymbol\omega_M]$ |

**Step 1. Decompose** along and across the velocity (EQ-KIN-1), with $s = \lVert\mathbf v_M\rVert$, $\hat{\mathbf v}_M = \mathbf v_M / s$:

$$a_t = \mathbf a_{raw}\cdot\hat{\mathbf v}_M, \qquad \mathbf a_n = \mathbf a_{raw} - a_t\,\hat{\mathbf v}_M$$

**Step 2. Envelope** (EQ-KIN-2).

Non-holonomic platform (fixed-wing, missile):

$$a_t \leftarrow \operatorname{clip}(a_t,\ a_{t,\min},\ a_{t,\max}), \qquad \mathbf a_n \leftarrow \operatorname{sat}(\mathbf a_n,\ a_{n,\max})$$

$$a_t \leftarrow 0 \ \text{ if } (s \le v_{\min} \text{ and } a_t < 0) \text{ or } (s \ge v_{\max} \text{ and } a_t > 0)$$

$$\mathbf a_{cmd} = a_t\,\hat{\mathbf v}_M + \mathbf a_n$$

Omnidirectional platform (quad): the total vector is limited, with no tangential/normal split:

$$\mathbf a_{cmd} = \operatorname{sat}(\mathbf a_{raw},\ a_{n,\max})$$

**Step 3. Acceleration lag and jerk limit** (EQ-KIN-3):

$$\dot{\mathbf a}_M = \operatorname{sat}\!\left(\frac{\mathbf a_{cmd} - \mathbf a_M}{\max(\tau_a, 10^{-3})},\ j_{\max}\right)$$

**Step 4. Integrate** (semi-implicit Euler):

$$\mathbf a_M \leftarrow \mathbf a_M + \dot{\mathbf a}_M\,\Delta t, \qquad \mathbf v_M \leftarrow \mathbf v_M + \mathbf a_M\,\Delta t$$

Speed clamp: if the platform cannot hover and $s < v_{\min}$, rescale $\mathbf v_M$ to $v_{\min}$; if $s > v_{\max}$, rescale to $v_{\max}$. Then

$$\mathbf p_M \leftarrow \mathbf p_M + \mathbf v_M\,\Delta t$$

**Step 5. Turn rate** (EQ-KIN-4), zero when $s < 10^{-3}$:

$$\boldsymbol\omega_M = \operatorname{sat}\!\left(\frac{\hat{\mathbf v}_M \times \mathbf a_M}{\max(s, 1)},\ \omega_{\max}\right)$$

**Closest approach within a step** (EQ-GEO-7). Code: `closest_approach_distance`. Inputs: target and interceptor positions at the previous and current step. Output: minimum distance over the step.

$$\mathbf r_0 = \mathbf p_T^{prev} - \mathbf p_M^{prev}, \qquad \Delta\mathbf r = (\mathbf p_T - \mathbf p_T^{prev}) - (\mathbf p_M - \mathbf p_M^{prev})$$

$$\lambda^* = \operatorname{clip}\!\left(-\frac{\mathbf r_0\cdot\Delta\mathbf r}{\lVert\Delta\mathbf r\rVert^2},\ 0,\ 1\right), \qquad d_{CPA} = \lVert\mathbf r_0 + \lambda^*\Delta\mathbf r\rVert$$

The environment uses $d_{eff} = \min(r, d_{CPA})$ and tracks the episode minimum $d_{miss}$.

### A5. What makes each platform specific

Sourced values: `configs/platforms.yaml`. Built-in presets: `interceptor/guidance/platform.py`.

| Quantity | Quad (Crazyflie) | Fixed-wing (Aerosonde) | Missile (AIM5) |
|---|---|---|---|
| Can hover | yes | no | no |
| Omnidirectional thrust | yes | no | no |
| $v_{\min}$ / $v_{\max}$ (m/s) | 0 / 45 | 15 / 40 | 180 / 1200 |
| $a_{t,\min}$ / $a_{t,\max}$ (m/s²) | −31.4 / 31.4 | −19.6 / 41.2 | −39.2 / 176.6 |
| $a_{n,\max}$ (m/s²) | 51.0 | 102.9 | 358.1 |
| $\tau_a$ (s) | 0.05 | 0.12 | 0.10 |
| $j_{\max}$ (m/s³) | 300 | 180 | 1000 |
| $\omega_{\max}$ (rad/s) | 6.0 | 3.5 | 12.0 |
| Kill radius (m) | 3 | 12 | 10 |
| Initial range (m) | 150–450 | 400–900 | 3500–7000 |
| Initial altitude (m) | 150 | 1000 | 4500 |
| Episode limit (s) | 18 | 36 | 20 |
| Target speed / accel limit | 18 m/s / 24.5 m/s² | 25 m/s / 41.2 m/s² | 400 m/s / 107.9 m/s² |

Notes:
- The environment uses the built-in presets unless limits are passed in. The quad preset equals the table. The presets differ for the fixed-wing ($v_{\min}$ 20, $v_{\max}$ 255 m/s, range 1200–2800 m, altitude 2000 m, target 105 m/s, 22 s) and the missile ($a_{n,\max} = 38g = 372.8$ m/s², $\tau_a = 0.07$ s).
- The quad row is the analytic kinematic envelope. The limits measured through the flight controller (C9) are lower: lateral 14.8 m/s², vertical 9.4 m/s², $v_{\max}$ 22 m/s, $\tau_a$ 0.17 s, jerk 85 m/s³.
- `a_n_down_max` is defined but is not applied by `apply_interceptor_constraints`.

Scenario rules:

$$r_{kill} = \max(2\,\ell_{span},\ r_{lethal}) \quad\text{(EQ-GEO-4)}$$

$$r_{0,\min} = v_{cl,0}\,t_{\min}, \qquad r_{0,\max} = v_{cl,0}\,t_{\max} \quad\text{(EQ-GEO-5)}$$

$$h_0 = 35 + \Delta h_{budget} \quad\text{(EQ-GEO-6)}$$

$$v_{T,0} = \rho_v\,v_{M,\max}, \qquad a_{T,\max} = \rho_a\,a_{M,n,\max} \quad\text{(EQ-TGT-1)}$$

### A6. Fixed-wing limits (Aerosonde)

Code: `interceptor/guidance/airframes.py`; data: `configs/airframes/aerosonde.yaml`

| | |
|---|---|
| **Inputs** | mass $m$, wing area $S$, air density $\rho$, $C_{L,\max}$, $C_{D0}$, Oswald factor $e$, aspect ratio $AR$, propeller $C_T(J)$, $C_Q(J)$, load-factor limit $n_{\max}$ |
| **Outputs** | $v_{\min}$, $v_{\max}$, $a_{t,\min}$, $a_{t,\max}$, $a_{n,\max}$, $\omega_{\max}$, $\tau_a$, $j_{\max}$ |

Point-mass equations of motion (EQ-FW-1), with speed $V$, flight-path angle $\gamma$, bank $\phi$, angle of attack $\alpha$:

$$\dot V = \frac{T\cos\alpha - D}{m} - g\sin\gamma, \qquad \dot\gamma = \frac{(L + T\sin\alpha)\cos\phi}{mV} - \frac{g\cos\gamma}{V}$$

Aerodynamic forces:

$$L = \tfrac12\rho V^2 S\,C_L, \qquad D = \tfrac12\rho V^2 S\,C_D, \qquad C_D = C_{D0} + \frac{C_L^2}{\pi e\,AR}$$

Propeller, with advance ratio $J = V/(nD_p)$:

$$T = \rho n^2 D_p^4\,C_T(J), \qquad Q = \rho n^2 D_p^5\,C_Q(J)$$

Stall speed gives $v_{\min}$ (EQ-FW-2, `aerosonde_stall_speed`):

$$V_{stall} = \sqrt{\frac{2mg}{\rho S\,C_{L,\max}}}$$

Maximum level speed is the root of (EQ-FW-3):

$$T(V) = D(V)$$

With the book propulsion model (Beard & McLain eqs. 4.17-4.22) and the mavsim parameters the root is 32.43 m/s, so $v_{\max} = 32.4$ m/s; at 25 m/s the same model gives $a_{t,\max} = 2.47$ and $a_{t,\min} = -3.02$ m/s² (BUGS #32).

Axial acceleration envelope (EQ-FW-4):

$$a_{t,\max} = \frac{T_{\max}(V) - D(V)}{m}, \qquad a_{t,\min} = \frac{T_{idle}(V) - D(V)}{m}$$

Load factor (EQ-FW-5, `aerosonde_load_factor`):

$$n = \frac{L}{mg} = \frac{\rho V^2 S\,C_L}{2mg}, \qquad -0.4\,n_{\max} \le n \le n_{\max} = 3.8\ \text{(14 CFR 23.337, normal category)}, \qquad a_{n,\max} = n_{\max}\,g$$

Coordinated-turn rate (EQ-FW-6, `aerosonde_turn_rate`):

$$\omega = \frac{g\sqrt{\max(n^2 - 1,\ 0)}}{V}$$

Lag and jerk (EQ-FW-7): $\tau_a = 0.12$ s (project choice, v1-notebook value; BUGS #32); $a_{n,\max}/\tau_a = 37.278/0.12 = 311$ m/s³, and the configured $j_{\max} = 180$ m/s³ is a project choice below it.

### A7. Missile limits (AIM5)

Code: `interceptor/guidance/airframes.py::aim5_normal_force_coefficient`, `aim5_gmax`, `bilinear`; data: `configs/airframes/aim5.yaml`

| | |
|---|---|
| **Inputs** | mass deck $m(t)$, thrust deck, aerodynamic decks $C_D(\alpha, M)$, $C_L(\alpha, M)$, reference area $S$, dynamic pressure $q$, $\alpha_{\max}$ |
| **Outputs** | $v_{\min}$, $v_{\max}$, $a_{t,\min}$, $a_{t,\max}$, $a_{n,\max}$, $\omega_{\max}$, $\tau_a$, $j_{\max}$ |

Boost-coast mass and thrust (EQ-MS-2), with nozzle exit area $a_e$:

$$m(t) = \text{deck}(t), \qquad T(t) = \text{deck}(t) + (p_{SL} - p)\,a_e$$

Axial dynamics (EQ-MS-3):

$$a_t = \frac{T - D}{m}, \qquad D = q\,S\,C_D(\alpha, M), \qquad \dot V = a_t - g\sin\gamma$$

Adopted axial limits (EQ-MS-4): $a_{t,\max} = 18g$, $a_{t,\min} = -4g$.

Table lookup: bilinear interpolation with edge clamping. For $x\in[x_i, x_{i+1}]$, $y\in[y_j, y_{j+1}]$, $t_x = (x - x_i)/(x_{i+1} - x_i)$, $t_y = (y - y_j)/(y_{j+1} - y_j)$:

$$z = (1 - t_x)(1 - t_y)\,z_{i,j} + t_x(1 - t_y)\,z_{i+1,j} + (1 - t_x)\,t_y\,z_{i,j+1} + t_xt_y\,z_{i+1,j+1}$$

Normal-force limit (EQ-MS-1):

$$C_{N,\max} = C_D(\alpha_{\max}, M)\sin\alpha_{\max} + C_L(\alpha_{\max}, M)\cos\alpha_{\max}$$

$$n_{\max} = \frac{C_{N,\max}\,q\,S}{mg}, \qquad a_{n,\max} = n_{\max}\,g$$

At 10 km, Mach 2, 38.6 kg: $C_{N,\max} = 10.865$, $q = 71.97$ kPa, $S = 0.01767$ m², so $n_{\max} = 36.5$ and $a_{n,\max} = 358.1$ m/s².

Turn rate (EQ-MS-5) and autopilot lag (EQ-MS-6):

$$\omega_{\max} = \frac{a_{n,\max}}{V}, \qquad \tau_a = 0.1\ \mathrm{s}, \qquad j_{\max} = \frac{a_{n,\max}}{4\tau_a}$$

### A8. Scenario generation

Code: `interceptor/guidance/env.py::_spawn`

| | |
|---|---|
| **Inputs** | platform limits, random generator |
| **Outputs** | initial $\mathbf p_M, \mathbf v_M$, $\mathbf p_{T,0}, \mathbf v_{T,0}$, target behaviour |

Interceptor start: $\mathbf p_M = (0, 0, h_0)$, $\mathbf a_M = \mathbf 0$.

Target position, with range $d\sim U(r_{0,\min}, r_{0,\max})$, azimuth $\psi_a\sim U(\pm35°)$, elevation $\epsilon\sim U(\pm15°)$:

$$\hat{\mathbf d} = (\cos\epsilon\cos\psi_a,\ -\cos\epsilon\sin\psi_a,\ \sin\epsilon), \qquad \mathbf p_{T,0} = \mathbf p_M + d\,\hat{\mathbf d}, \qquad z_{T,0} \ge 50\ \mathrm m$$

Target velocity, with heading $\psi\sim U(\pm\pi)$, pitch $\theta\sim U(\pm10°)$, speed $v\sim U(0.75, 1.15)\,v_{T,nom}$:

$$\mathbf v_{T,0} = v\,(\cos\theta\cos\psi,\ -\cos\theta\sin\psi,\ \sin\theta)$$

Interceptor velocity points at the target: $\mathbf v_M = s_0\,\hat{\mathbf d}$, $s_0\sim U(\max(v_{\min}, 0.65\,v_{\max}),\ 0.85\,v_{\max})$.

Behaviour sampling (EQ-MAN-1): each of the 16 laws has weight 1, except the hard set (spiral, barrel roll, bang-bang, terminal break, split-S, scissor, energy depletion) with weight 2.5; probabilities are the normalised weights.

### A9. Target motion

#### A9.1 Evasive target used by the environment

Code: `interceptor/guidance/targets.py::GenericEvasiveTarget`

| | |
|---|---|
| **Inputs** | target state, time $t$, geometry from A1 (true states), limit $a_{\max} = a_{T,\max}$, nominal speed $v_{nom}$ |
| **Outputs** | target $\mathbf p_T, \mathbf v_T, \mathbf a_T$ |

Path frame, with $V = \lVert\mathbf v_T\rVert$, $\hat{\mathbf t} = \mathbf v_T/V$ and $h = \sqrt{\hat t_x^2 + \hat t_y^2}$:

$$\mathbf n_h = \left(\frac{\hat t_y}{h},\ -\frac{\hat t_x}{h},\ 0\right), \qquad \mathbf n_v = -\frac{\hat{\mathbf t}\times\mathbf n_h}{\lVert\hat{\mathbf t}\times\mathbf n_h\rVert}$$

Heading and climb angle: $\psi = \operatorname{atan2}(-v_y, v_x)$, $\gamma = \operatorname{atan2}(v_z, \sqrt{v_x^2 + v_y^2})$. Angle differences are wrapped: $\operatorname{wrap}(x) = \operatorname{atan2}(\sin x, \cos x)$.

Per-episode random parameters: phase $\phi\sim U(0, 2\pi)$, frequency $\omega\sim U(1.1, 2.0)$ rad/s, scissor sign $\pm1$. Reference heading $\psi_{ref}$ is the initial heading. Helper: for a direction $\mathbf w$, $\operatorname{dir}_\perp(\mathbf w) = \dfrac{\mathbf w - (\mathbf w\cdot\hat{\mathbf t})\hat{\mathbf t}}{\lVert\cdot\rVert}$ (falls back to $\mathbf n_h$).

| # | Law | Commanded acceleration | ID |
|---|---|---|---|
| 1 | None | $\mathbf 0$ | EQ-MAN-2 |
| 2 | Constant g | $a_{\max}\mathbf n_h$ for $t \ge 0.25$ s | EQ-MAN-3 |
| 3 | Sinusoidal weave | $a_{\max}\sin(\omega t + \phi)\,\mathbf n_h$ | EQ-MAN-4 |
| 4 | Barrel roll | $a_{\max}[\cos(\omega t + \phi)\,\mathbf n_h + \sin(\omega t + \phi)\,\mathbf n_v]$ | EQ-MAN-5 |
| 5 | Terminal break | for $t_{go}\le 2.5$ s: $a_{\max}\,\dfrac{(\hat{\mathbf t}\times\hat{\mathbf r})\times\hat{\mathbf t}}{\lVert\cdot\rVert}$ | EQ-MAN-6 |
| 6 | Energy depletion | $0.25a_{\max}\hat{\mathbf t} + 0.7a_{\max}\operatorname{dir}_\perp(\hat{\mathbf r}) + 0.4a_{\max}\sin(\omega t)\,\mathbf n_h$ | EQ-MAN-7 |
| 7 | Split-S | $-a_{\max}\mathbf n_v + 0.2a_{\max}\hat{\mathbf t}$ until $\lvert\operatorname{wrap}(\psi - \psi_{ref})\rvert > 2.8$ and $\lvert v_z\rvert < 0.2\max(V, 1)$; then $0.2a_{\max}\hat{\mathbf t}$ | EQ-MAN-8 |
| 8 | Square bang-bang | $a_{\max}\operatorname{sign}(\sin(\omega t + \phi))\,\mathbf n_h$ | EQ-MAN-9 |
| 9 | Scissor | $-0.25a_{\max}\hat{\mathbf t} + \sigma a_{\max}\mathbf n_h$; $\sigma\to-1$ when $\operatorname{wrap}(\psi - \psi_{ref}) \ge 65°$, $\sigma\to+1$ when $\le -65°$ | EQ-MAN-10 |
| 10 | Notching | $\operatorname{clip}\!\left(-4\,\mathbf v_T\cdot\hat{\mathbf r} + \dfrac{V^2}{\max(r, 50)},\ \pm a_{\max}\right)\operatorname{dir}_\perp(\hat{\mathbf r})$ | EQ-MAN-11 |
| 11 | Turn into missile | $a_{\max}\operatorname{dir}_\perp(-\hat{\mathbf r}) + 0.2a_{\max}\hat{\mathbf t}$ | EQ-MAN-12 |
| 12 | Crank | $\psi_{des} = \operatorname{atan2}(\hat r_y, \hat r_x) \pm 60°$, side $= \operatorname{sign}(\sin(0.35t + \phi))$; $\operatorname{clip}(3\max(V, 5)\operatorname{wrap}(\psi_{des} - \psi),\ \pm a_{\max})\,\mathbf n_h$ | EQ-MAN-13 |
| 13 | Speed change | $0.8a_{\max}\operatorname{sign}(\sin(1.2t + \phi))\,\hat{\mathbf t}$ | EQ-MAN-14 |
| 14 | Minimum-range turn | for $t_{go}\le 4$ s: $-0.3a_{\max}\hat{\mathbf t} + a_{\max}\mathbf n_h$ | EQ-MAN-15 |
| 15 | Dive to dense air | $\gamma_d = -30°$ above 100 m, else 0; $0.25a_{\max}\hat{\mathbf t} + \operatorname{clip}(3\max(V, 5)(\gamma_d - \gamma),\ \pm a_{\max})\,\mathbf n_v$ | EQ-MAN-16 |
| 16 | Spiral corkscrew | $\theta = \omega t + 0.18t^2 + \phi$; $0.2a_{\max}\hat{\mathbf t} + a_{\max}[\cos\theta\,\mathbf n_h + \sin\theta\,\mathbf n_v]$ | EQ-MAN-17 |

Target integration (no lag, no jerk limit):

$$\mathbf a_T = \operatorname{sat}(\mathbf a_{law},\ a_{\max}), \qquad \mathbf v_T \leftarrow \operatorname{sat}(\mathbf v_T + \mathbf a_T\Delta t,\ 1.3\,v_{nom}), \qquad \mathbf p_T \leftarrow \mathbf p_T + \mathbf v_T\Delta t$$

Floor at 35 m: if $z_T < 35$, set $z_T = 35$ and, if descending, $v_z \leftarrow -0.5\,v_z$.

Constant-acceleration look-ahead used by the critic (`state_at_future`, horizon $15\Delta t$):

$$\mathbf p_T(t + \Delta) = \mathbf p_T + \mathbf v_T\Delta + \tfrac12\mathbf a_T\Delta^2, \qquad \mathbf v_T(t + \Delta) = \mathbf v_T + \mathbf a_T\Delta$$

#### A9.2 Static and polynomial targets (non-evasive set)

Code: `interceptor/guidance/targets.py::StaticTarget`, `PolynomialTarget`; `interceptor/envs/target_generator.py::PolynomialTargetPath`

Static: $\mathbf v_T = \mathbf a_T = \mathbf 0$.

Polynomial path of order 1 to 3, duration $T$, $\tau = \operatorname{clip}(t/T, 0, 1)$:

$$\mathbf p_T(t) = \mathbf p_0 + \boldsymbol\delta\,p(\tau), \qquad \mathbf v_T = \frac{\boldsymbol\delta\,p'(\tau)}{T}, \qquad \mathbf a_T = \frac{\boldsymbol\delta\,p''(\tau)}{T^2}, \qquad \mathbf j_T = \frac{6a\,\boldsymbol\delta}{T^3}\ (\text{order 3})$$

$$p(\tau) = \begin{cases}\tau & \text{order 1}\\ a\tau^2 + (1 - a)\tau & \text{order 2}\\ a\tau^3 + b\tau^2 + (1 - a - b)\tau & \text{order 3}\end{cases}$$

Coefficients $a\sim U(\pm0.8)$, $b\sim U(\pm1)$ are resampled until $p$ is non-negative and monotone on $[0, 1]$. End point: $\mathbf p_{end} = \mathbf p_0 + 0.5\,d\,\boldsymbol\xi$, $\boldsymbol\xi\sim\mathcal N(0, I)$, $z\ge 50$ m. Speed cap $v_{cap} = 1.15\,v_{T,nom}$:

$$\boldsymbol\delta = (\mathbf p_{end} - \mathbf p_0)\,\min\!\left(1,\ \frac{v_{cap}\,T}{\max_\tau\lvert p'(\tau)\rvert\,\lVert\mathbf p_{end} - \mathbf p_0\rVert}\right)$$

#### A9.3 Platform-constrained target generator (ported reference model)

Code: `interceptor/guidance/maneuvers.py::EvasiveTargetGenerator`, `ProportionalNavigationInterceptor`. It supplies the `ManeuverType` and `PlatformType` enumerations and is pinned by a parity test; the environment itself uses A9.1.

Target-side limits (its own presets):

| | Quad | Fixed-wing | Missile |
|---|---|---|---|
| $v_{\min}$ / $v_{\max}$ (m/s) | 0 / 25 | 20 / 220 | 180 / 1500 |
| $a_{t,\max}$ / $a_{t,\min}$ | $1.5g$ / $-2.5g$ | $1.0g$ / $-0.8g$ | $15g$ / $-3g$ |
| $a_{n,\max}$ / $a_{n,down}$ | $3.5g$ / $1g$ | $8g$ / $3g$ | $30g$ / $30g$ |
| $\tau_a$ (s) / $j_{\max}$ (m/s³) | 0.08 / 200 | 0.35 / 60 | 0.15 / 800 |

Its geometry adds the aspect and lead angles, and uses $t_{go} = r/\max(v_{cl}, 1)$ if closing, else 999 s:

$$\theta_{aspect} = \arccos(\hat{\mathbf v}_T\cdot\hat{\mathbf r}), \qquad \delta_{lead} = \arccos(\hat{\mathbf v}_M\cdot\hat{\mathbf r})$$

Laws that differ from A9.1 (amplitude $A = \min(n_g g, a_{n,\max})$):

| Law | Command |
|---|---|
| Constant g | $A\,\mathbf n_h$ for $t\ge t_0 = 0.5$ s |
| Weave | $A\sin(\omega_Tt + \phi_T)\,\mathbf n_h$, default $\omega_T = \dfrac{1}{\tau_M}\sqrt{\dfrac{\max(N' - 2,\ 0.5)}{2}}$, $\tau_M = 0.4$ s |
| Barrel roll, bang-bang | as A9.1 with $\omega_T = 1.5$ rad/s |
| Terminal break | trigger $t_{go}\le t_{break}$, $t_{break} = 2\tau_M$ if $N'\le3$ else $1.268\,\tau_M$, $\tau_M = 0.5$ s |
| Energy depletion | $a_{t,\max}\hat{\mathbf t} + 4V\operatorname{dir}_\perp(\hat{\mathbf r}) + 0.35a_{n,\max}\sin(1.2t)\,\mathbf n_h$ |
| Split-S | $-a_{n,\max}\mathbf n_v + 0.5a_{t,\max}\hat{\mathbf t}$ until heading change $> 2.95$ rad and $\lvert v_z\rvert < 0.15V$; then $a_{t,\max}\hat{\mathbf t}$ |
| Scissor | $a_{t,\min}\hat{\mathbf t} + \sigma a_{n,\max}\mathbf n_h$ |
| Notching | gain 5 in place of 4 |
| Turn into missile | $a_{n,\max}\operatorname{dir}_\perp(-\hat{\mathbf r}) + a_{t,\max}\hat{\mathbf t}$ |
| Crank | $\psi_{des} = \psi(-\hat{\mathbf r}) \pm 60°$, side $= \operatorname{sign}(\sin 0.25t)$, gain $3V$ |
| Speed change | $a_{t,\max}\hat{\mathbf t}$ if $\sin 0.8t\ge0$, else $a_{t,\min}\hat{\mathbf t}$ |
| Minimum-range turn | trigger $r\le\max(r_b, 1200)$, $r_b = \dfrac{2\lVert\mathbf v_M\rVert^2}{a_{M,\max}}\max(\lvert\sin\delta_{lead}\rvert, 0.25)$, $a_{M,\max} = 30g$ |
| Dive | $\gamma_d = -35°$ (0 below 150 m); $a_{t,\max}\hat{\mathbf t} + \operatorname{clip}(3V(\gamma_d - \gamma),\ -a_{n,down},\ a_{n,\max})\,\mathbf n_v$ |
| Spiral | $\theta = t + \tfrac12(0.35)t^2$; $0.3a_{t,\max}\hat{\mathbf t} + a_{n,\max}[\cos\theta\,\mathbf n_h + \sin\theta\,\mathbf n_v]$ |

Its constraint and update are A4 Steps 1 to 4 with the target limits; the omnidirectional case adds the free-fall floor $a_z \ge -a_{n,down}$.

Attitude derived from the motion, and body rates by finite difference:

$$\psi = \operatorname{atan2}(-v_y, v_x), \qquad \theta = \operatorname{atan2}(v_z, \sqrt{v_x^2 + v_y^2}), \qquad \phi = \operatorname{atan2}(\mathbf a\cdot\mathbf n_h,\ g)\ \ (\phi = 0\ \text{for the missile})$$

$$\boldsymbol\omega_B = \frac{\operatorname{wrap}(\boldsymbol\eta_{k+1} - \boldsymbol\eta_k)}{\Delta t}, \qquad \boldsymbol\alpha_B = \frac{\boldsymbol\omega_{B,k+1} - \boldsymbol\omega_{B,k}}{\Delta t}, \qquad \boldsymbol\eta = [\phi, \theta, \psi]$$

Reference pursuer in the same file (true PN with lag and optional drag):

$$\mathbf a_c = \operatorname{sat}\!\left(N'\max(v_{cl}, 0)\,\dot{\hat{\mathbf r}} + \tfrac12N'\mathbf a_{T,\perp},\ a_{M,\max}\right), \qquad \dot{\mathbf a} = \frac{\mathbf a_c - \mathbf a}{\tau_M}$$

$$\rho(h) = 1.225\,e^{-h/8500}, \qquad a_{drag} = 1.8\times10^{-4}\,\rho\,V^2 + 2.5\times10^{-4}\,\frac{\lVert\mathbf a\rVert^2}{\max(\rho, 0.1)}, \qquad \dot{\mathbf v} = \mathbf a - a_{drag}\,\hat{\mathbf v}$$

### A10. Seeker measurement and tracking

Code: `interceptor/guidance/seeker.py::SeekerSensorFilter`, `interceptor/guidance/tracker.py::KalmanTracker`

| | |
|---|---|
| **Inputs** | true target $\mathbf p_T, \mathbf v_T, \mathbf a_T$, interceptor position, noise scale $\kappa$, $\Delta t$ |
| **Outputs** | raw measurements $\mathbf y_p, \mathbf y_v, \mathbf y_a$; estimates $\hat{\mathbf p}_T, \hat{\mathbf v}_T, \hat{\mathbf a}_T$; visibility flag |

Measurement noise (EQ-SEEK-1, EQ-SEEK-2), with $\sigma_{p,0} = 0.5$ m, $\alpha_r = 0.0025$ rad, $r = \lVert\mathbf p_T - \mathbf p_M\rVert$:

$$\sigma_p = \kappa(\sigma_{p,0} + \alpha_r r), \qquad \sigma_v = 0.03\,\kappa\,v_{T,nom}, \qquad \sigma_a = 0.18\,\kappa\,a_{T,\max}$$

$$\mathbf y_p = \mathbf p_T + \mathcal N(0, \sigma_p^2 I), \qquad \mathbf y_v = \mathbf v_T + \mathcal N(0, \sigma_v^2 I), \qquad \mathbf y_a = \mathbf a_T + \mathcal N(0, \sigma_a^2 I)$$

Dropout (EQ-SEEK-3): each step is lost with probability $0.04\,\kappa$. The estimate then coasts:

$$\hat{\mathbf p} \leftarrow \hat{\mathbf p} + \hat{\mathbf v}\Delta t + \tfrac12\hat{\mathbf a}\Delta t^2, \qquad \hat{\mathbf v} \leftarrow \hat{\mathbf v} + \hat{\mathbf a}\Delta t$$

Fixed-gain filter (default; EQ-SEEK-4), gains $(\alpha_p, \alpha_v, \alpha_a) = (0.65, 0.55, 0.35)$. Predict with the coast equations to get $\hat{\mathbf p}^-, \hat{\mathbf v}^-$, then blend each measured quantity:

$$\hat{\mathbf p} = \alpha_p\mathbf y_p + (1 - \alpha_p)\hat{\mathbf p}^-, \qquad \hat{\mathbf v} = \alpha_v\mathbf y_v + (1 - \alpha_v)\hat{\mathbf v}^-, \qquad \hat{\mathbf a} = \alpha_a\mathbf y_a + (1 - \alpha_a)\hat{\mathbf a}$$

Kalman tracker (option `filter_type="kalman"`), per axis, state $x = [p, v, a]^\top$ (EQ-KF-1 to EQ-KF-5):

$$x_{k+1} = Fx_k + w_k, \qquad F = \begin{bmatrix}1 & \Delta t & \Delta t^2/2\\ 0 & 1 & \Delta t\\ 0 & 0 & 1\end{bmatrix}$$

$$Q = q_j\begin{bmatrix}\Delta t^5/20 & \Delta t^4/8 & \Delta t^3/6\\ \Delta t^4/8 & \Delta t^3/3 & \Delta t^2/2\\ \Delta t^3/6 & \Delta t^2/2 & \Delta t\end{bmatrix}, \qquad q_j = \left(\frac{a_{T,\max}}{\tau_{man}}\right)^2,\ \tau_{man} = 1\ \mathrm{s}$$

$$y_k = x_k + v_k, \qquad R = \operatorname{diag}(\sigma_p^2, \sigma_v^2, \sigma_a^2), \qquad H = I, \qquad P_0 = R(r = 0, \kappa = 1)$$

Predict: $x^- = Fx$, $P^- = FPF^\top + Q$. Update (Joseph form), innovation $\nu = y - x^-$:

$$S = P^- + R, \qquad K = P^-S^{-1}, \qquad x^+ = x^- + K\nu, \qquad P^+ = (I - K)P^-(I - K)^\top + KRK^\top$$

Consistency check, normalised innovation squared averaged over the three axes: $\mathrm{NIS} = \tfrac13\sum_{axes}\nu^\top S^{-1}\nu$.

Look-ahead used for the observation (`kalman_predict`): $\hat{\mathbf p}(\Delta) = \hat{\mathbf p} + \hat{\mathbf v}\Delta + \tfrac12\hat{\mathbf a}\Delta^2$, $\hat{\mathbf v}(\Delta) = \hat{\mathbf v} + \hat{\mathbf a}\Delta$.

### A11. Intercept oracle (critic input only)

Code: `interceptor/guidance/kinematics.py::compute_optimal_intercept_oracle`

| | |
|---|---|
| **Inputs** | $\mathbf p_M, \mathbf v_M$, true $\mathbf p_T, \mathbf v_T, \mathbf a_T$, $v_{\max}$, $a_{n,\max}$, $t_{go}$ |
| **Outputs** | reference acceleration $\mathbf a^*$ and velocity $\mathbf v^*$ |

With $T = \operatorname{clip}(t_{go}, 0.1, 20)$:

$$\mathbf p_f = \mathbf p_T + \mathbf v_T T + \tfrac12\mathbf a_T T^2, \qquad \mathbf v_f = \mathbf v_T + \mathbf a_T T$$

$$\mathbf a^* = \operatorname{sat}\!\left(\frac{6(\mathbf p_f - \mathbf p_M) - 4T\,\mathbf v_M - 2T\,\mathbf v_f}{T^2},\ a_{n,\max}\right), \qquad \mathbf v^* = \operatorname{sat}\!\left(\frac{\mathbf p_f - \mathbf p_M}{T},\ v_{\max}\right)$$

### A12. Observation

Code: `interceptor/guidance/observation.py::AsymmetricObsBuilder`

| | |
|---|---|
| **Inputs** | $\hat{\mathbf p}_T, \hat{\mathbf v}_T, \hat{\mathbf a}_T$, $\mathbf p_M, \mathbf v_M$, visibility, per-platform scales $s_\bullet$, oracle outputs, true target look-ahead |
| **Outputs** | actor vector (85), privileged critic block (16), total 101; every component clipped to $[-10, 10]$ |

One frame, 16 values (EQ-OBS-1), with $\Delta\mathbf p = \hat{\mathbf p}_T - \mathbf p_M$, $\Delta\mathbf v = \hat{\mathbf v}_T - \mathbf v_M$, $R = \lVert\Delta\mathbf p\rVert$:

$$\mathbf x_t = \left[\frac{\Delta\mathbf p}{s_{pos}},\ \frac{\Delta\mathbf v}{s_{vel}},\ \frac{\mathbf v_M}{s_{vel}},\ \frac{R}{s_{range}},\ \frac{v_c}{s_{closing}},\ \frac{\boldsymbol\omega_{LOS}}{s_{los}},\ \frac{t_{go}}{s_{tgo}},\ \mathbb 1_{visible}\right]$$

$$v_c = -\Delta\mathbf v\cdot\widehat{\Delta\mathbf p}, \qquad \boldsymbol\omega_{LOS} = \frac{\Delta\mathbf p\times\Delta\mathbf v}{\max(R, 15)^2}, \qquad t_{go} = \operatorname{clip}\!\left(\frac{R}{\max(v_c, 0.1)},\ 0,\ t_{\max}\right)$$

Future sample $k = 1..n$ at horizon $\Delta_k = kq\,\Delta t$, using the look-ahead of A10 and a constant-velocity interceptor:

$$\mathbf z_k = \left[\frac{\hat{\mathbf p}_T(\Delta_k) - (\mathbf p_M + \mathbf v_M\Delta_k)}{s_{fpos}},\ \frac{\hat{\mathbf v}_T(\Delta_k) - \mathbf v_M}{s_{fvel}},\ \mathbb 1_{visible}\right]$$

Actor vector (EQ-OBS-2), $m = 3$, $p = 2$, $n = 3$, $q = 5$; short history is padded with the oldest frame:

$$\mathbf o^{actor} = [\mathbf x_t, \mathbf x_{t-p}, \ldots, \mathbf x_{t-mp}]\ \Vert\ [\mathbf z_1, \ldots, \mathbf z_n], \qquad \dim = 16(1 + m) + 7n = 85$$

Privileged critic block (16), with $\mathbf p_T^+, \mathbf v_T^+$ the true target state $15\Delta t$ ahead:

$$\mathbf o^{priv} = \left[\frac{t_{rem}}{s_{trem}},\ \frac{\mathbf a^*}{s_{aopt}},\ \frac{\mathbf v^*}{s_{vopt}},\ \frac{\mathbf p_T^+ - \mathbf p_M}{s_{fpos}},\ \frac{\mathbf v_T^+ - \mathbf v_M}{s_{fvel}},\ \frac{\mathbf a_T}{s_{atgt}}\right]$$

Default scales: lengths $r_{0,\max}$; speeds $v_{\max}$; times $t_{\max}$; $s_{los} = 1$; $s_{aopt} = a_{n,\max}$; $s_{atgt} = \max(a_{T,\max}, 1)$.

### A13. Guidance reward and termination

Code: `interceptor/guidance/reward.py::GuidanceReward`, `interceptor/guidance/env.py::step`; weights: `configs/guidance.yaml`

| | |
|---|---|
| **Inputs** | true range $d$ before and after the step, $\hat{\mathbf v}_M$, $\hat{\mathbf r}$, $\dot{\hat{\mathbf r}}$, $\lVert\mathbf Z_\perp\rVert$, action $\mathbf u_t$, previous action, step $k$ of $K$, terminal flags, $d_{miss}$ |
| **Output** | scalar reward $r$ and its named terms |

Progress, potential-based (EQ-GUIDE-1, EQ-GUIDE-2):

$$\Phi(s) = -\frac{w_{progress}\,d}{r_{0,\max}}, \qquad r_{progress} = \gamma\,\Phi(s') - \Phi(s)$$

Shaping (EQ-GUIDE-3):

$$r_{aux} = w_{align}\,\hat{\mathbf v}_M\cdot\hat{\mathbf r} - w_{los}\lVert\dot{\hat{\mathbf r}}\rVert - w_{zem}\frac{\lVert\mathbf Z_\perp\rVert}{\max(d, 10)}$$

Action penalties (EQ-GUIDE-4) and time cost (EQ-GUIDE-5):

$$r_{act} = -w_{res}\lVert\mathbf u_t\rVert^2 - w_{smooth}\lVert\mathbf u_t - \mathbf u_{t-1}\rVert^2, \qquad r_{time} = -c_{time}$$

Terminal (EQ-GUIDE-6):

$$r_{term} = \begin{cases}B_{kill} + B_{speed}\left(1 - \dfrac{k}{K}\right) & \text{kill}\\ -B_{crash} & \text{crash}\\ -B_{div}\,\dfrac{d_{miss}}{r_{0,\max}} & \text{diverged}\end{cases}$$

Total (EQ-GUIDE-7) and discount from the horizon $H$ (EQ-GUIDE-8):

$$r = r_{progress} + r_{aux} + r_{act} + r_{time} + r_{term}, \qquad \gamma = \rho^{1/H} = 0.1^{1/1100} = 0.99791$$

Code defaults: $w_{progress} = 50$, $w_{align} = 0.35$, $w_{los} = 0.15$, $w_{zem} = 0.25$, $w_{res} = 0.04$ (APN+RL), $0.015$ (pure RL), $0$ (APN arms), $w_{smooth} = 0.03$, $c_{time} = 0.005$, $B_{kill} = 150$, $B_{speed} = 50$, $B_{crash} = 80$, $B_{div} = 25$.

Termination:

| Event | Condition |
|---|---|
| Kill | $d_{eff} \le r_{kill}$ |
| Crash | $z_M < 0$ |
| Diverged (APN arms only) | $k > 30$ and $v_{cl} < -0.2\,v_{\max}$ and $r > d_{miss} + 60$ m |
| Timeout | $k \ge K = t_{\max}/\Delta t$ |

Reward kept only for replaying recorded fixtures (`notebook_reward`): identical except $r_{progress} = 2.5\,\dfrac{d_{prev} - d}{\max(v_{\max}\Delta t,\ 10^{-3})}$.

---

## Part B. Quadrotor simulator (6-DOF dynamics)

Code: `interceptor/sim/pipeline.py`, `aero.py`, `wind.py`, `quaternion.py`, `constants.py`; sourced values: `configs/quadrotor.yaml`

| | |
|---|---|
| **State** | position $\mathbf p$, velocity $\mathbf v$ (world), quaternion $\mathbf q$, body rates $\boldsymbol\omega$, four rotor speeds $\Omega_j$ |
| **Inputs** | four normalised motor commands $c_j\in[0, 1]$, battery voltage $U$, wind $\mathbf v_{wind}$, step $h$ |
| **Outputs** | next state |

Constants in code: $m = 0.04085$ kg; $J = \operatorname{diag}(2.3951, 2.3951, 3.2347)\times10^{-5}$ kg m²; arm $a = 0.0397$ m; $c_l = 2.618\times10^{-8}$ N s²; $c_d = 5.45\times10^{-11}$ N m s²; $\tau_m = 0.02$ s; $J_{mp} = 2.0\times10^{-9}$ kg m²; $\Omega_{\max} = 2800$ rad/s.

### B1. Quaternion algebra

Rotation matrix, body to world (`quat2rot`):

$$R(\mathbf q) = \begin{bmatrix}1 - 2(y^2 + z^2) & 2(xy - wz) & 2(xz + wy)\\ 2(xy + wz) & 1 - 2(x^2 + z^2) & 2(yz - wx)\\ 2(xz - wy) & 2(yz + wx) & 1 - 2(x^2 + y^2)\end{bmatrix}$$

Hamilton product (`quat_mult`):

$$\mathbf q_1\otimes\mathbf q_2 = \begin{bmatrix}w_1w_2 - x_1x_2 - y_1y_2 - z_1z_2\\ w_1x_2 + x_1w_2 + y_1z_2 - z_1y_2\\ w_1y_2 - x_1z_2 + y_1w_2 + z_1x_2\\ w_1z_2 + x_1y_2 - y_1x_2 + z_1w_2\end{bmatrix}$$

Normalisation: $\mathbf q \leftarrow \mathbf q/\lVert\mathbf q\rVert$. Conjugate: $\mathbf q^{-1} = [w, -x, -y, -z]$.

### B2. Command path: stick mapping, rate PID, mixer

Stick mapping (`command_from_action`), action $[T, r, p, y]\in[-1, 1]^4$:

$$c_{cmd} = \max(0, T), \qquad \boldsymbol\omega_{cmd} = 2\,[r, p, y]\ \mathrm{rad/s}, \qquad \text{throttle cut if } T < 0.02$$

Built-in rate PID (EQ-CTRL-3, `pipeline.pid`), gains $K_p = (0.15, 0.15, 0.20)$, $K_i = (0.20, 0.20, 0.10)$, $K_d = (0.001, 0.001, 0)$:

$$\mathbf e = \boldsymbol\omega_{cmd} - \boldsymbol\omega_k, \qquad \mathbf I_k = \mathbf I_{k-1} + K_i\,\mathbf e\,\Delta t\ \ (\mathbf 0 \text{ on throttle cut}), \qquad \mathbf u = K_p\mathbf e + \mathbf I_k - K_d\frac{\boldsymbol\omega_k - \boldsymbol\omega_{k-1}}{\Delta t}$$

Mixer (`pipeline.mixer`). Inputs: $\mathbf u = (u_{roll}, u_{pitch}, u_{yaw})$, collective $c$. Output: four motor commands.

$$M = \begin{bmatrix}+1 & -1 & +1\\ +1 & +1 & -1\\ -1 & +1 & +1\\ -1 & -1 & -1\end{bmatrix}, \qquad \mathbf c_{raw} = c\,\mathbf 1 + M\mathbf u$$

Columns are $\operatorname{sign}(r_{j,y})$, $-\operatorname{sign}(r_{j,x})$ and the spin sign $\sigma_j$. Saturation-aware scaling: if $\max(\mathbf c_{raw}) > 1$,

$$\mathbf u \leftarrow \mathbf u\,\frac{1 - c}{\max(\mathbf c_{raw}) - c}, \qquad \mathbf c_{raw} = c\,\mathbf 1 + M\mathbf u; \qquad \mathbf c = \operatorname{clip}(\mathbf c_{raw}, 0, 1)$$

### B3. Motors and rotors

ESC steady-state speed (EQ-SIX-7), zero below the cut-off command 0.02:

$$\Omega_{ss}(c, U) = b_0 + b_1U + b_2\sqrt c + b_3c + b_4U\sqrt c$$

Current coefficients: $b_0 = 1170.37$, $b_2 = 1629.63$, others zero, so $\Omega_{ss} = b_0 + b_2\sqrt c$. This two-term fit is anchored on hover and full command.

Hover equilibrium:

$$\Omega_{hover} = \sqrt{\frac{mg}{4c_l}} = 1956.2\ \mathrm{rad/s}, \qquad c_{hover}:\ \Omega_{ss}(c_{hover}) = \Omega_{hover} \ \Rightarrow\ c_{hover} = 0.2325$$

Motor lag (EQ-SIX-6):

$$\dot\Omega_j = \frac{\Omega_{ss,j} - \Omega_j}{\tau_m}$$

Rotor thrust and reaction torque (EQ-SIX-5), spin signs $\sigma_j = +1, -1, +1, -1$:

$$\mathbf f_j = c_l\,\Omega_j^2\,\hat{\mathbf z}, \qquad \boldsymbol\tau_j = \sigma_j\,c_d\,\Omega_j^2\,\hat{\mathbf z}$$

Assembly at the arm positions $\mathbf r_j = (\pm a, \pm a, 0)$ (EQ-SIX-4):

$$\mathbf f_{prop} = \sum_j\mathbf f_j, \qquad \boldsymbol\tau_{prop} = \sum_j(\boldsymbol\tau_j + \mathbf r_j\times\mathbf f_j)$$

Motor reaction torque from rotor acceleration (EQ-SIX-8):

$$\boldsymbol\tau_{mot} = J_{mp}\sum_{j=1}^{4}\sigma_j\,\dot\Omega_j\,\hat{\mathbf z}$$

Maximum total thrust and thrust-to-weight ratio (`bridge/setpoint.py::max_thrust`):

$$T_{\max} = 4c_l(b_0 + b_2)^2, \qquad TW = \frac{T_{\max}}{mg} = 2.05$$

### B4. Aerodynamics

Air-relative body velocity: $\mathbf v_B = R(\mathbf q)^\top(\mathbf v - \mathbf v_{wind})$, components $(v_x, v_y, v_z)$.

Linear rotor drag, the adopted model (EQ-SIX-11), $k = (0.5, 0.5, 0.7)\ \mathrm{s^{-1}}$ (assumed, not fitted), no torque:

$$\mathbf f_{aero} = -m\,D\,\mathbf v_B, \qquad D = \operatorname{diag}(k_x, k_y, k_z)$$

Polynomial model, kept for fixture parity (EQ-AERO-8, EQ-AERO-9). With $v_{xy} = \sqrt{v_x^2 + v_y^2}$ and $\bar\Omega^2 = \tfrac14\sum_j\Omega_j^2$:

$$f_x = m\,\mathbf c_{fx}\cdot[v_x,\ v_x\lvert v_x\rvert,\ \bar\Omega^2,\ v_x\bar\Omega^2]$$

$$f_y = m\,\mathbf c_{fy}\cdot[v_y,\ v_y\lvert v_y\rvert,\ \bar\Omega^2,\ v_y\bar\Omega^2]$$

$$f_z = m\,\mathbf c_{fz}\cdot[v_z,\ v_z\lvert v_z\rvert,\ v_{xy}^2,\ v_{xy}\bar\Omega^2,\ v_z\bar\Omega^2,\ v_{xy}v_z\bar\Omega^2]$$

$$\tau_x = J_{xx}\,\mathbf c_{tx}\cdot[v_y,\ v_y\lvert v_y\rvert,\ \bar\Omega^2,\ v_y\bar\Omega^2,\ v_y\lvert v_y\rvert\bar\Omega^2], \qquad \tau_y = J_{yy}\,\mathbf c_{ty}\cdot[\text{same in } v_x]$$

$$\tau_z = J_{zz}\,(c_{z,0}v_x + c_{z,1}v_y)$$

Forces are clipped to $\pm50$ N and torques to $\pm5$ N m. $\mathbf c_{tx}$ and $\mathbf c_{ty}$ are zero.

Ground effect on the collective axis (EQ-SIX-10), $h_{ge} = 0.15$ m, $\kappa_{\max} = 0.20$; $f_{prop,z}$ is multiplied by

$$\kappa_{ge}(z) = 1 + \kappa_{\max}\left(1 - \frac{\operatorname{clip}(z, 0, h_{ge})}{h_{ge}}\right)$$

### B5. Rigid-body equations of motion

Code: `pipeline.rigid_body`, `pipeline.deriv`

Translation (EQ-SIX-1):

$$\dot{\mathbf p} = \mathbf v, \qquad \dot{\mathbf v} = \frac1m R(\mathbf q)(\mathbf f_{prop} + \mathbf f_{aero}) + \mathbf g$$

Rotation (EQ-SIX-2):

$$\dot{\boldsymbol\omega} = J^{-1}\left(\boldsymbol\tau_{prop} + \boldsymbol\tau_{mot} + \boldsymbol\tau_{aero} - \boldsymbol\omega\times J\boldsymbol\omega\right)$$

Attitude kinematics (EQ-SIX-3):

$$\dot{\mathbf q} = \tfrac12\,\mathbf q\otimes\begin{bmatrix}0\\ \boldsymbol\omega\end{bmatrix}$$

### B6. Integration

Classical RK4 (EQ-SIX-9, `pipeline.rk4`), command, voltage, wind and ground-effect gain held constant over the step; afterwards $\mathbf q$ is renormalised and $\Omega_j \ge 0$ enforced:

$$k_1 = f(x_n),\quad k_2 = f(x_n + \tfrac h2k_1),\quad k_3 = f(x_n + \tfrac h2k_2),\quad k_4 = f(x_n + hk_3)$$

$$x_{n+1} = x_n + \tfrac h6(k_1 + 2k_2 + 2k_3 + k_4)$$

Sub-stepping (`control_step`): $N = \Delta t_{control}/\Delta t_{physics}$ must be an integer; defaults 0.002 s / 0.001 s give $N = 2$.

Legacy option, used only for fixture replay: thrust and aerodynamics are evaluated at the half-step rotor speed $\Omega_{eff} = \max(\Omega + \tfrac h2\dot\Omega,\ 0)$. The validated path uses $\Omega_{eff} = \max(\Omega, 0)$.

### B7. Domain randomisation

Code: `pipeline.sample_physics_params`. Each parameter is its nominal value times an independent uniform factor per episode:

| Parameter | Factor |
|---|---|
| Mass $m$; each inertia $J_{ii}$ | $U(0.90, 1.10)$ |
| $c_l$; $c_d$ | $U(0.85, 1.15)$ |
| Each aerodynamic coefficient group; rotor-drag $k$ | $U(0.80, 1.20)$ |

### B8. Wind

Code: `interceptor/sim/wind.py::DrydenWind`, `dryden_spectrum`, `gauss_gust`; legacy `aero.wind_ou_step`

| | |
|---|---|
| **Inputs** | altitude $h$, mean wind $\bar{\mathbf w}$, wind speed at 20 ft $W_{20}$, airspeed $V = \max(V_{air}, 5)$, white noise, step $\Delta t$ |
| **Output** | wind velocity $\mathbf v_{wind} = \bar{\mathbf w} + (u, v, w)$ in world axes |

Legacy first-order process (EQ-WIND-1) and its discrete step, with per-episode $\bar{\mathbf w}\sim\mathcal N(0, 1.5^2)$, $\theta\sim U(0.2, 1.0)$, $\sigma\sim U(0.3, 2.0)$:

$$\dot u = -\theta u + \sigma\sqrt{2\theta}\,\xi(t), \qquad \mathbf w_{k+1} = \mathbf w_k + \theta(\bar{\mathbf w} - \mathbf w_k)\Delta t + \sigma\sqrt{\Delta t}\,\boldsymbol\xi_k$$

Low-altitude intensities and length scales (EQ-WIND-2), $h_{ft} = h/0.3048$, $\lambda = 0.177 + 0.000823\,h_{ft}$:

$$\sigma_w = 0.1\,W_{20}, \qquad \sigma_u = \sigma_v = \frac{\sigma_w}{\lambda^{0.4}}, \qquad L_w = h, \qquad L_u = L_v = \frac{h}{\lambda^{1.2}}$$

Forming filters (EQ-WIND-3); $H_w$ is $H_v$ with $(\sigma_w, L_w)$:

$$H_u(s) = \sigma_u\sqrt{\frac{2L_u}{\pi V}}\,\frac{1}{1 + (L_u/V)s}, \qquad H_v(s) = \sigma_v\sqrt{\frac{L_v}{\pi V}}\,\frac{1 + \sqrt3(L_v/V)s}{(1 + (L_v/V)s)^2}$$

One-sided spectra they realise ($\int_0^\infty S\,d\omega = \sigma^2$), used by the spectral test:

$$S_u(\omega) = \sigma_u^2\,\frac{2L_u}{\pi V}\,\frac{1}{1 + (L_u\omega/V)^2}, \qquad S_{v}(\omega) = \sigma_v^2\,\frac{L_v}{\pi V}\,\frac{1 + 3(L_v\omega/V)^2}{\left(1 + (L_v\omega/V)^2\right)^2}$$

Exact discretisation, white-noise intensity $\pi$, $\xi\sim\mathcal N(0, 1)$. Longitudinal axis, $\tau = L_u/V$, $a = e^{-\Delta t/\tau}$:

$$x \leftarrow a\,x + \sqrt{\frac{\pi(1 - a^2)}{2\tau}}\ \xi, \qquad u = \sigma_u\sqrt{\frac{2L_u}{\pi V}}\ x$$

Lateral and vertical axes, $\tau = L/V$, $\alpha = \Delta t/\tau$, $a = e^{-\alpha}$, noise $(n_1, n_2)\sim\mathcal N(0, \Sigma)$:

$$\Sigma = \begin{bmatrix}q_{11} & q_{12}\\ q_{12} & q_{22}\end{bmatrix}, \qquad q_{11} = \frac{\pi(1 - a^2)}{2\tau}$$

$$q_{12} = \frac{\pi\left(1 - a^2(1 + 2\alpha)\right)}{4\tau}, \qquad q_{22} = \frac{\pi\left(2 - a^2(2 + 4\alpha + 4\alpha^2)\right)}{8\tau}$$

$$x_1 \leftarrow a\,x_1 + n_1, \qquad x_2 \leftarrow a\,x_2 + \alpha a\,x_1^{old} + n_2, \qquad v = \sigma_v\sqrt{\frac{L_v}{\pi V}}\left(\sqrt3\,x_1 + (1 - \sqrt3)\,x_2\right)$$

Discrete 1-cosine gust (EQ-WIND-4), zero outside $0 < t < T_g$:

$$U(t) = \frac{U_{ds}}{2}\left(1 - \cos\frac{2\pi t}{T_g}\right)$$

---

## Part C. Flight controller, bridge and closed loop (quad only)

### C1. Guidance-to-flight-controller bridge

Code: `interceptor/bridge/setpoint.py::AccelBridge`, `interceptor/fc/attitude.py`, `interceptor/fc/thrust.py`

| | |
|---|---|
| **Inputs** | desired world acceleration $\mathbf a_{des}$, mass $m$, current attitude $\mathbf q$, yaw $\psi_d = 0$, tilt limit $\theta_{\max} = 60°$, rate limit 12 rad/s |
| **Outputs** | collective command $c$, body-rate setpoint $\boldsymbol\omega_{cmd}$, desired attitude $\mathbf q_d$ |

Thrust vector (EQ-BRIDGE-1):

$$\mathbf f = m(\mathbf a_{des} - \mathbf g)$$

Limits, vertical component has priority. With $f_z \leftarrow \max(f_z, 10^{-3})$ and lateral part $\mathbf f_{xy}$:

$$\mathbf f_{xy} \leftarrow \operatorname{sat}(\mathbf f_{xy},\ f_z\tan\theta_{\max})$$

If then $\lVert\mathbf f\rVert > T_{\max}$:

$$\mathbf f_{xy} \leftarrow \operatorname{sat}\!\left(\mathbf f_{xy},\ \sqrt{T_{\max}^2 - f_z^2}\right) \ \text{ if } f_z < T_{\max}, \qquad \mathbf f = (0, 0, T_{\max}) \ \text{ otherwise}$$

Collective, inverting the thrust and ESC curves (EQ-BRIDGE-2, `collective_for_thrust`):

$$c = \operatorname{clip}\!\left(\frac{\sqrt{\lVert\mathbf f\rVert/(4c_l)} - b_0}{b_2},\ 0,\ 1\right)^2$$

Desired attitude (EQ-BRIDGE-3, `quat_from_thrust_yaw`), with $\mathbf x_c = (\cos\psi_d, \sin\psi_d, 0)$:

$$\mathbf z_d = \frac{\mathbf f}{\lVert\mathbf f\rVert}, \qquad \mathbf y_d = \frac{\mathbf z_d\times\mathbf x_c}{\lVert\mathbf z_d\times\mathbf x_c\rVert}, \qquad \mathbf x_d = \mathbf y_d\times\mathbf z_d, \qquad R_d = [\mathbf x_d\ \mathbf y_d\ \mathbf z_d]$$

Rotation matrix to quaternion:

$$w = \tfrac12\sqrt{1 + R_{11} + R_{22} + R_{33}}, \qquad (x, y, z) = \frac{1}{4w}\left(R_{32} - R_{23},\ R_{13} - R_{31},\ R_{21} - R_{12}\right)$$

Rate setpoint (EQ-BRIDGE-4, `attitude_to_rate`), $K_p = 6\ \mathrm{s^{-1}}$, shortest rotation by flipping $\mathbf q_e$ when $q_{e,w} < 0$:

$$\mathbf q_e = \mathbf q_d^{-1}\otimes\mathbf q, \qquad \boldsymbol\omega_{cmd} = \operatorname{clip}\!\left(-2K_p\,\mathbf q_{e,xyz},\ \pm\omega_{lim}\right)$$

### C2. Position loop (outer loop for controller tests)

Code: `interceptor/fc/attitude.py::PositionLoop`. Inputs: position, velocity and their setpoints. Outputs: collective and desired attitude.

$$\mathbf a = \operatorname{clip}\!\left(k_p(\mathbf p_{des} - \mathbf p) + k_d(\mathbf v_{des} - \mathbf v),\ \pm a_{\max}\right), \qquad k_p = 6,\ k_d = 4,\ a_{\max} = 6\ \mathrm{m/s^2}$$

Then $\mathbf f = m(\mathbf a - \mathbf g)$ with tilt limit $35°$, and C1 for collective and attitude.

### C3. Baseline B1: project rate PID

Code: `interceptor/fc/pid.py`

| | |
|---|---|
| **Inputs** | $\boldsymbol\omega_{cmd}$, collective $c$, measured $\boldsymbol\omega$ |
| **Outputs** | four motor commands (through the mixer of B2) |

Plant gain per axis, measured from the model at hover by a perturbation $\delta u = 0.01$ (`axis_gain`):

$$K_i = \frac{\tau_{prop,i}\!\left(\Omega_{ss}(\text{mixer}(\delta u\,\mathbf e_i,\ c_{hover}))\right)}{J_{ii}\,\delta u}$$

Gain design by loop shaping (EQ-FC-1). Plant about one axis with motor lag $\tau = 0.02$ s:

$$G(s) = \frac{K}{s(\tau s + 1)}, \qquad C(s) = k_p\left(1 + \frac{\omega_i}{s}\right)$$

$$\omega_c = \frac{1}{3\tau}, \qquad \omega_i = \frac{\omega_c}{5}, \qquad k_p = \frac{\omega_c}{K}, \qquad k_i = k_p\,\omega_i$$

Phase at crossover: $-90° - \arctan(\omega_c\tau) - \arctan(\omega_i/\omega_c) = -119.7°$, a margin of about 60°.

Control law with conditional-integration anti-windup, $k_d = 0$ by default, $I_{lim} = 0.25$:

$$\mathbf e = \boldsymbol\omega_{cmd} - \boldsymbol\omega, \qquad \mathbf u = k_p\mathbf e + \mathbf I - k_d\frac{\boldsymbol\omega_k - \boldsymbol\omega_{k-1}}{\Delta t}, \qquad \mathbf c = \text{mixer}(\mathbf u, c)$$

$$\mathbf I \leftarrow \operatorname{clip}(\mathbf I + k_i\mathbf e\,\Delta t,\ \pm I_{lim}) \quad\text{only if no motor command is at 0 or 1}$$

Below the throttle cut ($c < 0.02$) the integrator is reset and the output is zero.

### C4. Baseline B2: Crazyflie firmware rate PID

Code: `interceptor/fc/firmware_pid.py`. Gains in firmware units (rates in deg/s, output in 16-bit counts): roll/pitch $K_p = 250$, $K_i = 500$, $K_d = 2.5$; yaw $K_p = 120$, $K_i = 16.7$, $K_d = 0$.

$$\mathbf e = \frac{180}{\pi}(\boldsymbol\omega_{cmd} - \boldsymbol\omega), \qquad \mathbf I \leftarrow \operatorname{clip}(\mathbf I + \mathbf e\,\Delta t,\ \pm33.3)$$

Derivative with a first-order low-pass at $f_c = 30$ Hz:

$$\alpha = \frac{\Delta t}{\Delta t + 1/(2\pi f_c)}, \qquad \mathbf d \leftarrow \mathbf d + \alpha\left(\frac{\mathbf e_k - \mathbf e_{k-1}}{\Delta t} - \mathbf d\right)$$

$$\mathbf n = \operatorname{clip}(K_p\mathbf e + K_i\mathbf I + K_d\mathbf d,\ \pm32767), \qquad \mathbf u = \frac{1}{65535}\left(\frac{n_{roll}}{2},\ \frac{n_{pitch}}{2},\ n_{yaw}\right)$$

$\mathbf u$ then goes through the mixer of B2.

### C5. Flight-controller RL environment

Code: `interceptor/fc/env.py`; values: `configs/fc.yaml`

| | |
|---|---|
| **Inputs** | action $\mathbf a\in[-1, 1]^4$, rate and collective setpoints $(\boldsymbol\omega_{cmd}, c_{sp})$ |
| **Outputs** | observation (33 actor + 16 privileged), reward, termination |

Action to motor commands: $\mathbf c = (\mathbf a + 1)/2$. The simulator then advances one control step (B6) at battery voltage 4.2 V.

Actor frame (11 values), three most recent frames stacked, with $r_{\max} = 8$ rad/s:

$$\mathbf x = \left[\frac{\boldsymbol\omega}{r_{\max}},\ \frac{\boldsymbol\omega_{cmd} - \boldsymbol\omega}{r_{\max}},\ \frac{c_{sp} - c_{hover}}{c_{hover}},\ \mathbf a_{prev}\right]$$

Privileged critic block (16 values):

$$\left[\frac{m}{m_0},\ \frac{J_{ii}}{J_{ii,0}},\ \frac{c_l}{c_{l,0}},\ \frac{c_d}{c_{d,0}},\ \frac{\Omega_j}{\Omega_{hover}} - 1,\ \bar{\mathbf w},\ \frac{R(\mathbf q)^\top\mathbf v}{5}\right]$$

Relative thrust error (actual total thrust against the thrust the commanded collective should give):

$$T_{err} = \frac{\sum_jc_l\,\Omega_j^2 - 4c_l\left(b_0 + b_2\sqrt{c_{sp}}\right)^2}{m_0\,g}$$

Reward (EQ-FC-2), with $\mathbf e = \boldsymbol\omega_{cmd} - \boldsymbol\omega$, $\Delta\mathbf a = \mathbf a_t - \mathbf a_{t-1}$ and $s$ the fraction of motors at 0 or 1:

$$r = -w_\omega\left\lVert\frac{\mathbf e}{r_{\max}}\right\rVert^2 - w_T\,T_{err}^2 - w_s\lVert\Delta\mathbf a\rVert^2 - w_{sat}\,s$$

Weights: $w_\omega = 1.0$, $w_T = 0.2$, $w_s = 0.02$, $w_{sat} = 0.1$.

Termination: diverged if $\max\lvert\omega_i\rvert > 40$ rad/s or non-finite, with an extra reward of $-10$; truncation after 4 s.

Initial conditions: tilt angle $\sim U(0, 0.5236)$ rad about a random horizontal axis with yaw $\sim U(\pm\pi)$; rates $\sim U(\pm2)$ rad/s; velocity $\sim\mathcal N(0, 1)$ m/s; rotors at $\Omega_{hover}$; altitude 100 m. The initial quaternion is the yaw rotation times the tilt rotation:

$$\mathbf q_0 = \left[\cos\tfrac\psi2,\ 0,\ 0,\ \sin\tfrac\psi2\right]\otimes\left[\cos\tfrac\theta2,\ \sin\tfrac\theta2\,\hat{\mathbf n}\right]$$

### C6. Setpoint generators

Code: `interceptor/fc/setpoints.py`. Each returns $[\omega_x, \omega_y, \omega_z, c]$ over time on one random axis with amplitude $A = \pm U(0.3, 1.0)\,r_{\max}$.

| Kind | Rate setpoint |
|---|---|
| Step | $A$ for $t \ge t_0$ |
| Ramp | $A\operatorname{clip}\!\left((t - t_0)/T_r,\ 0,\ 1\right)$ |
| Doublet | $+A$ on $[t_0, t_0 + w)$, $-A$ on $[t_0 + w, t_0 + 2w)$ |
| Sinusoid | $A\sin(2\pi ft)$ |
| Chirp | $A\sin\!\left(2\pi\left(f_0t + \tfrac12(f_1 - f_0)\,t^2/T\right)\right)$, $f_0 = 0.2$, $f_1 = 5$ Hz |
| Band-limited random | $a = e^{-2\pi B\Delta t}$, $x_{k+1} = a\,x_k + (1 - a)\sqrt{\tfrac{1 + a}{1 - a}}\ \xi_k$; setpoint $\operatorname{clip}(x_kA/3,\ \pm A)$, $B = 2$ Hz |

Collective profile: $c(t) = c_{hover}\left(1 + 0.15\sin(2\pi ft + \phi)\right)$, $f\sim U(0.3, 1.5)$ Hz.

### C7. Flight-controller metrics

Code: `interceptor/fc/metrics.py`. Inputs: time, setpoint, response and motor-command records. Outputs: scalars.

| Metric | Definition |
|---|---|
| Rise time | $t(y_{frac}\ge0.9) - t(y_{frac}\ge0.1)$, $y_{frac} = (y - y_0)/(y_{final} - y_0)$ |
| Overshoot | $\max\!\left(\max_t y_{frac} - 1,\ 0\right)$ |
| Settling time | first time after which $\lvert y - y_{final}\rvert \le 0.05\,\lvert y_{final} - y_0\rvert$ holds |
| Integral absolute error | $\sum_{axes}\int\lvert e\rvert\,dt$ (trapezoid) |
| RMS error | $\sqrt{\operatorname{mean}(e^2)}$ |
| Control effort | $\sum(u - \bar u)^2\,\Delta t$ |
| Action smoothness | $\operatorname{RMS}(\Delta u/\Delta t)$ |
| Saturation fraction | fraction of motor commands at 0 or 1 |

### C8. Closed loop: guidance through the flight controller

Code: `interceptor/bridge/closed_loop.py::run_closed_loop`

| | |
|---|---|
| **Inputs** | scenario seed, guidance arm, inner controller (B1, B2 or RL policy), wind on/off |
| **Outputs** | kill flag, miss distance, desired and achieved acceleration histories, tracking error |

Each 0.02 s guidance step: target and seeker update (A9, A10), guidance law (A1 to A3), bridge (C1); then 10 control steps of 0.002 s, each with 2 physics steps of 0.001 s. In the APN+RL arm the residual is $0.35\,a_{n,\max}\,\mathbf u$. Miss distance is the minimum of $d_{CPA}$ (A4) over all control steps.

Achieved acceleration and tracking error over the $N$ guidance steps:

$$\mathbf a_{act,k} = \frac{\mathbf v_{k+1} - \mathbf v_k}{0.02}, \qquad e_{RMS} = \sqrt{\frac1N\sum_k\lVert\mathbf a_{act,k} - \mathbf a_{des,k}\rVert^2}$$

Wind in this loop: Dryden at 100 m, mean $(1.0, 0.5, 0)$ m/s, $W_{20} = 3$ m/s, airspeed 5 m/s.

### C9. Measured quad limits

Code: `scripts/measure_quad_limits.py`; output: `results/tables/quad_limits.json`. Stack: bridge, B1, mixer, 1000 Hz physics with linear rotor drag.

Analytic bounds from the thrust-to-weight ratio:

$$a_{lat} = g\sqrt{TW^2 - 1} = 17.5, \qquad a_{up} = g(TW - 1) = 10.3, \qquad a_{down} = g = 9.81\ \ \mathrm{m/s^2}$$

Measured quantities:

| Quantity | Definition | Value |
|---|---|---|
| Specific force | $(\mathbf v_{k+1} - \mathbf v_k)/\Delta t - \mathbf g$, 25-sample moving average | |
| $\tau_a$ | time for a 4 m/s² lateral step to reach 63.2 % | 0.174 s |
| $a_{lat,\max}$ | peak lateral acceleration before saturation | 14.8 m/s² |
| $a_{up,\max}$, $a_{down,\max}$ | peak vertical acceleration at level attitude | 9.4 m/s² |
| $v_{\max}$ | terminal speed holding $0.9\,a_{lat,\max}$ | 22.0 m/s |
| $j_{\max}$ | $a_{lat,\max}/\tau_a$ | 85 m/s³ |
| Turn rate at top speed | $a_{lat,\max}/v_{\max}$ | 0.67 rad/s |
| Peak body rate | hard attitude step | 4.4 rad/s |

---

## Part D. Learning algorithms

Code: `training/algos.py` (Stable-Baselines3); derivations: `docs/algorithm_selection.md`. Inputs: transitions $(s_t, a_t, r_t, s_{t+1})$. Outputs: updated policy and value parameters. Networks: `[256, 256, 128]`, ReLU. The actor sees the first 85 (guidance) or 33 (flight controller) observation values; the critic sees all of them.

### D1. PPO (provisional choice)

Policy gradient:

$$\nabla_\theta J = \mathbb E\left[\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,\hat A_t\right]$$

Advantage estimate, $\lambda = 0.95$:

$$\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t), \qquad \hat A_t = \sum_{l=0}^{T-t-1}(\gamma\lambda)^l\,\delta_{t+l}$$

Clipped surrogate, $\epsilon = 0.2$:

$$\rho_t(\theta) = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{old}}(a_t\mid s_t)}, \qquad L^{CLIP} = \mathbb E\left[\min\left(\rho_t\hat A_t,\ \operatorname{clip}(\rho_t, 1 - \epsilon, 1 + \epsilon)\,\hat A_t\right)\right]$$

Total loss, $c_v = 0.5$, $c_e = 0$:

$$L = -L^{CLIP} + c_v\,\mathbb E\left[(V_\phi(s_t) - \hat R_t)^2\right] - c_e\,\mathbb E\left[\mathcal H(\pi_\theta(\cdot\mid s_t))\right]$$

### D2. TD3

Deterministic policy gradient:

$$\nabla_\theta J = \mathbb E_{s\sim\mathcal D}\left[\nabla_\theta\mu_\theta(s)\,\nabla_aQ_{\phi_1}(s, a)\big|_{a = \mu_\theta(s)}\right]$$

Clipped double-Q target, $\sigma = 0.2$, $c = 0.5$, $\epsilon\sim\mathcal N(0, \sigma^2)$:

$$y = r + \gamma(1 - d)\min_{i=1,2}Q_{\phi_i'}\!\left(s',\ \operatorname{clip}\left(\mu_{\theta'}(s') + \operatorname{clip}(\epsilon, -c, c),\ a_{\min},\ a_{\max}\right)\right)$$

Target update every 2 critic steps, $\tau = 0.005$:

$$\phi_i' \leftarrow \tau\phi_i + (1 - \tau)\phi_i', \qquad \theta' \leftarrow \tau\theta + (1 - \tau)\theta'$$

### D3. SAC

Maximum-entropy objective and soft value:

$$J(\theta) = \mathbb E_{\pi_\theta}\left[\sum_t\gamma^t\left(r_t + \alpha\,\mathcal H(\pi_\theta(\cdot\mid s_t))\right)\right], \qquad V^\pi(s) = \mathbb E_{a\sim\pi}\left[Q^\pi(s, a) - \alpha\log\pi(a\mid s)\right]$$

Critic target and loss, $a'\sim\pi_\theta(\cdot\mid s')$:

$$y = r + \gamma(1 - d)\left(\min_{i=1,2}Q_{\phi_i'}(s', a') - \alpha\log\pi_\theta(a'\mid s')\right), \qquad L_Q = \mathbb E\left[(Q_\phi(s, a) - y)^2\right]$$

Actor loss:

$$L_\pi = \mathbb E_{s\sim\mathcal D,\ a\sim\pi_\theta}\left[\alpha\log\pi_\theta(a\mid s) - Q_\phi(s, a)\right]$$

---

## Part E. Evaluation statistics

Code: `interceptor/eval/stats.py`, `interceptor/eval/sweeps.py`

| Statistic | Inputs | Output |
|---|---|---|
| Wilson interval (EQ-EVAL-1) | kills $k$, episodes $n$, $z = 1.96$ | 95 % interval on kill rate |
| Median and IQR (EQ-EVAL-2) | miss distances | location and spread |
| Empirical CDF | miss distances, grid | $F(x)$ |
| Exact McNemar (EQ-EVAL-3) | discordant pair counts $b$, $c$ | two-sided p-value |
| Paired bootstrap (EQ-EVAL-4) | paired samples, $B = 2000$ | 95 % interval on the difference |
| Seed summary | one value per training seed | mean, min, max, range |
| Kill rate against radius | miss distances, radii | kill rate at each radius |
| Residual split | RL residual, positions | along and across line-of-sight parts |

$$\frac{\hat p + \dfrac{z^2}{2n} \pm z\sqrt{\dfrac{\hat p(1 - \hat p)}{n} + \dfrac{z^2}{4n^2}}}{1 + \dfrac{z^2}{n}}, \qquad \hat p = \frac kn$$

$$\tilde x = \operatorname{median}(x), \qquad \mathrm{IQR} = Q_{0.75} - Q_{0.25}, \qquad F(x) = \frac{\#\{x_i \le x\}}{n}$$

$$p = \min\!\left(1,\ 2\sum_{i=0}^{\min(b,c)}\binom{b + c}{i}2^{-(b + c)}\right)$$

$$\Delta_b = \operatorname{stat}(\{x_i : i\in I_b\}) - \operatorname{stat}(\{y_i : i\in I_b\}), \qquad \mathrm{CI} = [\Delta_{(0.025)},\ \Delta_{(0.975)}]$$

Kill rate at radius $R$: $F(R)$ of the stored miss distances. Residual split, with $\hat{\mathbf l} = (\mathbf p_T - \mathbf p_M)/\lVert\cdot\rVert$:

$$a_\parallel = \mathbf a_{RL}\cdot\hat{\mathbf l}, \qquad a_\perp = \lVert\mathbf a_{RL} - a_\parallel\hat{\mathbf l}\rVert$$

Robustness sweeps replace one quantity and re-evaluate: seeker noise scale $\kappa$; target acceleration $a_{T,\max} = \rho_a\,a_{n,\max}$; target speed $v_{T,nom} = \rho_v\,v_{\max}$.

---

## Part F. Older code path still in the package (version 1 environment)

These modules predate the unified environment of Part A. They are still imported by `interceptor/envs/intercept_env.py`, the old training orchestrator, the visualiser command line and their tests. They are not used for the results reported for the three platforms.

### F1. APN module

Code: `interceptor/guidance/apn.py::compute_apn_command`. Same law as A2 with these helper definitions:

$$\boldsymbol\omega_{LOS} = \frac{\mathbf r\times\mathbf v_{rel}}{\lVert\mathbf r\rVert^2}, \qquad t_{go} = \operatorname{clip}\!\left(\frac{r}{\max(v_{cl}, 0.1)},\ t_{go,\min},\ t_{go,\max}\right), \qquad \mathbf a = N'\frac{\mathbf v_{rel,\perp}}{t_{go}} + \frac{N'}{2}\mathbf a_{T,\perp}$$

### F2. Simple kinematic model

Code: `interceptor/guidance/kinematic_model.py::KinematicModel`. Inputs: state, $\mathbf a_{cmd}$, $\boldsymbol\omega_{cmd}$. Output: next state. No lag, no jerk limit.

$$\mathbf a = \operatorname{sat}(\mathbf a_{cmd}, a_{\max}), \qquad \mathbf v \leftarrow \operatorname{sat}(\mathbf v + \mathbf a\Delta t,\ v_{\max}), \qquad \mathbf p \leftarrow \mathbf p + \mathbf v\Delta t, \qquad \boldsymbol\omega = \operatorname{sat}(\boldsymbol\omega_{cmd}, \omega_{\max})$$

Heading-rate command (`intercept_env._compute_heading_rate`): $\boldsymbol\omega_{cmd} = \operatorname{sat}\!\left(\dfrac{\hat{\mathbf v}\times\mathbf a}{\max(s, 1)},\ \omega_{\max}\right)$.

Attitude follows the velocity. Shortest-arc quaternion from unit $\mathbf u$ to unit $\mathbf v$, $c = \mathbf u\cdot\mathbf v$:

$$\mathbf q = \left[\frac s2,\ \frac{\mathbf u\times\mathbf v}{s}\right], \qquad s = \sqrt{2(1 + c)}$$

Spherical interpolation toward that attitude with fraction $t = 0.35$, $\theta_0 = \arccos(\mathbf q_0\cdot\mathbf q_1)$:

$$\mathbf q(t) = \frac{\sin((1 - t)\theta_0)}{\sin\theta_0}\mathbf q_0 + \frac{\sin(t\theta_0)}{\sin\theta_0}\mathbf q_1$$

Rate increment over one step: $\delta\mathbf q = \left[\cos\tfrac{\lVert\boldsymbol\omega\rVert\Delta t}{2},\ \hat{\boldsymbol\omega}\sin\tfrac{\lVert\boldsymbol\omega\rVert\Delta t}{2}\right]$, $\mathbf q \leftarrow \delta\mathbf q\otimes\mathbf q$.

### F3. Trajectory oracle

Code: `interceptor/guidance/optimal_trajectory.py`. Inputs: interceptor state, target path, $v_{\max}$, $a_{\max}$. Outputs: reference trajectory, its first acceleration and velocity, and a cost.

Minimum time to cover distance $d$ from initial speed $s_0$ (accelerate, then cruise):

$$t_{acc} = \frac{v_{\max} - s_0}{a}, \qquad d_{acc} = s_0t_{acc} + \tfrac12at_{acc}^2, \qquad t = \begin{cases}\dfrac{-s_0 + \sqrt{s_0^2 + 2ad}}{a} & d \le d_{acc}\\[2mm] t_{acc} + \dfrac{d - d_{acc}}{v_{\max}} & \text{otherwise}\end{cases}$$

Static target, accelerate-cruise-decelerate profile to rest: peak speed $v_p = \min(\sqrt{ad}, v_{\max})$, $t_{acc} = v_p/a$, $t_{cruise} = \max\!\left(0, (d - v_p^2/a)/v_p\right)$, total $2t_{acc} + t_{cruise}$.

Moving target: earliest time on a grid at which the reach time is no later than the grid time, then a cubic Hermite curve between the boundary states, $\tau = t/t_f$:

$$\mathbf p(\tau) = h_{00}\mathbf p_0 + h_{10}t_f\mathbf v_0 + h_{01}\mathbf p_f + h_{11}t_f\mathbf v_f$$

$$h_{00} = 2\tau^3 - 3\tau^2 + 1, \quad h_{10} = \tau^3 - 2\tau^2 + \tau, \quad h_{01} = -2\tau^3 + 3\tau^2, \quad h_{11} = \tau^3 - \tau^2$$

Velocity and acceleration are its first and second time derivatives. $t_f$ is multiplied by 1.10 until the peak speed and acceleration are within limits. Cost: $\sum_k\lVert\mathbf a_k\rVert^2\Delta t$.

### F4. Reactive target

Code: `interceptor/envs/target_generator.py::TargetGenerator`. Every 0.3 to 1.0 s a new jerk is drawn; with the evasive probability it points away from the interceptor:

$$\mathbf j = j_{\max}\,(\hat{\mathbf f} + 0.6\,\hat{\mathbf l}), \qquad \hat{\mathbf f} = \frac{\mathbf p_T - \mathbf p_M}{\lVert\cdot\rVert}, \qquad \hat{\mathbf l}\perp\hat{\mathbf f}\ \text{random}$$

$$\mathbf v \leftarrow \mathbf v + \mathbf a\Delta t, \qquad \mathbf p \leftarrow \mathbf p + \mathbf v\Delta t, \qquad \mathbf a \leftarrow \operatorname{sat}(\mathbf a + \mathbf j\Delta t,\ n_gg)$$

Containment: at the world radius the horizontal velocity is reflected, $\mathbf v_h \leftarrow \mathbf v_h - 2(\mathbf v_h\cdot\hat{\mathbf n})\hat{\mathbf n}$; speed is capped at 30 m/s.

### F5. Version 1 reward

Code: `interceptor/envs/reward.py::RewardFunction`, weights per stage in `configs/stages.yaml`. Accelerations are normalised by $a_{\max}$.

$$r = k_p\operatorname{clip}\!\left(\frac{d_{prev} - d}{\max(d, 1)}, \pm1\right) - k_{LOS}\lVert\boldsymbol\omega_{LOS}\rVert + k_{cv}\frac{\max(0, v_{cl})}{v_{\max}} - k_E\lVert\mathbf a_{tot}\rVert^2$$

$$\qquad - k_{RL}\lVert\mathbf a_{RL}\rVert^2 - k_s\lVert\Delta\mathbf u\rVert^2 - k_t - k_Z\frac{\lvert\mathrm{ZEM}\rvert}{\max(d, 1)} + r_{term}$$

$$r_{term} = \begin{cases}k_{kill}\max\!\left(0, 1 - \dfrac{d}{r_{kill}}\right)\max\!\left(0, 1 - \dfrac{k}{K}\right) & \text{kill}\\ k_{crash} & \text{out of bounds}\end{cases}$$

### F6. Target predictor

Code: `interceptor/prediction/kalman.py`, `const_vel.py`. Constant-acceleration look-ahead over $\Delta$, written as a state transition with the acceleration as input:

$$\begin{bmatrix}\mathbf p\\ \mathbf v\end{bmatrix}^+ = \begin{bmatrix}I & \Delta I\\ 0 & I\end{bmatrix}\begin{bmatrix}\mathbf p\\ \mathbf v\end{bmatrix} + \begin{bmatrix}\tfrac12\Delta^2I\\ \Delta I\end{bmatrix}\mathbf a$$

---

## Part G. Differences and coverage

### G1. Where the code differs from `docs/mathematics.md`

| Topic | Maths document | Code |
|---|---|---|
| Quad acceleration envelope | tangential and normal limits for every platform | quad limits only the total vector to $a_{n,\max}$ |
| Downward limit `a_n_down_max` | in the platform table | not applied to the interceptor |
| Time to go | $r/v_{cl}$ | $r/\max(v_{cl}, 1)$; the APN law also floors $t_{go}$ at 0.06 s |
| Fixed-gain seeker filter | position innovation drives position, velocity and acceleration | each of position, velocity, acceleration is measured and blended separately |
| Dropout | probability 0.04; acceleration decays by 0.98 per step | probability $0.04\kappa$; no decay |
| Terminal break direction | component of $\mathbf n_h$ perpendicular to the line of sight | component of the line of sight perpendicular to the target velocity |
| Missile preset used by default | $a_{n,\max} = 358.1$ m/s², $\tau_a = 0.1$ s | $38g = 372.8$ m/s², $\tau_a = 0.07$ s unless config limits are passed |
| Fixed-wing preset used by default | $v_{\max} = 40$ m/s, range 400–900 m | $v_{\max} = 255$ m/s, range 1200–2800 m unless config limits are passed |
| Reward equation IDs | shaping is EQ-GUIDE-3, action EQ-GUIDE-4, time EQ-GUIDE-5, terminal EQ-GUIDE-6 | comments in `reward.py` label them 4, 5, 6 and 3 |

### G2. Coverage

Read in full for this document: all of `interceptor/sim`, `interceptor/fc`, `interceptor/bridge`, `interceptor/eval`; in `interceptor/guidance`: `kinematics`, `apn`, `env`, `targets`, `maneuvers`, `seeker`, `tracker`, `observation`, `reward`, `airframes`, `platform`, `kinematic_model`, `optimal_trajectory`; `interceptor/envs/target_generator.py`, `envs/reward.py`; `interceptor/prediction/kalman.py`; `scripts/measure_quad_limits.py`.

Not transcribed from code:
- Fixed-wing thrust, drag and maximum-speed derivations (EQ-FW-1, 3, 4, 7) and missile decks and axial limits (EQ-MS-2 to 6): these exist as derivations and config values; only stall speed, load factor, turn rate and the normal-force limit are functions in code.
- `interceptor/envs/intercept_env.py` and `envs/obs_builder.py` (version 1 environment and its 16-value frame): only the heading-rate formula was checked.
- The remaining `scripts/` (benchmarks, plotting, evaluation drivers) and the visualiser: they call the functions above; any plotting or camera arithmetic in them is not listed.
- Network internals of Stable-Baselines3.
