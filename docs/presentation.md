# APN + RL Autonomous Interceptor — Technical Presentation

> **Audience:** Project supervisor / professor
> **Purpose:** Explain the migration from the previous physics-based setup to the
> current APN + RL path-planning setup, where APN and RL are each used, and what
> the training stages are.
> **Status:** Implementation complete (12/12 plan phases), 53/53 tests green,
> demo videos rendered.

---

## 0. TL;DR (30-second version)

The project **pivoted from a physics-based quadrotor simulator to a 3-D
path-planning interceptor.**

| | **OLD** (v6.0, archived) | **NEW** (v1.0, current) |
|---|---|---|
| **Question being answered** | "How do I make a Crazyflie fly?" | "How do I *intercept* a target?" |
| **Physics** | Full rigid-body: RK4 integration, PID, motor RPM, aerodynamic drag, ground effect, wind | **Kinematic only**: velocity + acceleration commands, clamped |
| **Control output** | `command_from_action` → `rpm_vector` (4 motor RPMs) | **`velocity_output` = `[vx, vy, vz, ωx, ωy, ωz]`** |
| **Guidance law** | *None* — RL alone | **Augmented Proportional Navigation (APN)** as the backbone |
| **RL's role** | Primary controller | **Residual correction** on top of APN |
| **Curriculum** | 8 stages | **3 stages** (static → polynomial → evasive) |
| **Difficulty driver** | Aerodynamic complexity / wind | **Target manoeuvrability** |

The single most important idea: **APN does the guidance; RL only fixes what APN
gets wrong.** This is why stage 1 and 2 are "free" (APN already solves them) and
stage 3 is the only stage where RL genuinely learns.

---

## 1. Why the pivot

### 1.1 The old setup's problem

The previous project answered *"can I make a quadrotor hover and track
waypoints?"* That is a **control** problem. It required:

- Rigid-body dynamics (Crazyflie 2.0 model), integrated with **RK4** at high rate.
- Cascaded **PID** loops for attitude and rate.
- A motor mixer mapping wrench → **4 motor RPMs**.
- Models for drag, ground effect, battery sag, wind.

That is a lot of physics — and almost none of it is relevant to the actual
research question, which is **interception**. Most of the effort went into
keeping the vehicle *airborne* rather than into *guidance*.

### 1.2 The new framing: path planning

An interceptor's job is purely geometric: **reach the target before it gets
away, and get within the kill radius.** Motors, RPM and aerodynamics are an
implementation detail of a real vehicle — they are *not* the research problem.

So the environment was rewritten as a **path planner**:

- State = positions and velocities (3-D points and vectors).
- Action = a **bounded residual acceleration** in the world frame.
- The environment converts that into a **velocity command** `[vx,vy,vz,ωx,ωy,ωz]`.
- No RPM, no PID, no RK4. Euler integration of acceleration → velocity → position.

This collapses ~400 lines of dynamics into a 100-line kinematic model, and frees
the whole training budget to be spent on **guidance**, which is the actual goal.

### 1.3 Why add APN at all (why not just RL)?

Proportional Navigation is a **classical missile-guidance law** from the 1940s.
It is *provably* near-optimal against a non-maneuvering target, and it needs
**zero training**. Our own measurements confirm it: APN alone intercepts
**99%** of static targets and **99%** of predictable polynomial targets.

Handing a tabula-rasa RL agent a problem that APN already solves is wasteful.
The insight is:

> **Let APN solve the solvable part, and let RL learn only the residual that APN
> cannot handle (evasive, reactive targets).**

That gives a much easier learning problem, better sample efficiency, and a policy
that is interpretable (you can look at the RL correction and see *where* the
classical law fails).

---

## 2. Architecture

```
                        ┌──────────────────────────────┐
  target state ───────► │   APN guidance law (apn.py)  │──►  a_APN
  interceptor state     │   a = N'·perp(V_rel)/t_go    │
                        │     + (N'/2)·perp(a_T)       │
                        └──────────────────────────────┘
                                     │
                                     ▼
                             a_total = a_APN + Δa_RL   ◄── clamp to a_max
                                     ▲
                        ┌──────────────────────────────┐
   observation ───────► │   PPO residual policy        │──►  Δa_RL
   (85-dim actor obs)   │   Box([-1,1], shape=(3,))    │
                        │   × stage rl_action_scale    │
                        └──────────────────────────────┘
                                     │
                                     ▼
                        ┌──────────────────────────────┐
                        │  Kinematic model (100 Hz)    │──►  velocity_output
                        │  integrate + clamp v_max     │      [vx,vy,vz,ωx,ωy,ωz]
                        └──────────────────────────────┘
```

**The hybrid equation (the whole project in one line):**

```
a_cmd  =  a_APN  +  Δa_RL
         └─── classical ───┘   └─ learned ─┘
```

### 2.1 Which subsystem does what

| Concern | Handled by | Where |
|---|---|---|
| Geometry of interception (leading the target) | **APN** | `src/guidance/apn.py` |
| Target acceleration compensation | **APN** (the `(N'/2)·perp(a_T)` term) | `src/guidance/apn.py` |
| Time-to-go estimation | **APN** | `compute_time_to_go()` |
| Residual correction against evasive targets | **RL** | `src/envs/intercept_env.py` + PPO |
| Speed / acceleration limits | **Kinematic model** (clamp) | `src/guidance/kinematic_model.py` |
| Turning the vehicle to face the commanded accel | **Kinematic model** (heading-rate + slerp) | `src/guidance/kinematic_model.py` |
| Reward shaping | Reward function (13 terms) | `src/envs/reward.py` |
| Privileged (cheating) critic input | Optimal-trajectory oracle | `src/guidance/optimal_trajectory.py` |

---

## 3. APN — Augmented Proportional Navigation

### 3.1 Plain-English intuition

Fly not *at* the target, but at where it *will be*. Proportional Navigation (PN)
does this by steering to zero the **line-of-sight (LOS) rate**: if the line from
you to the target stops rotating, you are on a collision course. "Augmented" (APN)
means we also feed in the target's **acceleration** so we can lead a *manoeuvring*
target, not just a coasting one.

### 3.2 The math (as implemented)

For interceptor position/velocity `p_M, v_M` and target `p_T, v_T, a_T`:

```
R    = p_T − p_M                              (range vector)
R̂    = R / |R|                                (unit LOS)
V_rel = v_T − v_M                             (relative velocity)
V_c  = −V_rel · R̂                             (closing velocity, >0 = closing)
t_go = clip( R / max(V_c, 0.1),  t_go_min,  t_go_max )     (time-to-go)

perp(x)  =  x − (x · R̂) R̂                    (component ⊥ to LOS)

a_APN  =  N' · perp(V_rel) / t_go   +   (N'/2) · perp(a_T)
         └─────── pure PN ───────┘         └─ APN lead term ─┘
```

**Navigation constant** `N'` ≈ **4.0** (raised to **4.5** in stage 3, where
manoeuvre is harder).

**Zero-Effort Miss (ZEM)** — how far you would miss if you did nothing:

```
ZEM  =  | perp( V_rel · t_go  +  ½ · a_T · t_go² ) |
```

ZEM is used both inside the APN law (implicitly) and as a **reward term** (to
push the policy toward small miss distance).

### 3.3 ⚠ The plan's formula was wrong (and what we did instead)

The written plan specified the cross-product form
`a_M = N'(V_rel × ω_LOS) + …`. Taken literally, that expression has a **non-zero
component along the LOS** (physically meaningless for a lateral steering command)
and an **inconsistent lateral magnitude** — it is not dimensionally consistent.

The mathematically equivalent and correct **zero-effort-miss form**,
`a_APN = N'·perp(V_rel)/t_go + (N'/2)·perp(a_T)`, is what is actually
implemented, and it was numerically verified against static / crossing /
accelerating targets. *(This is logged as Bug #1 in `BUGS.md`.)*

### 3.4 Two numerical bugs the plan didn't anticipate

1. **`t_go` explosion.** The plan's `t_go = R / max(V_c, 0.1)` is unbounded when
   the interceptor is momentarily *opening* the range (`V_c ≤ 0`): `t_go` jumps to
   `10·R`, so `ZEM ∝ V_rel·t_go` explodes to ~260,000 per step and the reward
   collapses to ≈ −1.75 million. Fixed by clipping `t_go ≤ t_go_max = 20 s`.
2. **Unbounded ZEM reward.** The plan used a raw metric ZEM penalty
   `−k_ZEM·ZEM`. Normalised to `−k_ZEM·|ZEM| / max(range, 1)` so it stays O(1).

### 3.5 Verification of APN alone

Rolling out APN with **zero RL action** (`--model apn`):

| Stage | Target type | Kill rate |
|---|---|---|
| 1 | static | **12/12 (100%)** |
| 2 | order-1 polynomial | **12/12 (100%)** |
| 3 | evasive | **0/12 (0%)** |

This is the empirical justification for the whole architecture: **APN is
sufficient for predictable targets and fails completely on evasive ones.**

---

## 4. RL — the PPO residual policy

### 4.1 What it learns

RL does **not** learn to fly. It learns a **bounded correction** to the APN
command — the part of the acceleration that the classical law cannot supply
against a reactive, dodging target.

```
Δa_RL = action × rl_action_scale       # action ∈ [-1,1]³
```

`rl_action_scale` grows with difficulty, so the policy's authority increases as
the target gets harder (see §5). Because the output is bounded and small relative
to `a_max`, a badly-behaved policy **cannot destabilise the episode** — it just
adds a bounded nudge to an already-decent APN command. This is a much safer,
better-conditioned learning problem than training a controller from scratch.

### 4.2 Algorithm & architecture

- **PPO** (Stable-Baselines3), continuous action space.
- **Asymmetric actor-critic.** The *actor* only sees what a real onboard sensor
  would plausibly provide (85 dims). The *critic* additionally sees **privileged
  ground-truth information** (16 dims) that would be unavailable in deployment —
  the target's true future trajectory, its true acceleration, and the optimal
  oracle solution. This is a standard technique to speed up training without
  leaking privileged information into the deployed policy. Implementation:
  `AsymmetricActorCriticPolicy` / `_SplitExtractor`, which slice
  `features[:, :85]` for the actor and feed all 101 to the critic.
- **Networks:** actor `[256,256,128]`, critic `[256,256,128]`, ReLU, `log_std_init=0`.
- **Hyperparameters:** `n_steps=4096`, `batch=512`, `γ=0.995`, `lr=3e-4`,
  `λ=0.95`, `clip=0.2`, `ent_coef=0.005`, `n_epochs=10`, `8` parallel envs.

### 4.3 Observation space (85 actor dims)

Built from a **16-dim instantaneous frame** plus temporal context:

**Single frame (16 dims)**

| idx | field | dim | meaning |
|---|---|---|---|
| 0–2 | `rel_pos` | 3 | target − interceptor |
| 3–5 | `rel_vel` | 3 | target velocity − interceptor velocity |
| 6–8 | `interceptor_vel` | 3 | own velocity |
| 9 | `range` | 1 | scalar distance |
| 10 | `closing_vel` | 1 | closing speed |
| 11–13 | `LOS_rate` | 3 | `(rel_pos × rel_vel)/range²` |
| 14 | `t_go_est` | 1 | estimated time-to-go |
| 15 | `visible` | 1 | 1/0 |

**Stacking**

| block | formula | dims |
|---|---|---|
| current frame | — | 16 |
| history (m=3, p=2) | `16 × (1+3)` | 64 |
| future predictions (n=3, q=5) | `7 × 3` | 21 |
| **actor total** | `64 + 21` | **85** |
| privileged critic (separate) | `time_remaining + optimal_accel + optimal_vel + target_true_future_pos/vel + target_true_accel` | **16** |
| **env total** | `85 + 16` | **101** |

The future block holds *predicted* target positions/velocities from a
**const-velocity** or **Kalman** predictor (the `future_source: pred` setting) —
i.e. it represents what the interceptor *infers*, whereas the critic's
`target_true_future_*` fields are the *actual* future (available only to the
critic). The gap between prediction and truth is exactly the residual the RL
policy must absorb.

### 4.4 Reward (13 terms)

Normalised progress, LOS-rate, closing-velocity, energy, RL-residual,
smoothness, time, ZEM, a **kill bonus** (scaled by miss accuracy and by how early
the kill happens), and an out-of-bounds crash penalty. All weights are per-stage
tunable in `configs/stages.yaml`.

```
r = k_progress·Δprogress  −  k_LOS·|LOS|  +  k_close·max(0,V_c)/v_max
    − k_energy·|a_total|²  −  k_rl·|Δa_RL|²  −  k_smooth·|Δaction|²
    − k_time  −  k_ZEM·|ZEM|/max(range,1)
    + (kill ? bonus : 0)  + (oob ? crash_penalty : 0)
```

The `−k_rl·|Δa_RL|²` term is deliberate: it **incentivises the policy to stay out
of APN's way** unless a correction is genuinely needed — which is precisely why
stages 1–2 show a reward improvement even though the kill rate is already at 99%.

---

## 5. The 3-stage curriculum (what stages exist)

The old project had 8 stages tied to aerodynamic difficulty. The new one has
**3 stages tied to target manoeuvrability**, and a single policy is trained
through them sequentially (warm-started from the previous stage).

| | **Stage 1** | **Stage 2** | **Stage 3** |
|---|---|---|---|
| **Target type** | `static` | `order_1` (polynomial) | `evasive` |
| **Target behaviour** | Frozen | Smooth ballistic arc, drone-independent | Reactive: re-samples a jerk-based maneuver, flees the interceptor |
| **Spawn range** | 20–60 m | 30–80 m | 40–100 m |
| **World radius** | 150 m | 200 m | 250 m |
| **Episode length** | 3–5 s | 5–10 s | 8–15 s |
| **Kill radius** | 1.0 m | 1.0 m | 1.5 m |
| **Target speed** | 0 | 5–20 m/s | 10–30 m/s |
| **Target accel** | 0 | 0–5 m/s² | up to 3 g, jerk ≤ 20 m/s³ |
| **Evasive probability** | — | 0 | 0.6 |
| **`rl_action_scale`** | **1.5** m/s² (0.1·a_max) | **4.5** m/s² (0.3·a_max) | **7.5** m/s² (0.5·a_max) |
| **`N'`** | 4.0 | 4.0 | 4.5 |
| **Training budget** | 2 M steps | 5 M steps | 10 M steps |
| **Advance threshold** (kill rate) | ≥ 0.95 | ≥ 0.90 | ≥ 0.75 |

### 5.1 What each stage is *for*

- **Stage 1 — static.** Sanity check. Teaches the pipeline to run end-to-end and
  the policy to produce a near-zero residual. *Expected outcome: trivially passed
  by APN; RL learns only to minimise its own residual magnitude.*
- **Stage 2 — polynomial (order 1).** A smooth, **predictable** moving target.
  The path is a closed-form cubic/quadratic/cubic-Hermite blend sampled at reset
  that the interceptor **cannot influence** (so it is exactly predictable from
  position + velocity). *APN still solves it; RL's job remains "don't interfere",
  now with a non-trivial LOS geometry.*
- **Stage 3 — evasive.** The real challenge. The target re-samples an
  acceleration/jerk maneuver every 0.3–1.0 s and, with probability 0.6, steers
  **away from the interceptor**. Pure APN collapses (0% kills). *This is the only
  stage where the RL residual genuinely learns guidance the classical law cannot
  provide.*

### 5.2 Curriculum scheduler

`CurriculumScheduler` inspects a sliding 100-episode window of kill rates:

```
should_advance:   recent_kill_rate(window=100)  ≥  stage.threshold
is_capped:        steps_trained ≥ max_training_steps  and  not advanced
```

The same PPO model is carried across stages (`model.set_env(...)`) so the policy
warm-starts each harder stage from a competent initialisation. Rollback is
supported (`rollback_threshold`) though unused by default.

---

## 6. Retained vs deleted code

The migration was a **clean rewrite**, not a refactor — deleted outright:

**Deleted:** `physics/` (RK4/PID/motors), `prediction/` (old), `stage-runtime/`,
`docker/`, the `compare_*` / `probe_*` / `validate*` / `self_test` scripts, the
`exp_*` notebooks, and the 8-stage configs and trained weights from v6.0.
(The full old tree was archived to a separate GitHub repo,
`interceptor-drone-rl`, commit `493d370`.)

**Retained (and adapted):** the *patterns* — `target_generator.py` (the
polynomial-target sampler was highly reusable), the `ObsBuilder` history/future
stacking scheme, the `stage_config` dataclass style, the reward-function
structure, and the **entire training infrastructure**: `orchestrator`,
`callbacks`, `curriculum`, `checkpoint_manager`, `weight_transfer`,
`asymmetric_policy`, `results`, `api`, `config_loader`, `logger`.

> Note: `weight_transfer.py` retains two constants from the *old* observation
> layout (`SHARED_STAGE1_PREFIX=12`, `STAGE1_OBS_DIM=14`) that no longer match
> the new 16-dim frame. It is **not imported** by the new orchestrator, so this
> is harmless, but it is flagged in `BUGS.md`.

---

## 7. Repo layout

```
run.py                    ← worked examples + CLI for train_model()
conftest.py               ← pytest sys.path bootstrap + 'slow' marker
requirements.txt          ← numpy, scipy, gymnasium, sb3, torch, pyvista,
                            trimesh, imageio, pyyaml, tensorboard, filterpy, pytest
configs/
  stages.yaml             ← the 3-stage curriculum (authoritative table)
  model.yaml              ← PPO architecture + 6 privileged critic fields
  config.yaml             ← global / observation / interceptor / apn settings
  visualiser.yaml         ← PyVista scene, camera, per-model scale & rotation
src/
  api.py                  ← train_model() / evaluate_model()
  results.py              ← TrainResult / ModelResult / StageOutcome
  guidance/               ← apn.py, kinematic_model.py, optimal_trajectory.py
  envs/                   ← intercept_env.py, obs_builder.py, reward.py,
                            stage_config.py, target_generator.py
  prediction/             ← base.py, const_vel.py, kalman.py
  training/               ← orchestrator, callbacks, curriculum,
                            asymmetric_policy, checkpoint_manager, weight_transfer
  utils/                  ← config_loader, model_loader, logger
visualiser/               ← renderer.py, animator.py, scene_builder.py, cli.py
scripts/                  ← train.py, evaluate.py, visualise.py, smoke_test.py,
                            download_models.py
tests/                    ← 53 pytest tests (unit + integration)
models/                   ← interceptor_drone.obj, target_drone.obj
data/                     ← runs/, videos/, evaluation/
```

---

## 8. Results & verification

| Check | Result |
|---|---|
| `python scripts/smoke_test.py` | **11 / 11 pass** |
| `pytest tests/ -m "not slow"` | **52 passed** |
| `pytest tests/` (incl. end-to-end) | **53 passed** |
| Observation dims (actor / critic / env) | 85 / 16 / 101 — asserted for all stages |
| Action space | `(3,)` continuous |
| Pure-APN kill rate (stages 1 / 2 / 3) | 100% / 100% / **0%** |
| Visualiser | valid MP4 produced off-screen (no GPU) |

### 8.1 Learning behaviour (benchmarked on this CPU host)

| Stage | FPS (8 envs) | APN-only kills | Random-action kills | RL learns quickly? |
|---|---|---|---|---|
| 1 | ~1,280 | 0.99 | 0.99 | No — already solved; reward rises 12→18 over ~8k steps |
| 2 | ~1,280 | 0.99 | 0.99 | No — APN dominates |
| 3 | ~280 | 0.04 | 0.03 | **Only real learning**, but needs the full 10 M budget (≈10 h CPU, GPU recommended) |

**Honest caveat for discussion:** stage 1's kill rate is *already* ~99% before
any training, because a static target is APN's ideal case. The visible "learning"
in stage 1 is a small reward gain (residual magnitude shrinking), not a kill-rate
curve. Stage 3 is where meaningful learning happens, and 500 k CPU steps only moved
kill rate 3% → ~12% — the designed 10 M-step curriculum (warm-started through
stages 1→2→3) is the intended path, ideally on a GPU.

---

## 9. Demo assets

Two orientations of the real drone meshes were calibrated (`I08` interceptor,
`T21` target) and rendered:

| Video | Stage | Camera |
|---|---|---|
| `data/videos/stage1_static.mp4` | 1 (static) | chase |
| `data/videos/stage2_polynomial.mp4` | 2 (order-1) | chase |
| `data/videos/stage1_fixedcam.mp4` | 1 (static) | **static** (framed on whole trajectory, zero camera drift) |
| `data/videos/stage2_fixedcam.mp4` | 2 (order-1) | **static** |

A model-orientation workflow exists for new meshes: the script renders all 24
proper cube rotations, you pick the code where the drone is upright and
nose-forward, and it is written into `configs/visualiser.yaml`
(`interceptor_rotation: I08`, `target_rotation: T21`, plus per-model scales
`0.1612` / `0.5114` because the two meshes use different raw units).

---

## 10. Open questions / discussion points

1. **Is the 3-stage split the right difficulty ladder?** Stage 1 is arguably too
   easy (APN-only ≈ 99%). We could skip it or replace it with a deliberately
   *tight-geometry* static case (short episode cap, large spawn distance) to make
   it actually exercise learning.
2. **Stage-3 sample efficiency.** 500 k steps bought only a small kill-rate gain.
   Options: denser reward around near-miss ZEM, curriculum within stage 3 (tune
   `evasive_probability` up gradually), asymmetric-advantage tuning, or simply
   more compute (GPU, larger `n_envs`).
3. **Should stages 1–2 be *learned* at all?** One could hand-off the APN solution
   directly (zero residual) and start training at stage 3, saving ~7 M steps.
4. **Realism.** The current model is pure kinematics. If the supervisor wants
   vehicle realism, a middle ground is to reintroduce first-order lag on the
   acceleration command (`τ·ȧ + a = a_cmd`) without bringing back full motor/RPM
   dynamics.

---

## 11. Key file → concept map (for Q&A)

| If asked about… | Look at… |
|---|---|
| APN guidance math | `src/guidance/apn.py` — `compute_apn_command()` |
| Why the ZEM-equivalent APN form | `BUGS.md` #1; `context.md` Issue 1 |
| Vehicle dynamics | `src/guidance/kinematic_model.py` — `KinematicModel` |
| Optimal-trajectory oracle (critic's cheat sheet) | `src/guidance/optimal_trajectory.py` |
| The Gym environment / step order | `src/envs/intercept_env.py` — `InterceptEnv.step()` |
| Observation building & stacking | `src/envs/obs_builder.py` |
| Reward | `src/envs/reward.py` |
| Stage definitions & tunables | `configs/stages.yaml` + `src/envs/stage_config.py` |
| Target generation (all 3 types) | `src/envs/target_generator.py` |
| Predictors (const-vel / Kalman) | `src/prediction/` |
| Asymmetric actor-critic | `src/training/asymmetric_policy.py` |
| Training loop & curriculum | `src/training/orchestrator.py`, `curriculum.py` |
| Public API | `src/api.py` — `train_model()`, `evaluate_model()` |
| Visualiser / MP4 | `visualiser/` |
| Known bugs & deviations | `BUGS.md` |
| Full context | `context.md`, `CURRENT_WORK.md` |
