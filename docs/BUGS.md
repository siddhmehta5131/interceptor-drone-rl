# BUGS — known issues

This project is a fresh implementation of `updation_plan.md` v1.0 (APN + RL
interceptor). Findings below are grouped by status. Per task instructions,
**non-blocking** defects are logged here rather than fixed mid-task.

Date: 2026-10-16.

---

## Fixed during implementation

### 1. Plan §2.3 APN cross-product is not dimensionally consistent — FIXED
**Area:** `src/guidance/apn.py` (`compute_apn_command`)
**Impact:** Would have produced a wrong guidance command. The plan's literal form
`a_M = N'·(V_rel × ω_LOS) + (N'/2)·a_T⊥` has a non-zero component **along the
line of sight** (a cross product of two vectors is perpendicular to both, but
`V_rel` is not generally perpendicular to the LOS) and an inconsistent lateral
magnitude. Plan §2.4 then derives the correct equivalent, `a_M = N'·ZEM⊥/t_go²`,
which is what PN standard theory gives.
**Fix:** implemented the mathematically equivalent **zero-effort-miss** form
`a_APN = N'·perp(V_rel)/t_go + (N'/2)·perp(a_T)` with
`perp(x) = x − (x·R̂)R̂`. Verified against static, crossing and accelerating
targets (see `tests/test_apn.py`).

### 2. `t_go` unbounded explosion when the range opens — FIXED
**Area:** `src/guidance/apn.py` (`compute_time_to_go`, `compute_apn_command`)
**Impact:** High. When the interceptor opens the range (`V_c ≤ 0`),
`t_go = R / max(V_c, 0.1)` collapses to `10·R` and the zero-effort-miss
(∝ `V_rel · t_go`) exploded to ≈ 2.6 × 10⁵ per step, driving the reward to
≈ −1.75 × 10⁶.
**Fix:** clamp `t_go = clip(R / max(V_c, 0.1), t_go_min, t_go_max)` with a new
`t_go_max = 20 s` (configurable in `configs/config.yaml`).

### 3. Raw ZEM reward term was unbounded — FIXED
**Area:** `src/envs/reward.py`
**Impact:** Medium. Plan §10.3 specified `r_ZEM = −k_ZEM · ZEM`, but ZEM is a
metric distance, so the term dominated the per-step reward as the range grew.
**Fix:** normalise to `r_ZEM = −k_ZEM · |ZEM| / max(distance, 1.0)`.

### 4. `stage_cfg.apn` does not exist (config-access gotcha) — test fixed
**Area:** `tests/test_integration.py`
**Impact:** Low (test only). A `StageConfig` exposes the APN values as **flat**
fields (`stage_cfg.apn_N_prime`, `stage_cfg.apn_t_go_min`); there is no
`stage_cfg.apn`. The global `cfg.apn` **is** an `APNSpec`. An early test used the
wrong attribute and failed to construct the env; corrected to the flat fields.
**Note for future code:** use `cfg.apn.*` for the global spec and
`stage_cfg.apn_N_prime` / `stage_cfg.apn_t_go_min` for per-stage overrides.

---

## Non-blocking (logged, not fixed)

### 5. VTK GLES shader warnings during off-screen render — RESOLVED
**Area:** `visualiser/renderer.py`
**Impact:** Cosmetic only. On this headless, GPU-less Windows host, VTK printed
`vtkShaderProgram` / `vtkOpenGLPolyDataMapper` ERROR messages during every
off-screen render. The MP4 output was written correctly and was valid.
**Root cause (found):** the spam came from **shadow-map shaders** — the
unconditional `enable_shadows()` call from plan §14.3. A headless GLES context
cannot compile them, so VTK errored on every frame.
**Resolution:** shadows are now **opt-in** via `visualiser.yaml`
(`lighting.shadows: false` by default); `enable_shadows()` is only called when
true. With shadows off the log drops to ~10 lines and `vtkShaderProgram` hits = 0.
This is a deliberate deviation from plan §14.3 (documented in `CURRENT_WORK.md`).

### 6. `weight_transfer.py` constants tied to the old (physics) observation layout
**Area:** `src/training/weight_transfer.py`
**Impact:** None today. `SHARED_STAGE1_PREFIX = 12` and `STAGE1_OBS_DIM = 14`
reflect the previous physics observation, not the new 16-dim relative-kinematics
frame (actor dim 85). The module is retained per the plan ("KEEP AS-IS") but is
**not imported** by the new orchestrator, so the stale constants are inert.
**Suggested fix (future):** update the prefix/dim to the new observation or delete
the module if weight transfer is not used.

### 7. Full training budgets not exercised on this host
**Area:** `configs/stages.yaml`
**Impact:** Low (verification gap, not a defect). The 2 M / 5 M / 10 M step
budgets were only validated with smoke-scale `--steps` overrides on CPU. The
curriculum scheduler advance/cap logic is covered by unit tests, but no full
kill-rate threshold has been reached here.
**Suggested fix:** run the real curriculum on a GPU host.

### 8. `models/*.obj` meshes — placeholder origin (superseded)
**Area:** `models/interceptor_drone.obj`, `models/target_drone.obj`
**Impact:** None now. Originally `scripts/download_models.py` could not fetch
real assets and wrote generated placeholder quadcopters. The user has since
supplied **real** `interceptor_drone.obj` (748,744 B) and `target_drone.obj`
(841,970 B). `scripts/download_models.py` will overwrite these only with
`--force`, so they are preserved.
**Note:** the two models use different raw units/bounds (interceptor
12.4×3.4×8.8, target 3.1×2.0×3.9), so **per-model scale** is required — see
item 9.

### 9. Per-model mesh scale + orientation needed for the visualiser
**Area:** `configs/visualiser.yaml`, `visualiser/scene_builder.py`
**Impact:** Visual only. The single `models.scale: 0.01` from the plan §14.3
reference assumed one asset unit system. The two supplied OBJ files differ in
scale and axis convention, so each needs its own transform:
`interceptor_scale: 0.1612`, `target_scale: 0.5114` (each normalises its mesh to
~2 m max dimension), plus `interceptor_rotation: I08` and `target_rotation: T21`
(proper cube-rotation codes) to make each model upright and front-facing.
**Fix:** `scene_builder` gained `cube_rotations()`, `rotation_matrix_from()` and
`apply_mesh_rotation()`; `renderer._setup_drones` bakes the rotation into the
actor base mesh and applies the per-frame pose matrix on top.
**Status:** Resolved for the two supplied models. New assets need fresh codes —
see the orientation-shot workflow in `CURRENT_WORK.md`.

### 10. Off-screen render is slow on CPU (software rasteriser)
**Area:** `visualiser/renderer.py`
**Impact:** Convenience only. With no GPU, renders run ~**5.4 s/frame** at
1920×1080 (stage-1 = 68 frames ≈ 6 min; stage-2 = 105 frames ≈ 14 min). Shadows
doubled this to ~11 s/frame and are now disabled by default.
**Mitigation:** pass `--resolution 1280x720` (or lower) to the CLI; use a
GPU-backed OpenGL context for production renders.
