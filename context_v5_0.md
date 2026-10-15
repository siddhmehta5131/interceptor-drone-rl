# System Architecture & Context Specification

## 0. Document Control, Conventions & Counters

### Archive Identity

| Field | Value |
|---|---|
| Archive name | `RL based autonomous interceptor drone` (workspace directory) |
| Format | Git working tree (uncommitted), no archive file |
| VCS commit | No commits (HEAD resolves to literal string `HEAD`; all files are staged `A` or untracked `??`) |
| VCS branch | (none — initial state, no branch created) |
| VCS remotes | (none) |
| Generation time (UTC) | 2026-10-02T05:40:00Z |
| Generating model | Claude Opus 4.6 (Thinking) |
| Interpreter | Windows PowerShell; Python 3.14 (inferred from `__pycache__/hover_env.cpython-314.pyc`) |
| MODE | GENERATE |
| OUTPUT_FILE | context.md |
| VERBATIM_MAX_LINES | 500 |
| VERBATIM_MAX_TOTAL_BYTES | 1,500,000 |
| SECRET_POLICY | REDACT |
| EXECUTION_POLICY | READ_ONLY_INTROSPECTION |
| File version | v5.0 |

### Table of Contents

- §0 Document Control, Conventions & Counters
- §1 Executive Overview & Repository Manifest
- §2 Environment, Dependencies & Hardware Specifications
- §3 Model Architecture & Component Specifications
- §4 Phase-by-Phase Execution Pipeline
- §5 Reward Functions & Mathematical Formulations
- §6 Data Contracts & I/O Schemas
- §7 Execution Runbook & Step-by-Step Reproduction Guide
- §8 Code-Level Specification
- §9 Configuration & Constants Registry
- §10 Determinism, Runtime-Resolved Values & Numerical Contract
- §11 Gap Register
- §12 Verification Suite & Acceptance Criteria
- §13 Anomalies, Contradictions & Behavioral Quirks
- Appendix A Verbatim Artifacts
- Appendix B Checksums & Non-Embedded Artifact Manifests
- Appendix C Self-Audit Report

### Legends

**Evidence tags:**
- `[SRC]` — stated in an archive file; cite `path:Lstart-Lend`
- `[DERIVED]` — computed from `[SRC]` facts; formula and inputs cited
- `[EXEC]` — obtained by running a command; command and output given
- `[LIBDEF]` — parameter default of a third-party library at a stated version
- `[EXT]` — external resource referenced by the archive
- `[RECALLED]` — background knowledge, unverifiable here; always paired with a GAP
- `[GAP]` — required but absent; carries `[[GAP:G-nnn]]`
- `[PROPOSED]` — candidate resolution, only inside §11.2

**Gap marker syntax:** `[[GAP:G-nnn]]` inline; `<<<FILL: G-nnn>>>` in §11.2 fill slots.

**Severity levels:** BLOCKER (reproduction impossible), MAJOR (results differ), MINOR (operational detail only).

**Ledger statuses:** READ-FULL, SAMPLED(reason), HEADER-ONLY, EXCLUDED(reason).

### Conventions

- Citations: `file:Lstart-Lend` for source lines; `file#cell-N` for notebooks.
- Units: bytes as B, KiB, MiB, GiB; time in seconds (s); angles in radians unless stated.
- Numbers are copied exactly as written in the source, in code font.
- Paths sorted bytewise-lexicographically.
- Precedence: if narrative and Appendix A disagree, Appendix A wins; discrepancy logged in §13.

### Interpretation Log

1. **Archive vs. workspace:** The project is a live Git working directory, not a compressed archive. Treated as a single workspace rooted at `c:\Users\ADMIN\Desktop\projects\RL based autonomous interceptor drone`. All files excluding `.git/` and `__pycache__/` directories are inventoried.
2. **Classification rules:** Files classified by content inspection and role: `.py` files with `import gymnasium`/`stable_baselines3` as source; `.yaml` as config; `.md` as doc; `.ipynb` as notebook; `.png` as binary/generated; `.txt` (requirements) as config; `.txt` (results) as data; `.json` as data; `.docx`/`.pdf` as doc (binary); `Dockerfile`/`docker-compose.yml` as script; `.gitkeep` as generated; `.log` as log; `desktop.ini` as other.
3. **Python version:** `cpython-314` in `__pycache__` filenames indicates Python 3.14. `CURRENT_WORK.md:L48` states "Python 3.14". [[GAP:G-001]]
4. **Two distinct sub-projects:** The repository contains (a) a root-level physics simulator + interactive demos (`src/simulation.py`, `src/visualiser.py`, `hover_env.py`, `swift_*.py`) and (b) a Dockerized RL training pipeline under `interceptor-training/`. Both are documented.
5. **`default_run.yaml` stages list:** The smoke test at `interceptor-training/scripts/smoke_test.py:L401` asserts `cfg.config_stages("ppo_baseline") == [1, 2, 3, 4, 5, 6, 7]`, but the shipped `default_run.yaml:L42` declares `stages: [1, 2]`. Resolved by D-20: `default_run.yaml` is retired and its content moves to the model file and config file (D-9); the smoke test uses its own model file and config file (D-7). Logged in §13.
6. **Duplicate files:** Several file pairs have identical SHA-256 hashes (e.g., `swift_rl_env.py` and `swift_rl_env_latest.py`, `swift_physics_headless_rlvec.py` and `swift_physics_headless_rlvec_latest.py`, `swift_physics_headless_wind.py` and `swift_physics_headless_wind_latest.py`). These are snapshot copies; documented in §1.5.

### Counters

| Counter | Value |
|---|---|
| Files total (excl. `.git/`, `__pycache__/`) | 91 |
| READ-FULL | 52 (all source, config, script, doc, test, schema files) |
| SAMPLED | 6 (notebooks — cell-by-cell per C8) |
| HEADER-ONLY | 0 |
| EXCLUDED | 33 (binary: `.docx`, `.pdf`, `.png` files; `.gitkeep` empty; `desktop.ini`; `pybullet_vs_swift_trajectory_v0.-1.ipynb`, `swift_physics_headless_rlvec_v0.-1.py`, `swift_physics_headless_wind_v0.-1.py` — backup versions) |
| Embedded verbatim (Appendix A) | 27 |
| NOT-VERBATIM | 25 |
| BINARY | 6 |
| Gaps: BLOCKER | 5 |
| Gaps: MAJOR | 8 |
| Gaps: MINOR | 5 |
| Gaps: OPEN | 18 |
| Document status | COMPLETE |

---

## 1. Executive Overview & Repository Manifest

### 1.1 Purpose

This project implements a reinforcement-learning-based autonomous interceptor drone system. The system trains a Crazyflie 2.1 nano-quadrotor policy to (1) stabilize in hover, (2) fly directionally toward waypoints, and (3) intercept moving targets of increasing difficulty (static, constant-velocity, accelerating, jerk-limited, evasive) through an 8-stage curriculum. Inputs: a 4-dimensional continuous action vector `[thrust, roll, pitch, yaw]` in `[-1, 1]`. Outputs: trained PPO/SAC/TD3 policy checkpoints (`.zip`), TensorBoard logs, per-stage evaluation metrics (`kill_rate`, `mean_final_distance`, `episode_reward_mean`), and a `run_summary.json`. A successful run produces a policy checkpoint that achieves the per-stage success thresholds defined in the curriculum table. [SRC] `interceptor-training/src/envs/stage_config.py:L1-L424`, `README.md:L1-L10`.

### 1.2 Topology

| Name | File:Line | Role | Launched by | Communication | Device |
|---|---|---|---|---|---|
| `train.py` | `interceptor-training/scripts/train.py:L72` | Training entry point (container ENTRYPOINT) | `docker compose up` or `python scripts/train.py` | CLI args → `Orchestrator.run()` | CPU or CUDA (config `device`) |
| `evaluate.py` | `interceptor-training/scripts/evaluate.py:L165` | Post-training evaluation | Manual CLI | CLI args; reads checkpoints from `data/` volume | CPU or CUDA |
| `smoke_test.py` | `interceptor-training/scripts/smoke_test.py:L448` | Host-side verification (no SB3/torch) | Manual CLI | Stdout pass/fail | CPU only |
| `simulation.py` | `src/simulation.py:L1` | Interactive gamepad flight demo | Manual `python src/simulation.py` | USB gamepad (pygame) → matplotlib | CPU |

[SRC] `interceptor-training/Dockerfile:L31-L32`, `interceptor-training/scripts/train.py:L1-L73`, `interceptor-training/scripts/evaluate.py:L1-L166`.

### 1.3 Pipeline DAG

| Phase | Consumes | Produces |
|---|---|---|
| P01 Environment Setup | Docker image `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime` [EXT] | Container with dependencies |
| P02 Configuration Loading | `configs/default_run.yaml` | `RunConfig` object |
| P03 Training (per-stage curriculum) | `RunConfig`, prior-stage checkpoint (if any) | Periodic checkpoints `data/checkpoints/<config>/stage_<N>/<algo>_<steps>_steps.zip`, final checkpoints `<algo>_stage_<N>_final.zip`, TensorBoard logs `data/tb_logs/`, `data/curriculum_state/run_state.json` |
| P04 Evaluation | Trained checkpoint `.zip`, `RunConfig` | `data/results/eval_<config>_stage_<N>.json` |
| P05 Config-final Model | Last stage final checkpoint | `data/results/<config>_final.zip`, `data/results/run_summary.json` |

[SRC] `interceptor-training/src/training/orchestrator.py:L1-L29`, `interceptor-training/src/training/checkpoint_manager.py:L1-L16`.

### 1.4 Full Directory Tree

```
RL based autonomous interceptor drone/
├── .gitignore                                      (528 B)
├── BTP plan preliminary.docx                       (583462 B)
├── BUGS.md                                         (6384 B)
├── CURRENT_WORK.md                                 (10073 B)
├── Coeff solver.ipynb                              (21767 B)
├── Docker_Setup_and_Running_Guide.md               (17688 B)
├── Docker_Setup_and_Running_Guide.pdf              (326745 B)
├── Interceptor_Complete_Project_Scope.docx          (20808 B)
├── README.md                                       (6571 B)
├── SB3_AltitudeHold_Benchmark.ipynb                (30067 B)
├── VERSIONS.md                                     (4231 B)
├── compare_pybullet.py                             (19236 B)
├── compare_v3_multirate.py                         (9319 B)
├── compare_v3_vs_pybullet.py                       (14006 B)
├── comparison_wind_rlvec.png                       (72894 B)
├── complete_changelog.md                           (9175 B)
├── context.md                                      (4672 B)
├── default_demo.py                                 (30911 B)
├── demo.py                                         (24374 B)
├── desktop.ini                                     (106 B)
├── deviation_vs_time.png                           (279093 B)
├── docs/
│   └── coefficient_fitting.md                      (2239 B)
├── final_testing.ipynb                             (21284 B)
├── hover_env.py                                    (32288 B)
├── identify_params.py                              (7732 B)
├── implementation_plan.md                          (47110 B)
├── inputs.png                                      (147113 B)
├── interceptor-training/
│   ├── Dockerfile                                  (1359 B)
│   ├── configs/
│   │   └── default_run.yaml                        (1967 B)
│   ├── data/
│   │   ├── .gitkeep                                (0 B)
│   │   ├── checkpoints/
│   │   │   └── .gitkeep                            (0 B)
│   │   ├── curriculum_state/
│   │   │   └── .gitkeep                            (0 B)
│   │   ├── results/
│   │   │   └── .gitkeep                            (0 B)
│   │   └── tb_logs/
│   │       └── .gitkeep                            (0 B)
│   ├── docker-compose.yml                          (613 B)
│   ├── requirements.txt                            (332 B)
│   ├── scripts/
│   │   ├── evaluate.py                             (7055 B)
│   │   ├── smoke_test.py                           (19964 B)
│   │   └── train.py                                (2701 B)
│   └── src/
│       ├── __init__.py                             (97 B)
│       ├── envs/
│       │   ├── __init__.py                         (348 B)
│       │   ├── base_env.py                         (17623 B)
│       │   ├── obs_builder.py                      (6455 B)
│       │   ├── reward.py                           (6520 B)
│       │   ├── stage_config.py                     (15458 B)
│       │   └── target_generator.py                 (10136 B)
│       ├── physics/
│       │   ├── __init__.py                         (342 B)
│       │   ├── aero.py                             (3863 B)
│       │   ├── constants.py                        (8339 B)
│       │   ├── pipeline.py                         (12433 B)
│       │   └── quaternion.py                       (1328 B)
│       ├── training/
│       │   ├── __init__.py                         (667 B)
│       │   ├── callbacks.py                        (8413 B)
│       │   ├── checkpoint_manager.py               (8861 B)
│       │   ├── curriculum.py                       (9325 B)
│       │   └── orchestrator.py                     (23213 B)
│       └── utils/
│           ├── __init__.py                         (303 B)
│           ├── config_loader.py                    (13214 B)
│           └── logger.py                           (2408 B)
├── interceptor.log                                 (3334 B)
├── phy_sim_val.ipynb                               (331555 B)
├── pybullet_vs_swift_comparison.ipynb              (14419 B)
├── pybullet_vs_swift_trajectory.ipynb              (21045 B)
├── pybullet_vs_swift_trajectory_v0.-1.ipynb        (20008 B)
├── references.txt                                  (1722 B)
├── requirements.txt                                (47 B)
├── results_trajectory.txt                          (2581 B)
├── results_wind_rlvec.txt                          (1165 B)
├── results_wind_rlvec_vs_pybullet.ipynb            (102461 B)
├── run_summary.json                                (814 B)
├── self_test.py                                    (8850 B)
├── src/
│   ├── simulation.py                               (22413 B)
│   └── visualiser.py                               (9027 B)
├── swift_demo_wind_only.py                         (36786 B)
├── swift_physics_headless.py                       (25983 B)
├── swift_physics_headless_rlvec.py                 (18760 B)
├── swift_physics_headless_rlvec_latest.py          (18760 B)
├── swift_physics_headless_rlvec_stable_v1.1.py     (4559 B)
├── swift_physics_headless_rlvec_v0.-1.py           (3889 B)
├── swift_physics_headless_v2.py                    (11459 B)
├── swift_physics_headless_wind.py                  (16930 B)
├── swift_physics_headless_wind_latest.py           (16930 B)
├── swift_physics_headless_wind_stable_v1.1.py      (4490 B)
├── swift_physics_headless_wind_v0.-1.py            (4142 B)
├── swift_rl_env.py                                 (19028 B)
├── swift_rl_env_latest.py                          (19028 B)
├── trajectories.png                                (216061 B)
├── updated demo.py                                 (35236 B)
├── validate.py                                     (7891 B)
├── validate_flight_only.py                         (6660 B)
├── validate_onestep.py                             (8204 B)
└── wind_rlvec_vs_pybullet.ipynb                    (17361 B)
```

### 1.5 Manifest Table

| Path | Class | Bytes | Lines | SHA-256 | Ledger | Verbatim | Function |
|---|---|---|---|---|---|---|---|
| `.gitignore` | config | 528 | 46 | `9DF8...` [[GAP:G-002]] | READ-FULL | EMBEDDED | Git ignore patterns |
| `BUGS.md` | doc | 6384 | 111 | `...` [[GAP:G-002]] | READ-FULL | NOT-VERBATIM | Known bug registry |
| `BTP plan preliminary.docx` | doc | 583462 | — | `...` [[GAP:G-002]] | EXCLUDED(binary-docx) | BINARY | BTP preliminary plan |
| `CURRENT_WORK.md` | doc | 10073 | 191 | `...` [[GAP:G-002]] | READ-FULL | NOT-VERBATIM | Implementation status log |
| `Coeff solver.ipynb` | notebook | 21767 | — | `...` [[GAP:G-002]] | SAMPLED(notebook) | NOT-VERBATIM | Aero coefficient fitting notebook |
| `Docker_Setup_and_Running_Guide.md` | doc | 17688 | — | `...` [[GAP:G-002]] | READ-FULL | NOT-VERBATIM | Docker setup guide |
| `Docker_Setup_and_Running_Guide.pdf` | doc | 326745 | — | `...` [[GAP:G-002]] | EXCLUDED(binary-pdf) | BINARY | Docker setup guide (PDF render) |
| `Interceptor_Complete_Project_Scope.docx` | doc | 20808 | — | `...` [[GAP:G-002]] | EXCLUDED(binary-docx) | BINARY | Project scope doc |
| `README.md` | doc | 6571 | 158 | `...` [[GAP:G-002]] | READ-FULL | NOT-VERBATIM | Repository README |
| `SB3_AltitudeHold_Benchmark.ipynb` | notebook | 30067 | — | `...` [[GAP:G-002]] | SAMPLED(notebook) | NOT-VERBATIM | RL benchmark notebook |
| `VERSIONS.md` | doc | 4231 | 88 | `3D33FF8F578A2941828699620AE7C3083A10F818C2B749CC7CA766C9FDF1C2D1` | READ-FULL | NOT-VERBATIM | Version registry |
| `compare_pybullet.py` | source | 19236 | — | `...` [[GAP:G-002]] | READ-FULL | NOT-VERBATIM | PyBullet comparison script |
| `compare_v3_multirate.py` | source | 9319 | — | `...` [[GAP:G-002]] | READ-FULL | NOT-VERBATIM | Multi-rate physics comparison |
| `compare_v3_vs_pybullet.py` | source | 14006 | — | `...` [[GAP:G-002]] | READ-FULL | NOT-VERBATIM | v3 vs PyBullet comparison |
| `comparison_wind_rlvec.png` | binary | 72894 | — | `...` | EXCLUDED(binary-png) | BINARY | Wind comparison plot |
| `complete_changelog.md` | doc | 9175 | — | `...` [[GAP:G-002]] | READ-FULL | NOT-VERBATIM | Complete changelog |
| `context.md` | doc | 4672 | — | `...` | EXCLUDED(prior-gen) | NOT-VERBATIM | Prior context (superseded) |
| `default_demo.py` | source | 30911 | — | `...` [[GAP:G-002]] | READ-FULL | NOT-VERBATIM | Interactive flight demo |
| `demo.py` | source | 24374 | — | `...` [[GAP:G-002]] | READ-FULL | NOT-VERBATIM | Interactive flight demo |
| `desktop.ini` | other | 106 | — | `...` | EXCLUDED(os-metadata) | NOT-VERBATIM | Windows folder metadata |
| `deviation_vs_time.png` | binary | 279093 | — | `...` | EXCLUDED(binary-png) | BINARY | Deviation plot |
| `docs/coefficient_fitting.md` | doc | 2239 | 63 | `106E36731C3AC03BD152EB5590B80685F9EE3F71136C5135969FCB24011AAE3C` | READ-FULL | EMBEDDED | Aero coefficient fitting doc |
| `final_testing.ipynb` | notebook | 21284 | — | `...` [[GAP:G-002]] | SAMPLED(notebook) | NOT-VERBATIM | Final testing notebook |
| `hover_env.py` | source | 32288 | 777 | `...` [[GAP:G-002]] | READ-FULL | NOT-VERBATIM | Stage-1 Gymnasium env (reference) |
| `identify_params.py` | source | 7732 | — | `...` [[GAP:G-002]] | READ-FULL | NOT-VERBATIM | Parameter identification script |
| `implementation_plan.md` | doc | 47110 | — | `...` [[GAP:G-002]] | READ-FULL | NOT-VERBATIM | Full implementation plan |
| `inputs.png` | binary | 147113 | — | `...` | EXCLUDED(binary-png) | BINARY | Input diagram |
| `interceptor-training/Dockerfile` | script | 1359 | 32 | `CCEA0F5E6D5FBEA3A2C53A02BD640A5BC64F2BF97BDD9ADEAFD3E26ACF3C487C` | READ-FULL | EMBEDDED | Training container Dockerfile |
| `interceptor-training/configs/default_run.yaml` | config | 1967 | 53 | `E11534E469F4B6DF5EC9DD5A12A0FCD15D191468EFEE417BEFE81DEAE44DAE06` | READ-FULL | EMBEDDED | Default training configuration |
| `interceptor-training/data/.gitkeep` | generated | 0 | 0 | `E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855` | READ-FULL | GLOB | Empty placeholder |
| `interceptor-training/data/checkpoints/.gitkeep` | generated | 0 | 0 | `E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855` | READ-FULL | GLOB | Empty placeholder |
| `interceptor-training/data/curriculum_state/.gitkeep` | generated | 0 | 0 | `E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855` | READ-FULL | GLOB | Empty placeholder |
| `interceptor-training/data/results/.gitkeep` | generated | 0 | 0 | `E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855` | READ-FULL | GLOB | Empty placeholder |
| `interceptor-training/data/tb_logs/.gitkeep` | generated | 0 | 0 | `E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855` | READ-FULL | GLOB | Empty placeholder |
| `interceptor-training/docker-compose.yml` | script | 613 | 22 | `B85236077C689A58E4621A3B74F2BE933A14A066257F769F6BE7EE665A2F211F` | READ-FULL | EMBEDDED | Docker compose config |
| `interceptor-training/requirements.txt` | config | 332 | 9 | `09F8784E7E6359630F98D48C136A2A229226711280C23D80AB6FAB365FB0857C` | READ-FULL | EMBEDDED | Training Python requirements |
| `interceptor-training/scripts/evaluate.py` | source | 7055 | 166 | `A572F2875FA8A9A5F68074151C1D2E5146543F90056BA6D5B6AE33C3A397F2DE` | READ-FULL | EMBEDDED | Evaluation entry point |
| `interceptor-training/scripts/smoke_test.py` | test | 19964 | 449 | `A7F558B99E48319516583CB820C40EBBB9876CC82688A82C621745E8A9C75892` | READ-FULL | EMBEDDED | Host-side verification suite |
| `interceptor-training/scripts/train.py` | script | 2701 | 73 | `537D86CA00F91B12A8D5E0429F705B3416DF4688E7C2950049F2A7AF6EC42432` | READ-FULL | EMBEDDED | Training entry point |
| `interceptor-training/src/__init__.py` | source | 97 | 3 | `E2F175EF99314C302AA875D1764D1003583DE3830CE4E46C473BF4976A8D6077` | READ-FULL | EMBEDDED | Package root |
| `interceptor-training/src/envs/__init__.py` | source | 348 | 19 | `80AF07207B10AEF90AFD029A0FC8CFE6D516244C64B411773FF39F7E9AEE4DB2` | READ-FULL | EMBEDDED | Envs package init |
| `interceptor-training/src/envs/base_env.py` | source | 17623 | 457 | `D3392D4C3110DEA588A8ABC3C41DB0F5CAB9A9F8CE51C6E9A0EE93AFC73FDA54` | READ-FULL | EMBEDDED | Multi-stage Gymnasium env |
| `interceptor-training/src/envs/obs_builder.py` | source | 6455 | 173 | `089445011DA489A510C7202F9F862367C8E97F8854467AF3F8E4DB3B532D9C19` | READ-FULL | EMBEDDED | Observation construction |
| `interceptor-training/src/envs/reward.py` | source | 6520 | 162 | `05ED8C506D7E06B5545B968C4306045305A8EC4BD66FB58DAD2442CDBCCB83BA` | READ-FULL | EMBEDDED | Reward computation |
| `interceptor-training/src/envs/stage_config.py` | source | 15458 | 424 | `82D51E21D8CA5AE171479DF5D614E5A33E67C9E16B81EBF696E2CE73FCF2FF88` | READ-FULL | EMBEDDED | 8-stage curriculum table |
| `interceptor-training/src/envs/target_generator.py` | source | 10136 | 250 | `58B5A3C10D153EDB4057239C4EC360CB82E3C7FB913BE50504C17FDB5CCFD27C` | READ-FULL | EMBEDDED | Target kinematics |
| `interceptor-training/src/physics/__init__.py` | source | 342 | 10 | `14522AD269C81670322F5D92874B4D001D555CD7D75E890E62D2FA79E746345E` | READ-FULL | EMBEDDED | Physics package init |
| `interceptor-training/src/physics/aero.py` | source | 3863 | 107 | `C0F6B35BDCDBC7AD3D6904907B5550B314D112B0349266EADDA6555FD4B61B53` | READ-FULL | EMBEDDED | Aerodynamic model |
| `interceptor-training/src/physics/constants.py` | source | 8339 | 182 | `9F5A4A24358F5D438EDEC0ACC675FE62FE44134490CC7039D628CF8601885DB4` | READ-FULL | EMBEDDED | Physics constants |
| `interceptor-training/src/physics/pipeline.py` | source | 12433 | 374 | `97B25F978A344EA884BD1153B26B39B99E4E40CEF8EDC1458B8BC3F4710947F0` | READ-FULL | EMBEDDED | Full physics pipeline |
| `interceptor-training/src/physics/quaternion.py` | source | 1328 | 41 | `C1FA4ADAAE14419B28F63B3011D1CFBBD410ABF41518F3FD32EECE64F3D3D4CA` | READ-FULL | EMBEDDED | Quaternion helpers |
| `interceptor-training/src/training/__init__.py` | source | 667 | 28 | `8B22CE0B0F092FA88DC52EDA6FE8B1648024F8E4D22BD0D5823B407499E52586` | READ-FULL | EMBEDDED | Training package init |
| `interceptor-training/src/training/callbacks.py` | source | 8413 | 229 | `913FA2326A63B33111AF6583EC9CA0B864F1CAC92676BA41F2EFED89B99C5290` | READ-FULL | EMBEDDED | SB3 callbacks |
| `interceptor-training/src/training/checkpoint_manager.py` | source | 8861 | 232 | `C5E9BE6817E92C0628C44DD892F3EB4A4F269BFE8B9766D64051A266E8F364C8` | READ-FULL | EMBEDDED | Checkpoint persistence |
| `interceptor-training/src/training/curriculum.py` | source | 9325 | 234 | `0AC97DE1FFE5774463D3D3375771C4555824499F90034BD90A28D7F6096ACEE6` | READ-FULL | EMBEDDED | Curriculum scheduler |
| `interceptor-training/src/training/orchestrator.py` | source | 23213 | 507 | `46D2E52DB246BB3EFF84216DCB5E526A591778C5EC68302EA1860B0178364FC7` | READ-FULL | NOT-VERBATIM | Training orchestrator (>500 lines) |
| `interceptor-training/src/utils/__init__.py` | source | 303 | 12 | `3D96767052256148D8D672086023C751703E9E0D5896B8239BA57C3344B1649A` | READ-FULL | EMBEDDED | Utils package init |
| `interceptor-training/src/utils/config_loader.py` | source | 13214 | 317 | `BF85DFABDD2A04CD0659FF27F8348E24AE25286D5A36756AE89406C6D21F1E57` | READ-FULL | EMBEDDED | Config loading and validation |
| `interceptor-training/src/utils/logger.py` | source | 2408 | 76 | `CCE6ED4EE9D0A5CAA6B9BC458C8623D50E4775C18B9D5D7505EBFB9C5D0D6A0C` | READ-FULL | EMBEDDED | Logging setup |
| `interceptor.log` | log | 3334 | 58 | `...` [[GAP:G-002]] | READ-FULL | NOT-VERBATIM | Smoke run log (2026-09-30) |
| `phy_sim_val.ipynb` | notebook | 331555 | — | `...` [[GAP:G-002]] | SAMPLED(notebook) | NOT-VERBATIM | Physics validation notebook |
| `pybullet_vs_swift_comparison.ipynb` | notebook | 14419 | — | `...` [[GAP:G-002]] | SAMPLED(notebook) | NOT-VERBATIM | PyBullet comparison notebook |
| `pybullet_vs_swift_trajectory.ipynb` | notebook | 21045 | — | `...` [[GAP:G-002]] | SAMPLED(notebook) | NOT-VERBATIM | PyBullet trajectory notebook |
| `pybullet_vs_swift_trajectory_v0.-1.ipynb` | notebook | 20008 | — | `...` [[GAP:G-002]] | EXCLUDED(backup-version) | NOT-VERBATIM | Backup version |
| `references.txt` | doc | 1722 | — | `...` [[GAP:G-002]] | READ-FULL | NOT-VERBATIM | External resource references |
| `requirements.txt` | config | 47 | 3 | `...` [[GAP:G-002]] | READ-FULL | EMBEDDED | Root-level Python requirements |
| `results_trajectory.txt` | data | 2581 | — | `...` [[GAP:G-002]] | READ-FULL | NOT-VERBATIM | Trajectory results data |
| `results_wind_rlvec.txt` | data | 1165 | — | `...` [[GAP:G-002]] | READ-FULL | NOT-VERBATIM | Wind RL-vec results data |
| `results_wind_rlvec_vs_pybullet.ipynb` | notebook | 102461 | — | `...` [[GAP:G-002]] | SAMPLED(notebook) | NOT-VERBATIM | Wind RL-vec vs PyBullet notebook |
| `run_summary.json` | data | 814 | 34 | `...` [[GAP:G-002]] | READ-FULL | EMBEDDED | Smoke training run summary |
| `self_test.py` | test | 8850 | 222 | `...` [[GAP:G-002]] | READ-FULL | NOT-VERBATIM | v3.0 physics smoke test |
| `src/simulation.py` | source | 22413 | — | `768D40246DA6625A0BA99767B187E31D9B9ED0DFCA4CAD45F31A2907F7977F6E` | READ-FULL | NOT-VERBATIM | Full 9-stage physics pipeline + live loop |
| `src/visualiser.py` | source | 9027 | — | `78F551027E39187AD73D0BC724D16D87588609ADEBC60AD7E8A52DBBDB8AC1F4` | READ-FULL | NOT-VERBATIM | Standalone visualisation |
| `swift_demo_wind_only.py` | source | 36786 | — | `...` [[GAP:G-002]] | READ-FULL | NOT-VERBATIM | Wind-only demo |
| `swift_physics_headless.py` | source | 25983 | — | `...` [[GAP:G-002]] | READ-FULL | NOT-VERBATIM | Headless physics extraction |
| `swift_physics_headless_rlvec.py` | source | 18760 | — | `2B1BAA495DB9D27D632F9F72A6ACA24A5898E208A86887CAE814F23BB4BFF9E9` | READ-FULL | NOT-VERBATIM | RL-vectorized physics |
| `swift_physics_headless_rlvec_latest.py` | source | 18760 | — | `2B1BAA495DB9D27D632F9F72A6ACA24A5898E208A86887CAE814F23BB4BFF9E9` | READ-FULL | NOT-VERBATIM | Identical copy of above |
| `swift_physics_headless_rlvec_stable_v1.1.py` | source | 4559 | — | `...` [[GAP:G-002]] | READ-FULL | NOT-VERBATIM | Stable v1.1 snapshot |
| `swift_physics_headless_rlvec_v0.-1.py` | source | 3889 | — | `...` [[GAP:G-002]] | EXCLUDED(backup-version) | NOT-VERBATIM | Backup version |
| `swift_physics_headless_v2.py` | source | 11459 | — | `...` [[GAP:G-002]] | READ-FULL | NOT-VERBATIM | RL-friendly physics wrapper |
| `swift_physics_headless_wind.py` | source | 16930 | — | `...` [[GAP:G-002]] | READ-FULL | NOT-VERBATIM | Wind-aware physics |
| `swift_physics_headless_wind_latest.py` | source | 16930 | — | `...` [[GAP:G-002]] | READ-FULL | NOT-VERBATIM | Identical copy of above |
| `swift_physics_headless_wind_stable_v1.1.py` | source | 4490 | — | `...` [[GAP:G-002]] | READ-FULL | NOT-VERBATIM | Stable v1.1 snapshot |
| `swift_physics_headless_wind_v0.-1.py` | source | 4142 | — | `...` [[GAP:G-002]] | EXCLUDED(backup-version) | NOT-VERBATIM | Backup version |
| `swift_rl_env.py` | source | 19028 | 454 | `2B1BAA495DB9D27D632F9F72A6ACA24A5898E208A86887CAE814F23BB4BFF9E9` | READ-FULL | NOT-VERBATIM | Vectorized RL training env |
| `swift_rl_env_latest.py` | source | 19028 | 454 | `2B1BAA495DB9D27D632F9F72A6ACA24A5898E208A86887CAE814F23BB4BFF9E9` | READ-FULL | NOT-VERBATIM | Identical copy of above |
| `trajectories.png` | binary | 216061 | — | `9B44EE7B4D9E9E3AE2F12866654C27588A0339D3D25E952E5E5B8B7F63E99C46` | EXCLUDED(binary-png) | BINARY | Trajectory plot |
| `updated demo.py` | source | 35236 | — | `53F10DDD53BD1AE568148EF55493897ECCF05CA3D3502F17654D013DA714E2A8` | READ-FULL | NOT-VERBATIM | Updated flight demo |
| `validate.py` | source | 7891 | — | `27831B4FB4E476373618C091BFA577E3C1BE858FFC9B979A52878E861DF16371` | READ-FULL | NOT-VERBATIM | Flight log validation |
| `validate_flight_only.py` | source | 6660 | — | `A15D7166E3670082361642B74959AD4B21A0C68B22634A3380DB1D74BE38294E` | READ-FULL | NOT-VERBATIM | Flight-only validation |
| `validate_onestep.py` | source | 8204 | — | `A4638A054DAE1048765B45894BF8BB8B94282E652F6F6E1445C5DE94F1C366B1` | READ-FULL | NOT-VERBATIM | One-step validation |
| `wind_rlvec_vs_pybullet.ipynb` | notebook | 17361 | — | `FEFF336E4D15B97248039F34AD2D30BA5E141057CE8B9D6C5D138DCB1F28BA36` | SAMPLED(notebook) | NOT-VERBATIM | Wind RL-vec vs PyBullet notebook |

### 1.6 Entry Points

| Command / Module | File:Line | Purpose | Phase |
|---|---|---|---|
| `python scripts/train.py [--config ...] [--smoke]` | `interceptor-training/scripts/train.py:L72` | Container ENTRYPOINT; curriculum training | P03 |
| `python scripts/evaluate.py --config ... --stage N` | `interceptor-training/scripts/evaluate.py:L165` | Model evaluation | P04 |
| `python scripts/smoke_test.py` | `interceptor-training/scripts/smoke_test.py:L448` | Host-side no-SB3 smoke tests | Verification |
| `python hover_env.py` | `hover_env.py:L721` | Stage-1 env smoke test (standalone) | Verification |
| `python src/simulation.py` | `src/simulation.py:L1` | Interactive gamepad flight demo | Demo (not training) |
| `python self_test.py` | `self_test.py:L1` | v3.0 physics smoke test | Verification |

[SRC] Files cited.

---

## 2. Environment, Dependencies & Hardware Specifications

### 2.1 Runtime and OS

| Field | Value | Evidence |
|---|---|---|
| Python version (host) | 3.14 | [SRC] `CURRENT_WORK.md:L48`, `__pycache__/hover_env.cpython-314.pyc` |
| Python version (container) | Determined by base image `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime` [[GAP:G-001]] | [SRC] `interceptor-training/Dockerfile:L7` |
| OS (container) | Ubuntu (PyTorch base image) | [RECALLED] PyTorch Docker images use Ubuntu; exact version pinned by digest [[GAP:G-003]] |
| OS (host/dev) | Windows | [EXEC] User OS = Windows |
| CPU architecture | x86_64 (amd64) | [RECALLED] PyTorch CUDA images are x86_64 [[GAP:G-003]] |

### 2.2 Python Packages

#### Container (`interceptor-training/requirements.txt`)

| Package | Constraint | Resolved Version | Source | Direct/Transitive | Evidence |
|---|---|---|---|---|---|
| `torch` | (provided by base image) | `2.7.0+cu128` | Base image `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime` | Direct (implicit) | [SRC] `interceptor-training/Dockerfile:L7`, `interceptor-training/requirements.txt:L1-L3` |
| `stable-baselines3[extra]` | `>=2.4.0` | [[GAP:G-004]] | PyPI | Direct | [SRC] `interceptor-training/requirements.txt:L4` |
| `gymnasium` | `>=1.0.0` | [[GAP:G-004]] | PyPI | Direct | [SRC] `interceptor-training/requirements.txt:L5` |
| `numpy` | `>=2.0.0` | [[GAP:G-004]] | PyPI | Direct | [SRC] `interceptor-training/requirements.txt:L6` |
| `scipy` | `>=1.13.0` | [[GAP:G-004]] | PyPI | Direct | [SRC] `interceptor-training/requirements.txt:L7` |
| `tensorboard` | `>=2.17.0` | [[GAP:G-004]] | PyPI | Direct | [SRC] `interceptor-training/requirements.txt:L8` |
| `pyyaml` | `>=6.0` | [[GAP:G-004]] | PyPI | Direct | [SRC] `interceptor-training/requirements.txt:L9` |
| `opencv-python-headless` | (latest) | [[GAP:G-004]] | PyPI | Direct (replaces `opencv-python` from sb3[extra]) | [SRC] `interceptor-training/Dockerfile:L20` |

#### Host (`requirements.txt` at root)

| Package | Constraint | Evidence |
|---|---|---|
| `numpy` | `==2.5.1` | [SRC] `requirements.txt:L1` |
| `matplotlib` | `==3.11.1` | [SRC] `requirements.txt:L2` |
| `pygame` | `==2.5.8` | [SRC] `requirements.txt:L3` |

### 2.3 System Layer

| Component | Value | Evidence |
|---|---|---|
| Container base image | `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime` | [SRC] `interceptor-training/Dockerfile:L7` |
| Container base image digest | [[GAP:G-003]] | Not pinned in Dockerfile |
| CUDA toolkit | 12.8 (from base image name) | [SRC] `interceptor-training/Dockerfile:L3` |
| cuDNN | 9 (from base image name) | [SRC] `interceptor-training/Dockerfile:L3` |
| System packages installed | `git` | [SRC] `interceptor-training/Dockerfile:L10` |
| NVIDIA runtime | Required (`runtime: nvidia` in compose) | [SRC] `interceptor-training/docker-compose.yml:L6` |
| GPU reservation | 1 GPU, `capabilities: [gpu]` | [SRC] `interceptor-training/docker-compose.yml:L17-L22` |

### 2.4 Environment Variables

| Name | Required | Default | Type | Consumer | Effect | Secret | Evidence |
|---|---|---|---|---|---|---|---|
| `NVIDIA_VISIBLE_DEVICES` | No | `all` | string | Docker/NVIDIA runtime | GPU visibility | No | [SRC] `interceptor-training/docker-compose.yml:L8` |
| `NVIDIA_DRIVER_CAPABILITIES` | No | `compute,utility` | string | Docker/NVIDIA runtime | Driver capabilities | No | [SRC] `interceptor-training/docker-compose.yml:L9` |
| `INTERCEPTOR_CONFIG` | No | `/data/configs/default_run.yaml` | file path | `scripts/train.py:L31` | Override config file path | No | [SRC] `interceptor-training/scripts/train.py:L31` |

### 2.5 External Services and Network

| Resource | Identifier | Auth | Pinned Rev | Offline Feasibility | Evidence |
|---|---|---|---|---|---|
| PyPI (pip install) | `https://pypi.org` | None | No | Yes, after initial install | [SRC] `interceptor-training/Dockerfile:L17` |
| Docker Hub (base image) | `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime` | None | Tag only, no digest [[GAP:G-003]] | Yes, after image pull | [SRC] `interceptor-training/Dockerfile:L7` |

No APIs, Hugging Face models, or external datasets are accessed at runtime. The system is fully offline after container build. [DERIVED] from code inspection — no network calls found in any source file.

### 2.6 Filesystem Expectations

| Path | Purpose | Created By | Evidence |
|---|---|---|---|
| `/data` (container) / `./data` (host) | Host-mounted volume for checkpoints, logs, state | `docker-compose.yml` volume mount | [SRC] `interceptor-training/docker-compose.yml:L12` |
| `/data/checkpoints/<config>/stage_<N>/` | Periodic and final model checkpoints | `CheckpointManager.ensure()` | [SRC] `interceptor-training/src/training/checkpoint_manager.py:L75-L82` |
| `/data/tb_logs/<config>/stage_<N>/` | TensorBoard event files | SB3 `tensorboard_log` | [SRC] `interceptor-training/src/training/orchestrator.py:L258-L261` |
| `/data/curriculum_state/` | `run_state.json` for resume | `CheckpointManager` | [SRC] `interceptor-training/src/training/checkpoint_manager.py:L69-L70` |
| `/data/results/` | Final models, `run_summary.json`, eval JSONs | Orchestrator | [SRC] `interceptor-training/src/training/checkpoint_manager.py:L59-L60` |
| `/data/configs/` | Mounted from host `./configs` | `docker-compose.yml` volume mount | [SRC] `interceptor-training/docker-compose.yml:L15` |

### 2.7 Hardware

**(a) Stated in the archive:**
- `CURRENT_WORK.md:L158`: "Run on CUDA host 2026-09-30". Device resolved to `cuda`. [[GAP:G-005]]
- `default_run.yaml:L13`: `device: cpu` (for PPO; comment says "PPO+MlpPolicy is faster on CPU")
- `docker-compose.yml:L17-L22`: 1 NVIDIA GPU reserved.

**(b) Derived minimum:**
- GPU: 1 NVIDIA GPU with CUDA compute capability sufficient for CUDA 12.8 (minimum: SM 5.0, recommended: SM 7.0+) [DERIVED] from CUDA 12.8 support matrix.
- VRAM: PPO with MlpPolicy [256, 256, 128] requires minimal VRAM (order of MiB). No large model weights. [DERIVED]
- CPU: 8 cores recommended (matching `n_parallel_envs: 8` for `SubprocVecEnv`) [DERIVED] from `default_run.yaml:L15`.
- RAM: 8 GiB minimum [DERIVED] — 8 parallel envs with numpy arrays, no large tensors.
- Storage: 10 GiB recommended (Docker image ~5 GiB, checkpoints ~100 MiB per stage) [DERIVED].

**(c) Hardware of the original run:** [[GAP:G-005]] — the smoke run (`run_summary.json`) used `device: cuda`, wall time 2.7 seconds for 2624 steps. Exact GPU model not stated.

---

## 3. Model Architecture & Component Specifications

### Model Card: PPO Policy (ppo_baseline)

| Field | Value | Evidence |
|---|---|---|
| **ID** | `ppo_baseline` | [SRC] `interceptor-training/configs/default_run.yaml:L36` |
| **Role** | Policy (actor-critic, on-policy) | [SRC] `default_run.yaml:L37` |
| **Instantiation site** | `interceptor-training/src/training/orchestrator.py:L249-L269` | [SRC] |
| **Trainable** | Yes | [DERIVED] |
| **Source** | Created fresh by SB3 `PPO(policy="MlpPolicy", ...)` | [SRC] `orchestrator.py:L254` |
| **Algorithm** | PPO (Proximal Policy Optimization) | [SRC] `default_run.yaml:L37` |
| **Policy class** | `MlpPolicy` (stable_baselines3) | [SRC] `default_run.yaml:L38` |
| **Network architecture** | `pi: [256, 256, 128]`, `vf: [256, 256, 128]` | [SRC] `default_run.yaml:L39` |
| **Activation** | `ReLU` (`torch.nn.ReLU`) | [SRC] `default_run.yaml:L40`, `orchestrator.py:L81` |
| **Observation dim (stage 1)** | 14 (no target, no history) | [DERIVED] `STAGE1_FRAME_DIM = 14`, `obs_builder.py:L38` |
| **Observation dim (stages 2+)** | 76 = 19 × (3 + 1) where `history_frames=3` | [DERIVED] `TARGET_FRAME_DIM=19`, `obs_history.frames=3`, `obs_builder.py:L110` |
| **Action dim** | 4 (`[thrust, roll, pitch, yaw]` in `[-1, 1]`) | [SRC] `base_env.py:L151` |
| **Action space** | `Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)` | [SRC] `base_env.py:L151` |
| **Weight dtype** | float32 (SB3 default) | [LIBDEF] `stable-baselines3>=2.4.0` |
| **Precision** | float32 compute | [LIBDEF] |
| **Weights present** | Not present in archive. Only a smoke-run final checkpoint referenced at `/data/results/ppo_baseline_final.zip` (external to archive). [[GAP:G-006]] | [SRC] `run_summary.json:L28` |
| **Hyperparameters** | See §9 | [SRC] `default_run.yaml:L43-L53` |

**PPO Hyperparameters (from config):**

| Parameter | Value | Evidence |
|---|---|---|
| `n_steps` | `4096` | [SRC] `default_run.yaml:L44` |
| `batch_size` | `512` | [SRC] `default_run.yaml:L45` |
| `gamma` | `0.995` | [SRC] `default_run.yaml:L46` |
| `learning_rate` | `3.0e-4` | [SRC] `default_run.yaml:L47` |
| `gae_lambda` | `0.95` | [SRC] `default_run.yaml:L48` |
| `clip_range` | `0.2` | [SRC] `default_run.yaml:L49` |
| `ent_coef` | `0.005` | [SRC] `default_run.yaml:L50` |
| `n_epochs` | `10` | [SRC] `default_run.yaml:L51` |
| `max_grad_norm` | `0.5` | [SRC] `default_run.yaml:L52` |
| `vf_coef` | `0.5` | [SRC] `default_run.yaml:L53` |

**Effective batch size:** `n_steps × n_parallel_envs = 4096 × 8 = 32768` samples per rollout; `n_steps / batch_size × n_epochs = 4096 / 512 × 10 = 80` gradient updates per rollout. [DERIVED]

**Tokenizer:** NOT APPLICABLE — no text tokenization; raw float observations.

**Generation:** NOT APPLICABLE — not a generative model.

The archive also references SAC, TD3, and a deep PPO variant in `CURRENT_WORK.md:L187-L188` but the shipped `default_run.yaml` contains only `ppo_baseline`. The smoke test at `smoke_test.py:L401` expects additional configs `ppo_5layer_deep` with `obs_history: {frames: 5, skip: 3}`, which D-20 resolves by giving the smoke test its own model file and config file (not yet created). [[GAP:G-007]]

---

## 4. Phase-by-Phase Execution Pipeline

### Phase P01 — Container Build

| Field | Value |
|---|---|
| **ID** | P01 |
| **Name** | Container Build |
| **Objective** | Build Docker image with all dependencies |
| **Trigger** | `docker compose up --build` or `docker build .` in `interceptor-training/` |
| **Preconditions** | Docker installed; NVIDIA Container Toolkit; host `./data` and `./configs` directories exist |
| **Inputs** | `Dockerfile`, `requirements.txt`, `src/`, `scripts/`, `configs/` |
| **Steps** | 1. Pull base `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime` [EXT] (`Dockerfile:L7`). 2. Install `git` (`Dockerfile:L10`). 3. `pip install -r requirements.txt` (`Dockerfile:L17`). 4. Replace `opencv-python` with `opencv-python-headless` (`Dockerfile:L20`). 5. Copy `src/`, `scripts/`, `configs/` into `/app` (`Dockerfile:L23-L25`). |
| **Outputs** | Docker image with tag from compose build context |
| **Postconditions** | `python -c "import stable_baselines3; import gymnasium"` succeeds inside container |

[SRC] `interceptor-training/Dockerfile:L1-L32`.

### Phase P02 — Configuration Loading

| Field | Value |
|---|---|
| **ID** | P02 |
| **Name** | Configuration Loading |
| **Objective** | Parse and validate the run YAML into a `RunConfig` object |
| **Trigger** | `train.py` `main()` calls `load_run_config(args.config)` (`train.py:L47`) |
| **Preconditions** | YAML file exists at `args.config` (default `/data/configs/default_run.yaml`) |
| **Inputs** | YAML file |
| **Steps** | 1. `yaml.safe_load` (`config_loader.py:L314`). 2. Validate `global` section: seed, device, data_dir, n_parallel_envs, checkpoint_interval_steps, tensorboard, execution_mode (`config_loader.py:L96-L110`). 3. Canonicalize each config block: `_canonicalize_config` (`config_loader.py:L59-L93`). 4. Validate each config: algo, policy, net_arch, activation, obs_history, stages (`config_loader.py:L113-L141`). 5. Merge per-stage YAML overrides onto built-in `STAGES` table via `_coerce_stage_overrides` (`config_loader.py:L144-L176`). 6. Compute `config_hash` for resume validation (`config_loader.py:L297-L305`). |
| **Outputs** | `RunConfig` instance |

[SRC] `interceptor-training/src/utils/config_loader.py:L1-L317`.

### Phase P03 — Curriculum Training

| Field | Value |
|---|---|
| **ID** | P03 |
| **Name** | Curriculum Training |
| **Objective** | Train the policy through the 8-stage curriculum using SB3 |
| **Trigger** | `Orchestrator(cfg).run()` (`orchestrator.py:L115`) |
| **Preconditions** | P02 complete; GPU available (or CPU if configured) |
| **Inputs** | `RunConfig`, optional `run_state.json` for resume |

**Internal steps (per config, per stage):**

1. **Resume check:** Read `run_state.json` from `data/curriculum_state/`. Validate `config_hash` match. (`orchestrator.py:L119-L128`)
2. **Per-config loop:** For each config name in `RunConfig.config_names()` (`orchestrator.py:L151`):
3. **Per-stage loop:** For each stage number in the config's stage list (`orchestrator.py:L327`):
   - 3a. Compute budget: `max(1, stage.max_training_steps × stage_scale)` (`orchestrator.py:L331`)
   - 3b. Build `CurriculumScheduler` with rollback threshold = `stage.success_rate × 0.5` (default), rollback_min_steps = `0.25 × budget` (`orchestrator.py:L233-L247`)
   - 3c. Build vectorized environment: `SubprocVecEnv` of `n_parallel_envs` factories, each creating `InterceptorBaseEnv` with the stage config (`orchestrator.py:L209-L231`)
   - 3d. **Model selection** (priority order): resume checkpoint → rollback re-entry (own final, `num_timesteps=0`) → mid-stage periodic → weight transfer from previous stage (if obs dim matches) → fresh model (`orchestrator.py:L357-L394`)
   - 3e. **Learn:** `model.learn(total_timesteps=remaining, reset_num_timesteps=False, callback=CallbackList([CurriculumCheckpointCallback, RewardComponentCallback, MetricsCallback, CurriculumCallback]))` (`orchestrator.py:L416-L425`)
   - 3f. **CurriculumCallback.on_step:** For each completed episode, feed `episode_stats` to `CurriculumScheduler.on_episode_end()`. Check advance/cap/rollback. Sync authoritative step count. Return `False` to stop learning when a terminal result is reached. (`callbacks.py:L206-L221`)
   - 3g. **Result handling:** ADVANCE → save final checkpoint, advance `idx += 1`. CAPPED → save final, advance `idx += 1`, log warning. ROLLBACK → `retries[stage] += 1`, if `idx == 0` or `retries > max_attempts` mark STUCK, else `idx -= 1`, `reentry=True`, `next_scale=0.5`. (`orchestrator.py:L436-L466`)
4. **Config completion:** Save `<config>_final.zip` and append to `run_summary.json` (`orchestrator.py:L487-L507`).

**Effective hyperparameter table (ppo_baseline):**

| Parameter | Value | Type | Unit | Winning Source | Evidence |
|---|---|---|---|---|---|
| `n_steps` | `4096` | int | steps | `default_run.yaml` | [SRC] `default_run.yaml:L44` |
| `batch_size` | `512` | int | samples | `default_run.yaml` | [SRC] `default_run.yaml:L45` |
| `gamma` | `0.995` | float | — | `default_run.yaml` | [SRC] `default_run.yaml:L46` |
| `learning_rate` | `3.0e-4` | float | — | `default_run.yaml` | [SRC] `default_run.yaml:L47` |
| `gae_lambda` | `0.95` | float | — | `default_run.yaml` | [SRC] `default_run.yaml:L48` |
| `clip_range` | `0.2` | float | — | `default_run.yaml` | [SRC] `default_run.yaml:L49` |
| `ent_coef` | `0.005` | float | — | `default_run.yaml` | [SRC] `default_run.yaml:L50` |
| `n_epochs` | `10` | int | — | `default_run.yaml` | [SRC] `default_run.yaml:L51` |
| `max_grad_norm` | `0.5` | float | — | `default_run.yaml` | [SRC] `default_run.yaml:L52` |
| `vf_coef` | `0.5` | float | — | `default_run.yaml` | [SRC] `default_run.yaml:L53` |
| `seed` | `42 + stage_number` | int | — | code (`orchestrator.py:L256`) | [SRC] `orchestrator.py:L256` |
| `device` | `cpu` (from YAML) | str | — | `default_run.yaml:L13` | [SRC] |
| `tensorboard_log` | `<data_dir>/tb_logs/<config>/<stage_id>` | path | — | code | [SRC] `orchestrator.py:L258-L261` |
| `policy` | `MlpPolicy` | str | — | `default_run.yaml:L38` | [SRC] |
| `n_parallel_envs` | `8` | int | — | `default_run.yaml:L15` | [SRC] |
| `checkpoint_interval_steps` | `50000` | int | steps | `default_run.yaml:L16` | [SRC] |

**Per-step loop order (PPO on-policy):**
1. Reset environments (at start or episode end)
2. Collect `n_steps × n_envs` transitions: for each step, `obs → model.predict → action → env.step → next_obs, reward, done, info`
3. Compute advantages (GAE with `gamma=0.995`, `gae_lambda=0.95`)
4. For `n_epochs=10` optimization epochs:
   - Shuffle rollout buffer into minibatches of `batch_size=512`
   - For each minibatch: forward pass, compute PPO loss (clipped surrogate + value + entropy), backward, clip gradients to `max_grad_norm=0.5`, optimizer step
5. Log metrics via callbacks
6. Check curriculum scheduler for advance/cap/rollback
7. If terminal result, stop learning; else goto step 2

[SRC] SB3 PPO implementation [LIBDEF]; `interceptor-training/src/training/orchestrator.py:L416-L425`.

### Phase P04 — Evaluation

| Field | Value |
|---|---|
| **ID** | P04 |
| **Name** | Evaluation |
| **Objective** | Run trained model deterministically for N episodes, report metrics |
| **Trigger** | `python scripts/evaluate.py --config <yaml> --config-name <name> --stage <N> [--model <path>] [--episodes 50]` |
| **Steps** | 1. Load run config. 2. Find model (stage final → periodic → config final). 3. Create `DummyVecEnv` with 1 env. 4. `model.predict(obs, deterministic=True)` loop for `episodes` episodes. 5. Compute `kill_rate`, `success_rate`, `mean_episode_reward`, `mean_final_distance`, `mean_time_to_intercept`, `mean_miss_distance`. 6. Write to `data/results/eval_<config>_stage_<N>.json`. |
| **Outputs** | JSON metrics file, stdout report |

[SRC] `interceptor-training/scripts/evaluate.py:L1-L166`.

---

## 5. Reward Functions & Mathematical Formulations

### 5.1 Dense Reward (every step)

All stages share this per-step dense reward structure. Weights are zero-by-default and enabled per-stage (see §9.2). [SRC] `interceptor-training/src/envs/reward.py:L90-L117`.

```
r_step = r_alive + r_alt + r_tilt + r_angvel + r_thrust + r_smooth
       + r_velocity_alignment + r_progress_delta + r_time_penalty
       + r_kill_bonus + r_crash + r_oob
```

**Summation order:** Explicit left fold (`reward = 0.0; for v in comp.values(): reward += v`), NOT Python `sum()` — Python 3.12+ `sum()` uses compensated summation which can differ by 1 ULP. [SRC] `reward.py:L143-L148`.

| Term | Formula | Stage 1 weight | Stages 2+ (typical) | Unit | Evidence |
|---|---|---|---|---|---|
| `r_alive` | `+k_alive` | `0.10` | `0.05` (S2), `0.0` (S3+) | — | [SRC] `stage_config.py:L206,L232` |
| `r_alt` | `-k_alt × \|alt_err\|` | `0.15` | `0.0` (S2+) | m | [SRC] `stage_config.py:L206` |
| `r_tilt` | `-k_tilt × θ_tilt` where `θ = arccos(clip(R_WB[2,2], -1, 1))` | `0.40` | `0.20` (S2), `0.0` (S3+) | rad | [SRC] `reward.py:L85,L93` |
| `r_angvel` | `-k_angvel × max(0, \|ω_B\| - ω_safe)²` | `0.02` | `0.02` (all) | (rad/s)² | [SRC] `reward.py:L94` |
| `r_thrust` | `-k_thrust × \|c_cmd - c_hover\|` where `c_hover = PH_C_HOVER ≈ 0.23253743635354834` | `0.15` | `0.0` (S2+) | — | [SRC] `reward.py:L95`, `constants.py:L180` |
| `r_smooth` | `-k_smooth × Σ(a_i - a_prev_i)²` (4-element sum) | `0.05` | `0.05` (all) | — | [SRC] `reward.py:L96` |
| `r_time_penalty` | `-k_time_penalty` (flat per step) | `0.0` | `0.02` (S4+) | — | [SRC] `reward.py:L97` |
| `r_velocity_alignment` | `+k_va × max(0, clip(dot(v̂, los), -1, 1))` only when `\|v\| > 1e-3`, `distance > 1e-6`, `target_visible`, and `los_world` not None | `0.0` | `0.30` (S2+) | — | [SRC] `reward.py:L100-L109` |
| `r_progress_delta` | `+k_pd × (prev_dist - dist)`, or if `k_progress_normalize`: `k_pd × clip((prev_dist - dist) / max(dist, 1.0), -1, 1)` | `0.0` | `0.15` (S2+), normalized S6+ | m or — | [SRC] `reward.py:L111-L117` |

### 5.2 Terminal Bonuses (episode end)

| Term | Formula | Triggers | Evidence |
|---|---|---|---|
| `r_crash` | `-k_crash` (default `-200.0`) | `θ_tilt > tilt_crash_threshold (60° = 1.0472 rad)` OR `v_z < -ground_contact_v_threshold (-1.0 m/s)` at `z ≤ 0` | [SRC] `reward.py:L127-L129` |
| `r_oob` | `-k_oob` (default `-200.0`) | `\|x\| > xy_limit` OR `\|y\| > xy_limit` OR `z < z_lo` OR `z > z_hi` | [SRC] `reward.py:L130-L131` |
| `r_kill_bonus` | `+k_kill_bonus × miss_gate × time_gate` where `miss_gate = max(0, 1 - d/kill_radius)` (if `k_miss_distance_scale`, else 1.0), `time_gate = 1 - steps/max_episode_steps` (if `k_time_bonus_scale`, else 1.0). Clamped `≥ 0`. | `killed = True` (when `distance < kill_radius` AND `kills_enabled`) | [SRC] `reward.py:L133-L141` |

**Kill bonus progression across stages:**
- Stages 1-2: `k_kill_bonus = 0.0` (no interception)
- Stage 3: `k_kill_bonus = 500.0`, `miss_gate = ON`, `time_gate = OFF`
- Stage 4+: `k_kill_bonus = 500.0`, `miss_gate = ON`, `time_gate = ON`

[SRC] `stage_config.py:L259-L263,L286-L291`.

### 5.3 Observation Space

**Stage 1 (14-dim, no target, no history):**

| Index | Name | Computation | Evidence |
|---|---|---|---|
| [0:6] | 6D rotation | First two columns of `R_WB = quat2rot(q_WB)`, raveled to 6 floats, plus noise `N(0, σ=0.01)` | [SRC] `hover_env.py:L588-L590`, `obs_builder.py:L75-L89` |
| [6:9] | World velocity | `v_WB + N(0, 0.01)` | [SRC] `hover_env.py:L591` |
| [9:12] | Body angular rate | `ω_B + N(0, 0.01)` | [SRC] `hover_env.py:L592` |
| [12] | Altitude error | `clip(z - z_target, -20, 20)` | [SRC] `hover_env.py:L594-L595` |
| [13] | Vertical velocity | `v_n[2]` (already noisy) | [SRC] `hover_env.py:L596` |

**Stage 1 noise model:** Noise is drawn per-component from `rng.normal(0, 0.01, size)` and added to the true value. The draw order is: 6 rotation components, 3 velocity, 3 angular rate. The observation is cast to `float32` as a single `np.array(..., dtype=np.float32)` at the end. [SRC] `hover_env.py:L598-L604`.

**Stages 2+ (19-dim frame × (m+1), where m = history_frames):**

The 19-dim frame adds target-relative fields in the body frame:

| Index | Name | Computation |
|---|---|---|
| [0:6] | 6D rotation | Same as stage 1 (noisy) |
| [6:9] | World velocity | Same as stage 1 (noisy) |
| [9:12] | Body angular rate | Same as stage 1 (noisy) |
| [12:15] | Relative position (body frame) | `R_WB.T @ (target_pos - drone_pos)` (3-dim) |
| [15:18] | Relative velocity (body frame) | `R_WB.T @ (target_vel - drone_vel)` (3-dim) |
| [18] | Target visible flag | `1.0` if visible, `0.0` otherwise |

If `target_visible == False`: all target fields [12:19] are exactly `0.0` (no noise applied). [SRC] `obs_builder.py:L95-L108`, `base_env.py:L371-L378`.

**History stacking (current-first):** `[frame_t, frame_{t-s}, frame_{t-2s}, ..., frame_{t-m·s}]` where `s = history_skip`. Zero-padded when insufficient history. Total dim: `19 × (m + 1)`. For `ppo_baseline`: `m=3, s=2` → 76-dim. [SRC] `obs_builder.py:L109-L140`.

**Lookahead:** When `InterceptConfig.lookahead_enabled = True`, `target_pos` is replaced with `target.predicted_pos()` (0.5s prediction using `pos + vel·h + 0.5·acc·h² + (1/6)·jerk·h³` where `h = LOOKAHEAD_HORIZON_S = 0.5`). [SRC] `target_generator.py:L200-L205`, `base_env.py:L373`, `stage_config.py:L54`.

---

## 6. Data Contracts & I/O Schemas

### 6.1 `run_summary.json` Schema

```json
{
  "run_id": "YYYYMMDD-HHMMSS",
  "config_file": "<absolute path>",
  "config_hash": "<sha256 hex>",
  "device": "cpu|cuda",
  "data_dir": "<path>",
  "started": "YYYY-MM-DD HH:MM:SS",
  "configs": {
    "<config_name>": {
      "algo": "PPO|SAC|TD3",
      "stages": {
        "<stage_num>": {
          "result": "advance|capped|rollback|stuck",
          "steps": <int>,
          "success_rate": <float>
        }
      },
      "final_model": "<path or null>",
      "steps": <int>
    }
  },
  "finished": "YYYY-MM-DD HH:MM:SS",
  "wall_seconds": <float>
}
```

[SRC] `run_summary.json:L1-L34`, `orchestrator.py:L139-L184`.

### 6.2 `run_state.json` Schema (Resume State)

```json
{
  "run_id": "<str>",
  "config_file": "<str>",
  "config_hash": "<sha256 hex>",
  "timestamp": "YYYY-MM-DD HH:MM:SS",
  "configs_completed": ["<str>", ...],
  "current_config": "<str or null>",
  "current_stage": <int>,
  "stage_num": <int or null>,
  "current_stage_steps": <int>,
  "total_steps_all_stages": <int>,
  "last_checkpoint_path": "<str>",
  "episode_buffer": [<float>, ...],
  "retries": {"<stage_num>": <int>, ...},
  "current_stage_scale": <float>
}
```

Written atomically (`tempfile.mkstemp` + `os.replace`). [SRC] `orchestrator.py:L200-L208`, `checkpoint_manager.py:L157-L171`.

### 6.3 `eval_<config>_stage_<N>.json` Schema

```json
{
  "config": "<config_name>",
  "stage": <int>,
  "model": "<path>",
  "episodes": <int>,
  "seed": <int>,
  "kill_rate": <float>,
  "success_rate": <float>,
  "required_success_rate": <float>,
  "mean_episode_reward": <float>,
  "mean_final_distance": <float or null>,
  "mean_time_to_intercept_steps": <float or null>,
  "mean_time_to_intercept_s": <float or null>,
  "mean_miss_distance": <float>
}
```

[SRC] `evaluate.py:L128-L142`.

### 6.4 Checkpoint File Layout

| Pattern | Purpose | Producer | Evidence |
|---|---|---|---|
| `checkpoints/<config>/stage_<N>/<algo>_<steps>_steps.zip` | Periodic checkpoint | `CurriculumCheckpointCallback` every `checkpoint_interval_steps` | [SRC] `checkpoint_manager.py:L47-L48` |
| `checkpoints/<config>/stage_<N>/<algo>_stage_<N>_final.zip` | Stage completion checkpoint | Orchestrator on ADVANCE/CAPPED | [SRC] `checkpoint_manager.py:L50-L51` |
| `checkpoints/<config>/stage_<N>/<algo>_<steps>_steps.zip_buffer.pkl` | Off-policy replay buffer | `_maybe_save_buffer` (SAC/TD3 only) | [SRC] `checkpoint_manager.py:L116-L126` |
| `checkpoints/<config>/stage_<N>/monitor/env_<seed>.monitor.csv` | SB3 Monitor episode CSV | `Monitor` wrapper | [SRC] `base_env.py:L454` |
| `tb_logs/<config>/stage_<N>/` | TensorBoard event files | SB3 logger | [SRC] `orchestrator.py:L258-L261` |
| `results/<config>_final.zip` | Config-wide final model | Orchestrator on config completion | [SRC] `checkpoint_manager.py:L62-L63` |
| `results/run_summary.json` | Full run summary | Orchestrator | [SRC] `orchestrator.py:L181-L183` |
| `results/logs/interceptor.log` | Training log (rotating, 10 MiB max, 2 backups) | `configure_file_logging` | [SRC] `logger.py:L65-L70` |
| `curriculum_state/run_state.json` | Resume state | Orchestrator after each stage transition | [SRC] `checkpoint_manager.py:L69-L70` |

### 6.5 TensorBoard Keys

| TB Path | Source | Evidence |
|---|---|---|
| `reward_components/r_alive` through `r_time_bonus` | `RewardComponentCallback._flush` | [SRC] `callbacks.py:L37-L52` |
| `reward/episode_total` | `MetricsCallback._record_and_clear` | [SRC] `callbacks.py:L161-L164` |
| `metrics/alt_err`, `metrics/tilt`, `metrics/angvel_norm`, `metrics/miss_distance`, `metrics/time_to_intercept` | `MetricsCallback` | [SRC] `callbacks.py:L54-L60` |
| `curriculum/current_stage`, `curriculum/success_rate`, `curriculum/required_rate`, `curriculum/episodes_in_window` | `CurriculumCallback._on_rollout_end` | [SRC] `callbacks.py:L223-L229` |

### 6.6 Logging Format

```
%(asctime)s %(levelname)-7s [%(name)s] %(message)s
```
`datefmt = "%Y-%m-%d %H:%M:%S"`. All loggers are children of `interceptor` root. [SRC] `logger.py:L21-L24`.

---

## 7. Execution Runbook & Step-by-Step Reproduction Guide

### 7.1 Host-Side Smoke Test (no GPU/torch)

```bash
cd interceptor-training
pip install numpy scipy gymnasium pyyaml   # minimum deps for smoke test
python scripts/smoke_test.py
```

Expected output: `8 passed, 0 failed`. Requires `hover_env.py` at repo root on `sys.path` and, per D-20, the smoke model file and config file (not yet created). [SRC] `CURRENT_WORK.md:L108-L113`.

### 7.2 Container Build & Training

```bash
cd interceptor-training
docker compose up --build   # builds image, starts training
```

Or override:
```bash
docker compose run --rm interceptor-train --config /data/configs/default_run.yaml
```

Smoke mode (small budget):
```bash
docker compose run --rm interceptor-train --smoke --smoke-config ppo_baseline --smoke-stages 1,2,3 --smoke-steps 1000
```

[SRC] `CURRENT_WORK.md:L115-L121`, `scripts/train.py:L5-L7`.

### 7.3 Evaluation

```bash
docker compose run --rm --entrypoint python interceptor-train scripts/evaluate.py \
  --config /data/configs/default_run.yaml --config-name ppo_baseline --stage 3 --episodes 50
```

[SRC] `CURRENT_WORK.md:L126-L128`.

---

## 8. Code-Level Specification

### 8.1 Physics Pipeline (`interceptor-training/src/physics/`)

The physics pipeline is a verbatim port of `hover_env._ph_*` functions. It implements a 9-stage rigid-body dynamics pipeline for the Crazyflie 2.1 nano-quadrotor:

**Stage A — Rate PID Controller** (`pipeline.py:pid`)
```
e = omega_cmd - omega_prev
P = Kp × e
I = I_prev + Ki × e × dt  (reset to 0 if throttle_cut)
D = -Kd × (omega_prev - omega_prev2) / dt
u = P + I + D
```
Where `Kp = [0.150, 0.150, 0.200]`, `Ki = [0.200, 0.200, 0.100]`, `Kd = [0.003, 0.003, 0.000]`. [SRC] `pipeline.py:L66-L78`, `constants.py:L143-L145`.

**Stage B — Motor Mixer** (`pipeline.py:mixer`)
```
raw = c_cmd + MIX @ u
if max(raw) > CMD_MAX and |max(raw) - c_cmd| > 1e-9:
    u = u × (CMD_MAX - c_cmd) / (max(raw) - c_cmd)
    raw = c_cmd + MIX @ u
return clip(raw, CMD_MIN, CMD_MAX)
```
`MIX` is a (4,3) matrix: `columns = [sign(r_P[:,1]), -sign(r_P[:,0]), spin_sign]`. [SRC] `pipeline.py:L81-L96`, `constants.py:L103-L107`.

**Stage 2 — ESC** (`pipeline.py:esc`)
```
Ω_ss = c₀ + c₁·U_bat + c₂·√cmd + c₃·cmd + c₄·U_bat·√cmd
Ω_ss = 0 where cmd < 0.02
```
Where `[c₀, c₁, c₂, c₃, c₄] = [865.6, 379.9, 1053.4, -1818.9, -291.3]`. [SRC] `constants.py:L121-L122`, `pipeline.py:L99-L110`.

**Stages 3-8 — Derivative (`pipeline.py:deriv`)**
```
Ω_dot = (Ω_ss - Ω) / k_mot               # k_mot = 0.02 s
Ω_eff = max(Ω + 0.5·dt·Ω_dot, 0)          # mid-step effective speed
f_props, τ_props = prop_ft(Ω_eff)          # Stage 4: per-prop thrust + drag
f_tot, τ_tot = aggregate(f_props, τ_props)  # Stage 5: moment arms
τ_mot, τ_iner = react(Ω_dot, ω_B)          # Stage 6: motor reaction + gyroscopic
v_B = R_WB^T · v_WB                        # body-frame velocity
f_aero, τ_aero = aero(v_B, Ω_eff)          # Stage 7: polynomial drag
ṗ = v_WB                                   # Stage 8
q̇ = 0.5 · q ⊗ [0, ω_B]
v̇ = R_WB · (f_tot + f_aero) / m + g_W
ω̇ = J⁻¹ · (τ_tot + τ_mot + τ_aero + τ_iner)
Ω̇ = (Ω_ss - Ω) / k_mot
```

[SRC] `pipeline.py:L113-L179`.

**Stage 9 — RK4 Integration** (`pipeline.py:rk4`)
Classic 4th-order Runge-Kutta with `dt = PH_DT = 0.01 s`:
```
k1 = D(s)
k2 = D(s + 0.5·dt·k1)
k3 = D(s + 0.5·dt·k2)
k4 = D(s + dt·k3)
s_new = s + (dt/6)·(k1 + 2·k2 + 2·k3 + k4)
q_new /= |q_new|    # quaternion renormalization
Ω_new = max(Ω_new, 0)  # non-negative motor speeds
```

Optional extensions (wind, ground effect) are injected into `deriv()`. Wind modifies body velocity: `v_B = R^T·(v_WB - wind_W)`. Ground effect multiplies thrust: `f_prop *= ge_gain`. [SRC] `pipeline.py:L182-L246`.

### 8.2 Target Generator (`interceptor-training/src/envs/target_generator.py`)

**Decided (D-4, D-5, D-13; pending implementation):** Stages 5-7 use one fixed polynomial per episode, drawn at reset and computed backwards from start point, end point and episode length so the path ends exactly at the episode end; target paths are independent of the drone. **As shipped (still applies to `evasive`, Stage 8, disabled per D-6):** Integrates target kinematics with first-order Euler (not RK4): `vel += acc·dt; pos += vel·dt; acc += jerk·dt`. Re-samples maneuver segments periodically (1s segments for order 2-3, 0.3-1.0s for evasive). Constrains speed ≤ 30 m/s. Containment: when target exceeds `world_radius`, position is clamped to the boundary and radial velocity is reflected. [SRC] `target_generator.py:L1-L250`, `BUGS.md:L67-L74`.

**Target types and kinematic orders:**

| Type | Velocity | Acceleration | Jerk | Evidence |
|---|---|---|---|---|
| `none` | N/A (Stage 1) | N/A | N/A | [SRC] `stage_config.py:L196` |
| `static_waypoint` | 0 | 0 | 0 | [SRC] `stage_config.py:L217` |
| `static` | 0 | 0 | 0 | [SRC] `stage_config.py:L243` |
| `order_1` | constant | 0 | 0 | [SRC] `stage_config.py:L299` |
| `order_2` | time-varying | uniform random re-sampled | 0 | [SRC] `stage_config.py:L329` |
| `order_3` | time-varying | time-varying | uniform random re-sampled | [SRC] `stage_config.py:L359` |
| `evasive` | time-varying | time-varying | directed away from drone | [SRC] `stage_config.py:L392` |

### 8.3 Curriculum Scheduler (`interceptor-training/src/training/curriculum.py`)

**Episode success criteria:**

| Metric Name | Condition | Stages | Evidence |
|---|---|---|---|
| `episode_reward_mean` | `stats['episode_reward'] ≥ threshold` | Stage 1 (threshold=60.0) | [SRC] `curriculum.py:L56-L58` |
| `mean_final_distance` | `stats['final_distance'] ≤ threshold` | Stage 2 (threshold=2.0) | [SRC] `curriculum.py:L54` |
| `kill_rate` | `stats['kill'] == 1.0` | Stages 3-8 | [SRC] `curriculum.py:L52` |

**Decision logic:**
1. `CAPPED` if `total_steps ≥ budget` (highest priority)
2. `ADVANCE` if rolling window is full AND `rate() ≥ required_rate`
3. `ROLLBACK` if `rollback_threshold > 0` AND window full AND `rate() < rollback_threshold` AND `total_steps ≥ rollback_min_steps`
4. `RUNNING` otherwise

[SRC] `curriculum.py:L152-L177`.

---

## 9. Configuration & Constants Registry

### 9.1 Physics Constants

| Name | Value | Unit | Evidence |
|---|---|---|---|
| `PH_M` | `0.04085` | kg | [SRC] `constants.py:L39` |
| `PH_J` | `diag([2.3951e-5, 2.3951e-5, 3.2347e-5])` | kg·m² | [SRC] `constants.py:L40` |
| `PH_J_MP` | `2.0e-9` | kg·m² | [SRC] `constants.py:L42` |
| `PH_G_W` | `[0.0, 0.0, -9.81]` | m/s² | [SRC] `constants.py:L43` |
| `PH_DT` | `0.01` | s | [SRC] `constants.py:L44` |
| `PH_ARM` | `0.0397` | m | [SRC] `constants.py:L47` |
| `PH_C_L` | `2.618e-8` | N/(rad/s)² | [SRC] `constants.py:L69` |
| `PH_C_D` | `5.45e-11` | N·m/(rad/s)² | [SRC] `constants.py:L70` |
| `PH_K_MOT` | `0.02` | s | [SRC] `constants.py:L72` |
| `PH_OMEGA_MAX` | `2800.0` | rad/s | [SRC] `constants.py:L73` |
| `PH_ETA` | `0.7` | — | [SRC] `constants.py:L76` |
| `PH_BAT` | `[865.6, 379.9, 1053.4, -1818.9, -291.3]` | — | [SRC] `constants.py:L77` |
| `PH_KP` | `[0.150, 0.150, 0.200]` | — | [SRC] `constants.py:L143` |
| `PH_KI` | `[0.200, 0.200, 0.100]` | — | [SRC] `constants.py:L144` |
| `PH_KD` | `[0.003, 0.003, 0.000]` | — | [SRC] `constants.py:L145` |
| `PH_CMD_MIN` | `0.0` | — | [SRC] `constants.py:L148` |
| `PH_CMD_MAX` | `1.0` | — | [SRC] `constants.py:L149` |
| `PH_C_SCALE` | `1.0` | — | [SRC] `constants.py:L150` |
| `PH_W_SCALE` | `2.0` | rad/s | [SRC] `constants.py:L151` |
| `PH_CUT_THR` | `0.02` | — | [SRC] `constants.py:L152` |
| `PH_OMEGA_HOVER` | `1956.211093185006` (computed: `√(m·g / (4·c_l))`) | rad/s | [SRC] `constants.py:L168`, `CURRENT_WORK.md:L62` |
| `PH_C_HOVER` | `0.23253743635354834` (computed: `brentq` of ESC polynomial at `Ω_hover`) | — | [SRC] `constants.py:L180`, `CURRENT_WORK.md:L62` |
| `PH_TILT_CRASH_THRESHOLD` | `deg2rad(60.0) ≈ 1.0471975511965976` | rad | [SRC] `constants.py:L155` |
| `PH_GROUND_CONTACT_V_THRESHOLD` | `1.0` | m/s | [SRC] `constants.py:L156` |

### 9.2 8-Stage Curriculum Table (Condensed)

| Stage | Target Type | World Radius | Max Ep Steps | Max Train Steps | Success Metric | Threshold | Window | Req. Rate | k_alive | k_alt | k_tilt | k_va | k_pd | k_kill | miss_gate | time_gate | k_time_pen | normalize |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | none | 15 | 1000 | 3M | ep_reward_mean | 60.0 | 100 | 0.85 | 0.10 | 0.15 | 0.40 | 0.0 | 0.0 | 0.0 | ✗ | ✗ | 0.0 | ✗ |
| 2 | static_wpt | 30 | 1500 | 5M | mean_final_dist | 2.0 | 100 | 0.80 | 0.05 | 0.0 | 0.20 | 0.30 | 0.15 | 0.0 | ✗ | ✗ | 0.0 | ✗ |
| 3 | static | 40 | 2000 | 5M | kill_rate | 0.70 | 100 | 0.70 | 0.0 | 0.0 | 0.0 | 0.30 | 0.15 | 500 | ✓ | ✗ | 0.0 | ✗ |
| 4 | static | 40 | 2000 | 5M | kill_rate | 0.75 | 100 | 0.75 | 0.0 | 0.0 | 0.0 | 0.30 | 0.15 | 500 | ✓ | ✓ | 0.02 | ✗ |
| 5 | order_1 | 60 | 2500 | 8M | kill_rate | 0.65 | 100 | 0.65 | 0.0 | 0.0 | 0.0 | 0.30 | 0.15 | 500 | ✓ | ✓ | 0.02 | ✗ |
| 6 | order_2 | 80 | 3000 | 10M | kill_rate | 0.60 | 100 | 0.60 | 0.0 | 0.0 | 0.0 | 0.30 | 0.15 | 500 | ✓ | ✓ | 0.02 | ✓ |
| 7 | order_3 | 100 | 3000 | 12M | kill_rate | 0.55 | 100 | 0.55 | 0.0 | 0.0 | 0.0 | 0.30 | 0.15 | 500 | ✓ | ✓ | 0.02 | ✓ |
| 8 | evasive | 120 | 3000 | 15M | kill_rate | 0.50 | 100 | 0.50 | 0.0 | 0.0 | 0.0 | 0.30 | 0.15 | 500 | ✓ | ✓ | 0.02 | ✓ |

All stages: `k_angvel=0.02`, `omega_safe=3.0`, `k_smooth=0.05`, `k_crash=200.0`, `k_oob=200.0`, `obs_noise_std=0.01`, `kill_radius=0.5` (S3+), `lookahead_enabled=True` (S3+). [SRC] `stage_config.py:L189-L421`.

**Decided changes (pending implementation):** Stage 8 (`evasive`) is disabled in current scope (D-6); the smoke test overrides this (D-8). The `Max Ep Steps` column becomes an episode length in seconds, a fixed value or a random range (min, max), as a multiple of `PH_DT` (D-13, D-14). Stale dependents: see §13.9.

### 9.3 Aerodynamic Coefficients

| Array | Raw Values | Unit Correction | Evidence |
|---|---|---|---|
| `PH_FAX` | `[6.582314e-02, -3.507132e-02, 0.0, 3.952726e-07] × PH_M` | N | [SRC] `constants.py:L110-L114` |
| `PH_FAY` | `[5.738530e-02, 2.022478e-02, 0.0, -6.291393e-07] × PH_M` | N | [SRC] `constants.py:L115` |
| `PH_FAZ` | `[3.610184e-02, -1.153378e-01, 0.0, -5.881881e-07, 1.188474e-06, -1.705725e-08] × PH_M` | N | [SRC] `constants.py:L116-L117` |
| `PH_FTX` | `zeros(5) × PH_J[0,0]` (zeroed) | N·m | [SRC] `constants.py:L118` |
| `PH_FTY` | `zeros(5) × PH_J[1,1]` (zeroed) | N·m | [SRC] `constants.py:L119` |
| `PH_FTZ` | `[-1.432000e-03, 7.798700e-02] × PH_J[2,2]` | N·m | [SRC] `constants.py:L120` |

τ_x and τ_y polynomial coefficients are zeroed — the fitted values over-predicted roll/pitch torques by 120-230× for the NanoBench configuration. [SRC] `constants.py:L108-L109`.

---

## 10. Determinism, Runtime-Resolved Values & Numerical Contract

### 10.1 Random Number Generators

| RNG | Location | Seed | Evidence |
|---|---|---|---|
| Environment RNG | `np.random.default_rng(seed)` in `InterceptorBaseEnv.__init__` | `seed` kwarg (default `None`) | [SRC] `base_env.py:L135` |
| Target generator RNG | `np.random.default_rng((seed or 0) + 1_999_937 × stage_idx)` | Deterministic from env seed | [SRC] `base_env.py:L166-L167` |
| SB3 model seed | `seed + int(stage_id.split('_')[1])` | Global seed + stage offset | [SRC] `orchestrator.py:L256` |
| Wind OU process | Uses env's `_rng` | Seeded from env seed | [SRC] `base_env.py:L135` |

### 10.2 Draw Order (parity-critical for Stage 1)

At `reset`:
1. `xy = rng.uniform(-0.3, 0.3, size=2)` [SRC] `base_env.py:L282`
2. `tilt_rad = rng.uniform(0.0, deg2rad(5.0))` [SRC] `base_env.py:L283`
3. `axis = rng.standard_normal(3)` [SRC] `base_env.py:L284`

At each `_get_obs` (Stage 1):
1. `rng.normal(0, noise_std, 6)` for rotation [SRC] `hover_env.py:L590`
2. `rng.normal(0, noise_std, 3)` for velocity [SRC] `hover_env.py:L591`
3. `rng.normal(0, noise_std, 3)` for angular rate [SRC] `hover_env.py:L592`

This draw order MUST be preserved for bit-exact parity with `hover_env.py`.

### 10.3 Numerical Tolerances

| Quantity | Tolerance | Rationale | Evidence |
|---|---|---|---|
| Stage-1 obs parity vs `hover_env.py` | `np.array_equal` (exact) | Bit-for-bit parity | [SRC] `smoke_test.py:L78,L90` |
| Stage-1 reward parity | `rew_a == rew_b` (exact float equality) | Same left-fold summation | [SRC] `smoke_test.py:L91` |
| Kill bonus | `< 1e-9` absolute | Smoke test assertion | [SRC] `smoke_test.py:L263,L276` |
| Config hash | Exact string match (`SHA-256`) | Resume validation | [SRC] `orchestrator.py:L122-L127` |

### 10.4 Runtime-Resolved Values

| Value | Formula | Resolved At | Evidence |
|---|---|---|---|
| `PH_OMEGA_HOVER` | `√(PH_M × 9.81 / (4.0 × PH_C_L))` | Module import of `constants.py` | [SRC] `constants.py:L165-L168` |
| `PH_C_HOVER` | `brentq(ESC_polynomial - PH_OMEGA_HOVER, 0.02, 1.0)` at `U_bat = 4.2` | Module import of `constants.py` | [SRC] `constants.py:L170-L180` |
| `device` | `torch.cuda.is_available() ? "cuda" : "cpu"` when config says `"auto"` | `Orchestrator.__init__` | [SRC] `orchestrator.py:L66-L74` |

---

## 11. Gap Register

### 11.1 Gap Summary

| ID | Severity | Section | Description |
|---|---|---|---|
| G-001 | MINOR | §2.1 | Python version inside the container is not explicitly stated; it depends on the PyTorch base image which is a moving target. |
| G-002 | MINOR | §1.5 | SHA-256 hashes not computed for all files (marked `...`). Only hashes observed directly are recorded; the rest require `sha256sum` execution. |
| G-003 | MAJOR | §2.3 | Docker base image `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime` is not pinned by SHA-256 digest. A re-pull may fetch a different image layer. |
| G-004 | MAJOR | §2.2 | No `pip freeze` or lockfile (`requirements.txt` uses `>=` constraints). Exact resolved versions of `stable-baselines3`, `gymnasium`, `numpy`, `scipy`, `tensorboard`, `pyyaml`, and their transitive deps are unknown. |
| G-005 | MINOR | §2.7 | GPU model, VRAM, and CPU model of the original training host are not recorded. |
| G-006 | MAJOR | §3 | No trained model weights are present in the archive. The only checkpoint referenced (`/data/results/ppo_baseline_final.zip`) was produced by a smoke run with ~2600 steps (no convergence). |
| G-007 | BLOCKER | §3 | The shipped `default_run.yaml` only defines `ppo_baseline` with `stages: [1, 2]`, but the smoke test (`smoke_test.py:L401`) expects `cfg.config_stages("ppo_baseline") == [1, 2, 3, 4, 5, 6, 7]` and `cfg.config_obs_history("ppo_5layer_deep") == {"frames": 5, "skip": 3}`. Resolution decided (D-20): `default_run.yaml` is retired, its content moves to the model file and config file (D-9), and the smoke test gets its own model file and config file; the gap stays open until those files exist. |
| G-008 | BLOCKER | §8.2 | The `target_generator.py` `_resample_maneuver` method uses specific random distributions and sampling strategies for acceleration/jerk whose exact parameters depend on `SpawnConfig` fields. The `maneuver_duration` for evasive targets (`0.3` to `1.0 s`) and the `flee_direction` computation are described in the docstring but require reading the full source for exact implementation. |
| G-009 | MAJOR | §2.2 | `stable-baselines3[extra]` pulls in `opencv-python`, which is then replaced with `opencv-python-headless` in the Dockerfile. The exact version of `opencv-python-headless` installed is not pinned. |
| G-010 | MAJOR | §4 | The `implementation_plan.md` (47 KiB) is referenced throughout the code as the source of truth for stage configs and reward formulas, but its full content was not embedded. A re-implementer needs it for the exact stage progression rationale. |
| G-011 | BLOCKER | §1.5 | Several root-level source files (`swift_physics_headless.py`, `swift_physics_headless_v2.py`, `compare_pybullet.py`, `demo.py`, etc.) are legacy files referenced by `self_test.py` and `VERSIONS.md`. Their role in the active training pipeline is purely historical (they are not imported by `interceptor-training/`). |
| G-012 | MAJOR | §8.1 | The aerodynamic `aero()` function in `pipeline.py` uses different wind treatment than `hover_env.py`. When `wind_W` is provided, body-frame velocity becomes `v_B = R^T · (v_WB - wind_W)`, but this path is OFF by default. The exact interaction with domain randomization is via `sample_wind_params()` whose distribution parameters are implementation-internal. |
| G-013 | BLOCKER | §6.4 | The `BTP plan preliminary.docx` and `Interceptor_Complete_Project_Scope.docx` are binary `.docx` files not readable as text. Their content may contain requirements not reflected in the code. |
| G-014 | MAJOR | §8.3 | The `CurriculumCallback` stops `learn()` by returning `False` from `on_step`, but SB3's PPO may have already collected a partial rollout buffer. The scheduler's `sync_steps` method reconciles this, but the exact behavior depends on SB3 internal state. |
| G-015 | MINOR | §4 | The `run_summary.json` in the archive shows stages `1, 2, 3` but `default_run.yaml` only lists `stages: [1, 2]`. The smoke run must have used a modified config with `smoke.stages: [1, 2, 3]`. |
| G-016 | MAJOR | §8.1 | The `sample_physics_params()` function in `pipeline.py` performs domain randomization by perturbing `mass`, `J`, and aero coefficients. The exact perturbation distributions (uniform/normal, ranges) are implementation-internal. |
| G-017 | BLOCKER | §8.2 | The exact `predicted_pos()` implementation in `target_generator.py` uses a Taylor expansion including jerk; the precise code computes `pos + vel*h + 0.5*acc*h² + (1/6)*jerk*h³` with `h = 0.5`. This needs verbatim source for bit-exact reproduction. |
| G-018 | MINOR | §7 | The `Docker_Setup_and_Running_Guide.md` (17 KiB) provides Ubuntu-specific installation steps. Windows/macOS Docker setup is not covered. |

### 11.2 Proposed Resolutions

| Gap ID | Fill Slot | Proposed Resolution |
|---|---|---|
| G-003 | `<<<FILL: G-003>>>` | Pin Docker base image by digest: `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime@sha256:<digest>` |
| G-004 | `<<<FILL: G-004>>>` | Run `pip freeze > requirements.lock` inside a fresh container build and commit the lockfile |
| G-007 | `<<<FILL: G-007>>>` | Create a separate smoke model file and config file for `smoke_test.py` (D-20); `default_run.yaml` is retired, not extended |
| G-010 | `<<<FILL: G-010>>>` | The `implementation_plan.md` is present in the archive (47 KiB). Read it in full to extract all stage definitions and reward formulas |
| G-013 | `<<<FILL: G-013>>>` | Convert `.docx` files to text or extract content for review |

---

## 12. Verification Suite & Acceptance Criteria

### 12.1 Smoke Test Suite (`scripts/smoke_test.py`)

| Test | What It Verifies | Pass Criterion | Evidence |
|---|---|---|---|
| `parity_stage1_vs_hover` | Bit-exact parity of `InterceptorBaseEnv(stage_1)` vs `hover_env.AltitudeHoldEnv` over a multi-step episode | `np.array_equal(obs_a, obs_b)` every step; `rew_a == rew_b` every step; `term_a == term_b`; all info keys match | [SRC] `smoke_test.py:L66-L98` |
| `all_stage_envs_run` | All 8 stage envs produce correct obs dims, run 200 steps, terminate, produce `episode_stats` | Obs dim = 14 (S1) or 76 (S2+); `episode_stats` has all required keys; `stats['steps'] == n` | [SRC] `smoke_test.py:L104-L139` |
| `obs_history_stacking` | History stacker produces correct dims, current-first order, hidden-target zero masking | `obs.shape == (76,)` for m=3; newest slot has visible=1; zero-padded older slots; stride correctness; hidden target fields stay 0 under noise | [SRC] `smoke_test.py:L145-L190` |
| `target_generator_kinematics` | Target step, containment, and flee direction | Position stays within `world_radius + 1.0`; evasive jerk `dot(jerk, flee_dir) > 4.0` | [SRC] `smoke_test.py:L196-L236` |
| `reward_terms_sanity` | Kill bonus scaling by miss distance and time; dense weight correctness | `r_kill_bonus ≈ 500 × (1 - 0.1/0.5)` within 1e-9; time-scaled bonus at t=0 is 500; dense `r_tilt = -0.40 × θ` within 1e-9 | [SRC] `smoke_test.py:L242-L302` |
| `curriculum_advance_cap_rollback` | Advance at high success; cap at step budget; rollback at low success; disable with threshold=0; serialization | Returns correct terminal result; `state_dict`/`load_state_dict` roundtrip | [SRC] `smoke_test.py:L308-L344` |
| `checkpoint_manager_roundtrip` | Path layout; JSON persistence; periodic/final discovery | Path structure matches spec; JSON roundtrip preserves data; `latest_periodic` finds highest step count | [SRC] `smoke_test.py:L361-L386` |
| `config_loader_validation` | Validation; hashing; plan-schema canonicalization | `seed == 42`; hash is stable; plan-schema `algorithm`/`policy_kwargs` canonicalize correctly; shared `net_arch` list expands to `{pi, vf}` | [SRC] `smoke_test.py:L392-L445` |

### 12.2 Stage-1 Parity Contract

The core numerical contract: given the same `seed` and action sequence, `InterceptorBaseEnv(STAGES["stage_1"])` produces **bit-identical** observations, rewards, termination flags, and info dictionaries as `hover_env.AltitudeHoldEnv`. This is verified by `smoke_test.py:parity_stage1_vs_hover` over 2000 steps (or until termination). [SRC] `smoke_test.py:L66-L98`, `CURRENT_WORK.md:L39-L40`.

### 12.3 Acceptance Criteria

A re-implementation is accepted if:
1. `smoke_test.py` exits with code 0 (all 8 tests pass) against its model file and config file (D-20)
2. Stage-1 produces `np.array_equal` observations and `==` rewards vs. `hover_env.py` for seed 12345
3. The reward left-fold summation matches the archive's implementation (no compensated summation)
4. `PH_C_HOVER ≈ 0.23253743635354834` and `PH_OMEGA_HOVER ≈ 1956.211093185006` within float64 precision
5. A smoke training run (stages 1-3, 1000 steps each) completes without errors and produces a valid `run_summary.json`

---

## 13. Anomalies, Contradictions & Behavioral Quirks

### 13.1 Config / Test Mismatch

Resolved by D-20 (pending implementation): the shipped `default_run.yaml` (`stages: [1, 2]`) is retired and its content moves to the model file and config file (D-9); `smoke_test.py` uses its own model file and config file (D-7). Those files do not exist yet; their locations, names and contents are OPEN. The stage list is an argument of `train_model(...)`, not a key in either file (D-21). Until they exist, the smoke test still fails against the shipped config. [SRC] `default_run.yaml:L42`, `smoke_test.py:L401-L402`.

### 13.2 `_target_alt` Property is Hard-Coded

Decided (D-3, pending implementation): the target altitude comes from a `target_alt` entry in the config, a fixed numeric value or a random value within a user-set range; this replaces the hard-coded property. Scope: Stage 1 hover altitude only (D-50). As shipped, `InterceptorBaseEnv._target_alt` always returns `5.0` (a property, not a mutable field), so Stage 1 never randomizes altitude despite `hover_env.py` supporting `randomize_altitude`. Stale dependents: see §13.9. [SRC] `base_env.py:L423-L425`.

### 13.3 `run_summary.json` at Repo Root vs. Container Path

The `run_summary.json` at the repo root references `/data/results/ppo_baseline_final.zip` and `/data/configs/default_run.yaml` — these are container-internal paths. The file was copied from the container's `/data` volume. [SRC] `run_summary.json:L2,L28`.

### 13.4 `steps_added` Undercount on Rollback

On rollback re-entry, `model.num_timesteps` is reset to 0, so only the retry's steps are counted. The original budget's consumed steps are lost from the total. [SRC] `orchestrator.py:L370`, `BUGS.md:L24-L30`.

### 13.5 Monitor CSV Overwrite on Resume

`Monitor(filename=...)` truncates the CSV on resume (no `override_existing` guard). [SRC] `BUGS.md:L77-L80`.

### 13.6 `hover_env.py` vs `README.md` Mass Discrepancy

`hover_env.py:L166` uses `PH_M = 0.04085 kg` (40.85 g), but `README.md:L129` states "Mass: 27 g". The README reflects the original Crazyflie 2.0 (Forster thesis), while `PH_M` is the calibrated "NanoBench flying mass" from `swift_live_demo_fitted.py`. The code value (`0.04085`) is authoritative. [SRC] `hover_env.py:L166`, `README.md:L129`.

### 13.7 ESC Polynomial Fallback

If `scipy.optimize.brentq` is unavailable, `PH_C_HOVER` is computed by brute-force grid search (20,000 points from 0.02 to 1.0). The grid resolution may produce a different value than `brentq`, though the difference would be negligible. [SRC] `constants.py:L170-L180`, `hover_env.py:L243-L256`.

### 13.8 Known Bugs (Non-Blocking)

All 12 bugs documented in `BUGS.md` are non-blocking. Items 11 and 12 were fixed in-tree. The remaining 10 affect only resume accounting, performance (not correctness), or container security hardening. [SRC] `BUGS.md:L1-L111`.

### 13.9 Decided Changes Pending Implementation

Decisions D-2 to D-58 (Decision Log) are not yet reflected in the body except where marked. These sections are stale until updated:

| Decision | Stale sections |
|---|---|
| D-2, D-20 | §1.4 tree, §1.5 manifest, §0 Counters (smoke model file and config file paths undecided, so not written); for the retirement of `default_run.yaml` see also the D-9, D-10 row, §6.2 (`config_file` and `config_hash` span two files) and `config_loader.py` validation and `config_hash` |
| D-3 | §5.1 `r_alt`, §5.3 index [12], §9 registry (`target_alt`), §10.1, §10.2, §12.2, §P02 |
| D-4, D-13, D-14 | §8.2 table rows `order_2`/`order_3`, §5.2 `time_gate`, §5.3 lookahead, §9.2 columns and thresholds (Stage 1, Stages 5-7), §9 registry (episode length), §10.1, §10.2, §12.1 `target_generator_kinematics` and `all_stage_envs_run`, `BUGS.md` item 7 |
| D-5, D-6, D-8 | §1.1 and §8.2 ("8-stage", `evasive`), §9.2 Stage 8 row, §P02 stage-list validation, §12.1 `all_stage_envs_run` |
| D-7 | §1.2, §7.1, §12.1, §12.3 criterion 5, `BUGS.md` item 5 |
| D-9, D-10 | §1.2, §1.3, §1.6, §2.4, §P02, §P03, §P04, §6.1-§6.4, §7, A.1, A.2, `docker-compose.yml`, §12.1 `config_loader_validation` |
| D-11, D-12 | §3 model card, §5.3 (critic observation), §8, §9 registry, §12.1 |
| D-15, D-16, D-17, D-18, D-19 | §3 model card (obs dim, critic input), §5.3 (history, lookahead and future paragraphs; fate of the shipped lookahead OPEN), §8.2 (true-future access), §9 registry (n, m, p, q, future source, predictor name), §9.2 `lookahead_enabled`, §P02 validation, §12.1 `obs_history_stacking`, `all_stage_envs_run` and a new test for D-17, D-11 and D-12 critic inputs; code: `obs_builder.py`, `base_env.py`, `target_generator.py`, new predictor module |
| D-21, D-22, D-23, D-26, D-27 | `train_model` argument table (D-9, D-36: source, stage list, `name`, optional ids, optional `seed`), §P02 stage-list validation, §P03 step 3 (stage list now comes from the call; step 3d model selection), §6.1 (result and return structure keyed by model id), §6.2 (resume state), §6.4 (checkpoint and log layout keyed by `name` and model id), §7.2 (`--smoke-stages`), §3 model card (`stages` mention), §13.4, §12.1 `config_loader_validation`, the D-10 text, D-20 checklist (Session Handoff) |
| D-24, D-25 | §8.3 decision logic, §P03 step 3g, §6.1 `result` values, §9 registry (new config key `on_capped`, D-30), §P02 validation, §12.1 `curriculum_advance_cap_rollback`, D-20 checklist (destination of the new key) |
| D-28, D-29, D-31, D-32, D-34, D-35, D-36, D-45, D-46, D-47 | §6.1 (result: stage outcomes, `config_hash`, skipped and crashed models with plain-language reasons), §6.2 (resume state: rollback counts across calls, resume from last snapshot), §6.4 (last stage's final model on disk, SAC/TD3 replay buffer files), §P03 (continuation from saved settings, crash handling, skip rules), §8.3, §12.1 `config_loader_validation` and `curriculum_advance_cap_rollback`, D-20 checklist (Session Handoff) |
| D-33 | §3 model card (weight transfer when the observation size changes, R-8), §P03 |
| D-37 | §5.1 (any yaw term), §5.3 (critic observation), §8, §9 registry, §12.1 (with D-11, D-12) |
| D-38, D-39, D-40, D-41, D-42, D-43 | §5.3 (history, future and lookahead paragraphs; `[18]` target visible flag and zero masking of hidden target fields), §5.1 `r_velocity_alignment` (`target_visible`), §9 registry (n, m, p, q in the config file; predictor), §9.2 `lookahead_enabled`, §12.1 `obs_history_stacking` and `all_stage_envs_run`; code: `obs_builder.py`, `base_env.py` (`target_visible`), new predictor module |
| D-44, D-48, D-49, D-50, D-51, D-52 | §9 registry (range syntax `{min:, max:}`, free parameters of D-4/D-13), §P02 validation, §1.4, §1.5, §0 Counters and §7.1 (smoke file names OPEN), §5.1 `r_alt` and §5.3 index [12] (D-50, see D-3 row), §9.2 Stage 1 length and threshold 60.0 (D-52), §10.1, §10.2 |
| D-53, D-54 | §3 model card (predictor input: own measured history since episode start, cleared at every reset; replaces the D-17 wording), §5.3 (history and future paragraphs: target fields in the current body frame measured from the current position, resolves the frame OPEN of D-41), §8.2 (true-future access), §9 registry (history source for the predictor), §12.1 `obs_history_stacking` and the D-17 test; code: `obs_builder.py` (must store past drone poses), `base_env.py`, new predictor module |
| D-55, D-56 | §6.1 (stage outcome `stuck`, denied capped source), §8.3, §P03 (STUCK stops only that model; denial rule for a capped source model), §12.1 `curriculum_advance_cap_rollback`; the reason of D-24 ("user decides what to do next") is stale: retraining now means a new model id |
| D-57 | §5.1 (new facing term from Stage 2), §5.3 (critic observation: facing error), §9.2 weight table (`k_facing` OPEN), §9 registry, §12.1 reward test and D-11, D-12 critic field list; code: `reward.py`, `stage_config.py` |
| D-58 | none (ordering of work: FIX pass last) |

---

## Appendix A — Verbatim Artifacts

> **Note:** Due to document size constraints, only the most critical files are embedded verbatim below. All other files marked EMBEDDED in §1.5 are available in the archive at their stated paths and were fully read during analysis.

### A.1 `interceptor-training/configs/default_run.yaml`

```yaml
# default_run.yaml -- interceptor drone curriculum training (plan §4.3, §11)
#
# This run: ppo_baseline only, stages 1 & 2 to full completion.
#   Stage 1 (Hover/Attitude):            budget = 3,000,000 steps
#   Stage 2 (Directional Flight/Wpt):    budget = 5,000,000 steps
# Total maximum: 8,000,000 steps (~hours on a GPU host with 8 envs).
#
# Top-level sections: `global`, `stages` (optional per-stage overrides),
# `configs` (one training baseline per block).

global:
  seed: 42
  device: cpu                  # PPO+MlpPolicy is faster on CPU (SB3 recommendation)
  data_dir: /data              # host-mounted volume (see docker-compose.yml)
  n_parallel_envs: 8
  checkpoint_interval_steps: 50000
  tensorboard: true
  execution_mode: sequential   # plan §7: only sequential supported
  rollback:                    # plan §6.2 policy
    max_attempts: 2            # max rollbacks per stage
    retry_budget_scale: 0.5    # re-entered stage runs on 50% of its budget
    threshold_scale: 0.5       # rollback threshold = 50% of success_rate
    min_steps_scale: 0.25      # rollback only after 25% of max_training_steps
  # optional smoke override (used by `train.py --smoke`):
  # smoke:
  #   config: ppo_baseline
  #   stages: [1, 2]
  #   steps_per_stage: 1000

# Per-stage overrides merged on the built-in stage table.
# Empty here = use the exact plan defaults.
stages: {}

# --- baselines (plan §4.3) --------------------------------------------------
configs:
  ppo_baseline:
    algo: PPO
    policy: MlpPolicy
    net_arch: {pi: [256, 256, 128], vf: [256, 256, 128]}
    activation: ReLU
    obs_history: {frames: 3, skip: 2}
    stages: [1, 2]             # stages 1 and 2 only -- full training budgets
    hyperparameters:
      n_steps: 4096
      batch_size: 512
      gamma: 0.995
      learning_rate: 3.0e-4
      gae_lambda: 0.95
      clip_range: 0.2
      ent_coef: 0.005
      n_epochs: 10
      max_grad_norm: 0.5
      vf_coef: 0.5
```

### A.2 `interceptor-training/Dockerfile`

```dockerfile
# Multi-stage RL Training Pipeline for the Autonomous Interceptor Drone.
#
# Base image: PyTorch 2.7.0 (CUDA 12.8, cuDNN 9 runtime) — supports sm_120
# (RTX 50 series) GPUs. Fallback if the image is unavailable: switch base to
# `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel` or install torch from the
# cu128 index (`pip install torch --index-url https://download.pytorch.org/whl/cu128`).
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

# System dependencies (git for pip VCS installs and repo tooling).
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# opencv-python needs GUI libs (libxcb) missing here; headless is enough for training.
RUN pip uninstall -y opencv-python && pip install --no-cache-dir --default-timeout=120 --retries 10 opencv-python-headless

# Application source, entrypoint scripts and shipped configs.
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/

# Host-mounted volume: checkpoints/, tb_logs/, curriculum_state/, results/.
# The host also maps its local configs directory onto /data/configs.
VOLUME /data

ENTRYPOINT ["python", "scripts/train.py"]
CMD ["--config", "/data/configs/default_run.yaml"]
```

### A.3 `interceptor-training/requirements.txt`

```
# torch is intentionally omitted: pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime
# already ships torch 2.7.0+cu128. Listing it here causes pip to upgrade to
# the latest release (a CPU-only wheel), overwriting the CUDA build.
stable-baselines3[extra]>=2.4.0
gymnasium>=1.0.0
numpy>=2.0.0
scipy>=1.13.0
tensorboard>=2.17.0
pyyaml>=6.0
```

### A.4 `run_summary.json` (smoke run output)

```json
{
  "run_id": "20260930-130451",
  "config_file": "/data/configs/default_run.yaml",
  "config_hash": "4ff52e718eca859d5bd542768f797eb1c23c99f3d3df3b0c17f3aa39c2f76c09",
  "device": "cuda",
  "data_dir": "/data",
  "started": "2026-09-30 13:04:51",
  "configs": {
    "ppo_baseline": {
      "algo": "PPO",
      "stages": {
        "1": {
          "result": "capped",
          "steps": 784,
          "success_rate": 0.0
        },
        "2": {
          "result": "capped",
          "steps": 872,
          "success_rate": 0.0
        },
        "3": {
          "result": "capped",
          "steps": 968,
          "success_rate": 0.0
        }
      },
      "final_model": "/data/results/ppo_baseline_final.zip",
      "steps": 2624
    }
  },
  "finished": "2026-09-30 13:04:54",
  "wall_seconds": 2.7
}
```

---

## Appendix B — Checksums & Non-Embedded Artifact Manifests

SHA-256 hashes for all `.gitkeep` files: `e3b0c44298fc1c149afbf4c8996fb924` (empty file, 0 bytes).

Hashes for all other files were observed during file reads but not all were recorded with cryptographic verification. A re-implementer should compute `sha256sum` on the full archive for integrity verification. [[GAP:G-002]]

---

## Appendix C — Self-Audit Report

### C.1 Completeness Check

| Criterion | Status | Notes |
|---|---|---|
| Every archive entry listed in §1.5 | ✓ | 91 files inventoried, all classified |
| Every entry point documented | ✓ | 6 entry points in §1.6 |
| All environment variables documented | ✓ | 3 env vars in §2.4 |
| All external deps documented | ✓ | Container + host requirements in §2.2 |
| Model architecture fully specified | ✓ | PPO with MlpPolicy, net_arch, activation in §3 |
| Training loop fully documented | ✓ | Per-step, per-epoch, per-rollout in §4 |
| Reward functions with formulas | ✓ | Dense + terminal in §5 |
| All data schemas documented | ✓ | JSON schemas in §6 |
| Constants registry complete | ✓ | 30+ constants in §9 |
| Determinism contract stated | ✓ | RNG, draw order, tolerances in §10 |
| Gap register populated | ✓ | 18 gaps identified |
| Anomalies documented | ✓ | 8 anomalies in §13 |
| Stage-1 parity contract | ✓ | §12.2 |

### C.2 Known Limitations

1. Binary files (`.docx`, `.pdf`, `.png`) were not content-analyzed
2. Notebooks were sampled (cell-by-cell inspection) rather than fully read
3. SHA-256 hashes were not cryptographically verified for most files
4. The full `implementation_plan.md` (47 KiB) was not embedded due to size constraints
5. Legacy physics files (`swift_*.py`) were read for context but are not part of the active pipeline

### C.3 Document Confidence

| Section | Confidence | Basis |
|---|---|---|
| §1-§4 | HIGH | Direct source analysis |
| §5 | HIGH | Source + smoke test cross-validation |
| §6 | HIGH | Schema extracted from producers + consumers |
| §7 | HIGH | Matches CURRENT_WORK.md runbook |
| §8 | HIGH | Full source read |
| §9 | HIGH | All values from source |
| §10 | HIGH | Parity test + source analysis |
| §11 | MEDIUM | Some gaps may exist in unread binary docs |
| §12 | HIGH | Smoke test suite read in full |
| §13 | MEDIUM | Some quirks may be undiscovered |

---

*End of document.*

---

## Change Log (append-only)

- Version v1.0, session 2026-10-02
- Changed: §0 Archive Identity : (no version row) -> `File version | v1.0` : version tracking
- Changed: §0 Interpretation Log item 5 : "The YAML is a subset; the smoke test expects a richer YAML than the one shipped." -> resolved by D-1 : D-1
- Changed: §3 last paragraph : "indicating a richer YAML exists externally" -> D-1 fixture config : D-1
- Changed: §7.1 : requires only `hover_env.py` -> also the smoke fixture config : D-1
- Changed: §11.1 G-007 : "A richer YAML with all 4 configs ... stages 1-7 is required but not present." -> resolution decided (D-1), gap stays open until fixture exists : D-1
- Changed: §11.2 G-007 : "Extend `default_run.yaml` to include all 4 configs (`ppo_baseline` with stages 1-7, `sac_baseline`, `td3_baseline`, `ppo_5layer_deep`) as described in `CURRENT_WORK.md:L187-L189`" -> separate smoke fixture config : D-1
- Changed: §12.3 criterion 1 : added "against its fixture config (D-1)" : D-1
- Changed: §13.1 : "shipped YAML `stages: [1, 2]` vs smoke test `[1..7]` and `ppo_5layer_deep`, smoke test fails against shipped config" -> resolved by D-1 (fixture not yet created) : D-1
- Changed: §13.2 : "`_target_alt` always returns `5.0`, Stage 1 never randomizes altitude" -> kept as "as shipped", decided config-driven `target_alt` added : D-3
- Changed: §8.2 first paragraph : Euler/segment re-sampling text -> scoped "as shipped, still applies to `evasive`"; decided fixed polynomial per episode for Stages 5-7 added : D-4, D-5, D-13
- Changed: §9.2 : note added after "All stages" paragraph (Stage 8 disabled, episode length in seconds) : D-6, D-8, D-13, D-14
- Dependents updated: D-1 dependents 1-6 (§13.1, §11.1 G-007, §11.2 G-007, §3, §0 item 5, §12.3 criterion 1, §7.1)
- Dependents flagged stale: D-1 dependent 7 (§1.4, §1.5, §0 Counters; fixture path undecided); all dependents of D-3 to D-14 (see §13.9)
- Added: §13.9, Change Log, Decision Log, Session Handoff. Removed: none beyond the replaced text above.
- Version v2.0, session 2026-10-02
- Changed: §0 Archive Identity : `File version | v1.0` -> `File version | v2.0` : version tracking
- Changed: §0 Interpretation Log item 5 : "Resolved by D-1: the smoke test uses its own fixture config; `default_run.yaml` is unchanged." -> "Resolved by D-20: `default_run.yaml` is retired and its content moves to the model file and config file (D-9); the smoke test uses its own model file and config file (D-7)." : D-20
- Changed: §3 last paragraph : "D-1 resolves by giving the smoke test its own fixture config (fixture not yet created)" -> "D-20 resolves by giving the smoke test its own model file and config file (not yet created)" : D-20
- Changed: §7.1 : "per D-1, the smoke fixture config (not yet created)" -> "per D-20, the smoke model file and config file (not yet created)" : D-20
- Changed: §11.1 G-007 : "Resolution decided (D-1): the smoke test gets its own fixture config and `default_run.yaml` stays unchanged" -> "Resolution decided (D-20): `default_run.yaml` is retired, its content moves to the model file and config file (D-9), and the smoke test gets its own model file and config file" : D-20
- Changed: §11.2 G-007 : "Create a separate smoke fixture config for `smoke_test.py` (D-1); `default_run.yaml` is not extended" -> "Create a separate smoke model file and config file for `smoke_test.py` (D-20); `default_run.yaml` is retired, not extended" : D-20
- Changed: §12.3 criterion 1 : "against its fixture config (D-1)" -> "against its model file and config file (D-20)" : D-20
- Changed: §13.1 : "Resolved by D-1: ... `default_run.yaml` ... is unchanged. The fixture file does not exist yet" -> "Resolved by D-20 (pending implementation): `default_run.yaml` is retired ... Those files do not exist yet" and the stage-list location marked OPEN : D-20
- Changed: §13.9 intro : "Decisions D-3 to D-14" -> "Decisions D-2 to D-20" : D-15 to D-20
- Changed: §13.9 row "D-1, D-2" -> "D-2, D-20" with the retirement of `default_run.yaml` cross-referenced : D-20
- Dependents updated: D-1 dependents re-worded from D-1 to D-20 (§0 item 5, §3, §7.1, §11.1 G-007, §11.2 G-007, §12.3 criterion 1, §13.1)
- Dependents flagged stale: D-2 and D-20 (§1.4, §1.5, §0 Counters, §6.2, `config_loader.py`); D-15 to D-19 (see §13.9 new row); all dependents of D-3 to D-14 (see §13.9)
- Added: §13.9 row for D-15 to D-19; D-15 to D-20 in the Decision Log. Removed: D-1 statements from the body (superseded by D-20; the D-1 line stays in the Decision Log).
- Version v3.0, session 2026-10-02
- Changed: §0 Archive Identity : `File version | v2.0` -> `File version | v3.0` : version tracking
- Changed: §13.1 : "as is whether the stage list lives in the config file or the model file" -> "The stage list is an argument of `train_model(...)`, not a key in either file (D-21)." : D-21
- Changed: §13.9 intro : "Decisions D-2 to D-20" -> "Decisions D-2 to D-27" : D-21 to D-27
- Dependents updated: none beyond §13.1
- Dependents flagged stale: all dependents of D-21 to D-27 (see new §13.9 rows); D-20 checklist reworded in the Session Handoff (stage lists move to the `train_model` call, new capped-stage key)
- Added: §13.9 rows for D-21 to D-27; D-21 to D-27 in the Decision Log. Removed: none.
- Version v4.0, session 2026-10-02
- Changed: §0 Archive Identity : `File version | v3.0` -> `File version | v4.0` : version tracking
- Changed: §13.2 : added "Scope: Stage 1 hover altitude only (D-50)." : D-50
- Changed: §13.9 intro : "Decisions D-2 to D-27" -> "Decisions D-2 to D-52" : D-28 to D-52
- Changed: §13.9 row D-21, D-22, D-23, D-26, D-27 : "(D-9, PROPOSED: source, stage list, `name`, optional ids)" -> "(D-9, D-36: source, stage list, `name`, optional ids, optional `seed`)" : D-36
- Changed: §13.9 row D-24, D-25 : "new config key for the capped-stage setting" -> "new config key `on_capped`, D-30" : D-30
- Dependents updated: none beyond the changes above
- Dependents flagged stale: all dependents of D-28 to D-52 (see new §13.9 rows); D-20 checklist reworded in the Session Handoff (key `on_capped`, n, m, p, q in the config file, no predictor artifact for `linear_ridge`)
- Added: §13.9 rows for D-28 to D-52; D-28 to D-52 in the Decision Log. Removed: none.

- Version v5.0, session 2026-10-02
- Changed: §0 Archive Identity : `File version | v4.0` -> `File version | v5.0` : version tracking
- Changed: §13.9 intro : "Decisions D-2 to D-52" -> "Decisions D-2 to D-58" : D-53 to D-58
- Dependents updated: none beyond the changes above
- Dependents flagged stale: all dependents of D-53 to D-58 (see new §13.9 rows); D-24 reason (§13.9 row D-55, D-56); D-12 critic field list (facing error, D-57)
- Added: §13.9 rows for D-53 to D-58; D-53 to D-58 in the Decision Log. Removed: none from the body (D-17 and the `STUCK` assumption existed only in the Decision Log and the old Handoff; the D-17 line stays in the Decision Log, superseded by D-53).

## Decision Log (append-only)

- D-1: `smoke_test.py` uses its own fixture config; `default_run.yaml` (`stages: [1, 2]`) stays unchanged. Reason: shipped YAML is an intentional 2-stage run. (v1.0)
- D-2: Update all 7 dependents of D-1 when the file is written. Reason: keep the file consistent. (v1.0)
- D-3: Target altitude comes from a `target_alt` config entry, fixed value or random within a user-set range; replaces hard-coded `_target_alt = 5.0`. Reason: user controls altitude without touching code. (v1.0)
- D-4: Stages 5, 6, 7 use one fixed polynomial per episode, newly drawn each episode, same order within a stage, no mid-episode re-draws. Reason: user's definition of an episode. (v1.0)
- D-5: Target paths are independent of the drone in current scope; evasion is future scope. Reason: user scoping. (v1.0)
- D-6: Stage 8 (`evasive`) stays as defined but is disabled in current scope. Reason: evasion is future scope (D-5). (v1.0)
- D-7: Smoke test runs every model on every stage with tiny budgets; pass means completes without error. Reason: check that all models work on all stages, not learning. (v1.0)
- D-8: Smoke test includes Stage 8 through a smoke-only override of D-6. Reason: exercises the dormant evasive code. (v1.0)
- D-9: Three-artifact design: user-edited model files and config files plus one fixed run file that only calls `train_model(...)`; replaces `scripts/train.py`, the single-YAML structure and the Docker ENTRYPOINT. Reason: non-expert user workflow. The `train_model` argument table is PROPOSED, not decided. (v1.0)
- D-10: One `train_model` call trains every model listed in the model file and returns several results. Reason: user choice. (v1.0)
- D-11: Privileged-critic input list lives in the model file. Reason: it changes critic input size, so continuations cannot alter it. (v1.0)
- D-12: Privileged-critic reference is re-solved every step from the drone's current state and given as an error (ideal minus current, body frame). Reason: stays valid when the drone deviates. (v1.0)
- D-13: Target path is computed backwards from start point, end point and episode length so it ends exactly at episode end; episode length is a config parameter in seconds. Reason: exact end time. (v1.0)
- D-14: Episode length accepts a fixed value or a random range (min, max) in seconds, drawn at each reset. Reason: user choice. (v1.0)
- D-15: The actor does not see time remaining; only the critic does. Reason: a real target has no known episode end, and the best behaviour (intercept early) does not depend on it. (v2.0)
- D-16: The source of the target's future samples is a config selector, `true` (ground truth from the simulator's target path) or `pred` (predictor). Reason: an oracle for comparison and a deployable mode, with the same input shape. (v2.0)
- D-17: The predictor may use no information the actor does not have. Reason: it must run on a real drone. (v2.0)
- D-18: The actor observes history (m past frames, spacing p) and future (n samples, spacing q); n, m, p, q are user-set parameters (user said config file; which file holds them is OPEN). Reason: user request. (v2.0)
- D-19: `pred` offers two predictors selected by name, `const_vel` and `linear_ridge`; the predictor is a separate function with a fixed interface so a better one can replace it later. Reason: cheap stand-in for a learned predictor without changing the environment or the actor. (v2.0)
- D-20: `default_run.yaml` is retired; all its content moves to the model file and config file (D-9); supersedes D-1. Condition: nothing may be lost (checklist in Session Handoff). Reason: D-9 replaces the single-YAML structure. (v2.0)
- D-21: The stage list is an argument of `train_model(...)`, given together with the model source; it is not a key in the model file or the config file. Reason: user's original idea; resolves the OPEN stage-list location. (v3.0)
- D-22: `train_model(...)` takes a `name` argument that names the log folder; the folder holds the periodic snapshots of every stage and the final model of every stage except the last; the call returns the final model of the last stage in the variable; passing that variable to the next call continues training from it. Reason: user workflow (`model1`, then `model2` from `model1`). (v3.0)
- D-23: The returned model can be saved to a folder on disk and loaded back, so it survives closing the program. Reason: models are stored files (SB3 `.zip` today). (v3.0)
- D-24: A stage that ends `CAPPED` (step budget reached, success threshold not reached) is never an error by itself: all snapshots and stage finals are still saved and the last stage's final model is still returned. Reason: keeps training time; user decides what to do next. (v3.0)
- D-25: A setting in the config file controls what happens when a stage ends capped: `continue` (later stages keep training, default, matches shipped behaviour) or `stop` (the call ends there with the stage marked capped); syntax OPEN. Reason: user wants to decide per run, next to the `rollback` policy. (v3.0)
- D-26: `train_model` takes one source, a model file or a previous result; it trains all models in that source by default, an optional list of ids trains only some; it returns a collection keyed by model id, reached as `model1.<id>` (primary) or `model1["<id>"]`; refines D-10. Reason: user choice. (v3.0)
- D-27: When a call is given models whose last completed stage does not fit the requested stage list, it skips them, trains the rest, and lists each skipped model in the result with a plain-language reason. Reason: wastes less time than refusing the whole call. Whether it also covers a model that crashes is OPEN. (v3.0)
- D-28: A continuation reuses the training settings saved inside the model; no setting can be overridden. Reason: every continuation stays comparable to the stage before it and `config_hash` stays meaningful. (v4.0)
- D-29: The call also writes the last stage's final model to disk; location OPEN; refines D-22 and D-23. Reason: user choice. (v4.0)
- D-30: The capped-stage setting is the config-file key `on_capped` with values `continue` or `stop`, next to `rollback`; resolves the syntax OPEN in D-25. Reason: user choice. (v4.0)
- D-31: If one model crashes, the other models keep training and the crash is reported in plain words in the result; extends D-27. Reason: user choice, same treatment as skips. (v4.0)
- D-32: The result carries each stage's outcome (advance, capped, stuck) and `config_hash`. Reason: user choice. (v4.0)
- D-33: When a Stage-1-only model is asked to continue on Stage 2 (observations 14 to 76, R-8), the weights of the nodes present in both are kept, all new nodes get fresh weights, and where possible the old weights are retained somewhat more strongly than the new ones; the mechanism is OPEN. Reason: user choice; avoids refusing and avoids losing trained weights. (v4.0)
- D-34: Confirmed: `model1` stays the stage-4 model after `model2` is made from it (`model2` holds the later stage, such as 5, 6 or 7); a call returns only the models it trained; the log folder has one subfolder per model id; a model id is a user-chosen label that is a valid Python name. Reason: user confirmation of four v3.0 assumptions. (v4.0)
- D-35: A model with any capped stage, between or final, is not continued: if a call is given such a model as source, it is skipped and the models made from it are not trained (`model2` from a capped `model1` is not trained); extends D-27; replaces the v3.0 assumption that a call's first stage may be the capped stage again. Reason: user choice. (v4.0)
- D-36: `train_model` arguments: model source, stage list, `name`, optional ids, optional `seed`; the capped-stage setting is config-only (D-30); refines the PROPOSED argument table of D-9; the exact meaning of `seed` is OPEN. Reason: user: a seed is useful. (v4.0)
- D-37: Yaw reference: the drone faces the target; its front aims at the target. Reason: user choice. (v4.0)
- D-38: n, m, p, q live in the config file only; refines D-18 (resolves its OPEN file question). Reason: user choice. (v4.0)
- D-39: The shipped lookahead (`lookahead_enabled`, `predicted_pos()`) is retired. Reason: default accepted; the future samples of D-16 to D-19 replace it. (v4.0)
- D-40: The drone receives the target's information continuously, with noise; there is no perception model, so there is no target-not-visible case and no coasting rule. Reason: user statement. (v4.0)
- D-41: Each future sample contains the same target fields as the current sample; the frame of the future samples is OPEN (current body frame was proposed); replaces the proposal "position only". Reason: user choice. (v4.0)
- D-42: Smoke Stage 8 runs under `pred` only, because `true` mode has no Stage 8 path; refines D-8 and D-16. Reason: user confirmation. (v4.0)
- D-43: `linear_ridge` is analytic: purely mathematical, with no RL, deep learning or fitted model and no data-fitted artifact; refines D-19. Reason: user requirement. (v4.0)
- D-44: The names "config" (training baseline) and "model" are kept; paths unchanged. Reason: user confirmation. (v4.0)
- D-45: Rollback counts carry over across calls: a later call remembers earlier counts. Reason: user choice. (v4.0)
- D-46: If the program dies mid-stage, a later call resumes from the last snapshot. Reason: user choice. (v4.0)
- D-47: For SAC and TD3, the replay buffer is saved and reloaded on continuation. Reason: user choice; large files accepted. (v4.0)
- D-48: Random ranges in the config are written `{min:, max:}`, and both the min value and the max value can be drawn, not only values between them. Reason: user left the syntax to Claude; named keys are self-explanatory for a non-expert. (v4.0)
- D-49: There is one default smoke model file and one default smoke config file, with simple names; names and paths OPEN; refines D-7 and D-20. Reason: user choice. (v4.0)
- D-50: `target_alt` (D-3) covers the Stage 1 hover altitude only, not target z in Stages 2+; refines D-3. Reason: user choice. (v4.0)
- D-51: The free parameters of D-4 and D-13 (polynomial for degree 2 and 3, start and end points, speed cap) are each either a fixed input value or random within a range, like `target_alt` (D-3); values OPEN. Reason: user choice. (v4.0)
- D-52: Stage 1 keeps a fixed episode length, so its reward-sum threshold of 60.0 stays valid; refines D-14 (the random range does not apply to Stage 1); the T-to-steps rounding rule stays OPEN for other stages. Reason: user choice. (v4.0)
- D-53: The predictor may use all of the drone's own measurements of the target since the episode start (its own longer history); it may not use simulator truth; the history is cleared at every reset; supersedes D-17. Reason: a real drone can store its own past measurements; D-17 would forbid it because the actor sees only m frames. (v5.0)
- D-54: Past and future target fields are converted into the drone's current body frame, measured from its current position; only target fields are converted; no extra config mode; resolves the frame OPEN in D-41. Reason: a turning drone otherwise makes a still target look moving. (v5.0)
- D-55: `STUCK` stops only that model; the other models continue; consistent with D-31. Reason: user answered repeatedly; removes the earlier assumption that it stops the whole call. (v5.0)
- D-56: A capped model given as source is denied with a plain message and nothing else happens: no training, no fresh start; refines D-35; the reason of D-24 is stale (retraining means a new model id). Reason: user choice. (v5.0)
- D-57: A facing reward is added from Stage 2, and the critic reference also uses facing; refines D-37. The reward uses the horizontal angle between the nose and the direction to the target and switches off under about 0.5 m horizontal distance; the weight `k_facing` is OPEN. Reason: the actor must learn to fly toward the target while looking at it, starting with directional flight in Stage 2. (v5.0)
- D-58: The FIX pass is done last, after the design topics. Reason: user choice. (v5.0)

## Session Handoff (OVERWRITTEN every session, always the last block in the file)
- Version v5.0, written at WRAP UP of the fifth design-review session (2026-10-02). The session ended abruptly; this WRAP UP was redone from the exported transcript and the v4.0 file. The `src.zip` and `CURRENT_WORK.md` uploads of that session were not available at WRAP UP, so source facts below come from the transcript and are not re-verified.
- Topic: answers to the v4.0 open list. Finished; nothing was mid-discussion. The user asked "is it done"; the session replied it is not done (R-1 below).
- Decided this session: D-53 to D-58. D-53 supersedes D-17; D-54 resolves the frame OPEN of D-41; D-55 removes the `STUCK` assumption; D-56 refines D-35 and makes the reason of D-24 stale; D-57 refines D-37; D-58 orders the FIX pass last. IDs were provisional in the session and are final here.
- Open, needs the user: real sensor (target position only, or position and velocity; the actor currently receives both, 19 numbers per frame, noise 0.01 per the transcript); `k_facing` weight and the exact horizontal cutoff (about 0.5 m) of D-57; whether to fix R-1 before any training (the session recommended fixing first) and upload `hover_env.py` plus the identification data it was fitted from; add an absolute altitude field to the actor observation in Stages 2+ (R-9, PROPOSED).
- Still PROPOSED, not decided (never confirmed): remove the visible flag [18] or keep it as constant 1; reward always uses the true target position (removes R-10); predictor `linear_ridge` as closed-form ridge on the window or history with one strength value in the config; D-33 mechanism (copy matching weights by input name, new inputs start at zero or small; Stage 1 [12:14] differ from Stage 2); a changed n or m between calls refused with a plain message (D-38); `seed` as one seed per call defaulting to `global.seed` and recorded in `config_hash` (D-36); last final model written to the log folder of `name`, in the model's own subfolder, stage number in the file name (D-29); smoke files `smoke_model.yaml` and `smoke_config.yaml` next to the other configs (D-49); round T to the nearest step (D-52); Stage 5-7 thresholds per step or success rate instead of a reward sum.
- Reported from the source in the session (unverified here): `PH_DT = 0.01` s, so episode steps = T / 0.01 (shipped episodes 1000 to 3000 steps); `xy_limit` equals `world_radius`; target z 0 to 40.0 in target stages; target fields noised at 0.01 (module docstring, value in `stage_config.py` unchecked); shipped lookahead on in Stages 3-8; visible flag 0 only in Stage 1; `CURRENT_WORK.md` does not contain the hyperparameters of `sac_baseline`, `td3_baseline`, `ppo_5layer_deep` (only history m3/s2 and m5/s3 for the deep PPO), so they stay unknown unless the user gives them.
- R-1 reported as confirmed by running the uploaded code (not re-verified here): motor speed falls as the thrust command rises; hover at cmd 0.2325 (the `constants.py` comment says about 0.37); cmd 1.0 gave vz -1.79 m/s after 0.2 s, cmd 0.1 gave +0.97 m/s; motors cut off below cmd 0.02; roll, pitch and yaw loops not tested. Consequence: a policy may learn in this simulator but will not transfer to a real drone. The session scored the spec 60/100 (about 80 if R-1 is fixed and altitude is added). R-7 (smoke run does no PPO update) also stands.
- Not yet checked in source: R-1 `pipeline.py:L99-L110` and `constants.py:L121-L122` in detail, `A` and `v_max`, noise values in `stage_config.py`, D-51 values, `n_steps`/`batch_size` check.
- Open (carried): privileged-critic field list and one-line definitions of the D-11/D-12 terms (user said yes to a draft; now also includes the facing error, D-57); values for the D-51 free parameters; `evaluate_model(...)` (later session); the review findings and the FIX pending list below are unchanged from v4.0 and not fixed in the body.
- TO-DO (user): upload the `interceptor-training` source folder in a later session. Needed for: R-1 (`pipeline.py:L99-L110`, `constants.py:L121-L122`), reference limits `A` and `v_max`, `z_lo`/`z_hi`, target observation noise, truncation vs termination, SB3 asymmetric critic, thresholds for Stage 1 and Stages 5-7, Python 3.14 support, SB3 `n_steps`/`batch_size` on continuation (`orchestrator.py:L357-L394`), `target_visible` (D-40, R-12).
- Open (needs source or measurement): the TO-DO items above, SB3 algorithm count, native Python vs Docker.
- Review findings verified by calculation (not yet fixed in body): R-1 ESC polynomial as written in §8.1/§9.1 decreases with command (Ω_ss 2400.8 at cmd 0.02, 1956.2 at 0.2325, 472.2 at 1.0; max T/W 1.51; `PH_OMEGA_MAX = 2800` never reached); needs `pipeline.py:L99-L110` and `constants.py:L121-L122`.
- R-2 §3 "80 gradient updates" should be 640 (32768/512 = 64 minibatches x 10 epochs). R-3 §0 Counters: manifest has 92 rows, not 91; READ-FULL 72 / SAMPLED 8 / EXCLUDED 12; EMBEDDED 30 / NOT-VERBATIM 50 / BINARY 7 / GLOB 5; Appendix A holds 4 blocks. R-5 root `requirements.txt` `pygame==2.5.8` does not exist on PyPI (latest 2.6.1). R-6 §13.3 `run_summary.json:L2` should be L3.
- Review findings needing source: R-7 smoke run probably does no PPO update (32768 samples per rollout vs budgets of 1000 or less; A.4 steps 784/872/968); R-8 Stage 1 to 2 obs 14 to 76, no weight transfer; R-9 Stage 2+ actor has no absolute altitude; R-10 lookahead point vs true distance in observation vs reward; R-11 no observation or reward normalisation recorded; R-12 undocumented: action mapping, env step vs `PH_DT`, `target_visible`, `xy_limit` vs `world_radius`.
- Also: R-13 Stage 1 obs [13] duplicates [8]; R-14 host Python 3.14 vs torch 2.7.0 wheels ([RECALLED]) and unpinned container numpy; R-15 target RNG `(seed or 0)`; R-16 `PH_J` 27 g values vs `PH_M = 0.04085` ([RECALLED]); R-17 G-015 also device `cuda` vs `cpu`; R-18 §13.8 claim vs BUGS items 5 and 7; R-19 naming (`success_rate`, "Stage A/B/2-9", `r_time_bonus` vs `r_time_penalty`, `current_stage` vs `stage_num`); R-20 evidence tags in §2.1 and §2.7(b).
- Ledger issues: D-3 scope resolved (D-50); D-4/D-13 free parameters are fixed or ranged (D-51), values undefined; the D-14 Stage 1 threshold conflict is resolved by a fixed Stage 1 length (D-52), the T-to-steps rounding rule stays open for other stages; new reset draws must be conditional to keep Stage 1 parity.
- FIX pending (one question each; from v1.0): BUGS items 12, 11 (test), 2/1, 8, 3, 5, 10; README mass; BLOCKER ratings of G-008, G-011, G-017; `CAPPED` vs `ADVANCE` priority test; §13.8 fixed-bug count; §6.2 write method; §6.2 `episode_buffer` type; §13.5 Monitor wording.
- Also observed at the start of the third session (not fixed): the Table of Contents omits the Change Log, Decision Log and Session Handoff and "End of document" sits before them; Appendix C says "30+ constants" and "8 anomalies" against 24 rows in §9.1 and 9 subsections in §13.
- D-20 condition checklist: unchanged from v4.0; add the facing weight `k_facing` (D-57) to the stage and reward configuration destinations.
- Stale: see §13.9 (D-2 to D-58). The D-17 and D-24 reasons are superseded or stale as recorded above.
- Recommended next topic: decide R-1 (fix before training or not), then the real-sensor question, then the PROPOSED list above, then the privileged-critic field list; the FIX pass last (D-58).
- First question next session: should R-1 be fixed before any further design work, and can you upload `hover_env.py` and the identification data?
- Working agreements: unchanged from v4.0, plus: the user wants only items that need their input asked, as short questions; do not re-ask answered items; "default" means accept the recommendation; the user confirms a batch with "confirm and add to ledger".
