"""smoke_test.py -- fast validation suite (plan §16).

Runs a sequence of dependency-light checks covering the guidance law, the
environment, observations, reward, target generation, the oracle and an
end-to-end APN-only intercept.  Prints a PASS/FAIL report and exits non-zero if
anything fails.  No GPU and no training required.

Run::

    python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Callable, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

_RESULTS: List[Tuple[str, bool, str]] = []


def _check(name: str, fn: Callable[[], None]) -> None:
    try:
        fn()
        _RESULTS.append((name, True, ""))
    except Exception as exc:  # noqa: BLE001
        _RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))
        traceback.print_exc()


# --------------------------------------------------------------------- checks


def check_imports() -> None:
    import src.api  # noqa: F401
    import src.envs.intercept_env  # noqa: F401
    import src.envs.obs_builder  # noqa: F401
    import src.envs.reward  # noqa: F401
    import src.guidance.apn  # noqa: F401
    import src.guidance.kinematic_model  # noqa: F401
    import src.guidance.optimal_trajectory  # noqa: F401
    import src.prediction.base  # noqa: F401
    import src.training.orchestrator  # noqa: F401
    import visualiser  # noqa: F401


def check_apn_head_on() -> None:
    from src.guidance.apn import compute_apn_command

    # Head-on, target static straight ahead -> APN command ~ 0.
    a_cmd, info = compute_apn_command(
        p_interceptor=np.array([0.0, 0.0, 0.0]),
        v_interceptor=np.array([20.0, 0.0, 0.0]),
        p_target=np.array([50.0, 0.0, 0.0]),
        v_target=np.zeros(3),
        a_target=np.zeros(3),
        N_prime=4.0,
    )
    assert a_cmd.shape == (3,)
    assert np.linalg.norm(a_cmd) < 1e-6, f"expected ~0, got {a_cmd}"
    assert info["closing_vel"] > 0.0


def check_apn_crossing() -> None:
    from src.guidance.apn import compute_apn_command

    # Target crossing laterally -> APN command points along +y (lead the target).
    a_cmd, _ = compute_apn_command(
        p_interceptor=np.array([0.0, 0.0, 0.0]),
        v_interceptor=np.array([20.0, 0.0, 0.0]),
        p_target=np.array([40.0, 0.0, 0.0]),
        v_target=np.array([0.0, 8.0, 0.0]),
        a_target=np.zeros(3),
        N_prime=4.0,
    )
    assert a_cmd[1] > 0.0, f"expected +y lead, got {a_cmd}"


def check_kinematic() -> None:
    from src.guidance.kinematic_model import KinematicModel

    km = KinematicModel(v_max=50.0, a_max=15.0, omega_max=3.0, dt=0.02)
    state = km.reset(position=np.zeros(3), velocity=np.zeros(3))
    for _ in range(50):  # 1 second at a_max forward
        state = km.step(state, np.array([15.0, 0.0, 0.0]))
    # v = a t = 15, x = 0.5 a t^2 = 7.5
    assert abs(np.linalg.norm(state.velocity) - 15.0) < 1.0, state.velocity
    assert abs(state.position[0] - 7.5) < 0.5, state.position


def check_stage_config() -> None:
    from src.envs.stage_config import STAGES

    assert set(STAGES) == {"stage_1", "stage_2", "stage_3"}, STAGES.keys()
    for sid, stage in STAGES.items():
        assert stage.obs.actor_dim == 85, (sid, stage.obs.actor_dim)
        assert 0.0 < stage.success_rate <= 1.0


def check_obs_dims() -> None:
    from src.utils.config_loader import load_config
    from src.envs.intercept_env import InterceptEnv

    cfg = load_config("configs/config.yaml")
    for n in cfg.stage_numbers():
        stage = cfg.stage(n)
        env = InterceptEnv(stage, obs_config=stage.obs)
        assert env.observation_space.shape == (101,), (n, env.observation_space.shape)
        assert env.action_space.shape == (3,)
        env.close()


def check_reward() -> None:
    from src.envs.reward import RewardFunction
    from src.envs.stage_config import RewardConfig

    rf = RewardFunction(RewardConfig(), v_max=50.0, a_max=15.0, kill_radius=1.0)
    r_kill, term, _ = rf.compute(
        distance=0.5, prev_distance=5.0, closing_vel=20.0, LOS_rate_norm=0.0,
        a_total_norm=1.0, a_rl_norm=0.5, action=np.zeros(3), prev_action=np.zeros(3),
        ZEM=0.0, step_count=10, max_steps=200, killed=True, out_of_bounds=False,
    )
    assert term and r_kill > 20.0, r_kill
    r_oob, term2, _ = rf.compute(
        distance=10.0, prev_distance=9.0, closing_vel=0.0, LOS_rate_norm=0.0,
        a_total_norm=0.0, a_rl_norm=0.0, action=np.zeros(3), prev_action=np.zeros(3),
        ZEM=0.0, step_count=10, max_steps=200, killed=False, out_of_bounds=True,
    )
    assert term2 and r_oob < 0.0, r_oob


def check_target_generator() -> None:
    from src.envs.stage_config import STAGES
    from src.envs.target_generator import TargetGenerator

    for sid, stage in STAGES.items():
        gen = TargetGenerator(stage, np.random.default_rng(0))
        gen.reset(np.zeros(3), episode_length_s=8.0)
        for _ in range(20):
            gen.step(0.02, np.zeros(3))
        assert np.all(np.isfinite(gen.pos)), sid
        assert np.all(np.isfinite(gen.vel)), sid
        pos_future, vel_future, acc_future = gen.state_at(gen.elapsed + 1.0)
        assert np.all(np.isfinite(pos_future)) and np.all(np.isfinite(acc_future)), sid


def check_oracle() -> None:
    from src.guidance.optimal_trajectory import compute_optimal_intercept
    from src.envs.target_generator import PolynomialTargetPath

    path = PolynomialTargetPath(
        start=np.zeros(3), end=np.array([60.0, 0.0, 0.0]),
        duration=6.0, order=1, rng=np.random.default_rng(0),
    )
    result = compute_optimal_intercept(
        p_interceptor=np.zeros(3), v_interceptor=np.array([15.0, 0.0, 0.0]),
        target_path=path, v_max=50.0, a_max=15.0, dt=0.02,
    )
    assert np.isfinite(result["optimal_time"]) and result["optimal_time"] > 0.0
    assert np.all(np.isfinite(result["optimal_accel"]))


def check_history_stacking() -> None:
    from src.envs.obs_builder import ObsBuilder
    from src.envs.stage_config import ObsStageConfig

    builder = ObsBuilder(ObsStageConfig())
    builder.reset()
    dims = None
    for i in range(4):
        pos = np.array([float(i), 0.0, 0.0])
        obs, _ = builder.build(
            interceptor_pos=pos, interceptor_vel=np.zeros(3),
            target_pos=pos + np.array([10.0, 0.0, 0.0]), target_vel=np.zeros(3),
            target_acc=np.zeros(3), target_visible=True, dt=0.02, time_remaining=3.0,
            optimal_accel=np.zeros(3), optimal_vel=np.zeros(3),
            target_true_future_pos=pos, target_true_future_vel=np.zeros(3),
            target_true_accel=np.zeros(3),
        )
        dims = obs.shape
    assert dims == (85,), dims


def check_apn_only_intercept() -> None:
    from src.envs.stage_config import STAGES
    from src.envs.intercept_env import InterceptEnv

    stage = STAGES["stage_1"]
    kills = 0
    for seed in range(5):
        env = InterceptEnv(stage, obs_config=stage.obs, seed=seed)
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            obs, _r, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
            done = terminated or truncated
        kills += int(info["killed"])
        env.close()
    assert kills >= 4, f"APN-only stage-1 kills={kills}/5"


# ----------------------------------------------------------------------- main


def main() -> int:
    checks = [
        ("imports", check_imports),
        ("apn head-on", check_apn_head_on),
        ("apn crossing", check_apn_crossing),
        ("kinematic model", check_kinematic),
        ("stage config", check_stage_config),
        ("observation dims", check_obs_dims),
        ("reward terms", check_reward),
        ("target generator", check_target_generator),
        ("optimal trajectory", check_oracle),
        ("history stacking", check_history_stacking),
        ("APN-only intercept", check_apn_only_intercept),
    ]
    for name, fn in checks:
        _check(name, fn)

    print("\n=== smoke test report ===")
    failed = 0
    for name, ok, msg in _RESULTS:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}" + (f"  -- {msg}" if msg else ""))
        failed += int(not ok)
    print(f"\n{len(_RESULTS) - failed}/{len(_RESULTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
