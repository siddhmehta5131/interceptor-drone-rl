"""P1.7/P1.8 parity: the unified z-up environment replays notebook fixtures.

The fixture environment is NED.  The consolidated world is z-up; the two are
related by ``T = diag(1, -1, -1)``.  Positions, velocities, accelerations and
the body-rate vector transform with ``T``; scalars (range, closing speed,
t_go, reward, visible) are unchanged.  This is the same convention already
used by the manoeuvre parity test.

P5.9 replaced the notebook reward with the potential-based guidance reward, so
this replay passes ``legacy_reward=True``; observations, trajectories and
velocity outputs are still compared bit-for-bit against the notebook.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from interceptor.guidance.env import UnifiedInterceptEnv
from interceptor.guidance.seeker import SensorNoiseConfig
from interceptor.guidance.maneuvers import ManeuverType, PlatformType
from interceptor.guidance.platform import InterceptorLimits

FIX = Path(__file__).resolve().parents[1] / "fixtures"
T = np.array([1.0, -1.0, -1.0])
TOL = 1e-9
MODES = {"apn_raw": "apn_raw", "apn_filtered": "apn_filtered"}


def _obs_to_zup(obs: np.ndarray) -> np.ndarray:
    """Transform every vector-valued field of the 101-dim observation."""
    out = np.array(obs, dtype=np.float64, copy=True)
    for base in (0, 16, 32, 48):
        out[..., base + 0:base + 3] *= T
        out[..., base + 3:base + 6] *= T
        out[..., base + 6:base + 9] *= T
        out[..., base + 11:base + 14] *= T
    for base in (64, 71, 78):
        out[..., base + 0:base + 3] *= T
        out[..., base + 3:base + 6] *= T
    out[..., 85 + 1:85 + 4] *= T
    out[..., 85 + 4:85 + 7] *= T
    out[..., 85 + 7:85 + 10] *= T
    out[..., 85 + 10:85 + 13] *= T
    out[..., 85 + 13:85 + 16] *= T
    return out


@pytest.mark.parametrize("platform", list(PlatformType),
                         ids=[p.value for p in PlatformType])
@pytest.mark.parametrize("mode", list(MODES),
                         ids=list(MODES))
@pytest.mark.parametrize("seed", [101, 202, 303])
def test_notebook_environment_replay(platform, mode, seed):
    fx = np.load(FIX / f"nbenv_{platform.value}_{mode}_s{seed}.npz")
    env = UnifiedInterceptEnv(
        platform=platform, mode=MODES[mode], target_set="evasive",
        noise_scale=float(fx["noise_scale"]),
        fixed_maneuver=ManeuverType(str(fx["maneuver"])), seed=seed,
        legacy_reward=True, limits=InterceptorLimits.from_preset(platform),
        noise_config=SensorNoiseConfig())  # notebook-v2 noise, not configs/seeker.yaml
    obs, _ = env.reset()
    obs_hist, rewards, vout = [obs.copy()], [], []
    done = False
    while not done:
        obs, r, term, trunc, _ = env.step(np.zeros(3, dtype=np.float32))
        obs_hist.append(obs.copy())
        rewards.append(r)
        vout.append(env.last_info["velocity_output"])
        done = term or trunc

    assert np.max(np.abs(np.asarray(env.traj_M) * T - fx["traj_M"])) <= TOL * 1e3
    assert np.max(np.abs(np.asarray(env.traj_T) * T - fx["traj_T"])) <= TOL * 1e4
    assert np.max(np.abs(np.asarray(env.traj_T_noisy) * T - fx["traj_T_noisy"])) <= TOL * 1e4
    got_v = np.asarray(vout).copy()
    got_v[:, :3] *= T
    got_v[:, 3:] *= T
    assert np.max(np.abs(got_v - fx["velocity_output"])) <= TOL * 1e4
    got_obs = np.asarray(obs_hist, dtype=np.float64)
    want_obs = _obs_to_zup(fx["obs"])
    assert np.max(np.abs(got_obs - want_obs)) <= 1e-6
    assert np.max(np.abs(np.asarray(rewards) - fx["reward"])) <= 1e-8
    assert env.step_count == int(fx["steps"])
    assert bool(env.last_info["killed"]) == bool(fx["killed"])
