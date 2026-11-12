"""cli.py -- command-line interface for the 3D visualiser (plan §14.5).

Usage::

    python -m visualiser --model apn --stage 2 --output video.mp4
    python -m visualiser --model data/runs/apn_rl_v1/ppo_interceptor/final.zip \\
        --stage 3 --output-dir videos/ --fps 60

The ``--model`` argument is either ``apn`` (APN-only, no RL residual), a run
name under ``data/runs``, or a path to a saved SB3 ``.zip`` policy.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .animator import InterceptAnimator
from .scene_builder import have_pyvista, load_visualiser_config

__all__ = ["generate_trajectory", "main"]


def _load_stage_configs(config_path: str, model_yaml: str):
    from src.utils.config_loader import load_config

    cfg = load_config(config_path, model_source=model_yaml)
    return cfg


def _resolve_model_path(model: str) -> Optional[str]:
    """Return a ``.zip`` path when ``model`` names a trained policy, else None."""
    if model in ("", "apn", "none", "apn_only"):
        return None
    p = Path(model)
    if p.exists() and p.suffix == ".zip":
        return str(p)
    # a run name under data/runs
    for candidate in Path("data/runs").glob(f"{model}/**/*.zip"):
        return str(candidate)
    return None


def generate_trajectory(
    stage: int,
    *,
    model_path: Optional[str] = None,
    config_path: str = "configs/config.yaml",
    model_yaml: str = "configs/model.yaml",
    seed: int = 0,
    deterministic: bool = True,
) -> Dict[str, Any]:
    """Roll one episode out and return ``{"trajectory": [...], "kill_radius": r}``."""
    from src.envs.intercept_env import InterceptEnv, InterceptorConfig, APNConfig

    cfg = _load_stage_configs(config_path, model_yaml)
    stage_cfg = cfg.stage(stage)

    ic = cfg.interceptor
    env = InterceptEnv(
        stage_cfg,
        obs_config=stage_cfg.obs,
        apn_config=APNConfig(N_prime=stage_cfg.apn_N_prime, t_go_min=stage_cfg.apn_t_go_min, t_go_max=cfg.apn.t_go_max),
        interceptor_config=InterceptorConfig(
            v_max=ic.v_max, a_max=ic.a_max, omega_max=ic.omega_max,
            speed_ratio=ic.speed_ratio, dt=ic.dt,
        ),
    )

    model = None
    if model_path:
        from stable_baselines3 import PPO

        model = PPO.load(model_path, device=cfg.device)

    obs, info = env.reset(seed=seed)
    trajectory: List[Dict[str, Any]] = [_initial_frame(env, info)]
    done = False
    while not done:
        if model is not None:
            action, _ = model.predict(obs, deterministic=deterministic)
        else:
            action = np.zeros(env.action_space.shape, dtype=np.float32)
        obs, _reward, terminated, truncated, info = env.step(action)
        trajectory.append(env.get_trajectory_log()[-1])
        done = bool(terminated or truncated)

    kill_radius = float(getattr(env, "kill_radius", stage_cfg.intercept.kill_radius))
    env.close()
    return {"trajectory": trajectory, "kill_radius": kill_radius, "stage": stage}


def _initial_frame(env, info: Dict[str, Any]) -> Dict[str, Any]:
    state = getattr(env, "state", None)
    target = getattr(env, "target", None)
    return {
        "step": 0,
        "interceptor_pos": np.asarray(getattr(state, "position", info.get("interceptor_pos", np.zeros(3))), dtype=np.float64),
        "interceptor_vel": np.asarray(getattr(state, "velocity", np.zeros(3)), dtype=np.float64),
        "interceptor_quat": np.asarray(getattr(state, "orientation", np.array([1.0, 0.0, 0.0, 0.0])), dtype=np.float64),
        "target_pos": np.asarray(getattr(target, "pos", np.zeros(3)), dtype=np.float64),
        "target_vel": np.asarray(getattr(target, "vel", np.zeros(3)), dtype=np.float64),
        "a_apn": np.zeros(3),
        "a_rl": np.zeros(3),
        "a_total": np.zeros(3),
        "velocity_output": np.zeros(6),
        "distance": float(info.get("distance", 0.0)),
        "closing_vel": float(info.get("closing_vel", 0.0)),
        "LOS_rate": float(info.get("LOS_rate", 0.0)),
        "t_go": float(info.get("t_go", 0.0)),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m visualiser",
        description="Render APN+RL interceptor episodes to MP4 with PyVista.",
    )
    parser.add_argument("--model", default="apn", help="'apn', a run name, or a .zip path")
    parser.add_argument("--stage", default="2", help="stage number, or 'all'")
    parser.add_argument("--output", default=None, help="output .mp4 (single stage)")
    parser.add_argument("--output-dir", default="data/videos", help="output directory")
    parser.add_argument("--config", default="configs/config.yaml", help="training config YAML")
    parser.add_argument("--model-yaml", default="configs/model.yaml", help="model definition YAML")
    parser.add_argument("--visualiser-config", default="configs/visualiser.yaml")
    parser.add_argument("--resolution", default=None, help="output WxH, e.g. 1280x720 (overrides config)")
    parser.add_argument(
        "--camera", default=None,
        help="chase|top-down|fixed|los|orbit|static (static = fixed framing, never moves)",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--quality", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true", help="sample actions (default deterministic)")
    args = parser.parse_args(argv)

    if not have_pyvista():
        raise SystemExit("pyvista is not installed: pip install pyvista trimesh")

    stages = _parse_stages(args.stage)
    model_path = _resolve_model_path(args.model)
    model_label = Path(model_path).stem if model_path else "apn"

    for stage in stages:
        rollout = generate_trajectory(
            stage,
            model_path=model_path,
            config_path=args.config,
            model_yaml=args.model_yaml,
            seed=args.seed,
            deterministic=not args.stochastic,
        )
        if args.output and len(stages) == 1:
            out = args.output
        else:
            out = str(Path(args.output_dir) / f"{model_label}_stage{stage}.mp4")

        vis_config = load_visualiser_config(args.visualiser_config)
        if args.resolution:
            try:
                w, h = (int(v) for v in str(args.resolution).lower().replace("x", ",").split(",")[:2])
                vis_config["resolution"] = [w, h]
            except Exception:
                raise SystemExit(f"--resolution: expected WxH, got {args.resolution!r}")

        animator = InterceptAnimator(
            model_name=model_label,
            stage=stage,
            trajectory_data=rollout["trajectory"],
            config_path=args.visualiser_config,
            config=vis_config,
            camera=args.camera,
            kill_radius=rollout["kill_radius"],
        )
        path = animator.render(out, fps=args.fps, quality=args.quality)
        print(f"rendered stage {stage}: {path}  ({len(rollout['trajectory'])} frames)")
    return 0


def _parse_stages(text: str) -> List[int]:
    if str(text).strip().lower() == "all":
        return [1, 2, 3]
    return [int(c) for c in str(text).replace(" ", "").split(",") if c]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
