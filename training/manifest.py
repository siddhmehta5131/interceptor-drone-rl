"""Checkpoint manifest for the GitHub Release (plan P7.5).

The manifest is the single index of released checkpoints: run id, file,
SHA-256, git commit and configuration hash.  ``verify`` re-hashes every file
and optionally loads the model with Stable-Baselines3.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

__all__ = ["sha256", "build_manifest", "verify_manifest"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(run_root: Path, output: Optional[Path] = None) -> dict:
    run_root = Path(run_root)
    entries = []
    for run_dir in sorted(p for p in run_root.iterdir() if p.is_dir()):
        done = run_dir / "DONE"
        ckpt = run_dir / "checkpoint.zip"
        if not done.exists() or not ckpt.exists():
            continue
        summary = json.loads(done.read_text(encoding="utf-8"))
        commit = (run_dir / "git.txt").read_text(encoding="utf-8").strip() if (
            run_dir / "git.txt").exists() else "unknown"
        config_hash = (run_dir / "config.sha256").read_text(encoding="utf-8").strip() if (
            run_dir / "config.sha256").exists() else ""
        entries.append({
            "run_id": run_dir.name,
            "file": str(ckpt.relative_to(run_root)).replace("\\", "/"),
            "sha256": sha256(ckpt),
            "commit": commit,
            "config_sha256": config_hash,
            "algo": summary.get("algo"),
            "task": summary.get("task"),
            "seed": summary.get("seed"),
            "num_timesteps": summary.get("num_timesteps"),
        })
    manifest = {"version": 1, "runs": entries}
    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def verify_manifest(manifest: dict, run_root: Path, load: bool = False) -> list[str]:
    """Return a list of problems; an empty list means the manifest is valid."""
    problems: list[str] = []
    for entry in manifest.get("runs", []):
        path = Path(run_root) / entry["file"]
        if not path.exists():
            problems.append(f"{entry['run_id']}: missing {path}")
            continue
        if sha256(path) != entry["sha256"]:
            problems.append(f"{entry['run_id']}: sha256 mismatch")
            continue
        if load:
            try:
                from training.runner import load_run_model
                load_run_model(path.parent)
            except Exception as exc:  # pragma: no cover - defensive
                problems.append(f"{entry['run_id']}: load failed: {exc}")
    return problems
