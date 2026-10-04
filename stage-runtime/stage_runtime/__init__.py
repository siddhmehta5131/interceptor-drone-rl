"""Stage-mode runtime: offline per-stage demos rendered to MP4.

The runtime is deliberately *separate* from the training image.  It imports the
training package read-only (env + physics + prediction + checkpoint layout) but
never trains anything and never writes to the run directory.

Two control sources, selected by the per-stage JSON config:

``instructions``
    A fixed instruction set.  Either an open-loop action script
    (``open_loop``) or a deterministic proportional guidance law
    (``guidance``).  No torch / stable-baselines3 needed.

``model``
    A checkpoint uploaded (bind-mounted) for that stage.  The stage's
    observations are fed to the policy and its response drives the drone.

Output: one MP4 per stage, built from ``segments`` consecutive ~3 s segments.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

__version__ = "1.0.0"

# ---------------------------------------------------------------------------
# Locate the training package and put it on sys.path.
#
# Resolution order:
#   1. $INTERCEPTOR_ROOT                       (set by the stage image/compose)
#   2. ../interceptor-training                 (repo checkout layout)
#   3. ../../interceptor-training              (src/stage_runtime layout)
#   4. /opt/interceptor-training               (baked into the stage image)
# ---------------------------------------------------------------------------

_MARKER = Path("src") / "envs" / "base_env.py"


def _looks_like_training_root(path: Path) -> bool:
    return (path / _MARKER).is_file()


def resolve_training_root() -> Path:
    """Return the directory that must be on ``sys.path`` to import ``src.*``."""
    candidates: list[Path] = []
    env = os.environ.get("INTERCEPTOR_ROOT", "").strip()
    if env:
        candidates.append(Path(env).expanduser())
    here = Path(__file__).resolve().parent
    for base in (here.parent.parent, here.parent.parent.parent):
        candidates.append(base)                              # package at root
        candidates.append(base / "interceptor-training")     # sibling package
    candidates += [
        Path("/opt/interceptor-training"),
        Path("/repo/interceptor-training"),
        Path("/repo"),
    ]
    for cand in candidates:
        if _looks_like_training_root(cand):
            return cand.resolve()
    raise RuntimeError(
        "cannot locate the interceptor-training package; set INTERCEPTOR_ROOT "
        "to the directory that contains src/envs/base_env.py"
    )


def bootstrap_path() -> Path:
    """Idempotently add the training package root to ``sys.path``."""
    root = resolve_training_root()
    text = str(root)
    if text not in sys.path:
        sys.path.insert(0, text)
    return root


TRAINING_ROOT = bootstrap_path()

__all__ = ["TRAINING_ROOT", "bootstrap_path", "resolve_training_root", "__version__"]
