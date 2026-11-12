"""download_models.py -- fetch / generate the drone OBJ meshes (plan §14.2).

The visualiser wants ``models/interceptor_drone.obj`` and
``models/target_drone.obj``.  Free quadcopter meshes can be downloaded from
Free3D / GrabCAD / Sketchfab, but those pages are not scriptable in general, so
this helper:

1. leaves any mesh that already exists untouched;
2. optionally tries a small list of direct OBJ URLs;
3. otherwise generates a stylised placeholder quadcopter with ``trimesh`` and
   writes it as OBJ so the visualiser works out of the box.

Run::

    python scripts/download_models.py
    python scripts/download_models.py --force
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path("models")
INTERCEPTOR = "interceptor_drone.obj"
TARGET = "target_drone.obj"

# Direct OBJ URLs (best-effort; may be unavailable/offline).
_CANDIDATES = {
    INTERCEPTOR: [],
    TARGET: [],
}

README = """# Drone models

`interceptor_drone.obj` and `target_drone.obj` are loaded by the PyVista
visualiser (see `visualiser/scene_builder.py`).

Source options (pick free / CC-licensed assets):

- Free3D      https://free3d.com/3d-models/quadcopter
- GrabCAD     https://grabcad.com/library?query=quadcopter
- Sketchfab   https://sketchfab.com/tags/drone

If a file is missing, `scripts/download_models.py` writes a **placeholder**
quadcopter built from primitives so the visualiser still runs.  Drop a real
OBJ at the same path to replace it (meshes are auto-scaled by
`models.scale` in `configs/visualiser.yaml`).

## Licence

Real downloaded meshes keep their own licence -- check the source page before
redistribution.  The placeholder geometry generated here is public domain.
"""


def _placeholder(size: float = 1.5):
    """A small stylised quadcopter built from trimesh primitives."""
    import trimesh

    body = trimesh.creation.box(extents=(size, size * 0.5, size * 0.25))
    parts = [body]
    arm = size * 0.7
    for sx in (-1, 1):
        for sy in (-1, 1):
            rotor = trimesh.creation.cylinder(radius=size * 0.28, height=size * 0.06, sections=24)
            rotor.apply_translation((sx * arm, sy * arm, 0.0))
            parts.append(rotor)
    mesh = trimesh.util.concatenate(parts)
    return mesh


def _try_download(url: str, dest: Path) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:  # noqa: S310
            data = resp.read()
        if not data:
            return False
        dest.write_bytes(data)
        return True
    except Exception:
        return False


def _ensure(name: str, dest: Path, force: bool) -> str:
    if dest.exists() and not force:
        return f"exists  {dest}"
    for url in _CANDIDATES.get(name, []):
        if _try_download(url, dest):
            return f"downloaded {dest}  <- {url}"
    try:
        import trimesh  # noqa: F401
    except Exception:
        return f"SKIPPED {dest} (trimesh not installed and no download available)"
    mesh = _placeholder()
    mesh.export(dest)
    return f"placeholder {dest}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fetch or generate drone OBJ meshes.")
    parser.add_argument("--force", action="store_true", help="overwrite existing files")
    parser.add_argument("--out-dir", default=str(MODELS_DIR))
    args = parser.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name in (INTERCEPTOR, TARGET):
        print(_ensure(name, out / name, args.force))

    readme = out / "README.md"
    if not readme.exists() or args.force:
        readme.write_text(README, encoding="utf-8")
        print(f"wrote   {readme}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
