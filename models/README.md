# Drone models

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
