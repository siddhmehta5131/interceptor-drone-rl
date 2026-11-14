# Meshes (P10.5)

| Mesh | Used for | Source | Licence |
|---|---|---|---|
| `interceptor_drone.obj` | quad interceptor | **Unrecorded.** Origin not documented when it was added (finding F21). | **Unknown: do not redistribute until the source page is identified, or replace it.** |
| `target_drone.obj` | quad target | **Unrecorded** (F21). | **Unknown**, as above. |
| `primitive:fixed_wing` | fixed-wing interceptor | Generated in code by `visualiser/scene_builder.py::primitive_mesh` (cylinder, cone, box wing/tail). | Project code, no third-party asset. |
| `primitive:missile` | missile interceptor | Same function (cylinder body, cone nose, four box fins). | Project code, no third-party asset. |
| placeholder quad | fallback when an OBJ is missing | `scene_builder.placeholder_drone` (box body, four cylinder rotors). | Project code. |

The two OBJ files are the only items with an open licence question. Resolve it
by either (a) finding the original download page and recording its licence here,
or (b) deleting the two files and pointing `configs/visualiser.yaml` at
`primitive:`-style meshes (the renderer then falls back to `placeholder_drone`
for the quad). Videos in `results/videos/` use the OBJ meshes for the quad
until this is decided.
