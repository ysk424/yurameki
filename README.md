# Yurameki2 0.2.1

Yurameki is a Blender extension for simulating VR-character-style long straight
hair with NVIDIA Warp. Each strand is solved as a Stable Cosserat elastic rod:
per-segment quaternion frames with stretch/shear and bend/twist energies. The
stretch/shear energy keeps every segment at rest length intrinsically, which
replaces the earlier "XPBD solve, then reconnect the rod by FK" pipeline.

Tokoya owns planting, cutting, reset, and `Settle Hair Back`. Yurameki starts
from that already-groomed Curves object and simulates motion only.

Tokoya and Yurameki are designed as one workflow: Tokoya generates and prepares
the hair, then Yurameki simulates it.

## Installation

- Blender 5.1 or newer. Current testing is on Blender 5.2 beta.
- Windows x64.
- NVIDIA CUDA-capable GPU.
- Blender's Python environment must be able to import NVIDIA `warp-lang`.

Install the release ZIP through Blender's extension/add-on installer:

```text
dist/yurameki2-0.2.1.zip
```

## Direction

- Use NVIDIA Warp as the compute backend.
- Use the existing Blender Curves joints and their current segment lengths.
- Do not resample hair into 1 cm cylinder chains.
- Keep the first `Root Locked Points` joints of every strand constrained to the
  evaluated Curves pose. The default is `3`.
- Let the remaining joints move as a Stable Cosserat elastic rod under gravity,
  damping, stretch/shear and bend/twist rod energies, and Warp Mesh collision.
  The rod preserves segment length, so no FK reconnection is required.
- Keep automatic substeps based on constrained-point motion plus estimated
  gravity/velocity motion of free joints.
- Use a filled Body proxy when needed; keep Clothes as an evaluated mesh so
  Alembic / Mesh Sequence Cache clothing can update per frame.

## UI

### Yurameki Assistant

`Yurameki Assistant` is a session-based conversation for tuning the current
simulation settings. Register an OpenAI API key once; on Windows it is stored as
a generic credential named `Yurameki/OpenAI API Key` in Credential Manager and
is never written to the Blend file or Yurameki configuration. Enter requests
such as `make it softer but settle sooner`, then press `Send`. The assistant
uses `gpt-5.4-nano` and applies only validated, allow-listed Warp and Collision
parameters. API use is billed by OpenAI according to the account behind the key.

The conversation and setting snapshots live only for the current Blender
session. `Previous` restores the state before the last assistant change, and
phrases such as `前の方がよかった` do the same without making an API request.
`New` clears the conversation and its undo history. Requests run in the
background so Blender's interface remains responsive. Current parameter values,
their documented ranges, and recent conversation text are sent to OpenAI; mesh
geometry, object names, and the API key are not included in the request body.

The panel has three command buttons:

- `Check`: validates Hair, Body, optional Clothes, builds/reuses the Body proxy,
  and verifies Warp CUDA / Mesh initialization.
- `Simulate`: runs the Warp joint-chain simulation for the frame range and
  stores every simulated frame in the Yurameki runtime cache by default. Each
  completed frame is shown in the viewport during the run, and on success
  Blender is left on the end frame so the simulated result is visible.
- `Bake Cache`: converts the current Yurameki runtime cache to Curves position
  keyframes when you are ready to commit the result.

Hair, Body, and Clothes are selected by eyedropper fields.

`Start Frame` is the unchanged initial hair state. Simulation and correction
start at `Start Frame + 1` and continue through `End Frame`, inclusive. `End
Frame` must be greater than `Start Frame`.

All extension source labels are written in English. Yurameki does not ship or
register its own translation dictionary. Blender may localize standard UI terms,
including parameter names, according to its Interface Translation settings;
Japanese UI text therefore comes from Blender, not from Yurameki.

`Output: Runtime Cache` is the default. It keeps simulation playback in a cache
instead of immediately creating millions of Curves `position` F-Curves.
`Position Keyframes` keeps the direct F-Curve bake path for final Blender-native
animation output. `Final Preview` is only for a static look at the last simulated
frame; it overwrites the Curves data, so every timeline frame will show that
final shape. When you do bake keyframes, Yurameki writes the Curves `position`
F-Curves directly in bulk instead of calling Blender's per-point keyframe
operator.

`KEEP LENGTH` is a redundant safety pass. The elastic rod already preserves
segment length, so the reconnection is a near-identity. The flag stays default-on
only because the CPU Body FK hard repair -- which pushes strands that would
otherwise stay inside the Body proxy back out along their rest lengths -- reuses
the same frame-1 rest lengths. Turning it off runs the pure rod with the Body
seed guard still active.

## Tuning

Every control has a tooltip in the N-panel. The three that shape hair motion are:

- `Bend Stiffness log10`: rod bend/twist stiffness. Higher is stiffer and
  straighter; lower is floppier and whips more.
- `Particle Mass g`: momentum. Lower is lighter and snappier and stores less
  energy (less overshoot); higher is heavier and swings more.
- `Damping`: how quickly motion settles.

Stretch is fixed -- the rod is inextensible -- so there is no stretch knob.
Collision, substep, and workflow controls are unchanged from the Warp path.

## Warp Requirement

The Blender Python environment must be able to import `warp`.
If `Check` reports `Warp import failed`, install NVIDIA `warp-lang` for the
Python used by Blender.

## Collision Notes

Body uses the Yurameki filled proxy, because closed body collision needs stable
inside/outside queries and fewer holes. Clothes are not proxied by default. This
keeps Marvelous Designer Alembic / Mesh Sequence Cache clothes evaluated at the
current frame.
The proxy preserves the complete source mesh, including ears, and caps existing
boundary loops such as eye openings with explicit triangulated cap vertices
instead of relying on large single hole-fill faces. It contains no
model-specific absolute-coordinate face removal.

Collision uses Warp Mesh ray and nearest-point queries. Body and Clothes are
kept as separate Warp meshes: Body uses signed nearest-surface push-out, Clothes
uses two-sided unsigned push-out. Segment collision corrections are clamped so a
long strand cannot teleport an endpoint across the character in one pass.
When `KEEP LENGTH` is enabled, Yurameki records contact strands during the Warp
collision step and runs a post-KEEP collision projection on those active strands
plus short-lived contact history. This preserves rod lengths while avoiding a
full strand/triangle brute-force pass after FK length restoration.
Collision-corrected points are marked on the GPU and their derived velocity is
zeroed by default, so collider repair does not become rebound energy on the next
substep.
After KEEP/post-KEEP correction, Yurameki also runs a Body proxy hard guard from
the Head bone end point. If the ray from that internal seed to a hair point does
not cross the Body proxy, the point is treated as still inside the head and is
pushed to the first outward skin crossing plus collision margin.
Hair-hair collision is not implemented.

The operator report includes the actual number of frame transitions and
substeps, plus the selected CUDA device and SM architecture, for example
`steps=23, substeps=207, cuda:0 sm_120`.

## Repository Notes

The release package is Warp-only. The retired 0.6.x native CUDA cylinder-chain
implementation is kept in Git history, not in the current source tree or build
package.
