# Yurameki 0.7.15

Yurameki is a Blender extension for simulating VR-character-style long straight
hair with NVIDIA Warp.

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
dist/yurameki-0.7.15.zip
```

## Direction

- Use NVIDIA Warp as the compute backend.
- Use the existing Blender Curves joints and their current segment lengths.
- Do not resample hair into 1 cm cylinder chains.
- Keep the first `Root Locked Points` joints of every strand constrained to the
  evaluated Curves pose. The default is `3`.
- Let the remaining joints move under gravity, damping, stretch distance
  constraints, bend constraints, and Warp Mesh collision.
- Keep automatic substeps based on constrained-point motion plus estimated
  gravity/velocity motion of free joints.
- Use a filled Body proxy when needed; keep Clothes as an evaluated mesh so
  Alembic / Mesh Sequence Cache clothing can update per frame.

## UI

The panel has two command buttons:

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

`Output: Runtime Cache` is the default. It keeps simulation playback in a cache
instead of immediately creating millions of Curves `position` F-Curves.
`Position Keyframes` keeps the direct F-Curve bake path for final Blender-native
animation output. `Final Preview` is only for a static look at the last simulated
frame; it overwrites the Curves data, so every timeline frame will show that
final shape. When you do bake keyframes, Yurameki writes the Curves `position`
F-Curves directly in bulk instead of calling Blender's per-point keyframe
operator.

`KEEP LENGTH` is the default length-safety path. Frame 1 is treated as the rest
shape for every strand segment. After each simulated frame, Yurameki keeps the
simulated rod directions but rebuilds the free joints after the locked prefix by
FK from the frame-1 segment lengths. All `Root Locked Points` remain at the
evaluated pose. The corrected result is used for live preview, cache playback,
final preview, and keyframe baking, and the corrected guide state is fed back
into the next frame.

0.7.13 adds a Body FK hard repair pass after the seed hard guard. Active strands
are reconstructed root-to-tip from their rest lengths; when a joint would remain
inside the Body proxy, Yurameki searches from the XPBD direction toward the
strand's straight continuation before falling back to a seed-ray escape point.

## Parameters

- `Root Locked Points`: number of joints from the root constrained to the
  evaluated pose.
- `Guide Decimation`: simulate one guide strand for every N strands, then
  interpolate the full Curves cache. `1` simulates all strands; `100` simulates
  about one percent of the strands.
- `KEEP LENGTH`: rebuild every simulated strand from its frame-1 segment
  lengths before live preview, cache output, final preview, or keyframe baking.
  The complete locked prefix stays at the evaluated pose; later joints are
  FK-rebuilt along the simulated rod directions, and the corrected guide state
  is fed back into the next frame.
- `Gravity m/s2`: Warp-style acceleration vector.
- `Damping`: velocity damping after prediction.
- `Max Velocity m/s`: speed limit for free joints. `0` disables the clamp.
- `Particle Mass g`: mass used for inverse mass of free joints.
- `Iterations`: CUDA Warp distance/bend constraint iterations. The default is
  `30` for stiff, aligned long hair tests, and the UI allows up to `256`.
- `Stretch Compliance log10`: base-10 exponent for XPBD-style compliance of
  adjacent joint lengths. `-8` means `1e-8`.
- `Bend Compliance log10`: base-10 exponent for XPBD-style compliance of
  two-joint bend distances. `-5` means `1e-5`.
- `Collision Margin mm`: target separation from collider mesh.
- `Collision Search mm`: nearest-surface search radius. The default is wide
  enough for Body inside/outside repair, while each correction step is clamped
  to a small distance for stability.
- `Collision Max Correction mm`: maximum collider push-out per collision pass.
- `Collision Response`: fraction of the collision position correction applied
  per pass. The default `1.0` fully repairs penetration.
- `Collision Velocity Damping`: extra damping for points touched by collision.
  The default `1.0` stores zero velocity after collider repair, treating the
  repair as position-error correction rather than a bouncing physical impact.
- `Collision Passes`: segment collision passes after point collision.
- `Post Collision Iterations`: extra constraint/collision reconciliation passes.
- `Auto Substep mm`: maximum constrained-joint motion per substep.
- `Max Substeps`: cap for automatic substeps.

## Warp Requirement

The Blender Python environment must be able to import `warp`.
If `Check` reports `Warp import failed`, install NVIDIA `warp-lang` for the
Python used by Blender.

## Collision Notes

Body uses the Yurameki filled proxy, because closed body collision needs stable
inside/outside queries and fewer holes. Clothes are not proxied by default. This
keeps Marvelous Designer Alembic / Mesh Sequence Cache clothes evaluated at the
current frame.
The proxy caps boundary loops, including eye openings and ear-cut openings, with
explicit triangulated cap vertices instead of relying on large single hole-fill
faces.

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

The 0.7.8 MCP validation run used `KEEP LENGTH` on a 6000-strand / 12-point
Curves test from frame 1 to frame 2. Evaluated viewport/cache playback lengths
matched frame 1 with maximum absolute total-strand error below `0.001 mm`.

## Repository Notes

The release package is Warp-only. The retired 0.6.x native CUDA cylinder-chain
implementation is kept in Git history, not in the current source tree or build
package.

## 0.7.15 Release

- `Start Frame` is now an unchanged initial state; simulation and correction
  run from the following frame through `End Frame`.
- Body correction obtains its Head seed from each simulated frame.
- Removed overlapping GPU writes from bend constraints by using four
  non-conflicting constraint colors.
- `KEEP LENGTH` and the Body hard guard now preserve every configured
  `Root Locked Points` joint at the evaluated pose.
- Runtime caches now restore the original Curves data outside the cached frame
  range before Blender evaluates existing animation.
- Runtime caches are isolated by Blend file and are cleared safely when a file
  is loaded or saved under another name.
- Runtime caches are invalidated safely when Curves point counts or curve span
  layouts change.

## 0.7.14 Release

- Public labels are English-only, even when Blender's UI language is Japanese.
- `Iterations` defaults to `30` for stiff straight-hair tests.
- Completed frames are shown in the viewport during simulation.
- Yurameki Body proxy colliders are hidden in the viewport after creation or
  reuse.
- The 0.7.13 Body FK hard repair remains unchanged to avoid result drift.
