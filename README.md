# Yurameki 0.7.6

Yurameki is now a NVIDIA Warp long-straight-hair simulation prototype.

Tokoya owns planting, cutting, reset, and `Settle Hair Back`. Yurameki starts
from that already-groomed Curves object and simulates motion only.

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
  stores every simulated frame in the Yurameki runtime cache by default. On
  success, Blender is left on the end frame so the simulated result is visible.
- `Bake Cache`: converts the current Yurameki runtime cache to Curves position
  keyframes when you are ready to commit the result.

Hair, Body, and Clothes are selected by eyedropper fields.

`Output: Cache` is the default. It keeps simulation playback in a cache instead
of immediately creating millions of Curves `position` F-Curves. `Keyframes`
keeps the direct F-Curve bake path for final Blender-native animation output.
`Final Preview` is only for a static look at the last simulated frame; it
overwrites the Curves data, so every timeline frame will show that final shape.
When you do bake keyframes, Yurameki writes the Curves `position` F-Curves
directly in bulk instead of calling Blender's per-point keyframe operator.

## Parameters

- `Root Locked Points`: number of joints from the root constrained to the
  evaluated pose.
- `Guide Decimation`: simulate one guide strand for every N strands, then
  interpolate the full Curves cache. `1` simulates all strands; `100` simulates
  about one percent of the strands.
- `Gravity m/s2`: Warp-style acceleration vector.
- `Damping`: velocity damping after prediction.
- `Max Velocity m/s`: speed limit for free joints. `0` disables the clamp.
- `Particle Mass g`: mass used for inverse mass of free joints.
- `Iterations`: distance/bend constraint iterations. The UI allows up to `256`
  for stiff, aligned long hair tests.
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

Collision uses Warp Mesh ray and nearest-point queries. Body and Clothes are
kept as separate Warp meshes: Body uses signed nearest-surface push-out, Clothes
uses two-sided unsigned push-out. Segment collision corrections are clamped so a
long strand cannot teleport an endpoint across the character in one pass.
Collision-corrected points are marked on the GPU and their derived velocity is
zeroed by default, so collider repair does not become rebound energy on the next
substep.
Hair-hair collision is not implemented.

The operator report includes the actual number of frame transitions and
substeps, plus the selected CUDA device and SM architecture, for example
`steps=23, substeps=207, cuda:0 sm_120`.
