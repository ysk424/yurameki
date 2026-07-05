# Yurameki 0.7.2

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
  bakes every simulated frame by default. On success, Blender is left on the end
  frame so the simulated result is visible.

Hair, Body, and Clothes are selected by eyedropper fields.

`Bake: Keyframes` is the default. `Final Preview` is only for a static look at
the last simulated frame; it overwrites the Curves data, so every timeline frame
will show that final shape.
Keyframe baking writes the Curves `position` F-Curves directly in bulk instead
of calling Blender's per-point keyframe operator.

## Parameters

- `Root Locked Points`: number of joints from the root constrained to the
  evaluated pose.
- `Gravity m/s2`: Warp-style acceleration vector.
- `Damping`: velocity damping after prediction.
- `Particle Mass kg`: mass used for inverse mass of free joints.
- `Iterations`: distance/bend constraint iterations.
- `Stretch Compliance`: XPBD-style compliance for adjacent joint lengths.
- `Bend Compliance`: XPBD-style compliance for two-joint bend distances.
- `Collision Margin mm`: target separation from collider mesh.
- `Collision Search mm`: nearest-surface search radius. The default is wide
  enough for Body inside/outside repair, while each correction step is clamped
  to a small distance for stability.
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
Hair-hair collision is not implemented.

The operator report includes the actual number of frame transitions and
substeps, plus the selected CUDA device and SM architecture, for example
`steps=23, substeps=207, cuda:0 sm_120`.
