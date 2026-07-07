# Yurameki Warp Prototype Notes

Status: active development fork.

Current version: 0.7.7.

Branch: custom-cpp-cuda.

Current direction: use NVIDIA Warp through its Python kernel, array, and Mesh
query APIs. The native C++ CUDA cylinder-chain path is no longer part of the
active extension package.

## Current State

- Yurameki expects hair already planted and settled by Tokoya.
- The panel exposes `Check`, `Simulate`, and `Bake Cache` command buttons.
- `Check` validates Hair/Body/Clothes, builds or reuses the filled Body proxy,
  and verifies Warp CUDA plus Warp Mesh initialization.
- `Simulate` uses the existing Blender Curves joints and original adjacent
  segment lengths. No 1 cm cylinder resampling is performed.
- The first `Root Locked Points` joints of each strand are kinematic and follow
  the evaluated Curves pose. Default is `3`.
- Free joints are integrated with Warp kernels using gravity, damping, distance
  constraints, bend constraints, point sweep collision, nearest-surface
  push-out, and segment ray collision.
- Automatic substeps are based on constrained-point motion plus estimated
  gravity/velocity motion of free joints. `Auto Substep mm` sets the target
  movement per substep and `Max Substeps` caps the result.
- Body collision uses the filled Body proxy. Clothes are evaluated directly each
  frame so Marvelous Designer Alembic / Mesh Sequence Cache meshes can update.
- Latest package target: source tree `0.7.7`; no native DLL build is required.

## 0.7.7 live frame preview

- `Simulate` now writes each completed frame to the visible Curves object and
  forces a viewport refresh during the run, so long simulations show progress in
  Blender instead of staying visually frozen until the end.
- Target Curves poses are read before live preview writes begin, keeping the
  preview output from feeding back into the next frame's simulation input.

## 0.7.6 guide decimation cache test

- Added `Guide Decimation`. `1` keeps the previous all-strands simulation path;
  `100` simulates one guide strand for every 100 strands.
- Decimated simulation still uses the existing Warp solver. Only the input
  strand set is reduced.
- After each simulated frame, guide motion is converted back into a full Curves
  cache by blending nearby guide deltas from root-position proximity. The
  visible cache, preview playback, and `Bake Cache` path remain full strand
  data.

## 0.7.5 UI unit cleanup

- `Particle Mass` is now entered in grams in the UI and converted back to kg
  before being passed to the Warp solver.
- `Stretch Compliance` and `Bend Compliance` are now entered as base-10
  exponents. For example, `-8` passes `1e-8` to the solver.

## 0.7.4 runtime cache output

- `Output: Cache` is now the default simulation output mode.
- `Simulate` stores per-frame Curves positions in a Yurameki runtime cache and
  replays them on frame changes instead of immediately creating Curves position
  F-Curves.
- `Bake Cache` converts the current runtime cache to Curves position F-Curves
  only when the result needs to be committed as Blender-native keyframes.

## 0.7.0 Warp rewrite

- Replaced the active simulation path with `_warp_sim.py`.
- Replaced the parameter list with Warp-style controls: gravity, mass, damping,
  stretch compliance, bend compliance, collision margin/search/passes, and
  automatic substep limits.
- Removed active UI access to solver-step debugging, CUDA detection, cylinder
  length, propagation distance, and upper-shape memory. Those were tied to the
  discarded cylinder-chain path.
- The old native CUDA and cylinder files remain in repository history and may
  still exist in the working tree, but they are not included in the 0.7.0
  extension manifest.

## 0.7.0 explosion-stability pass

- Hair-hair collision is not active in the Warp prototype. Explosion observed
  after the first frame was not caused by self collision.
- Body and Clothes collision are now split into separate Warp meshes. Body uses
  signed nearest-surface repair suitable for the filled proxy; Clothes use
  two-sided unsigned repair for open Alembic / Mesh Sequence Cache surfaces.
- Nearest-surface push-out no longer trusts a raw face normal for every
  collider. Clothes orient the normal toward the hair point, and Body uses
  Warp's signed-normal query.
- Segment ray collision no longer teleports an endpoint all the way to the hit
  plane when the correction is large. Each correction pass is clamped to a few
  millimeters.
- Velocity is derived after constraint and collision reconciliation, not before
  collision. This prevents stale pre-collision velocity from feeding the next
  substep.
- Automatic substeps now include estimated free-joint motion from gravity and
  existing velocity, not only kinematic root/locked-point motion. This matters
  when the body barely moves but long hair is falling under gravity.

## 0.7.1 CUDA/frame visibility pass

- Warp initialization now explicitly selects `cuda:0` and rejects non-CUDA
  devices before allocating simulation arrays.
- Simulate now leaves the scene on the requested end frame after a successful
  run. The previous code always restored the original frame, which made a
  multi-frame run look as if it had stopped at frame 1.
- Operator reports now include frame transition count, total substep count, and
  the CUDA device / SM architecture. Console output also prints one progress
  line per simulated frame transition.

## 0.7.2 bake default

- `Bake: Keyframes` is now the default. The previous default, `Final Only`,
  wrote the last simulated frame directly to the Curves datablock, so frame 1
  and frame 24 both displayed the frame-24 shape after simulation.
- The static mode remains available as `Final Preview` for quick last-frame
  inspection, but it is intentionally not an animation bake.
- Keyframe baking now writes Curves `position` F-Curves through Blender 5.2's
  `Action.fcurve_ensure_for_datablock()` API in bulk. This avoids millions of
  per-point `keyframe_insert()` calls on large hair tests.

## 0.7.3 velocity limit pass

- Added `Max Velocity m/s`. Free joints are clamped both after gravity
  prediction and after post-collision velocity derivation. `0` disables the
  clamp.
- Exposed the previously hard-coded collision push-out clamp as
  `Collision Max Correction mm`. The default is `5 mm`, close to the old
  `max(margin * 6, 2 mm)` behavior for the default `0.8 mm` margin.
- Baked Curves position F-Curves now use `LINEAR` interpolation. The previous
  default Blender `BEZIER` interpolation could add unintended between-frame
  overshoot to a simulation cache.
- `Iterations` can now be set up to `256`. Other tuning controls also have
  wider UI ranges so long-straight-hair stiffness, damping, collision push-out,
  and substep limits can be explored without code edits.
- Collision repair now writes a GPU contact mask. Velocity derivation uses that
  mask to apply `Collision Velocity Damping`, which defaults to `1.0`; contacted
  points therefore keep zero velocity after penetration repair. This treats
  collider correction as position-error repair, not as a spring-like impact that
  stores rebound energy.
- Added `Collision Response` for the position-correction fraction. The default
  remains `1.0` so body/clothes penetration is fully repaired; lowering it is a
  tuning option for softer but less strict collider response.

## 0.6.x archive

- The 0.6.0 path used native CUDA with fixed-length cylinder chains.
- The 0.6.1 package fixes Blender 5.x Curves position baking by keyframing
  `attributes["position"].data[i].vector` from the Curves datablock instead of
  calling `keyframe_insert("vector")` on the attribute value itself.
- The 0.6.2 package adds a Simulate bake mode. `Final Only` is the default and
  writes only the final simulated frame to Hair Curves data. `Keyframes` keeps
  the expensive full Curves position keyframe bake for deliberate animation
  exports.
- Simulate bake now stores each frame's evaluated-minus-original Curves offset
  before writing any results and subtracts that offset when writing back. This
  prevents surface-deform-style modifiers from being applied twice to simulated
  world-space points.
- The 0.6.3 package raises the default Simulate gravity from `1 mm` to `5 mm`
  and the UI maximum from `10 mm` to `50 mm`. This keeps the current solver in
  the intended non-XPBD direction: a non-stretch FK chain that always wants to
  fall downward, follows root motion with falloff/delay, and leaves aerodynamic
  drag for a later layer.
- The 0.6.4 package adds Simulate upper-shape memory. At the start frame,
  subdivided points above `Memory Height m` default `1.5 m` are marked as the
  saved Settle/top groom shape. Each simulated frame reads the evaluated Curves
  shape as the moving style target, then CUDA pulls only those marked upper
  points toward that target by `Memory Strength`. Below the saved height, hair
  remains an unconstrained non-stretch chain with downward gravity and collider
  avoidance, so shoulder hair can fall when arms move down.
- The 0.6.5 package changes Simulate collider response from only short
  endpoint push-out toward traditional continuous segment/triangle collision
  for cloth-like surfaces. A segment that crosses a collider triangle now
  chooses the earliest crossing, orients the contact normal against motion, and
  projects the fixed-length chain direction to slide along the surface. The
  preferred fallback slide is down the surface, then backward for near-horizontal
  cloth/shoulder surfaces. `COLLIDER_SUBSTEPS` defaults to `1`; raise it only
  when multiple close surfaces must be resolved in a single segment step.
- The 0.6.6 package fixes Simulate subdivision so the chain is resampled by
  `Cylinder Length cm` as a maximum link length. `Interpolation` is kept as the
  minimum number of subdivisions per original source segment, but no longer
  prevents 1 cm/2 cm chain lengths from being honored. CUDA collision also
  checks the endpoint sweep from the previous tip position to the candidate tip
  position, so cloth/body surfaces are less likely to be crossed between
  frames without contact.
- The 0.6.7 package removes `Settle Hair Back` from Yurameki. Initial grooming
  is owned by Tokoya; this repository now treats groom history as out of scope.

## Retired Initial Groom Archive

Detailed Yurameki-era initial-groom notes were removed because they described
obsolete Settle behavior and encouraged future debugging to trust old visual
acceptance records. For current grooming behavior, inspect Tokoya's active
implementation and its evaluated-coordinate notes directly.

## Verified Before Break

FK root pull test through MCP:

```text
n_strands: 6000
n_cylinders: 280105
max_len_err_mm: about 0.00006
max_chain_gap_mm: 0.0
root pull: 0.3 mm default
```

CUDA native test:

```text
detect: hit_count = 1 on a minimal triangle test
avoidance: adjusted tip returned, length error = 0.0
```

## Next Manual Test

In Blender, install/use the 0.6.6 source add-on after rebuilding
`native/yurameki_cuda_collide.dll`, then:

1. Select `CC_Base_Body` or the intended mesh collider.
2. Press `Pick Body`.
3. Press `Check`; it validates Curves hair and builds the filled collider proxy.
4. Optionally select a clothes mesh and press `Pick Clothes`.
5. Press `Detect CUDA Collider`; this is the preparation/check step.
6. Set `Start Frame` and `End Frame`.
7. Keep `Memory Height m` near `1.5` so only the upper groom is remembered.
8. Press `Simulate` to compute the frame range in memory and bake
   Curves position keyframes after completion.

Important: broadphase uniform grid is not implemented yet.  The 0.6.0 CUDA
collider path uses capsule/triangle AABB rejection before exact distance tests,
but still scans collider triangles inside each kernel.

## Retired 0.5/0.4 Groom Notes

Detailed 0.5 and 0.4 initial-groom investigation notes were removed from this
active Yurameki development log. They referred to a grooming path that is no
longer part of Yurameki and contained visual acceptance language that can
mislead later debugging.
