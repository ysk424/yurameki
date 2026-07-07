# Yurameki Warp Prototype Notes

Status: active development fork.

Current version: 0.7.11.

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
- Latest package target: source tree `0.7.11`; no native DLL build is required.

## 0.7.11 proxy boundary caps

- Replaced the proxy's generic boundary `holes_fill` step with explicit
  per-loop cap construction. Each boundary loop gets a center cap vertex and a
  fan of triangles, then face normals are recalculated.
- This closes the original Body eye openings and the Yurameki ear-cut openings
  as collision proxy geometry instead of leaving large wire-invisible n-gon
  caps.
- `Check` reports the number of cap vertices added as `capv=...` when a proxy
  is rebuilt.

## 0.7.10 repository cleanup

- Removed the retired 0.6.x native CUDA cylinder-chain implementation from the
  working tree: `gravity_sim.py`, `cuda_collider.py`, `solver_interface.py`,
  and `native/`.
- Removed older generated package archives from `dist/`. The current package is
  rebuilt from the manifest paths only.
- Replaced stale native-DLL manual test notes with this cleanup note. The old
  implementation remains available through Git history if archaeology is
  needed, but it is no longer an active source file.

## 0.7.9 post-KEEP collision projection

- Existing Warp collision now records the guide strands that touched Body or
  Clothes during each simulated frame. The record is a compact strand mask, not
  a triangle scan.
- After `KEEP LENGTH` rebuilds rods from frame-1 segment lengths, active guide
  strands are run through a post-KEEP collision projection. One GPU thread walks
  one strand sequentially, queries the Warp Mesh BVH for each rod segment, and
  rotates the rod tip while preserving the segment length.
- A short contact TTL keeps recently touched strands active for a few frames, so
  continuing shoulder/body contact does not need a fresh broad collision hit on
  every frame.
- The post-KEEP pass uses the existing Body signed query and Clothes two-sided
  query, shares the existing correction clamp/response settings, and zeroes
  velocity on strands it actually corrects.
- No all-hair/all-triangle brute force pass is added. Inactive strands only pay
  a cheap mask check in the post-KEEP kernel.

## 0.7.8 keep length FK

- Added `KEEP LENGTH`, enabled by default. It records frame-1 per-segment rod
  lengths, then before live preview/cache/final/keyframe output it rebuilds each
  strand from point 0 using the current simulated rod directions and those
  frame-1 lengths.
- The corrected guide/full strand positions are written back into the Warp
  simulator state after every frame so subsequent frames continue from the
  length-fixed result.
- MCP validation on the Lumi test file with 6000 strands x 12 points, frames
  1-2, `Guide Decimation=100`, and `KEEP LENGTH=on`: evaluated cache playback
  had maximum absolute total-strand length difference of about `0.000203 mm`
  versus frame 1, with zero strands over `0.001 mm`.

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
- The old native CUDA and cylinder files remain in repository history, but they
  are no longer present in the active working tree.

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

## Retired Archives

Detailed 0.6.x native CUDA and older initial-groom notes were removed from this
active development log because they described code paths that are no longer
present. Use Git history for those details when needed.
