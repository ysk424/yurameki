# Yurameki CUDA Prototype Notes

Status: active development fork.

Current version: 0.5.11.

Branch: custom-cpp-cuda.

Hard rule: do not restore the previous solver path.  The prototype uses native
CUDA with CUB/CCCL.  PyTorch is reserved for future matrix/tensor work only if
it becomes useful.

## Current State

- Previous solver files were removed from this fork.
- Blender Curves hair is read and split into fixed 1 cm cylinders.
- A deterministic solve order array is generated from root/cylinder position.
- `Apply FK Root Pull` verifies the 1 cm cylinder FK chain.
- CUDA collider detection is implemented in `native/yurameki_cuda_collide.cu`.
- `Apply CUDA Avoidance` runs CUDA collider avoidance with substeps and a capped
  movement per substep.
- Latest package: `dist/yurameki-0.5.11.zip`.
- `Check` now creates a copied collider proxy and fills all boundary holes on
  the proxy mesh. Collider operations prefer this proxy when it exists, giving
  parity checks a closed collision target without changing the groom solver
  heuristics.
- `Simulate Gravity` is the first frame-range gravity bake path. It buffers
  simulated frames in memory and bakes Curves position keyframes after compute.
- `Settle Hair Back` no longer exposes `Outside mm` in the UI. Each segment now
  has a final penetration guard after normal push iterations: endpoint inside,
  direct segment ray hit, and 25%/50%/75% inside samples are checked before the
  candidate is accepted.
- Top-of-head root emergence is protected: for the first two rods, an original
  outward-growing strand direction is preserved before blending back to the
  normal back/down groom curve. Those protected root rods also use shallow
  5%/10%/20% final-guard samples.
- When a valid head-region collider normal is found, rod 1 is now locked as a
  scalp-emergence anchor: point 0 remains the root and point 1 is placed one
  segment length along the oriented collider normal. Later settle/refinement
  passes skip rod 1.
- Curves-to-cylinder decoding no longer assumes equal point counts per strand.
  It uses each Curve span directly, stores flattened original arclength targets,
  makes the last cylinder use its actual remaining length, and reconstructs
  Blender points from the solved chain's real arclength.
- Adjacent rod direction changes are limited to `1 radian` during normal groom
  direction selection. This limit is intentionally not re-applied after collider
  push-out or final penetration repair.
- The locked root rod keeps the original root position and uses the collider
  normal direction, but its sign is forced toward the head-outward radial
  direction before placing point 1.
- Root-emergence direction checks now use the same head-outward normal sign
  correction before comparing the original root rod direction with the nearest
  collider normal. This keeps the cylinder-0 direction fix scoped to normal
  orientation only.
- The settle pass now caches each strand's original cylinder-0 direction before
  changing any Curves points. This cached source direction is used first when
  choosing the sign of collider normals at the root; the head radial direction is
  only a fallback when the source rod direction is unavailable.

## Rejected 0.5.10 direct root-source direction experiment

After 0.5.10 was pushed, an unpushed test changed the locked root rod so point 1
used the cached original cylinder-0 direction directly whenever that direction
moved farther from the approximate head center.  The intent was to stop treating
the scalp/root rod through collider-normal interpretation.

This was pulled back.  The visual result increased scalp penetration.  Keep the
pushed 0.5.10 behavior: use the cached source cylinder-0 direction to orient the
collider normal sign, but still place the locked root rod along the oriented
collider normal.  Do not reintroduce the direct source-direction root lock
without a separate proof that it reduces penetration on the scalp test scene.

## Rejected 0.5.6 final-guard experiment

After 0.5.5, a bounded fixed-length sphere search was tested as a final-guard
repair: after a push-out, the candidate was reprojected to the rod length,
rechecked, and then several directions on the sphere around the previous joint
were searched if the candidate was still penetrating.

This was pulled back.  It was finite, not an infinite loop, but it was slow in
the single-threaded Python initial groom and it did not produce usable hair.
The inside-view result showed two different failure modes:

- front/bangs: short hairs became hooked or curled inside the scalp;
- side/back: longer hairs still appeared inside the head region, including near
  the eye-side interior view.

Do not reintroduce this method as-is.  The next approach should treat these as
separate cases instead of trying to repair every failure by broad angular search
at the final guard.

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

In Blender, install/use `yurameki-0.5.11.zip`, then:

1. Select `CC_Base_Body` or the intended mesh collider.
2. Press `Pick Collider`.
3. Press `Check`; it validates Curves hair and builds the filled collider proxy.
4. Press `Detect CUDA Collider`; this is the preparation/check step.
5. Set `Start Frame` and `End Frame`.
6. Press `Simulate Gravity` to compute the frame range in memory and bake
   Curves position keyframes after completion.

Important: broadphase uniform grid is not implemented yet.  The current CUDA
collider path can be slow on full 400k+ triangle meshes because it is still the
first correctness pass.

## Intended Order

1. Finish 1 cm cylinder FK.
2. Make CUDA collider avoidance visually usable.
3. Add hair-vs-hair collision using the fixed solve order.
4. Optimize collider broadphase with CUB radix sort / grid cells.

## 0.5.0 gravity bake

The first gravity simulation pass starts from the lowered 0.4 groom result.
Roots are not fixed in world space: for each frame, evaluated Curves root
positions are read so root motion follows body/modifier animation. The Mesh
collider is also evaluated every frame.

Subframes are selected from the world-space motion of the last root in root-Z
order. Up to `1 mm` runs one substep; `4.5 mm` runs five. This root is the
current "north pole" representative point. A second equator representative may
be added later if rotation cases need it.

Each substep applies gravity, preserves segment length in FK order, and sends
same-joint segments across all strands to CUDA capsule/mesh avoidance. Hair-hair
collision is intentionally absent in this first pass.

## Initial groom BVH experiment

The promising initial-groom path is not full simulation.  It places strands one by one from the root using a deterministic curve:

- process lower-root strands in fixed Z order for tests;
- keep each original segment length exactly;
- near the root, move briefly toward +Y to escape the face/head;
- after that, aim mostly down (-Z), producing straight long hair;
- build a CPU BVH from the evaluated body collider;
- if a candidate point or segment hits the body, push it to a small clearance;
- when close to the body, project the desired direction to the body tangent plane so hair can slide along ears/neck/shoulders;
- do not let it stick forever: probe 2cm downward, and if that path is clear, release from surface sliding back to downward motion;
- if surface following runs longer than about 3cm, bias outward plus downward to force release.

Best 100-strand test so far:

- collision radius: 2.5mm
- follow radius: 30mm
- release probe: 20mm
- release clearance: 4mm
- outside signed clearance: 4mm
- max surface run: 30mm
- result: length error about 0.000065mm, tip direction dot(-Z) about 0.997

This looks much better around the ear: it can slide on the body briefly, then leave the body and fall vertically instead of sticking to the neck.

## Next topic: hair overlap / stacking

Hair-vs-hair is deliberately not solved in 0.4.11.  Two candidate directions are under consideration:

1. Post-groom lift/stacking correction:
   - detect visually overlapped regions after body grooming;
   - lift or offset the upper hair bundle so it stacks cleanly;
   - hard part: every rod/segment length must remain fixed after the lift.

2. Treat already-placed hair as collision geometry:
   - process strands in fixed root Z order;
   - register placed hair into a spatial structure;
   - detect candidate strand collisions like body collider collisions;
   - hard part: efficient and stable collision detection for many hair rods.

The current intuition is to test a simple occupancy grid first, because the fixed solve order naturally supports "placed lower hair becomes a soft collider for later hair".

## 2026-07-02 pause note

Current working build: 0.4.12.

What changed today:

- `Settle Hair Back` is the current CPU BVH initial groom path.
- `Groom Until` is the temporary debug limit.  It processes fixed root-Z order up to that count:
  - 500 = first 500 strands;
  - 600 = first 600 strands;
  - 0 = all strands.
- Release from surface sliding no longer checks only the 2cm endpoint.  It samples the release path at 25%, 50%, 75%, and 100% and requires every sample to be outside by signed nearest-normal clearance.
- `Outside mm` controls that signed outside clearance.  Current default is 4mm.
- Ear/scalp issues are still not fully solved.  The key suspicion is premature release around convex head/ear geometry.  The new path sampling should reduce but may not eliminate this.
- Hair-vs-hair stacking is still experimental only.  The MCP tests showed that body and hair collision sources can be separated cleanly:
  - body = Blender BVHTree from `CC_Base_Body`;
  - hair = fixed placed-strand spatial grid.
- Lateral-only hair avoidance preserved downward falling better than free 3D avoidance:
  - previous naive hair avoidance made tips too sideways;
  - lateral-only test kept avg tip down dot around 0.992.

Tomorrow's likely next steps:

1. Install/test `dist/yurameki-0.4.12.zip`.
2. Use `Groom Until = 500` to compare with the known good 500-strand state.
3. Use `Groom Until = 600` to inspect the next layer.
4. If ears/scalp still fail, improve inside/outside release using a stronger global test such as ray parity or multiple ray directions, not only nearest-normal signed distance.
5. After body groom is stable, integrate hair-vs-hair stacking from the lateral-only grid experiment.

## 2026-07-04 build note

Current working build: 0.4.13.

What changed:

- `Settle Hair Back` now uses a multi-direction ray parity check to detect points inside the static collider.
- Head-region penetrations are pushed outward from an approximate head center, with the center shifted downward by the collider's XY head/body width estimate.
- Penetration pushes try to ray-cast to the outward exit point, then place the hair point just outside by the collision radius.
- Return stats now include `inside_pushes` and `head_radial_pushes`.

## 2026-07-04 release fix

Current working build: 0.4.14.

What changed:

- Confirmed strand `4718` recalculates exactly to the current scene shape from the backup data.
- The outward straight segment was caused by `release_path_outside_enough()` requiring the full `Outside mm` clearance before releasing from surface sliding.
- A release path now only has to stay outside the collider, so a strand returns to gravity direction once the downward probe path is clear and non-penetrating.

## 2026-07-04 upward clamp

Current working build: 0.4.15.

What changed:

- Confirmed strand `512` recalculates exactly to the current scene shape from the backup data.
- The upward segment was caused by refinement passes reusing `new[j + 1] - new[j]` even when that existing segment pointed upward.
- Refinement desired directions now clamp upward segments back to `_base_drop_direction()`.

## 2026-07-04 weak-downward clamp

Current working build: 0.4.16.

What changed:

- Reviewed the same refinement path for remaining straight-line carryover after collider avoidance.
- Refinement desired directions now clamp when `direction.z > -0.35`, catching upward, horizontal, and weakly downward carryover directions.

## 2026-07-04 final 0.4 build

Current working build: 0.4.19.

What changed:

- Restored the code behavior to the `0.4.16` weak-downward clamp baseline.
- Later experimental `0.4.17` and `0.4.18` approaches were rejected for this line.
- `0.4.19` is the final 0.4 build; remaining small raised areas should be handled by the next gravity simulation step.
