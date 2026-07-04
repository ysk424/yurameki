# Yurameki 0.5.13

Development fork status: active prototype.  This branch is not a production
release.

Yurameki 0.5.13 starts the gravity-bake pass on top of the finalized 0.4.19
initial groom.

This version intentionally removes the previous solver implementation.  It only
contains the Blender interface needed before CUDA work starts:

- read one Blender Curves hair object
- split every strand into fixed 1 cm cylinders
- build a deterministic solve order from root position
- export solver input arrays as `.npz` plus a small `.json` summary
- apply one length-preserving probe step back to the Curves object
- apply a root-pull FK test where only strand roots move and 1 cm cylinders follow
- call a native CUDA collider detection DLL from Blender
- apply CUDA collider avoidance with substeps and capped per-step movement
- simulate a frame range in memory and bake Curves position keyframes

The probe step moves each cylinder target by `+0.5 mm` on Y and `-3 cm` on Z,
then reprojects the cylinder back to its fixed length.  This is only a check for
the interface and length constraint.  It is not the final CUDA solver.

## CUDA Collider

The native collider entry point lives in `native/yurameki_cuda_collide.cu`.
The first kernel detects contacts between 1 cm hair cylinders and collider
triangles on CUDA, then uses CUB to reduce the per-cylinder hit flags into a
total hit count.

Collider avoidance is currently used by both the single-step debug operator and
the 0.5 gravity bake.

## 0.5.x gravity bake

`Simulate Gravity` starts from the current lowered hair state. It follows the
evaluated root positions for each frame, applies gravity, preserves segment
length in FK order, resolves the moving Mesh collider through CUDA capsule/mesh
avoidance, buffers every simulated frame in memory, then bakes Curves `position`
keyframes after computation completes.

`Check` validates both the Curves object and selected Mesh collider, then builds
a filled collider proxy. The proxy keeps the source object's modifiers but uses
its own mesh copy with boundary holes filled, so parity-based inside/outside
tests can treat the collider as closed. Later collider operations prefer this
proxy when it is available.
`Detect CUDA Collider` is the preparation/check step before simulation.

The current subframe metric is the world-space movement of the last root in
root-Z order. Up to `1 mm` uses one substep; `4.5 mm` uses five substeps.

Hair-hair collision is not implemented yet. The first 0.5 pass prioritizes
body collider avoidance and preventing segment stretch.

## 0.4.19 initial groom

`Settle Hair Back` is now a CPU BVH initial-groom pass.  It is not the old repeated gravity settle.  It lays selected lower-Z root strands behind the body, keeps segment lengths fixed, slides briefly along body surfaces, and releases back to vertical falling only when the whole 2cm probe path is outside the body by signed nearest-normal clearance.

This build strengthens penetration handling.  Points are checked with multi-ray
inside/outside tests, and penetrated head-region points are pushed outward from
an approximate head center instead of using the local surface normal.

When a downward release probe is clear and non-penetrating, this build now
returns to gravity direction without requiring the full outside clearance.

The 0.5.1 package removes the old `Outside mm` UI parameter and adds a final
guard at the end of each `solve_candidate()` segment: endpoint inside, direct
segment ray hit, and 25%/50%/75% inside samples are checked after the normal
push iterations. Only failed candidates receive an extra push or safe-direction
fallback.

The 0.5.2 package protects scalp emergence at the top of the head. For the first
two rods, if the original strand direction points away from the collider, that
direction is preserved and then blended back into the normal back/down groom
curve. The final guard also checks shallow 5%/10%/20% samples on those protected
root rods.

The 0.5.3 package makes the first rod a fixed scalp-emergence anchor when a
valid head-region collider normal is found: point 0 stays fixed, point 1 is
placed one segment length along the oriented collider normal, and later passes
do not move that first rod.

The 0.5.4 package fixes the Curves-to-cylinder round trip for debugging. The
decoder reads each Blender Curves strand by its own `first_point_index` and
`points_length`, so 8-12 point strands can coexist. The last cylinder uses the
actual remaining length instead of always 1 cm. Reconstruction samples the
actual solved cylinder chain by relative arclength and writes back to the
original Blender point offsets.

The 0.5.5 package limits normal groom direction changes between adjacent rods to
`1 radian`. The limit is applied only before collision solving; inside-collider
push-out and final penetration repair remain higher priority and are not clamped
back by this angle rule.

The 0.5.7 package keeps the cylinder-0/root rod start point unchanged, but
forces the locked root normal to face the head-outward direction before placing
point 1.

The 0.5.9 package applies the same head-outward normal sign correction to the
root-emergence direction test, so cylinder 0 cannot be accepted as outward
because of a flipped nearest-surface normal.

The 0.5.10 package caches each strand's original cylinder-0 direction before
the settle pass changes any Curves points.  That cached source direction is used
as the primary guide for root normal sign, with the head radial direction kept
only as a fallback.

The 0.5.11 package changes the former `Check Hair` button to `Check`. Pressing
it creates a copied collider proxy, fills all boundary holes on the proxy mesh,
and stores that proxy for later collider checks. This is the first foundation
step for using a closed manifold-style collision target while leaving the
existing groom heuristics unchanged.

The 0.5.12 package keeps root emergence protection only on cylinder 0. Cylinder
1 now returns to the normal back/down groom direction, so the first bend can use
the existing 1 radian turn limit instead of preserving an upward source
emergence direction for two rods. This was chosen after comparing a raised
eye-side patch with nearby lower roots: the raised patch had only about
`24-27 degrees` between cylinders 0 and 1, while the OK comparison strands were
already at the `57.3 degree` turn limit. The 0.5.12 preview changed the raised
strands and left the OK comparison strands unchanged.

The 0.5.13 package makes surface release require the existing internal
`release_clearance_m` instead of only checking that probe samples are outside
the collider. This keeps shallow outside paths from being accepted when they are
visually too close to the skin.

Refinement passes no longer reuse an upward-pointing segment direction as the
next target direction; upward candidates are reset to the base falling curve.
The same clamp now also catches nearly horizontal and weakly downward
directions, so surface-avoidance remnants do not continue straight for several
segments after they are clear of the collider.

This is the final 0.4 groom build. Remaining small raised areas are expected to
be handled by a later gravity simulation pass rather than more initial-groom
heuristics.

Hair-vs-hair stacking is intentionally left for the next step.
