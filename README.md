# Yurameki 0.6.6

Development fork status: active prototype.  This branch is not a production
release.

Yurameki 0.6.6 redesigns `Simulate` on top of the finalized 0.4.19 initial
groom and the 0.5.x CUDA collider path.

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
- simulate a frame range with a subdivided non-stretch chain and bake Curves
  position keyframes

The probe step moves each cylinder target by `+0.5 mm` on Y and `-3 cm` on Z,
then reprojects the cylinder back to its fixed length.  This is only a check for
the interface and length constraint.  It is not the final CUDA solver.

## CUDA Collider

The native collider entry point lives in `native/yurameki_cuda_collide.cu`.
The first kernel detects contacts between 1 cm hair cylinders and collider
triangles on CUDA, then uses CUB to reduce the per-cylinder hit flags into a
total hit count.

Collider avoidance is currently used by both the single-step debug operator and
the 0.6 chain simulation.

## 0.6.0 Simulate

`Simulate` starts from the current lowered hair state. It subdivides every
source strand internally (`Interpolation` defaults to `1`), follows evaluated
root positions for each frame, keeps cylinder 0 in its current direction, then
propagates the cylinder-0 tip movement down the chain with a linear falloff.
The default propagation distance is `50 cm`, so root motion reaches zero around
that chain distance.

The chain is non-stretch. Each segment is solved from root to tip in FK order by
projecting the desired motion back to the stored segment length. This is not an
XPBD model. Hair always has a downward bias through `Gravity Step mm`; root
motion is followed with distance falloff, while aerodynamic drag is reserved for
a later solver layer.

CUDA resolves collider avoidance after an AABB overlap prefilter between the
hair capsule and collider triangle. Hair-hair collision is not implemented in
0.6.0; strands are independent except for the deterministic root-Z ordering used
by the existing solver preparation path.

`Check` validates the Curves object, Body mesh, and optional Clothes mesh, then
builds a filled Body collider proxy. The proxy keeps the source object's
modifiers but uses its own mesh copy with boundary holes filled, so
parity-based inside/outside tests can treat the Body collider as closed. Later
collider operations prefer this proxy when it is available and add the Clothes
mesh as a second collider source.
`Detect CUDA Collider` is the preparation/check step before simulation.

The current subframe metric is the maximum world-space root movement across the
hair. Up to `1 mm` uses one substep; `4.5 mm` uses five substeps.

The 0.6.1 package fixes Curves position baking on Blender 5.x by inserting
keyframes from the Curves datablock with explicit attribute data paths.

The 0.6.2 package adds `Bake: Final Only` as the default. This writes only the
final simulated frame back to the Hair Curves data and avoids creating hundreds
of thousands of Blender FCurves during quick tests. Use `Bake: Keyframes` only
when an actual frame-by-frame Curves animation bake is needed.

It also writes simulated world-space points back through the stored evaluated
minus original Curves offset for each frame, so surface-deform style modifiers
are not applied twice during bake.

The 0.6.3 package raises default gravity from `1 mm` to `5 mm` per simulation
frame and allows up to `50 mm` for quick tuning. For short test ranges, try
`5-20 mm` before changing other parameters.

The 0.6.4 package adds upper-shape memory for long straight hair. At simulation
start, points above `Memory Height m` (`1.5 m` by default) are marked as the
saved top groom. During CUDA chain solving, those upper points are pulled toward
the evaluated Curves shape with `Memory Strength`, while lower points are left
to fall by gravity as non-stretch chain links and only avoid colliders. This is
intentional style physics for waist-length straight hair, not a general hair
simulation model.

The 0.6.5 package changes collider crossing response to a traditional
segment/triangle continuous collision style. When a chain segment crosses a
collider triangle, CUDA now treats the earliest crossing as contact and slides
the fixed-length segment along the surface instead of merely pushing the
already-crossed endpoint by a small distance. The preferred slide direction is
down the surface, then backward if the surface is horizontal. Collider
`Substeps` now defaults to `1`; increase it only when a strand has to resolve
multiple nearby surfaces in one step.

The 0.6.6 package fixes the Simulate chain subdivision. Simulation points are
now resampled by `Cylinder Length cm` as a maximum link length, with
`Interpolation` kept as a minimum per-source-segment subdivision count. A 1 cm
setting therefore creates about 1 cm links instead of merely splitting Blender's
original 5 cm curve spans in half. CUDA collision also checks the tip motion
from the previous position to the candidate position, reducing missed cloth/body
crossings when a point jumps across a surface between frames.

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

The 0.5.14 temporary package removes the left/right ear protrusion region from
the generated collider proxy before hole filling. The source body mesh is not
changed. This intentionally sacrifices ear collision in favor of cleaner hair
shaping around the sides of the head.

The neck-up groom is accepted for this phase. Collider setup now continues with
a Clothes collider alongside the current Body collider, with the top input area
organized as Hair, Body, and Clothes fields.

The 0.5.15 package adds that Hair/Body/Clothes input layout. The old generic
Collider field is now the Body field, and Clothes is an optional second Mesh
collider included in initial groom, CUDA detection, solver step, and gravity
bake collision input.

The 0.5.16 package adds a lower free-groom region below world `Z = 1.30 m`.
Segments in that region use the original strand direction as their target,
skip the back/down release direction, skip forced surface release, and skip the
adjacent-rod turn limiter. Collider push-out and final penetration guards still
run, so shoulder and clothes collision is avoided without making the lower hair
snap into straight groom directions.

The 0.5.17 package changes that lower free-groom target. Below world
`Z = 1.30 m`, the source strand direction is no longer preserved. The target is
mostly straight down with a small blend from the previous segment direction,
while collider push-out and final penetration guards remain active. This reduces
the radial straightening that can happen after lower hair is pushed away from
clothes.

The 0.5.18 package raises the lower free-groom threshold from world
`Z = 1.30 m` to `Z = 1.42 m`, applying the shoulder/clothes down-flow behavior a
little higher while leaving the accepted head/face region unchanged.

Refinement passes no longer reuse an upward-pointing segment direction as the
next target direction; upward candidates are reset to the base falling curve.
The same clamp now also catches nearly horizontal and weakly downward
directions, so surface-avoidance remnants do not continue straight for several
segments after they are clear of the collider.

This is the final 0.4 groom build. Remaining small raised areas are expected to
be handled by a later gravity simulation pass rather than more initial-groom
heuristics.

Hair-vs-hair stacking is intentionally left for the next step.
