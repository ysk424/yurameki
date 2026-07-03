# Yurameki 0.4.17

Development fork status: active prototype.  This branch is not a production
release.

Yurameki 0.4.17 is the first prototype step for the new CUDA hair solver.

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

The probe step moves each cylinder target by `+0.5 mm` on Y and `-3 cm` on Z,
then reprojects the cylinder back to its fixed length.  This is only a check for
the interface and length constraint.  It is not the final CUDA solver.

## CUDA Collider

The native collider entry point lives in `native/yurameki_cuda_collide.cu`.
The first kernel detects contacts between 1 cm hair cylinders and collider
triangles on CUDA, then uses CUB to reduce the per-cylinder hit flags into a
total hit count.

This is a detection-only milestone.  Hair response and hair-vs-hair contact
rules come after collider detection is visible and measurable.

## 0.4.17 initial groom

`Settle Hair Back` is now a CPU BVH initial-groom pass.  It is not the old repeated gravity settle.  It lays selected lower-Z root strands behind the body, keeps segment lengths fixed, slides briefly along body surfaces, and releases back to vertical falling only when the whole 2cm probe path is outside the body by signed nearest-normal clearance.

This build strengthens penetration handling.  Points are checked with multi-ray
inside/outside tests, and penetrated head-region points are pushed outward from
an approximate head center instead of using the local surface normal.

When a downward release probe is clear and non-penetrating, this build now
returns to gravity direction without requiring the full outside clearance.

Refinement passes no longer reuse an upward-pointing segment direction as the
next target direction; upward candidates are reset to the base falling curve.
The same clamp now also catches nearly horizontal and weakly downward
directions, so surface-avoidance remnants do not continue straight for several
segments after they are clear of the collider.
When refinement points are already far from the collider, the desired direction
now returns to the base falling curve instead of preserving the previous
avoidance direction.

Recommended first test values:

- `Groom Until`: 500
- `Groom Radius mm`: 2.5
- `Follow mm`: 30
- `Release Probe mm`: 20
- `Outside mm`: 4

Hair-vs-hair stacking is intentionally left for the next step.
