# Yurameki Warp Notes

Status: public release. Current version: 0.2.1 (Yurameki2 line). Branch:
`custom-cpp-cuda`.

Direction: NVIDIA Warp through its Python kernel, array, and Mesh query APIs.
There is no native C++/CUDA library and no cylinder resampling; the existing
Blender Curves joints and their current segment lengths are used directly.

## Stable Cosserat elastic-rod solver

Each strand is solved as a Stable Cosserat elastic rod (`_cosserat.py`). Hair is
a rod, so only the rod core of the method is used -- no cloth warp/weft lattice
or membrane. Each segment carries a quaternion material frame; the stretch/shear
energy `C = (p_{i+1}-p_i) - l_i*d3(q_i)` couples the segment vector to its frame
tangent, so the rod stays at rest length intrinsically. That is why no FK "keep
length" reconnection is required after the solve.

- Structure (per substep, `Iterations` outer loops): an **exact** per-strand
  tridiagonal position solve (Thomas algorithm; orientations fixed, scalar
  tridiagonal Hessian along the chain) alternated with a quasi-static local
  Gauss-Newton orientation sweep (positions fixed, rotational inertia neglected,
  red/black colouring, `q <- q*exp(omega)`). The paper's closed-form `lambda`
  orientation update is not used -- it suppresses quaternion bookkeeping, so a
  finite-difference-validated local Gauss-Newton update reaching the same
  minimiser is used instead.
- Because the position solve is exact, segment length is preserved regardless of
  how stiff `k_ss` is relative to inertia. After collision (which moves points
  last and would otherwise leave the rod stretched) a final rod solve restores
  length while its weak inertia term anchors the result to the just-collided,
  pushed-out shape.
- Stiffnesses (`_warp_sim.py`): bend/twist stiffness `k_bt` is the UI knob
  **Bend Stiffness log10** (`k_bt = 10^value`, higher = stiffer). Stretch/shear
  stiffness is fixed at `STRETCH_STIFFNESS = 1e4`: the exact position solve makes
  the rod inextensible for any large value, so it is not exposed as a near-inert
  knob; it only needs to stay well above `k_bt` so the tangent tracks the frame.
- The interesting tuning axes are **Bend Stiffness** (shape / whip), **Particle
  Mass** (momentum: lower = lighter and snappier, less stored energy), and
  **Damping** (settling). Stretch is intentionally not a tuning axis.
- Defaults (0.2.1) are tuned for the rod: `Particle Mass = 3 g`, `Damping = 0.05`,
  `Max Velocity = 5 m/s`, `Collision Velocity Damping = 0.5`. The old XPBD mass
  (0.01 g) was far too small for the rod: gravity (proportional to mass) could
  not overcome the stiff stretch/shear coupling in the implicit solve, so a
  groomed -- even upward-styled -- strand would not fall. Raising mass beyond a
  few grams does not increase the fall (it only adds stretch); the fall is a
  mass/stiffness balance, not free-fall. At grams-scale mass the hair collapses
  under gravity and drapes on the body collider. Validated on the live scene by
  dropping the upward-styled Lumi groom to a natural drape (see docs history).
- Validation: `tools/validate_cosserat.py` (`python tools/validate_cosserat.py
  cpu` or `cuda:0`) checks orientation/position gradients by finite difference,
  that the Warp kernels reproduce an independent numpy reference, rest-state
  stability, and a horizontal cantilever drooping under gravity with segment
  length preserved to ~0.01% and no FK. On the live 6000-strand scene the rod
  holds length to <0.07% including collision.

## Runtime

- Yurameki expects hair already planted and settled by Tokoya; it starts from
  that groomed Curves object and simulates motion only.
- Panel commands `Check`, `Simulate`, `Bake Cache`:
  - `Check` validates Hair / Body / optional Clothes, builds or reuses the filled
    Body proxy, and verifies Warp CUDA plus Warp Mesh initialization.
  - `Simulate` runs the elastic-rod solve over the frame range and stores every
    frame in the Yurameki runtime cache by default, showing each frame live and
    leaving Blender on the end frame.
  - `Bake Cache` converts the runtime cache to Curves position keyframes.
- The first `Root Locked Points` joints of each strand are kinematic and follow
  the evaluated Curves pose (default `3`); the remaining joints are the rod.
- `Start Frame` is an unchanged initial state; simulation runs from `Start + 1`
  through `End Frame`.
- Automatic substeps size `dt` from constrained-point motion plus estimated
  gravity/velocity motion of the free joints (`Auto Substep mm`, `Max Substeps`).
- Runtime caches are isolated by Blend file and Curves topology, restore the
  original groom outside the cached range, and store per-frame local positions
  plus `restore_local_values` in an `.npz` next to the blend.

## Collision

- Body uses the Yurameki filled proxy for stable inside/outside queries; Clothes
  are evaluated directly each frame so Marvelous Designer Alembic / Mesh Sequence
  Cache garments update. Body uses signed nearest-surface push-out, Clothes
  two-sided unsigned push-out.
- Collision uses Warp Mesh ray and nearest-point queries; segment corrections are
  clamped so a long strand cannot teleport across the character in one pass.
  Collision-corrected points are marked so their derived velocity is zeroed
  rather than becoming rebound energy.
- After the solve a Body seed hard guard from the Head bone end pushes any point
  still inside the head out to the first skin crossing plus margin. When `KEEP
  LENGTH` is on (default), a CPU Body FK hard repair reconstructs contacting
  strands from frame-1 rest lengths; this is now the only reason the flag exists,
  since the rod already preserves length. Hair-hair collision is not implemented.
- The Body proxy preserves the source mesh (including ears) and caps existing
  boundary loops such as eye openings with explicit triangulated cap vertices.
