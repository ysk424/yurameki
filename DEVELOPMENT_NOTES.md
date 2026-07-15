# Yurameki Warp Notes

Status: public release. Current version: 0.2.8 (Yurameki2 line). Branch:
`custom-cpp-cuda`.

## Interactive bake + workflow trim (0.2.8)

- **Progressive, stoppable Simulate.** `_warp_sim.simulate()` is now a thin
  blocking wrapper over `simulate_iter()`, a generator that draws each frame to
  the Curves and yields once per computed frame. `YURAMEKI_OT_simulate` is a modal
  timer operator that steps it one frame per tick, so the user watches the bake
  live and can stop early with the **Stop** button (`YURAMEKI_OT_stop_simulate`
  sets `yurameki_sim_cancel`) or **Esc**. On stop the generator catches
  `GeneratorExit` and registers a runtime cache of the frames computed so far (so
  they stay viewable/bakeable) instead of reverting; a genuine error still
  reverts. `execute()` remains a headless run-to-completion fallback.
- **No body proxy.** The collider is the source Body mesh directly. The proxy only
  capped holes for a closed inside/outside test, but collision uses the
  closest-face **normal** sign (`mesh_query_point_sign_normal`), not a winding
  number, so watertightness is not required for hair on the outside of the head.
  `collider_proxy.py` and the proxy property are removed.
- **Check button and the OpenAI Assistant removed** to streamline toward the main
  release (network permission dropped from the manifest).

## Adaptive root lock (0.2.8)

The root lock is no longer a single uniform `Root Locked Points` count. When
`Adaptive Root Lock` is on (default), on the side the head is advancing toward,
each strand's joints are made kinematic all the way down to the earlobe line so
they ride the head instead of being headbutted and flung, while trailing hair
keeps swinging.

- **Direction only, magnitude ignored.** The trigger is the *direction* the head
  is moving, not the speed/acceleration. A strand can fold badly during slow head
  motion (observed: strand 4104 kinks to >100 deg at joint 6 while the head moves
  only ~0.05 m/s), so a speed gate would miss it. As long as the head is moving at
  all (`|v_head| >= ADAPTIVE_LOCK_MIN_SPEED`), the leading side gets the full
  earlobe lock; a slow turn and a fast snap lock the same joints.
- **Head advance velocity `v_head`** is sampled at the head-bone *tip* (top of
  skull), not its root, so a nod or head-shake (which rotates the skull about the
  near-stationary neck pivot) still registers as motion into the hair. Measured
  per frame in world space (prev tip -> current tip) and EMA-smoothed
  (`ADAPTIVE_LOCK_VHEAD_SMOOTH`) so the direction does not jitter frame to frame.
- **Cone selection.** Each strand's root direction (from the head-bone root) is
  compared to `v_head`: full lock inside `ADAPTIVE_LOCK_INNER_DEG` (75 deg) with a
  soft smoothstep skirt out to `ADAPTIVE_LOCK_OUTER_DEG` (90 deg) so the lock set
  does not pop hard at the cone boundary.
- **Earlobe prefix.** The lock count is the contiguous run of root-side joints
  sitting above the earlobe line along the head-up axis, measured on the animated
  groom. The earlobe line is the head-local up-height of the jaw-hinge anchor
  `CC_Base_JawRoot` (`_ear_anchor_world`; eye bones are the fallback) -- the jaw
  hinge sits at ear-canal/earlobe height and rides the head. Extending the lock
  from the old "above the head-bone root" line (which stopped one joint short of
  the joint-6 kink) down to the earlobe is what lets the lock actually cover the
  fold. It is a valid kinematic prefix (the solver assumes locked vertices form a
  root prefix) and never drops below the uniform `Root Locked Points` baseline.
  Nothing assumes a fixed facing (currently -Y).
- **Cost.** On the leading side the locked joints ride the head, so there is no
  penetration to resolve there -- the preventive alternative to substepping into
  the contact. Implemented driver-side in NumPy (`_adaptive_lock_counts`), which
  rebuilds and re-uploads `inv_mass` each frame via `set_lock_counts`; the solver
  kernels are unchanged. `keep_length` FK still uses the uniform baseline count;
  extending it to the per-strand count is a follow-up if the locked span drifts.

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
- `Internal Damping` (0.2.2) is a strain-rate velocity Laplacian between a
  strand's free joints (`_cosserat._internal_damp_kernel`, Jacobi form, applied
  once per substep after the velocity update). It damps the internal deformation
  velocity -- ringing, jitter, frizz -- while a spatially smooth
  (rigid/translational) velocity field is a fixed point, so the bulk
  follow-through and gravity fall are preserved; each free-free edge contributes
  equal-and-opposite corrections, so the per-strand mean velocity (bulk momentum)
  is exact. This realises the "hair moves passively (driven by the root and
  gravity), not from its own stored energy" model. Prefer it over raising the
  global `Damping`, which suppresses the wanted motion too. Validated offline
  (uniform velocity unchanged, alternating jitter fully decays, mean drift
  ~1e-8) in `tools/validate_cosserat.py` test [6].
- Collision push-out velocity fix (0.2.5). A collision push-out is a *position*
  correction, not a force, but it leaked into the derived velocity: velocity was
  taken from the pushed position (`v = (predicted - pos)/dt`), so a ~5 mm push
  injected ~1 m/s and the stiff rod flung the strand. Primary fix: velocity is
  now derived from the **pre-collision** solved position (`self.pre_collide`,
  snapshotted right after the main solve, before `_collide`), so the push-out
  contributes zero velocity. Secondary: `_mark_strand_contacts` flags any strand
  pushed this substep and `_derive_velocity_kernel` damps the whole flagged
  strand (`Collision Velocity Damping`, `substep_contact_strands`) to settle
  resting contact -- the push propagates along the rod, so per-point damping was
  not enough. This targets the top-of-head long-segment "cat-ear" strands (rest
  segments `[1.4, 1.7, 5.2x9] cm`, styled up), which the scalp's upward normal
  pushed up until they coiled over ~19 frames; turning `KEEP LENGTH` off made it
  worse, pointing at the push-out rather than the elastic solve. Confirmed by
  isolation: the same strands have ~0 upward rise with collision disabled and
  rise with it on. The strand-wide velocity damping (0.2.4) reduced but did not
  eliminate the fly (residual push acceleration remained), which motivated
  deriving velocity from the pre-collision position. An earlier air-drag
  experiment (0.2.3) was reverted -- the cause was the push-out injecting
  velocity, not free inertial coasting.
- Buckle resistance + root bend boost (0.2.6). A floppy (low `k_bt`) rod clamped
  at the root behaves like a cantilever: the bending moment is largest at the
  most-loaded free joint next to the clamp and ~zero toward the tip, so instead of
  curving smoothly the rod concentrates all its bend at that one joint and finally
  *buckles* -- folds back past 90 deg (segment-to-segment `cos < 0`) -- there. Two
  cooperating fixes, exposed as one UI knob **Buckle Resistance** (default `1.0`):
  (1) an effective bend stiffness that is boosted near the root and falls off
  linearly to the tip (`BEND_ROOT_BOOST = 4.0`,
  `k_bt_eff = k_bt*(1 + boost*(1 - seg/segs))` in `cosserat_orientation_kernel`),
  so the highly loaded root joints resist bending more and the curve spreads
  toward the tip; and (2) a direct position-space hinge constraint
  (`cosserat_buckle_kernel`): for every interior joint whose
  `cos = dot(d_hat_in, d_hat_out)` drops below `BUCKLE_THRESHOLD` (`0.0` = 90 deg),
  a length-preserving PBD correction moves the joint's three points apart along the
  gradient of `cos` (perpendicular to the segments), inverse-mass weighted, until
  the joint reopens to the threshold. It is 3-colour scheduled over joint index
  `(local+1)%3` because each joint writes three points. Computing the angle
  directly (dot product + reciprocal length) was chosen over a normalised
  integer-theta lookup table: at 6000 strands it is already trivially cheap on the
  GPU and a table adds quantisation error for no measurable speed gain. The hinge
  correction is perpendicular, hence length-preserving to first order, but a finite
  step leaves a small second-order length drift (~2% inside the solver); this is
  **not** repaired by reordering the solve -- the `KEEP LENGTH` Body FK already
  rebuilds each strand from frame-1 rest lengths using the simulated (now opened,
  un-folded) segment directions at the end of every frame, so it restores exact
  length while keeping the un-buckled shape. **Superseded in 0.2.7 and now
  dormant (`Buckle Resistance` default `0`).** A correct same-pipeline A/B (buckle
  off vs on, both with `KEEP LENGTH` feedback) showed the buckle does not fix the
  real defect: the first 14 frames are bit-identical (the bend cosine never crosses
  the threshold, so the constraint never fires), and where it does fire it slightly
  *worsens* the minimum joint cosine (`-0.046` -> `-0.15`). The earlier "fold fixed"
  reading compared mismatched baselines (buckle-off measured without `KEEP LENGTH`,
  buckle-on with it), so the improvement was `KEEP LENGTH`, not the buckle. The kink
  is caused by collision, not by the rod's own bending (see next), so it is fixed
  there instead. The root-boost term (`BEND_ROOT_BOOST = 4.0`) is left on as a mild
  stiffen-near-the-root; the buckle kernel is retained but off by default.
- Anti-kink collision distribution (0.2.7). The actual cause of the one
  scalp-adjacent strand that folded and whipped (`strand 2229`, a top/back-of-head
  strand draping down the skull) is the **collision push-out**, isolated by a
  same-pipeline A/B: with collision on, its J4 bend cosine collapses from ~`0.36`
  to `~0.06` (a ~90 deg kink) at frame 16 and its tip whips `37.6 mm` in one frame;
  with collision off the same strand stays smooth (`cos` 0.8-0.99, gentle drape,
  tip `3 mm`). This is distinct from the 0.2.5 velocity fix: the push-out *position*
  itself forces the kink. Because the rod is nearly inextensible, a push that moves
  one interior joint out much more than its neighbours becomes a sharp bend the weak
  `k_bt` cannot relax before the next substep re-applies it, and the folded part
  actually dives *into* the body. Fix: instead of committing each point's push-out
  independently, `_collide` snapshots the pre-collision positions, captures the net
  push-out as a per-point correction field, diffuses it along each strand with a few
  Jacobi passes (`_corr_capture_kernel` / `_corr_smooth_kernel` / `_corr_apply_kernel`;
  locked points hold zero, anchoring the diffusion at the root), and re-applies the
  smoothed field. Diffusion only ever moves points further out (never inward), so it
  cannot add penetration. Exposed as the **Collision Smoothing** knob (default `0.5`
  -> `COLLISION_SMOOTH_WEIGHT = 0.5`, `round(strength*COLLISION_SMOOTH_MAX_PASSES)`
  = 4 passes; `0` disables). Validated on the live scene: strand 2229's minimum J4
  cosine goes `0.006` (kinked) -> `0.456` (~63 deg, no fold), its frame-15 tip whip
  `37.6 mm` -> `5.4 mm`, and its penetration `-25 mm` -> `-6 mm`; globally `zmax` is
  unchanged (`1.781 m` -> `1.779 m`) and a 3600-point penetration sample is unchanged
  (deepest `-2.5 mm` -> `-3.1 mm`, no point past `-5 mm` either way), so the healthy
  majority is untouched while the pathological strand is fixed.
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
  strands from frame-1 rest lengths. The rod solve already preserves length, so the
  flag mainly cleans up the small residual drift from the collision push-out and the
  buckle hinge correction (see Buckle resistance above). Hair-hair collision is not
  implemented.
- The Body proxy preserves the source mesh (including ears) and caps existing
  boundary loops such as eye openings with explicit triangulated cap vertices.
