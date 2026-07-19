# Yurameki 0.3.0 native implementation notes

## Ownership boundary

- `_native_sim.py`: Blender main-thread orchestration only. It reads evaluated Curves and meshes,
  calls one native state object, writes Curves, and owns runtime cache/keyframe output.
- `native/src/yurameki_native.cpp` (`_yurameki_native_0_3_0`): vector/quaternion math, Cosserat integration, guide
  interpolation, BVH, point/segment collision, length projection, quality evaluation,
  grooming, convergence, and OpenMP parallel work.
- `__init__.py` and `ui.py`: operator and properties only.

No Python numerical fallback exists. A missing or ABI-incompatible native module is a setup
error instead of silently selecting a different simulation.

## Whole-system editing rule

This solver is small but tightly coupled across C++, its Python binding, Blender orchestration,
UI properties, cache semantics, and release packaging. For an architectural rewrite, keep one
end-to-end design and update all affected layers coherently before judging intermediate states.
Do not turn a human-sized sequence of partially complete milestones into separate architectural
decisions that can drift from one another. After the implementation pass, reread every affected
source and document, then build and run the complete validation suite. The current C++/OpenMP
design is authoritative; do not reconstruct the removed Warp/CUDA implementation from git
history or old release archives unless a user explicitly requests that rollback.

## Persistent identities and rest data

Blender's `data.curves` order is the strand number. Uniform `points_length` is required; the
local point number is the rod-chain index. The constructor copies evaluated world positions
from frame 1 for every strand and stores:

- each rod length;
- each adjacent-rod angle;
- mean absolute second difference of normalized rod tangents (roughness);
- guide Cosserat material frames and rest Darboux vectors.

The current start-frame positions initialize dynamic state, but never replace frame-1 rest
data. This distinction matters when `Start Frame != 1`.

## Per-frame state transition

```text
evaluated target + animated collider meshes
  -> adaptive root locks
  -> automatic substeps
  -> Cosserat position/orientation solve
  -> BVH point and segment collision
  -> full-strand guide restoration
  -> shape evaluation / grooming / collision evaluation loop
  -> write source Curves
  -> evaluate Blender modifiers
  -> external feedback settle loop
  -> sync actual evaluated state and store actual source-local positions
```

The inner loop has both `settle_iterations` and `settle_stagnation`. Each accepted repair is
relaxed by `settle_relaxation`; the best score is copied aside and restored if no candidate
fully converges. The Python feedback loop has an independent
`surface_feedback_iterations` bound.

## Shape metric

For normalized rod tangents `t`, define discrete bend `b[i]=t[i+1]-t[i]`. Roughness is the
mean absolute second difference `sum(|b[i+1]-b[i]|)/count`. Its limit combines an absolute
floor with the frame-1 value times `roughness_factor`, so gentle intentional arcs remain valid
while sharp or alternating curvature is rejected. Acceptance also checks maximum joint fold, deviation from each frame-1 joint angle,
and, when KEEP LENGTH is enabled, maximum absolute frame-1 rod-length error.

## Threading

OpenMP runs only over native arrays. Blender APIs are not thread-safe and remain outside the
GIL-released C++ calls. Current parallel regions cover rest-data generation, guide mapping,
BVH triangle preparation, per-guide dynamics, and per-full-strand settle/evaluation. A strand
is owned by one worker throughout a region, so point writes do not race. Aggregated counters
use OpenMP atomic/critical operations compatible with MSVC's OpenMP mode.

## 6000-strand tuning observation

The current scene can run all 6000 strands with `Guide Decimation = 1`; guide reduction is
an optional speed/quality tradeoff, not a requirement. A first full run took about 40 minutes
and was already visually clean, with the remaining fast "fly" motion looking primarily like
excess inertia rather than a strand-count failure.

`particle_mass` weights the inertial target against the fixed rod stiffness. Reducing it lowers
that inertial weight and is the first tuning direction for this motion. Gravity is integrated as
acceleration, but the equilibrium gravity load relative to fixed elastic stiffness scales roughly
with `mass * |gravity|`; increasing gravity after reducing mass can therefore recover lost sag
while keeping the smaller inertial weight. This is not an exact parameter equivalence because
damping, velocity limiting, collision, substeps, and the implicit solve also control the transient.
Tune mass first, then gravity magnitude, and finally damping/velocity limits from the resulting run.

## Validation

`tools/validate_native.py` checks exact rest stability, frame-1 length ownership and rod counts,
bounded folded-strand grooming, deep Body penetration despite a small search radius, a smooth
inextensible gravity cantilever, 1-thread/4-thread determinism, and rejection of malformed native
inputs. Build and run it with Blender's matching CPython ABI. Blender integration tests
additionally need a scene because evaluated Curves and modifiers are Blender-owned.
