# Body Proxy Hard Repair Plan

This document exists so the hard anti-penetration implementation can survive
context loss. Do not start the implementation by improvising inside the solver.
Keep the goal and the responsibilities separated.

## Implementation Status

Implemented in 0.7.13 as a CPU reference pass:

- Body FK hard repair runs after KEEP/post-KEEP and the seed hard guard.
- Active strands come from current/post-KEEP contact, recent FK repair TTL, and
  strands corrected by the seed hard guard.
- Repair reconstructs each active strand root-to-tip from rest lengths.
- Invalid candidates are searched from `xpbd_dir` toward `straight_dir` with
  slerp and a small binary refinement.
- Seed-ray escape remains the fallback when no straightness-preserving candidate
  is valid.
- Repaired or seed-guarded guide strands have Warp velocities zeroed.
- The same guard-plus-FK correction is applied during preview/cache/keyframe
  conversion when the guard changes a strand.

Still open:

- GPU/Warp acceleration for the seed tests and FK repair.
- More seed selection validation in difficult characters.
- Tuning segment validation if it proves too conservative near roots.

## Current State

Yurameki 0.7.12 already has these layers:

- XPBD/Warp simulation produces candidate particle positions.
- `KEEP LENGTH` rebuilds hair strands from frame-1 segment lengths.
- Post-KEEP collision projection repairs guide strands that had recent contact.
- The Body proxy is capped and closed with triangulated boundary caps.
- A Head-bone-end seed ray hard guard moves hair points that are still inside
  the Body proxy to the first skin crossing plus collision margin.

This has reduced penetration, but not eliminated it.

The remaining problem is not simply "stronger collision response". The next
step is a separate rod reconstruction repair pass:

```text
XPBD creates motion intent
FK preserves rod length and straightness
Body proxy tests decide inside/outside
Hard repair reconstructs invalid strand segments
The repaired coordinates are written back to XPBD state
```

## Primary Goal

For Body proxy penetration, final visible/cache hair must be outside the proxy
as much as physically possible.

If constraints conflict, priority is:

1. Avoid obvious Body proxy penetration.
2. Preserve a straight, non-twisted strand shape.
3. Preserve segment lengths.
4. Preserve XPBD motion.
5. Preserve local collision response details.

The repair must not use collider normals as the main direction of strand shape.
Normals may be used only for final escape placement and margin.

## Non-Goals

- Do not solve hair-hair collision here.
- Do not use eye, eyelash, tearline, eye occlusion, or clothes meshes for the
  Body inside/outside decision.
- Do not depend on generic parity through all character meshes.
- Do not make XPBD itself responsible for zero penetration.
- Do not add a broad UI surface until the algorithm is stable.

## Key Insight

When a joint candidate is inside the Body proxy, pushing along the collision
normal can make the strand worse:

- It can increase the bend at the previous joint.
- It can twist the strand into unstable planes.
- Repeated normal pushes can make the hair look wavy or crumpled.

The repair direction should therefore be based on the strand's own rod
continuation, not on the collision normal.

For joint `i`, with previous repaired joints already known:

```text
p[i-2] = repaired previous-previous joint
p[i-1] = repaired previous joint
xpbd[i] = XPBD candidate for this joint
L[i-1] = rest segment length from p[i-1] to p[i]

straight_dir = normalize(p[i-1] - p[i-2])
xpbd_dir     = normalize(xpbd[i] - p[i-1])
```

If the XPBD candidate penetrates, the first repair should rotate from
`xpbd_dir` toward `straight_dir`, because that reduces bend and preserves the
idea of long straight hair.

## Inside/Outside Decision

Use Body proxy only.

Use one or more internal seed points. The first implementation can use the Head
bone end, but the long-term design should support nearest seed selection:

- Head bone end for scalp, face, and eye-height hair.
- Spine/neck bone endpoints for lower hair.
- Additional internal endpoints can be added later if needed.

For a test point `p`:

```text
seed = chosen internal bone endpoint
ray from seed to p against Body proxy

if ray crosses Body proxy at least once before p:
    p is outside
else:
    p is inside or invalid
```

For an inside point, cast farther along the same direction:

```text
ray from seed along normalize(p - seed), distance = large
first hit = skin exit
escape point = first_hit + outward_normal * margin
```

This is not generic mesh parity. It is a body-specific "did we pass through
skin from an internal seed?" test.

## Segment Decision

Point tests are not enough. A segment can cross the proxy even if both endpoints
look acceptable.

For segment `p0 -> p1`:

- `p0` should already be repaired and outside.
- Test `p1` with the seed method.
- Test segment ray `p0 -> p1` against Body proxy.
- If either test fails, the candidate segment is invalid.

The first implementation may focus on point inside repair if segment testing is
too expensive, but the design target is point plus segment validation.

## Repair Algorithm

Process each active strand from root to tip.

```text
for each active strand:
    repaired[0..locked-1] = input locked/root positions

    for i from locked to points_per_strand - 1:
        prev = repaired[i - 1]
        prevprev = repaired[i - 2] if i >= 2 else fallback root direction
        L = rest_length[i - 1]

        xpbd_dir = normalize(xpbd[i] - prev)
        straight_dir = normalize(prev - prevprev)

        candidate_dir = xpbd_dir
        candidate = prev + candidate_dir * L

        if candidate is valid:
            repaired[i] = candidate
            continue

        zero velocity for this point or whole strand

        repaired_candidate = search direction from xpbd_dir toward straight_dir
        if found:
            repaired[i] = repaired_candidate
            continue

        repaired[i] = escape point from seed ray plus margin
```

After repair, write `repaired` back to:

- visible/cache output positions
- Warp simulator position array
- Warp velocity array, with repaired points or strands zeroed

## Direction Search

Avoid fixed 5-degree stepping as the final design. It is easy to understand but
can be slow and branchy.

Preferred search:

```text
try t values along slerp(xpbd_dir, straight_dir, t)
t = 0.25, 0.50, 0.75, 1.00

if this is not enough:
    binary search near the last valid/invalid boundary
```

The goal is not to push along the surface normal. The goal is to find the
least-bent valid continuation of the strand.

If `xpbd_dir` and `straight_dir` are almost identical:

- Try `straight_dir` directly.
- If still inside, use seed escape point.

## Repair Plane

The repair should avoid twisting.

Preferred plane normal:

```text
plane_normal = cross(straight_dir, xpbd_dir)
```

If too small:

1. Use previous frame or previous segment bend plane normal.
2. Use a strand-level cached plane normal.
3. Use `cross(straight_dir, seed_to_prev_dir)`.
4. As a final fallback, skip plane-specific rotation and use direct slerp.

Do not build the primary repair plane from the collision normal. Collision
normal can be noisy and can increase visible bending.

## Active Strand Selection

Do not run the expensive repair on every strand forever.

Candidate activation sources:

- Current Body collision contact.
- Post-KEEP contact history.
- Head-seed hard guard corrected this strand.
- Any point failed the seed outside test.
- Strand root is near the Body proxy AABB.
- Large XPBD movement in a frame.

Use a short TTL, similar to existing post-KEEP contact history.

For debugging and final validation, support a full-resolution all-strands mode
internally. It can be slower but is useful to prove correctness.

## GPU/CPU Strategy

Short term:

- Implement the first version in Python/CPU using Blender ray casts or existing
  evaluated proxy geometry.
- Keep it behind a small, well-named internal function.
- Use it to validate the algorithm and correctness.

Medium term:

- Move inside/outside tests to Warp kernels.
- Keep Body proxy mesh resident on GPU.
- Send only hair point/segment candidates.
- Return compact flags and escape positions.

Long term:

- One GPU thread per active strand can repair joints sequentially.
- Alternatively, one GPU thread per candidate point can classify points, then a
  strand pass reconstructs FK.

The algorithm has sequential dependency along a strand. That is acceptable:
one thread per active strand is often a better fit than trying to parallelize
every joint and then reconciling dependencies.

## Velocity Handling

If a joint or strand is repaired:

- Zero velocity for the repaired point.
- Consider zeroing the whole strand if many points were repaired.
- Later, tangential velocity may be preserved, but start with zero velocity.

Interpretation: soft skin absorbed the invalid kinetic energy.

## Fallbacks

Some states are physically contradictory:

- Root/locked joint is already invalid.
- Segment length cannot reach outside without crossing the Body proxy.
- Hair is initialized through the head.
- Bone seed is outside or too close to the surface.

Fallback order:

1. Try straightness-preserving FK direction repair.
2. Try seed escape point plus margin.
3. If length preservation conflicts, prioritize outside placement.
4. Mark this as a hard repair count in stats.

## Statistics To Add

Add counters so testing can see what happened:

- `body_hard_guard_points`
- `body_fk_repair_strands`
- `body_fk_repair_points`
- `body_fk_escape_points`
- `body_fk_failed_points`
- `body_fk_velocity_zeroed`

The operator report should stay compact, for example:

```text
bodyGuard=123, bodyFK=54/801, escape=12, fail=0
```

## Implementation Phases

### Phase 1: Documentation and scaffolding

- This document.
- Add a small set of pure helper functions with clear names.
- No solver behavior change until helpers are testable.

### Phase 2: CPU reference repair

- Implement point outside test using nearest internal bone endpoint seed.
- Implement FK strand repair on CPU for active or all strands.
- Run after KEEP/post-KEEP and before cache/write.
- Write repaired positions back to simulator state.
- Add stats.

### Phase 3: Validation in Blender

- Use current difficult head scene.
- Compare before/after visible penetration.
- Check that hair does not become wavy or twisted.
- Confirm `bodyFK` counters correlate with visible fixes.
- Confirm frame-to-frame stability.

### Phase 4: GPU acceleration

- Port inside/outside classification to Warp.
- Keep the Body proxy Warp mesh alive.
- Return inside flags and escape positions.
- Decide whether FK reconstruction stays CPU or becomes one-thread-per-strand
  Warp kernel.

### Phase 5: Refinement

- Add multiple seed endpoints.
- Choose nearest seed per strand or per point.
- Add active strand TTL.
- Add segment crossing validation.
- Tune fallback thresholds.

## Bone Seed Selection Plan

Initial implementation:

```text
seed = Head bone tail
```

Next implementation:

```text
candidate seeds = tails of bones whose names include:
    Head
    Neck
    Spine
    Chest

choose seed nearest to current point or strand root
```

Do not use arbitrary external object centers. The seed must be inside the body.

## Important Invariants

- Body proxy only for Body inside/outside decisions.
- Repair proceeds root to tip.
- Previous repaired joint must be considered authoritative.
- Length is preserved when possible.
- Straightness is preferred over normal push-out.
- Collision normal is not the primary repair direction.
- Repaired positions must be fed back into the simulator state.
- Repaired velocities must be damped or zeroed.

## Suggested Function Names

```python
_body_internal_seed_points(body_obj) -> np.ndarray
_choose_body_seed(point, strand_root, seeds) -> Vector
_body_seed_outside_test(point, body_obj, seed) -> bool
_body_seed_escape_point(point, body_obj, seed, margin) -> tuple[Vector, bool]
_valid_body_segment(prev, candidate, body_obj, seed) -> bool
_repair_strand_fk_body(...)
_apply_body_fk_hard_repair(...)
```

## Stop Conditions

Stop and reassess if:

- Repair removes penetration but creates severe curl or twist.
- More than a small percentage of all points require escape fallback.
- The selected seed is outside the proxy.
- CPU repair becomes too slow before proving correctness.
- Segment validation contradicts point validation too often.

If any of these happen, inspect the Body proxy and seed selection before adding
more collision passes.
