# Yurameki CUDA Prototype Notes

Status: active development fork.

Current version: 0.4.10.

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
- Latest package: `dist/yurameki-0.4.10.zip`.

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

In Blender, install/use `yurameki-0.4.10.zip`, then:

1. Press `Check Hair`.
2. Use `Apply FK Root Pull` if FK needs a quick sanity check.
3. Select `CC_Base_Body` or the intended mesh collider.
4. Press `Pick Collider`.
5. Use `Detect CUDA Collider` to confirm hit count.
6. Use `Apply CUDA Avoidance` and inspect whether hair moves away from the body.

Important: broadphase uniform grid is not implemented yet.  The current CUDA
collider path can be slow on full 400k+ triangle meshes because it is still the
first correctness pass.

## Intended Order

1. Finish 1 cm cylinder FK.
2. Make CUDA collider avoidance visually usable.
3. Add hair-vs-hair collision using the fixed solve order.
4. Optimize collider broadphase with CUB radix sort / grid cells.

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
- max surface run: 30mm
- result: length error about 0.000065mm, tip direction dot(-Z) about 0.997

This looks much better around the ear: it can slide on the body briefly, then leave the body and fall vertically instead of sticking to the neck.

## Next topic: hair overlap / stacking

Hair-vs-hair is deliberately not solved in 0.4.10.  Two candidate directions are under consideration:

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
