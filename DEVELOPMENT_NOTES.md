# Yurameki CUDA Prototype Notes

Status: active development fork.

Current version: 0.4.6.

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
- Latest package: `dist/yurameki-0.4.6.zip`.

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

In Blender, install/use `yurameki-0.4.6.zip`, then:

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
