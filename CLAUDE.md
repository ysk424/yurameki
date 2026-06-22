# Yurameki — Project Working Notes

This file is a handoff log for Claude Code sessions.
**Read this first** before touching anything in this repo.

---

## ⚠️ START HERE — Stage 1 server separation (2026-06-22 PM)

**PUBLIC? No — PRIVATE repo** `https://github.com/ysk424/yurameki` (branch
`master`). Push checkpoints often; PULL to roll back if a direction fails.
Discipline (user's rule): if you fail the same way a few times on coordinate
transforms, STOP and rethink — don't thrash (that burns tokens).

### Done this session (all validated, all pushed)

1. **Test data extracted from the live test project** (Blender MCP) →
   `testdata/`. Scene: armature `HD Neutral F`, body `CC_Base_Body`, hair
   `Curves` (36000 pts = 4000×9), head bone `CC_Base_Head`, frames 1..450, fps 24.
   - `collider.abc` — tanabata recipe (`bpy.ops.wm.alembic_export`,
     selected, RENDER eval, global_scale 1.0). **6 GB, gitignored** (`*.abc`).
   - `head_world.npy` (450,4,4) ground truth = `arm.matrix_world @ pb.matrix`.
   - `fk_channels.npy` (450,8,9) **tsudura-equivalent** loc/euler-XYZ/scale,
     decomposed from `pose_bone.matrix_basis`. `armature_world.npy` is constant.
   - `fk_meta.json` — chain (root..head), rest matrices, inherit flags.
   - `groom_rest.npy` (36000,3) frame-1 hair world. `eval_roots_sample.npy`.
   - No tsudura clip existed; the FK data was extracted by hand.

2. **The math (server/, pure numpy, Blender-independent):**
   - `fk.py` — FK: `pose[b]=pose[parent]·(rest[parent]⁻¹·rest[b])·basis[b]`,
     Blender Euler XYZ = `Rz·Ry·Rx`. `validate_fk.py`: reconstructed head world
     matches ground truth to **4e-7 (0.0004 mm, 1.4e-5°)** over 450 frames.
   - `validate_root_follow.py` — rigid model `root(f)=head_M(f)·head_M(1)⁻¹·root(1)`:
     **95% of roots within 0.002 mm**, worst 2.8 mm at the hairline (mixed
     neck/head weights). Model is valid.
   - NOTE: a rotation-error metric first showed 120° — that was a METRIC bug
     (0.01 world scale baked in the matrices), not FK. Caught by reasoning.

3. **Headless engine + server:**
   - `engine.py` — drives kinematic roots by the head-follow transform, runs the
     extension's **Taichi (CPU)** solver with `body_collision_fn=None`. Imports
     `_sim_taichi` (its only bpy use is lazy inside `build_body_bvh`). Full
     450-frame clip: finite, **8 s on CPU**.
   - `app.py` — TCP JSON-RPC `127.0.0.1:7780` (ping/simulate/shutdown), tanabata
     pattern. `cli.py` — client. Tested from **CLI and Blender MCP** (Blender as
     a thin client). Hair sim runs fully outside Blender.

### Environment parity (2026-06-22 PM, done)

Policy: install BPY's pip packages into the bare Python 3.13 so PY mirrors BPY.
Blender 5.1 BPY = Python 3.13.9, **warp-lang 1.13.0** (bundled, CUDA 12.9,
RTX 5070 Ti sm_120), numpy 2.3.4, NO taichi. Production ran the **Warp CUDA**
path (`_sim_warp` + `_collision_warp`); taichi was only the CPU fallback.
- Installed `warp-lang==1.13.0` into bare Python → CUDA works headless.
- `engine.py` now selects Warp for backend CUDA, Taichi for CPU/VULKAN.
- Cross-check: `--backend CUDA` vs CPU agree to **0.12 mm max** (120 frames).
- `server/requirements.txt` pins the env.

### Headless body collision — DONE & VALIDATED (2026-06-23)

Collision is now wired into the engine, headless, with the **Tokoya algorithm
unchanged** (not one kernel line touched). Validated to **~0 penetration over
the full 450-frame real animation drive**.

**Root cause of the "hair pokes through head" we saw:** NOT the collider being
unset, and NOT the collision algorithm. Two things, both documented in Tokoya
`CLAUDE.md` (sections "毛根の0.5mmオフセット", v0.3.2/3.3):
1. **Roots not 0.5 mm outside the collider.** Tokoya guaranteed this at PLANT
   time; Yurameki doesn't plant (external groom in), so it was missing. Measured:
   **1847/4000 roots (46%) buried up to 29 mm inside** CC_Base_Body. Roots are
   pinned (inverse_mass=0) AND collision-excluded (point<2), so a buried root =
   strand starts inside = must pass through to get out.
2. **substeps=1** (hardcoded in `__init__._snapshot_sim_params`) vs the proven
   **8 substeps / 20 iterations** of the zero-penetration MCP run.

**Fix (proven method, data + settings only):**
- `testdata/groom_rest_conditioned.npy` — groom pre-conditioned so every point
  with signed-dist < 0.5 mm (same normal-sign predicate the collision uses) is
  projected to `closest + normal*0.5mm`. 13.7% of points moved, 0 roots inside.
- `_collision_warp.py` — `import bpy` made lazy (inside `_evaluated_body_arrays`);
  added `triangles=(verts,indices)` ctor and `update_mesh(verts)` (refit per
  frame). **Kernels unchanged.**
- `engine.py` — `simulate(..., collider, body_frames)`; `run_from_testdata(...,
  collision=True)` seeds from the conditioned groom, feeds per-frame world body
  verts, uses substeps=8/iter=20. CLI: `--collision`.
- Body geometry extracted from the live scene (not the 6 GB abc):
  `testdata/body_verts_world.npy` (450,225184,3 world, ~1.2 GB) +
  `body_tris_idx.npy` (449472,3). Topology fixed; verts per frame.

**Validation:** `python server/engine.py --collision --start 0 --end 450`.
Robust ray-parity inside test, roots excluded, per-frame body: penetration
**0.00–0.04%** (≈1 stray point / 2800 sampled), deepest ~0–1 mm. One lone
10 mm outlier at frame 450 (1 point) — transient at the tail; note, not chase.
Before the fix: **45% inside, 69 mm deep.**

### Next (not done yet)

- **Apply the same fix to the Blender side** so the user can eyeball it: a
  `yurameki.condition_groom` operator (roots → 0.5 mm outside the Body) + set
  the interactive/bake path to substeps=8. Then real-machine visual check in
  Blender (Simulate / bake range, confirm no head poke-through).
- Investigate the single frame-450 10 mm outlier if it recurs.
- Alembic OUTPUT (geometry cache) — deferred; npz for now.
- Stage 2 = C++ (Taichi AOT) only after the algorithm is frozen. Not now.

---

## Yurameki v0.1.0 (2026-06-22)

### このプロジェクトは何か

**Yurameki（揺らめき）**: Blender 5.1 用ヘアシミュレーション拡張。
`blender-tokoya-extension` の**シミュレーション核だけ**を分岐した派生プロジェクト。
Tokoya は引き続き公開・維持する。Yurameki はそこからスピンオフした別物。

- 植毛(`_mask_plant.py`)・カット(`_mesh_ops.py`)は**持ち込まない**。
  入力は「既にある Hair Curves オブジェクト」。毛は Tokoya 等で用意する。
- 履歴を残すクローンではなく、Tokoya のシミュレーション核5ファイルを
  コピー＋`tokoya`→`yurameki` リネームして開始した。

### v0.1.0 の到達点（最初のプッシュ）

- 指定レンジ（Start..End）または全体のシミュレーションを一括実行。
- Start/End 欄。初期値はシーンのフレーム範囲（`frame_start`〜`frame_end`）。
  `register()` と `load_post` で `_sync_bake_range_to_scene()` がセットする。
  `Use Scene Range` ボタンでも再取得可能。
- Export Path 欄（`//hair.abc`、subtype FILE_PATH）と Export ボタン。
- **Alembic 書き出しは未実装**（UIのみ）。`YURAMEKI_OT_export_alembic` は
  WARNING を出して CANCELLED を返すスタブ。ベイクは C++ サーバーへ移すため、
  オンデバイス書き出しは意図的に未実装。

### 構成（Tokoya から引き継いだ核）

```
_sim_taichi.py        — Taichi XPBD ソルバー、可変9点ストランド
_sim_warp.py          — Warp CUDA 共有状態ソルバー
_collision_warp.py    — Warp CUDA Body 衝突バッチ処理
_world_passthrough.py — 現在フレームの静的 Simulate と Body 衝突
_recording.py         — REC/PLAYBACK、フレーム補間、RAM・NPZ キャッシュ
                        + bake_range()（v0.1.0 で追加：レンジ一括ベイク）
__init__.py           — Operator、WM Property、persistent handler
ui.py                 — Yurameki N-パネル（タブ "Yurameki"）
yurameki_defaults.json— 物理パラメーターデフォルト
blender_manifest.toml — 拡張 manifest と配布対象
```

### bake_range の仕組み（_recording.py）

- 対話録画と**同じ** `_simulate_next()` 経路を再利用する。
- `scene.frame_set(start)` → `start()` で録画開始 → `start+1..end` を
  `_simulate_next` でループ → `stop()`。結果は `self.frames` に入り、
  タイムライン再生で baked モーションがそのまま再生される。
- POINTS_PER_STRAND=9 固定。CACHE_SUFFIX は `.yurameki-cache.npz`。

### Operators（6個）

| bl_idname | 役割 |
|---|---|
| `yurameki.simulate` | 現在フレームの静的整髪（N steps） |
| `yurameki.bake_range` | Start..End を一括シミュレーション |
| `yurameki.use_scene_range` | Start/End にシーン範囲をセット |
| `yurameki.record` | タイムライン REC トグル |
| `yurameki.export_alembic` | **スタブ**（未実装） |
| `yurameki.pick_body` | Active を Body に設定 |

### 配布物

- `dist/yurameki-0.1.0.zip`
- ビルド: Blender の `blender --command extension build`、または
  manifest の `[build].paths` を ZIP 直下に固めるだけ（ルートに
  `__init__.py` が来る構造）。

### 次の作業（サーバー化）

- ベイク計算を **C++ サーバー** へ移す方針。仕様は v0.1.0 確定後に決める。
- Alembic 書き出しはサーバー側で実装する。Blender 側はシーン/カーブの
  入力出しと結果の取り込みに専念する想定。

### Tokoya 由来の地雷（引き継ぎ）

- `from __future__ import annotations` は `@ti.kernel` の型注釈を壊す
  （PEP 563）。**ただし `_sim_taichi.py` 内のカーネル定義に対してのみ**。
  他ファイルの `from __future__` は問題ない。
- `@ti.kernel` 引数はスカラーのみ。ndarray は from_numpy/to_numpy 経由。
- Taichi のキャッシュ更新はアンインストール→再インストールが確実。
- POINTS_PER_STRAND は `_recording.py` と `_world_passthrough.py` で一致必須。
- Body BVH はワールド座標で構築（CC Body は world scale 0.01）。

---

## オーナー情報

Owner: `azoo` / `ysk424` (ysk424@hotmail.com)
Communication: 主に日本語
Platform: Windows 11, RTX 5070 Ti (CUDA sm_120), Blender 5.1
