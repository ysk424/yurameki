# Yurameki（揺らめき）

Yurameki（揺らめき、*shimmer/sway*）は、Blender 5.1用のヘアシミュレーション拡張です。
名前は日本語の動詞「揺らめく」の名詞形です。
[Tokoya](https://github.com/ysk424/blender-tokoya-extension) からシミュレーション核だけを
取り出した派生プロジェクトで、既存のHair CurvesオブジェクトをアニメーションするBodyメッシュに
沿って計算し、フレームレンジを一括ベイクします。

植毛・カット・スタイリングは行いません。毛はTokoyaなどで用意してください。

## Demo video

- [YURAMEKI: Free Long Hair Simulation for Blender](https://youtu.be/a5TuKmiJEBw)

## 主な機能

- NVIDIA Warp による CUDA XPBD ソルバー
- NVIDIA Warp による CUDA Body / Cloth 衝突のバッチ処理
- 現在フレームの静的整髪（`Simulate`）
- **指定レンジの一括シミュレーション（`Simulate Range`）**
- Start / End フレーム指定。初期値はシーンのフレーム範囲（1〜最終フレーム）
- `Simulate Range` は録画経路でレンジをベイクし、圧縮キャッシュで再生します
- v0.3.1: `Auto Frame Interpolation` is enabled by default. During range
  simulation, Yurameki chooses sub-frame physics steps from root motion and root
  spacing, and shows bake progress in Blender's progress meter and the panel.
- Alembic 書き出し欄（v0.1.0 ではUIのみ。実処理は後続のサーバーで実装予定）
- v0.3.0: strandごとの点数固定を解除しました。同一Curves内の全strandが
  同じpoints数なら、9 points以外でもシミュレーションできます。
- v0.2.1 CUDA version: CUDA/Warp 専用化。CPU / Vulkan / Taichi fallback と
  Python Clean up / Condition Groom UI を削除しました。
- v0.1.10: Clean 2 now selects strands by total bend angle over all internal
  joints, and Clean 3 adds an aggressive even/odd strand-number smoothing pass.
- v0.1.9: CUDA/Warp collision meshes are reused during range bake; animated
  Body/Cloth evaluated vertices are updated per subframe and the mesh BVH is
  refit instead of rebuilt.
- v0.1.8: optional `Cloth` collider picker. The selected mesh is read as an
  evaluated mesh each subframe, so Alembic geometry-cache deformation is used
  for CUDA/Warp collision together with the Body mesh.
- v0.1.7: `Comb` を `Clean` に改名し、Clean 1 / Clean 2 を Start-End のベイク範囲へ適用するようにしました。RECボタンは削除し、`Simulate Range` が録画開始を兼ねます。
- v0.1.6: ベイク済みの現在フレームに対する修復ボタンを追加しました。
- v0.1.5: Warp CUDA 経路に rest pose からの角度LIMITを追加しました。
- v0.1.4: 頭部の急な移動で Body が毛へ入り込むケースを抑えるため、
  自由点を毛根移動へ事前追従させます。

セルフコリジョンは実装していません。

## 必要環境

- Blender 5.1以降 / Windows x64
- NVIDIA Warp（Blender 5.1 同梱版）
- 対応する NVIDIA GPU とドライバー

## 入力ヘアの制約 / Input hair requirements

- Yurameki v0.3.0 supports variable points per strand.
- All strands in one Curves object must have the same point count.
- The minimum supported count is `3 points per strand`.
- Each strand uses `p0` as the root, `p1` as the root anchor, and `p2..tip`
  as free points.
- `Check Hair` verifies these conditions before simulation. `Simulate` and
  `Simulate Range` also run the same check and stop with a warning if it fails.
- `Root Min Distance mm` separates roots that are too close by moving the whole
  strand slightly, preserving its shape as much as possible.

## 基本操作

1. シーンに Hair Curves オブジェクトを1つ用意します（Tokoya で植毛したものなど）。
2. `Body` にアニメーション追従兼コライダーの Mesh を設定します。
   必要なら `Cloth` に Alembic geometry-cache Mesh を設定します。
3. `Check Hair` でstrand構造とroot距離を確認します。
4. 必要なら `Simulate` で現在フレームの形を整えます。
5. `Bake & Export` で Start / End を指定（`Use Scene Range` でシーン範囲を流用）。
6. `Auto Frame Interpolation` をオンにしたまま `Simulate Range` を実行します。
   速いフレームでは物理サブステップ数が自動で増え、進捗はパネルと
   Blenderのプログレスメーターに表示されます。
7. タイムラインを再生すると、ベイク結果が再生されます。

## Alembic 書き出し

v0.1.0 では Export 欄とボタンを用意するのみで、実処理は実装していません。
ベイク計算は別途 C++ サーバーへ移行する予定で、書き出しはそちら側で扱います。

## Angle limit

v0.1.5 adds an experimental Warp CUDA angle limit. For each strand triplet
`p0, p1, p2`, Yurameki compares the current internal vector angle against the
rest-pose angle captured when the solver is built. The default global limit is
`1.0` radian, configured by `ANGLE_LIMIT_RAD` in `yurameki_defaults.json` or
`_world_passthrough.py`.

The implementation constrains the equivalent `p0-p2` chord range on the GPU
after the existing segment and bending springs. It is intentionally a single
global value for now, so the effect of the angle limit can be evaluated before
adding UI controls or per-point/texture-style maps.

## Collision tuning

The Physics panel exposes two collision controls in millimeters:

- `Collision Radius mm`: the surface clearance used when pushing hair outside
  the collider. The default is `0.5 mm`.
- `Collision Search mm`: the nearest-surface search distance used by Warp mesh
  collision. The default is `3.0 mm`.

Larger values can reduce visible penetration, but may also make the hair look
slightly more inflated around the scalp or clothing.

## Clean repair notes (v0.1 archived)

The v0.1 UI name was `Clean`; the original working name was `Comb`. These
Python/Numpy repair buttons are archived in the v0.1 series. The unstable
v0.2.0 branch removes them from the active UI so the next solver can stay on
Warp CUDA.

Clean 1 is the local-neighbour repair. For every strand, build a
stable neighbour list from `surface_uv_coordinate` using the nearest 16 strand
roots in UV space. The separation score is:

```text
score(strand) =
  max over points p1..p5 [
    median distance from this strand point to the same point index on
    the 16 UV-neighbour strands
  ]
```

The default repair threshold should be `20 mm`. In the current test scene this
keeps the normal root-zone spread near the median range while selecting the
clear outliers: at frame 125, the Python/Numpy prototype selected 414 of 6000
strands (6.9%) with `score > 20 mm`. The same prototype took about 0.46 seconds
including evaluated-curve readback and neighbour-list construction; with the UV
neighbour list cached, the score calculation itself was about 0.02 seconds.

The repair target should be built only from valid neighbours whose own score is
at or below the threshold. Detection uses the root-zone points `p1..p5`, but the
actual repair must rebuild the full strand shape through the tip. Copy the
valid neighbours' root-relative curves for `p1..p8` back onto the broken strand
root, preferably as an inverse-distance weighted blend of the nearest 2-4 valid
neighbours. Keep the broken strand root fixed and blend the result by a user
strength value.

Clean 2 is the aggressive long-straight-hair repair. A strand is selected when
the sum of all internal bend angles at `p1..p7` is greater than `210` degrees,
where each bend angle is:

```text
angle(pJ) = acos(dot(normalize(pJ - pJ-1), normalize(pJ+1 - pJ)))
```

Straight continuation is `0` radians. The default total threshold is
`210` degrees (`3.665191429` radians), or an average of `30` degrees over the
seven internal joints of a 9-point strand. Once selected, the repair is the
same root-preserving neighbour interpolation used by Clean 1: find valid nearby
UV-neighbour strands, blend the nearest 2-4 valid root-relative curves, and
rebuild `p1..p8` through the tip while keeping `p0` fixed. This is intentionally
strong and intended for long straight hair.

Clean 3 is an even stronger smoothing pass for baked cache frames. It treats the
first and last strand numbers as fixed boundaries. First it rewrites every even
interior strand number, then every odd interior strand number. Each rewritten
strand gets all points `p0..p8` from the midpoint of the previous and next
strand numbers. This intentionally moves roots as well as tips, and is meant as
a user-triggered seasoning pass rather than an always-on simulation rule.

Clean up buttons are intentionally manual and taste-dependent. They are closer
to optional seasoning than to a physically neutral simulation step. In the
current long-straight-hair test, Clean up is visibly effective, but the preferred
amount depends on the look of the shot.

One current performance reference is about 4 minutes for a 240-frame simulation
on the author's test scene. This is a scene-specific measurement, not a
guarantee, but it is useful as a rough baseline while tuning the CUDA/Warp path.

Clean repairs must verify the evaluated Curves result after writing. A single
write to the original Curves datablock may not survive the Deform Curves on
Surface / Surface Deform round-trip for large shape changes. The practical
writeback path is iterative: write the desired evaluated world curve using the
current evaluated-original offset, update the depsgraph, re-read the evaluated
curve, then repeat until the measured error threshold is satisfied.

The Empty is only an expensive interactive label, not the final detection
method. It can still be used as a root-finder/debug probe: given an Empty or
picked world-space point near a suspicious hair point, report the owning strand
number, point index, root position, and tip position.

## Development note

If you want to modify the simulation, using Codex is recommended. It is useful
for tracing the solver code, comparing cleanup strategies, and making small
experimental changes safely.

## ライセンス

MIT License. Tokoya 由来のコードを含みます。
