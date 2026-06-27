# Yurameki（揺らめき）

Yurameki（揺らめき、*shimmer/sway*）は、Blender 5.1用のヘアシミュレーション拡張です。
[Tokoya](https://github.com/ysk424/blender-tokoya-extension) からシミュレーション核だけを
取り出した派生プロジェクトで、既存のHair CurvesオブジェクトをアニメーションするBodyメッシュに
沿って計算し、フレームレンジを一括ベイクします。

植毛・カット・スタイリングは行いません。毛はTokoyaなどで用意してください。

## 主な機能

- Taichi XPBD ソルバー（CUDA 既定 / Vulkan / CPU）
- NVIDIA Warp による CUDA Body 衝突のバッチ処理
- 現在フレームの静的整髪（`Simulate`）
- **指定レンジの一括シミュレーション（`Simulate Range`）**
- Start / End フレーム指定。初期値はシーンのフレーム範囲（1〜最終フレーム）
- `Simulate Range` は録画経路でレンジをベイクし、圧縮キャッシュで再生します
- Alembic 書き出し欄（v0.1.0 ではUIのみ。実処理は後続のサーバーで実装予定）
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
- Python パッケージ `taichi`（Blender の Python 3.13 が参照する user site-packages へ）
- NVIDIA Warp（Blender 5.1 同梱版）
- CUDA 利用時は対応する NVIDIA GPU とドライバー

## 基本操作

1. シーンに Hair Curves オブジェクトを1つ用意します（Tokoya で植毛したものなど）。
2. `Body` にアニメーション追従兼コライダーの Mesh を設定します。
   必要なら `Cloth` に Alembic geometry-cache Mesh を設定します。
3. 必要なら `Simulate` で現在フレームの形を整えます。
4. `Bake & Export` で Start / End を指定（`Use Scene Range` でシーン範囲を流用）。
5. `Simulate Range` でレンジ全体を計算し、各フレームをキャッシュします。
6. 必要なら `Clean 1` / `Clean 2` を押して、Start / End のキャッシュ範囲を修復します。
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

## Clean repair notes

The UI name is `Clean`; the original working name was `Comb`. `Clean 1` and
`Clean 2` operate on the baked cache for the configured Start-End range. The
current frame display is refreshed after the range cache is repaired. `Clean 3`
is still a reserved button.

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

Clean 2 is the tail-bend repair. A strand is selected when at least
one tail bend angle at `p5`, `p6`, or `p7` is `>= 0.5` radians, where a bend
angle is:

```text
angle(pJ) = acos(dot(normalize(pJ - pJ-1), normalize(pJ+1 - pJ)))
```

Straight continuation is `0` radians. The default threshold is `0.5` radians
(about 28.65 degrees). Once selected, the repair is the same root-preserving
neighbour interpolation used by Clean 1: find valid nearby UV-neighbour strands,
blend the nearest 2-4 valid root-relative curves, and rebuild `p1..p8` through
the tip while keeping `p0` fixed. In the current frame-123 MCP test, Clean 2
selected 151 strands, repaired all 151, and reduced the selected tail-bend
maximum below `0.5` radians.

Clean repairs must verify the evaluated Curves result after writing. A single
write to the original Curves datablock may not survive the Deform Curves on
Surface / Surface Deform round-trip for large shape changes. The practical
writeback path is iterative: write the desired evaluated world curve using the
current evaluated-original offset, update the depsgraph, re-read the evaluated
curve, then repeat until the measured tail-bend/error threshold is satisfied.
In the frame-123 Clean-2 test, one-shot writeback left visible failures, while
iterative writeback reached zero `p5..p7 >= 0.5 rad` strands after 7 iterations.

The Empty is only an expensive interactive label, not the final detection
method. It can still be used as a root-finder/debug probe: given an Empty or
picked world-space point near a suspicious hair point, report the owning strand
number, point index, root position, and tip position.

## ライセンス

MIT License. Tokoya 由来のコードを含みます。
