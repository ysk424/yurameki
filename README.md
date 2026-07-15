# Yurameki 1.0.0

NVIDIA Warp elastic-rod long straight-hair simulator for Blender.
Blender 用 NVIDIA Warp 弾性ロッド・ロングストレートヘア シミュレータ。

License: MIT. Requires an NVIDIA CUDA GPU. / ライセンス: MIT。NVIDIA CUDA GPU が必要です。

---

## 日本語

Yurameki は、VR キャラクター風のロングストレートヘアを NVIDIA Warp で揺らす
Blender 拡張です。各ストランドを **Stable Cosserat 弾性ロッド**として解きます
（セグメントごとのクォータニオンフレーム＋伸び/せん断・曲げ/ねじれエネルギー）。
伸び/せん断エネルギーが各セグメントを本来の長さに保つため、従来の「XPBD で解いて
FK で繋ぎ直す」処理は不要です。

Yurameki は**すでにグルーミング済みの Blender Curves オブジェクト**を入力に取り、
**動きだけ**をシミュレートします（植毛・カット・整形は Tokoya 側の役割）。

### 動作要件

- Blender 5.1 以降（Windows x64）
- **NVIDIA CUDA 対応 GPU**（`cuda:0` で動作します）
- Blender の Python 環境から NVIDIA `warp-lang` を import できること

### インストール

Blender の拡張機能インストーラからリリース ZIP を入れてください:

```text
dist/yurameki-1.0.0.zip
```

### 使い方

N パネル（3D ビューポート右の「Yurameki」タブ）で操作します。

1. **Input** — スポイトで対象を指定
   - `Hair`: シミュレートする Curves
   - `Body`: 衝突用のボディメッシュ（元メッシュをそのまま使用）
   - `Clothes`: 任意の衣装メッシュ
2. **Simulate** — フレーム範囲を計算
   - **各フレームを計算しながらその場で描画**します。見ながら確認でき、
     結果が NG なら **Stop ボタン**または **Esc** で中断できます。
   - 中断しても**それまでに計算済みのフレームは残ります**（そのまま再生・Bake 可能）。
   - `Start Frame` は初期状態。シミュレートは `Start Frame + 1` から
     `End Frame` まで（`End Frame > Start Frame`）。
3. **Bake Cache** — ランタイムキャッシュを Curves の位置キーフレームに焼き込み

出力モード（`Output`）:
- `Runtime Cache`（既定）: 再生をキャッシュで保持（大量の F-Curve を作らない）
- `Position Keyframes`: Curves の `position` F-Curve に直接ベイク
- `Final Preview`: 最終フレームの静止形状だけを表示

### 適応ルートロック（Adaptive Root Lock）

頭が突っ込んでくる側の毛は、頭蓋骨と髪の抵抗で押さえられて動きにくくなります。
Yurameki はこれを模し、**頭が進んでいる方向側**のストランドについて、根本から
**耳たぶの高さまでの関節をキネマティック（頭に追従）化**します。突っ込まれる側の
毛は頭に乗って弾かれず、後ろ側の毛は自由に揺れます。

- トリガーは頭の**運動の向きのみ**（速度の大きさは無視）。ゆっくりした動きでも作動。
- 頭の進行方向 `v_head` は頭ボーン先端（頭頂）の速度で測るため、うなずき・首振り
  （回転）も拾います。
- 進行方向を軸にした 3D 円錐で選別（75° 以内はフル、90° まで滑らかに減衰）。
- 耳たぶ基準は頭ボーン付近の顎関節（`CC_Base_JawRoot`、無ければ目ボーン）。
- ロック数はベースラインの `Root Locked Points`（既定 3）を下回りません。

### チューニング

各項目には N パネル上にツールチップがあります。動きを決める主なもの:

- `Bend Stiffness log10`: 曲げ/ねじれ剛性。高いほど硬く直毛的、低いほど柔らかく暴れる。
- `Particle Mass g`: 運動量。低いほど軽快でオーバーシュート少、高いほど大きく揺れる。
- `Damping`: 全体の減衰（すべての動きの収束速度）。
- `Internal Damping`: ストランド内部のひずみ速度減衰。リンギング・ジッタ・毛羽立ちを
  抑えつつ、頭と重力による受動的な追従は残します。`Damping` を上げるより先にこちらを。

伸びは固定（ロッドは伸縮しない）ため、伸びのつまみはありません。

### 衝突について

Warp Mesh のレイ／最近傍クエリを使用。Body と Clothes は別々の Warp メッシュ:
Body は最近傍面の**符号付き**押し出し（法線符号による内外判定）、Clothes は両面の
符号なし押し出し。セグメント補正はクランプされ、1 パスで端点がキャラクターを
横断してテレポートすることはありません。毛同士の衝突は未実装です。

Body には**元のメッシュをそのまま**使います（穴を塞ぐプロキシは廃止）。頭皮の外側を
覆う髪では、法線符号の内外判定で十分に正しく動作します。

### ライセンス

MIT License. Copyright (c) 2026 Yoshihiko Tsukamoto. 同梱の `LICENSE` を参照。

---

## English

Yurameki is a Blender extension that simulates VR-character-style long straight
hair with NVIDIA Warp. Each strand is solved as a **Stable Cosserat elastic rod**
(per-segment quaternion frames with stretch/shear and bend/twist energies). The
stretch/shear energy keeps every segment at rest length intrinsically, so the old
"XPBD solve, then reconnect the rod by FK" step is unnecessary.

Yurameki takes an **already-groomed Blender Curves object** as input and simulates
**motion only** (planting, cutting, and styling belong to Tokoya).

### Requirements

- Blender 5.1 or newer (Windows x64)
- An **NVIDIA CUDA-capable GPU** (it runs on `cuda:0`)
- The Blender Python environment must be able to import NVIDIA `warp-lang`

### Installation

Install the release ZIP through Blender's extension/add-on installer:

```text
dist/yurameki-1.0.0.zip
```

### Usage

Everything lives in the N-panel ("Yurameki" tab in the 3D viewport).

1. **Input** — pick objects with the eyedropper fields
   - `Hair`: the Curves to simulate
   - `Body`: the body mesh collider (used directly)
   - `Clothes`: an optional garment mesh
2. **Simulate** — bake the frame range
   - **Each frame is drawn as it is computed**, so you can watch the result and,
     if it looks wrong, end early with the **Stop** button or **Esc**.
   - Stopping **keeps the frames computed so far** (still playable and bakeable).
   - `Start Frame` is the unchanged initial state; simulation runs from
     `Start Frame + 1` through `End Frame` (`End Frame > Start Frame`).
3. **Bake Cache** — convert the runtime cache to Curves position keyframes

Output modes (`Output`):
- `Runtime Cache` (default): keeps playback in a cache instead of creating
  millions of Curves `position` F-Curves
- `Position Keyframes`: bakes the Curves `position` F-Curves directly
- `Final Preview`: shows only the static final-frame shape

### Adaptive Root Lock

On the side the head is advancing toward, the skull and drag pin the hair so it
cannot move. Yurameki models this: for strands in the head's direction of motion,
it makes the joints from the root **down to the earlobe line** kinematic (they ride
the head). Hair the head drives into is pinned and cannot be flung, while trailing
hair keeps swinging.

- The trigger is the head's **direction of motion only** (magnitude ignored), so a
  slow turn still engages it.
- The head advance velocity `v_head` is sampled at the head-bone tip (top of skull),
  so nods and head-shakes (rotation) register too.
- Strands are selected by a 3D cone around `v_head` (full lock within 75°, a soft
  skirt to 90°).
- The earlobe line uses the jaw-hinge bone `CC_Base_JawRoot` (eye bones as a
  fallback). The lock never drops below the `Root Locked Points` baseline (default 3).

### Tuning

Every control has a tooltip. The ones that shape motion:

- `Bend Stiffness log10`: rod bend/twist stiffness (higher = stiffer/straighter).
- `Particle Mass g`: momentum (lower = lighter/snappier, less overshoot).
- `Damping`: how quickly all motion settles.
- `Internal Damping`: strain-rate damping inside a strand; removes ringing, jitter,
  and frizz while keeping the passive follow-through. Prefer it over raising
  `Damping`.

Stretch is fixed (the rod is inextensible), so there is no stretch knob.

### Collision

Collision uses Warp Mesh ray and nearest-point queries. Body and Clothes are kept
as separate Warp meshes: Body uses signed nearest-surface push-out (inside/outside
from the closest-face normal), Clothes uses two-sided unsigned push-out. Segment
corrections are clamped so an endpoint cannot teleport across the character in one
pass. Hair-hair collision is not implemented.

The Body collider is the **source mesh used directly** (the hole-capping proxy has
been removed): the closest-face normal test is reliable for hair on the outside of
the head.

### License

MIT License. Copyright (c) 2026 Yoshihiko Tsukamoto. See the bundled `LICENSE`.
