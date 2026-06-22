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
- タイムライン録画（`REC`）と圧縮キャッシュ再生
- Alembic 書き出し欄（v0.1.0 ではUIのみ。実処理は後続のサーバーで実装予定）

セルフコリジョンは実装していません。

## 必要環境

- Blender 5.1以降 / Windows x64
- Python パッケージ `taichi`（Blender の Python 3.13 が参照する user site-packages へ）
- NVIDIA Warp（Blender 5.1 同梱版）
- CUDA 利用時は対応する NVIDIA GPU とドライバー

## 基本操作

1. シーンに Hair Curves オブジェクトを1つ用意します（Tokoya で植毛したものなど）。
2. `Body` にアニメーション追従兼コライダーの Mesh を設定します。
3. 必要なら `Simulate` で現在フレームの形を整えます。
4. `Bake & Export` で Start / End を指定（`Use Scene Range` でシーン範囲を流用）。
5. `Simulate Range` でレンジ全体を計算し、各フレームをキャッシュします。
6. タイムラインを再生すると、ベイク結果が再生されます。

## Alembic 書き出し

v0.1.0 では Export 欄とボタンを用意するのみで、実処理は実装していません。
ベイク計算は別途 C++ サーバーへ移行する予定で、書き出しはそちら側で扱います。

## ライセンス

MIT License. Tokoya 由来のコードを含みます。
