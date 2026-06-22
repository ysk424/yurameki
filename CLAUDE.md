# Yurameki — Project Working Notes

This file is a handoff log for Claude Code sessions.
**Read this first** before touching anything in this repo.

---

## ⚠️ START HERE — Yurameki v0.1.0 (2026-06-22)

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
