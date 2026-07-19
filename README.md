# Yurameki 0.3.0

Blender 5.2+ 向けの C++20 / OpenMP ロングヘア・シミュレータです。Blender
Curves の各カーブを番号順のストランドとして扱い、各隣接点を1本のロッドとします。
UI、Depsgraph、Curves 入出力だけを Python に残し、数値計算は
`_yurameki_native_0_3_0` が行います。

## インストール

Blender の拡張機能インストーラから `dist/yurameki-0.3.0.zip` を指定します。このZIPには
Windows x64 / Blender 5.2 / CPython 3.13用のネイティブモジュールが含まれます。

## シミュレーションの順序

1. 対象範囲、フレーム1、元の表示フレームについて、評価済み Curves を先に読みます。
2. **フレーム1の全ストランド・全ロッド長、関節角、曲率変動**を不変な基準値として
   C++ に保存します。開始フレームが1以外でも基準は変わりません。
3. 各フレームをサブステップ化し、ガイドストランドの Stable Cosserat rod を解きます。
   `Guide Decimation > 1` なら近い最大4本のガイド変位から全ストランドを復元します。
4. Body は面法線の符号付き距離、Clothes は両面距離として、点とロッド区間を BVH で
   衝突判定します。
5. 各ストランドについて次の有限ループを実行します。
   - 形状評価: ロッド長誤差、隣接ロッド角、フレーム1からの角度変化、曲率変動
   - NG: 接線を局所的に平滑化し、必要ならフレーム1のロッド長へ投影
   - 貫通評価: 点のマージン違反とロッド区間の三角形交差
   - NG: 衝突修復、変位のストランド方向平滑化、再評価
   - 最大反復数、改善停止回数、最良候補の保存で無限ループを防止
6. 結果を Curves に書き、Surface Deform 等を含む評価済み結果を読み直します。変形後に
   再び NG なら、設定回数まで C++ の整髪・貫通評価へ戻します。
7. 最終的に実際に Curves へ書かれたローカル座標をキャッシュまたはキーフレームへ保存
   します。途中停止時も計算済みフレームをランタイムキャッシュに残します。

## なめらかさの評価

点列を単にベジェカーブと仮定して微分するのではなく、Blender Curves の制御点列から
正規化ロッド接線 `t[i]` を作ります。一次差分 `b[i] = t[i+1] - t[i]` が曲率に相当し、
二次差分の平均絶対量

```text
roughness = sum(length(b[i+1] - b[i])) / number_of_differences
```

を乱れの指標にします。フレーム1自身の値に係数を掛けた上限と絶対下限を併用し、
意図された曲率を残しながら急な二次差分を検出します。これに、最大関節角、フレーム1からの角度変化、
ロッド長誤差を組み合わせて OK/NG を判定します。したがって「絶対値を合計する」という
発想は使っていますが、ワールド座標の微分値そのものではなく、弧長に依存しにくい接線
とその差分を使います。

## 要件と開発ビルド

- Windows x64
- Blender 5.2 以降。現在の開発 ABI は Blender 5.2 / CPython 3.13
- Visual Studio 2022、CMake 3.24+、pybind11、OpenMP
- CUDA や `warp-lang` は不要

```powershell
$python = 'C:\Users\azoo\git\build_windows_Release_x64_vc17_Release\bin\5.2\python\bin\python.exe'
$pythonLib = 'C:\Users\azoo\git\blender\lib\windows_x64\python\313\libs\python313.lib'
cmake -S native -B build/native -G 'Visual Studio 17 2022' -A x64 `
  "-DPython3_EXECUTABLE:FILEPATH=$python" `
  "-DPython3_LIBRARY:FILEPATH=$pythonLib"
cmake --build build/native --config Release
& $python tools/validate_native.py
```

生成物はアドオン直下の `_yurameki_native_0_3_0.cp313-win_amd64.pyd` です。この手順は
拡張 ZIP を作りません。

## UI

- `Input`: Hair Curves、Body、任意の Clothes
- `Simulate`: フレーム範囲、出力、ガイド間引き、ロッド長保持
- `Elastic Rod`: 質量、重力、減衰、曲げ剛性、反復
- `Collision`: マージン、探索距離、補正上限、衝突後反復
- `Per-frame Grooming`: 評価閾値、整髪強度、収束と停滞の上限
- `Compute`: OpenMP スレッド数。0 は論理プロセッサをすべて使用

Body と Clothes の衝突は実装済みです。毛同士の衝突は未実装です。Blender の RNA と
Depsgraph はメインスレッドだけから操作し、OpenMP は独立したストランド、BVH の
三角形準備とクエリ、評価・整髪に使用します。BVH の木構築自体は逐次処理です。

MIT License. Copyright (c) 2026 Yoshihiko Tsukamoto.
