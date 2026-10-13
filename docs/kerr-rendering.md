# 数値電場を使った Blender 動画の再生成

`scripts/render_kerr_films.py` は、[数値計算の仕様](kerr-physics.md)で作った
`artifacts/kerr-physics/` の複素電場を読み、接写した光ファイバの内部に
三つの独立した動画として表示する。動画の形状を描くために、伝搬計算を
作り直したり、測定結果を調整したりしない。

| ファイル | 入力ケース |
| --- | --- |
| `01_self_phase_modulation.mp4` / `.blend` | 分散を切った Kerr 自己位相変調 |
| `02_chromatic_dispersion.mp4` / `.blend` | Kerr を切った分散 |
| `03_kerr_dispersion_interplay.mp4` / `.blend` | Kerr と分散の同時作用 |

保存先は `reports/animations/kerr_physics/`。三つの 1920×1080、24 fps の
動画に、ポスター `poster_01.png`〜`poster_03.png`、ネイティブ `.blend`、
`manifest.json`、字幕用フォントのライセンスを添える。説明資料は、この三つの
生成済み動画と数値・描画 manifest を入力として別途作成する。

## 再生成コマンド

既存の数値 artifact の manifest と NPZ の SHA-256 を照合してから開始する。
Blender 5.2.1 LTS、NVIDIA OptiX に対応した GPU、FFmpeg、FFprobe、Python の
Pillow と IPAexGothic が必要。Blender 自身に同梱された NumPy を使って
電場を可視化する。CUDA と CPU への無言のフォールバックは行わない。

```bash
# まず各ケースの冒頭・中間・80km の静止画を作る。
.venv/bin/python scripts/render_kerr_films.py --mode stills

# 同じ静止画と、実データによる2秒の短い動作検証を .work/ に作る。
.venv/bin/python scripts/render_kerr_films.py --mode proof

# ネイティブ Blender シーンだけを保存する。
.venv/bin/python scripts/render_kerr_films.py --mode build

# 3本の本編をレンダリングして MP4、ポスター、manifest を保存する。
.venv/bin/python scripts/render_kerr_films.py --mode render
```

`--blender` で実行ファイルを指定できる。WSL から Windows Blender を呼ぶときは
入力・出力のパスを `wslpath` で変換する。現在の環境では次の実行ファイルを
自動検出する。

```bash
.venv/bin/python scripts/render_kerr_films.py --mode stills \
  --blender '/mnt/c/Program Files/Blender Foundation/Blender 5.2/blender.exe'
```

`--case pure_spm` などで対象を一つに限定できる。`--mode proof` は選択に combined が
あればそれを、なければ最後に選択したケースを2秒の動作検証に使う。外部の
`--data-dir` も使え、manifest にその絶対パスを明示する。`--duration` は 10〜20 秒、
標準は 12 秒。`--samples` と `--adaptive-threshold` は品質設定で、標準は
128 samples と 0.014。`--resume` は source・データ・Blender の build hash・
OptiX のデバイス識別子・GPU driver・品質設定・動画長が同じ場合だけ既存の PNG を
再利用する。SHA-256、PNG の chunk CRC、寸法、圧縮された画素ストリームの
decode と行数も照合する。破損または置き換えられたフレームは再描画する。
driver 情報を取得できない環境では再開を許可しない。設定の変更後は新しい scratch
ディレクトリを使うか、`--resume` を外して再生成する。

本編のフレーム、レンダリングログ、所要時間、短い検証動画は `.work/kerr-cinema/`
に置く。大量の連続 PNG は成果物に含めない。MP4 の全フレームを Cycles で
描画し、動画の光学フローによる補間や、フレーム複製による高速化は行わない。

## 表示の意味

冒頭はコア 8.2 µm、クラッド 125 µm の断面比を保つ。値の出典は
[Corning SMF-28 ULL のデータシート](https://www.corning.com/media/worldwide/coc/documents/Fiber/SMF-28%20ULL.pdf)。
その後は、ガラスを切り欠いた内部にある電場を拡大する。拡大された形状は
電場値を読むための図示であり、コアが Kerr 効果で膨張するという意味ではない。
具体的な倍率と移行フレームは動画 manifest の `visual_encoding` に保存する。

長手方向の位置は、表示された伝搬距離でのパルス内の相対時刻を表す。
生成する電場形状の範囲は −240〜+240 ps で、カメラはその内部を接写する。
映像の試料の長さを 80 km の実寸比に対応させたり、異なる位置を同時に
観測した物理的な写真として扱ったりしない。接写では低い強度の外側の形状が
画面の外に出る場合がある。manifest の欠落エネルギーは、この数値データ窓に
よる欠落を指し、画面投影の完全性を保証する値ではない。

電場の曲線は各偏波の実部
`Re[A_pol(z,t) exp(-i 2πt / 12ps)]` から作る。X と Y の電場成分を互いに
直交する方向へ配置する。同じ複素位相を持つ `Ax=Ay` は 45 度の直線偏光を
保ち、位相の変化を偏光の回転に置き換えない。曲線は光子の軌跡ではない。

実際の 1550 nm の光は赤外線で、搬送波の周期は非常に短い。映像では搬送波を
12 ps の模式的な周期に広げる。包絡線と包絡線の位相には数値データを使う。
滑らかな輪郭を作るため、複素包絡線の実部・虚部を相対時刻方向へ4倍の密度で
線形補間し、その細かい点で模式的な搬送波を評価する。元データの標本点の値を
保った描画上の補間であり、物理計算の精度を追加で主張しない。
位相色は `arg(Ax)` の −π〜+π の循環値を、0 のシアン、正側のオレンジ、
負側の青、±π のマゼンタへ割り当てる。弱すぎる裾の色は元データの
`phase_valid` で隠し、未定義の位相を着色しない。

半透明の外形は `sqrt(Px+Py)` の値を半径に対応させた振幅の図示であり、
物理的なモード半径が強度とともに変化することを表さない。横断方向の
電磁場分布を解いた映像でもない。三つのケースは同じ電場・相対時刻の固定倍率と
同じカメラ経路を使う。距離ごとの自動正規化や補償利得は加えず、80 km での
16 dB の損失を表示に残す。

ネイティブの shape key と位相 atlas は、計算済みの 0,1,…,80 km の各出力を
使う。その間は Blender の線形補間で滑らかにつなぐ。これは表示の補間であり、
伝搬ソルバーの数値精度を追加で主張するものではない。距離の進行は観察のための
時間配分で、実際の光の飛行時間とは異なる。開始・終了フレームと正確な対応式を
manifest に記録する。

## シーンの編集と検証

`.blend` の内部には、電場の shape key、位相画像、材質、照明、カメラの
アニメーションが保存される。位相画像は pack 済みで、外部の NPZ や Python
handler がなくても保存シーンを開き、任意のフレームを描画できる。電場の shape
key 名には元の伝搬距離を付け、scene と object のプロパティに入力 SHA-256 と
表示の意味を記録する。字幕は小さな透明画像として後処理で重ねるため、3D
シーン自体を編集しても字幕の数学的説明が物体に固定されることはない。

Cycles / OptiX はガラス、治具、床をスタジオ照明で撮影するために使う。
非線形な光の伝搬を Cycles が計算したものではない。照明の反射が内部の電場を
隠さないよう、ガラスの caustics を切り、adaptive sampling と denoising を
用いる。[Cycles GPU の条件](https://docs.blender.org/manual/en/latest/render/cycles/gpu_rendering.html)
と、保存時の実際の GPU・Blender 版・samples を manifest に残す。

renderer は完成した数値 manifest と NPZ のハッシュを照合し、保存した動画を
FFprobe で検査する。解像度、フレームレート、フレーム数、動画長、各出力の
SHA-256 を記録する。通常検証として、ネイティブシーンを再読込したフレームの
電場・位相・画像を確認する。独立 validator の回数はこの作業では 0。

## 素材とライセンス

ガラス・治具・床・照明はコードで作った形状と材質で、第三者の HDRI、モデル、
画像、音楽、動画を取り込まない。字幕は未改変の
[IPAexGothic](https://moji.or.jp/ipafont/)を使って raster 化する。
フォントのバイナリを `.blend` に埋め込まず、使用した font のハッシュと
インストール済みパッケージのライセンスを描画前に確認し、保存する。独自の
`--font` を使う場合は対応する `--font-license` も指定する。映像の色は可視化上の
false color で、赤外線が肉眼でその色に見えるという主張をしない。
