# 円形コアの計算場を使うBlender描画

Issue [#25](https://github.com/Motoki0705/optical-comm-lab/issues/25) の描画CLIは
`scripts/render_spatial_kerr.py`、固定した表示設定は
`configs/spatial_kerr_render.json` にある。物理モデルと数値計算は
[spatial-kerr.md](spatial-kerr.md) と `artifacts/spatial_kerr/` が担当する。
rendererはその場を読み取るだけで、伝搬計算、Kerr係数の選定、結果の補正を行わない。
数値入力がない場合は停止し、仮の光や代替画像を作らない。
完成した本編は各6秒・24 fps・1920×1080の
[反射](../reports/animations/spatial_kerr/01_reflection.mp4)、
[Kerr OFF/ON比較](../reports/animations/spatial_kerr/02_kerr_comparison.mp4)、
[位相と明暗](../reports/animations/spatial_kerr/03_phase_change.mp4)。
反射と変化を読める教育用のモデルとして採用し、強い仮係数と表示の発光・色による
強調を含む。通信SMFの実パワーや材料n₂を再現した映像ではない。

## 入力と表示

`--data-dir` はリポジトリ外のフォルダーも指定できる。`manifest.json` の
schema_version=1、ファイルのbytes/SHA256、µm単位と配列軸を確認してから読む。
入力の4ケースは `linear`、`weak_reference`、`intermediate_kerr`、
`educational_kerr`。主映像は κ=0 / 3e-4 の同一入射条件による比較で、
κは正規化強度に対する仮係数である。材料のn₂や実パワーW、元の80 kmや
1600 kmの単一モード通信研究に対応づけない。

| 入力 | 表示で使う値 |
| --- | --- |
| `<case>_sections.npz` | `(z,x)` の中心y=0のcomplex128場、`(snapshot,y,x)`の断面とその座標 |
| `<case>_flows.npz` | 25本の流れ、物理XYZ位置、valid、種のquantile、J/I速度、強度 |
| `<case>_iso_0p1/0p3/0p6.npz` | 物理XYZの頂点・法線とint32三角面、入射ピーク比の固定しきい値 |

Blenderの座標は厳密に `(world X,world Y,world Z)=(z,x,y)/100`。
XYZで同じ100 µm/Blender単位を使い、伝搬方向だけ引き伸ばさない。
入力法線の軸・長さを確認する。発光表示面の控えめな表示用陰影には保存された
三角面から得るsmooth normalsを使い、nativeファイルへの巨大なcustom loop normal配列の複製を省く。
元の頂点・三角面は全て保持し、元法線はNPZとSHA256で保存する。
`I=|ψ|²` は入射ピークを1とする。
ケースや各zの最大値への正規化はしない。

3D分布は入力から抽出された **I=0.1 / 0.3 / 0.6 の等強度面** を、青／緑／橙で
表示する。この色は面のしきい値の凡例であり、連続強度のカラーバーではない。
opacity、輝度、線幅、色は全ケースで固定する。面は強度分布を立体で見るための
可視化で、写真の光散乱を測定した映像ではない。
面の法線に共通の方向光による軽い陰影を付け、立体の形を読みやすくする。
この陰影は表示用で、定量の散乱輝度や連続強度を示さない。色の凡例は等強度面のしきい値である。

白い線は、数値場のJ/Iから得られた25本のパワー流線を、そのままPOLYで描く。
全ての種のIDを残し、最初のinvalid以後を描かず、再開しているvalidマスクは拒否する。
曲線を手で曲げたり、光子の軌道やパルス先端へ読み替えたりしない。
標準の表示strideは1であり、元の有効点を間引かない。

現在の物理artifactの流線の積分刻み比較では、位置差の最大値はlinearで
0.0191 µm、強条件で0.0355 µm。一方、linear対cubicの補間比較では最大
2.03 µm / 0.777 µmだった。流線は流れを追う近似表示として扱い、個々の線の
微小な反射位置の移動量は測定しない。native頂点の再現誤差の小ささは、
流線の物理精度を保証しない。これらの差を元metricsからmanifestへ引き継ぐ。

紫の輪と小さな十字は**観察断面の注記**。各フレームの観察zに最も近い保存済みの
共通z標本で、同じ断面を横切る流線の位置を指す。種が停止した後は十字を隠す。
カメラとこの注記だけが距離を走査し、強度面・流線・定常の光場は動かない。
「距離を進めて観察」と表示し、観察の6秒に光の伝搬速度や物理時間の単位を付けない。

反射接写とOFF/ON比較には、主反射面y=0で円形コアの境界に当たる
x=±25 µmへ細い直線のガイドを置く。これはコアの位置を示す表示用注記で、
場の等強度線や光の経路ではない。元の強度面と流線の位置は変えない。

内部の中心縦断面は連続強度を共通の `[0,3]` 入射ピーク比カラーバーで表示する。
3を超える値は同じ上端色にクリップし、上端を超えた割合と元データの最大値を記録する。
位相差は `arg(ψ_Kerr conj(ψ_linear))` の主値 `[-π,π]`。
両方の強度が1e-4以上の場所だけを着色し、弱い場所を灰色へ置く。
位相をunwrapしたり、合わせ込みの位相補正をしない。
位相は伝搬中の強度に応じて変わり、反射点だけで発生するという説明をしない。
位相0を暗い青緑へ置く共通の循環色を使い、0に近い場を白い巨大な面へしない。
03の代表画は約1.8 mmの観察位置で、距離に沿って蓄積した位相差と明暗を読む。
本編用の保存シーンには250〜2600 µmの全観察経路を残す。
下段の強度2パネルは上段と同じ距離を見ているため、下段の世界距離ラベルを省き、
固定画面のカラーバーと数値への重なりを避ける。距離の目盛りは上段に残す。

## 保存した三つの映像

| 本編ファイル | 内容 |
| --- | --- |
| `01_reflection.mp4` | linearの立体分布と25本の流れ、観察断面、±x側のコア境界 |
| `02_kerr_comparison.mp4` | **上がOFF、下がON**。同じカメラ・入射条件・固定表示尺度で2パネル比較 |
| `03_phase_change.mp4` | 上に「波のずれ」の内部位相差、下に同じ距離を観察するOFF/ONの連続強度断面 |

本編は各144フレームをCycles96 samples＋denoiseで描いた。
各filmの `.blend`、`poster_01/02/03.png`、`render_manifest.json`、フォントライセンスを
同じフォルダーへ保存する。`entrance_01.png` は本番01の最初の1920×1080 PNGを
そのままコピーした構造の静止画で、再描画やproofの拡大ではない。
初期previewの短いproofは `.work/` の履歴へ分離し、本編の案内には使わない。

```bash
# 物理入力と座標・尺度・マスクの通常確認。Blenderを起動しない。
.venv/bin/python scripts/render_spatial_kerr.py --mode inspect

# 実データ入り代表画3点と、各filmのnative .blend。
.venv/bin/python scripts/render_spatial_kerr.py --mode posters

# 上記と最大3秒・960×540のlinear現象proof。
.venv/bin/python scripts/render_spatial_kerr.py --mode proof

# 保存後のsceneを再読込し、頂点・流線・packed textureを入力と再照合。
.venv/bin/python scripts/render_spatial_kerr.py --mode verify-scene

# 入口のカメラ構図だけを1枚で確認する。既存previewを上書きしない。
.venv/bin/python scripts/render_spatial_kerr.py --mode posters --film 01_reflection --poster-frame 1 --output-dir .work/spatial-kerr-entrance-review

# 修正済みのカメラ経路と全componentのactive sceneを使い、本編を全量再生成する。
.venv/bin/python scripts/render_spatial_kerr.py --mode render
```

`--film 01_reflection` で制作対象を限定できる。`--data-dir`、`--output-dir`、
`--scratch-dir`、`--config`、`--blender` を指定できる。Windows BlenderはWSLから
wslpathでパスを変換し、引数配列で呼び出す。OptiXがなければ別装置へ黙って変更せず停止する。

## 可搬性・再現と確認

入力manifestの全ファイルをSHA256に照合したうえで、実行ごとに `.work/` の新しい
scratchへ読み取り専用入力のコピーを固定する。元入力とコピーのハッシュを確認し、
コードもsnapshotへ固定する。全量renderは新規cacheを使い、一般的なresumeは実装しない。
設定、元入力、コード、フォント、装置、Blender build、GPU driver取得結果、
各フレームのSHAを記録し、古いフレームを混ぜない。

各blendには、そのfilmの名前付きcomponent scene、固定3D形状、保存流線、
観察注記のフレーム位置、packed float texture、日本語フォント、設定と入力manifestがある。
2/3本目の最終レイアウトはCLIがcomponentのネイティブ画素を並べて作る。
画像の引き伸ばしや光学フロー、フレーム複製で動画を作らない。
nativeファイルを開くと最初のcomponent sceneが表示されるため、比較のもう片方は
Blenderのscene選択で切り替えられる。再現CLIはこれらを毎フレーム同じ規則で合成する。

最初の斜め投影では細い直線のように見えて折り返しが読めなかったため、
物理倍率を保持し、軸方向を短縮する接写の正投影へ変更した。
画面の上下は主な折り返し方向の物理x（world Y）を保つ。
01の代表画は最初の反射付近のおよそ0.2〜0.9 mmを大きく見る。
映画02の両ケースにも同じ投影・カメラ・尺度を用いる。
これらは投影を調整した観察画で、画像で物理反射角を測定するものではない。
01は冒頭に円形端面・ガラスの窓と計算分布の関係を見せ、カメラだけで反射接写へ移る。
入口は形状lookdevの端面が読める向きを基礎に、最初のおよそ1 mmを大きく写す。
最初の12フレームは構造を観察し、24フレームまでに接写へ寄る。その間、観察zは250 µmに固定する。
以後は250〜2600 µmを距離に沿って観察する。字幕はnative driverで画面サイズを保つ。
この導入と接写への移動も光パルスの移動ではない。
導入の13〜24フレームは、位置とEuler角を独立に補間せず、各フレームの注視点を
追う回転キーで視線を保つ。全componentの検査・描画では対象sceneをactiveにしてから
frameを設定し、別sceneに残った終端frameの評価が混入しないようにする。

代表画を保存したnativeファイルを再読込して、入力頂点と有効流線の位置差を
1e-3 µm未満、RGBA texture差を1e-6未満に照合する。位相のmaskと入射ピーク基準は
GPUなしのrenderer testsでも確認する。動画はFFprobeで寸法・fps・フレーム数・長さを確認し、
FFmpegで全フレームをデコードする。これはrendererの通常確認であり、数値伝搬の
検証・独立validator・完全なMaxwell解の保証ではない。今回のvalidatorは0回。
全componentの144フレームで、実際のcamera位置・姿勢・画角・注視点の投影を照合し、
比較panelの距離と姿勢が一致することも確認する。各render前後にOptiX設定と
GPU指定・CPU無効を検査する。factory reset後はOptiXを再設定し、CPU fallbackを拒否する。
FFmpeg/FFprobeの実行path、binary SHA、version表示は取得したrunについて保存する。

Pillow、NumPy、Blender 5.2/OptiX、FFmpeg/FFprobe、IPAexGothicを使う。
標準はインストール済みIPAexGothicを未改変でblendにpackし、ライセンスを同梱する。
独自フォントでは `--font` と `--font-license` を指定する。
外部のHDRI、他者のモデル、音声、AI生成画像は使用しない。

## カメラ不具合の正式補修と生成来歴

元本番source `a6b623c`（SHA `2f34c912…`）には、導入途中で視線が外れる
[Issue31](https://github.com/Motoki0705/optical-comm-lab/issues/31) と、非先頭sceneが
終端frameで評価される [Issue34](https://github.com/Motoki0705/optical-comm-lab/issues/34) があった。
補修は01の13〜24、02のON全144、03下段のOFF/ON各144をBlenderで再描画した。
02のOFF、03の位相、01の補修外は、元の同じframe番号のPNGをhash照合して保持する。
全ポスターとnativeは修正後の状態から生成し、3動画を全frameから再符号化する。
古い画像の生成元を新sourceへ付け替えず、manifestの元run・補修run・component別
置換frame・新旧source/scene/frame SHAへ分けて記録する。
保存した補修画像/nativeの実producerは `b98a0e1`（SHA `57cbad279085…`）、
補修外の元画像は `a6b623c` である。その後の
[Issue35](https://github.com/Motoki0705/optical-comm-lab/issues/35) はhost引数検査と説明のみを
変更した。既存mediaをこの後続sourceの生成物として付け替えず、再描画もしない。

専用CLIは既知の元source SHAとconfig SHA `f37d3116…` に限定する。
元の完成manifest、source/input snapshot、全864 native＋432 composed frameのhash、
Python/NumPy/既知GPU driver、Blender build/device、フォントを照合する。
カメラ以外の場・表示geometry・材質のfingerprintを比較し、補修外のcamera、
全位置/画角が不変であることを確認する。入力data、係数、材質は補修で変更しない。
元cacheを残した環境で、同じ元runから補修を再実行する手順は次のとおり。

```bash
.venv/bin/python scripts/render_spatial_kerr.py --mode repair-cameras --preflight-only --repair-manifest /path/to/original-completed-render_manifest.json --output-dir /path/to/original-deliverables --repair-cache /path/to/original-cache
# 表示されたrepair_stateの指定サンプルと通常検査を確認してから実際の補修へ進む。
.venv/bin/python scripts/render_spatial_kerr.py --mode repair-cameras --repair-state /path/to/repair-cache/repair_state.json --output-dir /path/to/original-deliverables
```

preflightと完了段階でsource/config/input/environmentとsample hashを再照合する。
別versionのpreflightを流用しない。元cacheがない場合は、修正済みsourceの
`--mode render` で全量再生成する。これは広いresumeや任意frameの置換機能ではない。
prepareとfinishの同時指定、repairの `--film` / `--poster-frame`、finishにprepare用の
manifest/cacheを追加する指定、他modeのrepair専用指定は、入力読取りや出力更新前に拒否する。
[Issue33](https://github.com/Motoki0705/optical-comm-lab/issues/33) のencoder識別は補修時に
実際に取得したpath/version/SHAを保存し、記録がない元runへ遡って帰属させない。
H.264は非可逆なので、保持したPNGが同じでも再符号化後の画素やGOPのbyteは変わり得る。

通常の全量 `--mode render` で01を含む場合、hostが実rendercacheの
`01_reflection/reflection/frame_0001.png` をbyteを変えず `entrance_01.png` へ自動コピーする。
元frameの保存SHAを照合し、manifestの `entrance_image` にfile/SHA/元film/元frameを、
`geometry_context` に実frameの解像度・samples・source/config/input SHAを記録する。
01を含まないrenderでは入口生成を主張しない。この
[Issue36](https://github.com/Motoki0705/optical-comm-lab/issues/36) のhost改善は未来の再生成用で、
凍結済みmediaの生成元やbyteを変更しない。
`contact_sheet_01/02/03.png` は実際の完成MP4のframe1/18/40/72/110/144を縮小配置した
通常の目視確認用画像で、数値検証の代わりではない。再作成例:

```bash
ffmpeg -i reports/animations/spatial_kerr/01_reflection.mp4 -vf "select='eq(n,0)+eq(n,17)+eq(n,39)+eq(n,71)+eq(n,109)+eq(n,143)',scale=960:540,tile=3x2" -frames:v 1 reports/animations/spatial_kerr/contact_sheet_01.png
```

## 観察範囲とガラス表面

物理計算は無限クラッド近似で、外径125 µmのガラス／空気境界の反射は含まない。
外皮、観察窓、支持具、表面照明は表示用。データのROI・間引き・数値精度と
資格確認の限界を元manifestから引き継ぐ。弱い参照条件や格子誤差を超える精度は
rendererの滑らかな表示から主張しない。

一体の充実した円形ガラスを観察窓で開き、内部の細かな屈折率差の描画面を省略する。
表面は弱い曲面反射と直進透過の表示材質で、数値分布をレンズでずらさない。
01導入では端面と充実したガラスの観察窓を読むため、表示材質の表面反射と
小さな中性色の面照明を強め、接写へ移る間に共通の弱い表面材へ戻す。
1.08〜1.6 mmの外皮は、剥いだ観察領域を示す導入用のannular jacketで、
接写開始時に隠す。外径190 µmの外皮とその表示切替は計算外の装飾である。
この材質・外皮の変更は計算場の色尺度や座標を変更せず、光が動く演出にも使わない。
計算場の色・流線は直接観察のrayにだけ表示し、ガラスの鏡像や架空の光散乱へ増殖させない。
色は赤外光のfalse color、照明・投影・opacityは表示設定であり、反射角の測定や
物理的な散乱輝度の測定には使えない。
