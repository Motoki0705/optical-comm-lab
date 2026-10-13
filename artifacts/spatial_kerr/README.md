# 円形コアの空間Kerr場と描画入力（Issue #24 / PR #26）

横2次元＋zのスカラー近軸CWモデルです。円形コア半径25 µm、波長1.55 µm、コア/クラッド屈折率1.450/1.445、入射幅12 µm、近軸傾き3°、長さ3 mmを固定し、4ケースを保持しています。強度は解析的な入射ピークを1とした無次元量です。実材料のn2、W単位、実通信SMF、時間パルス、個々の光子を再現した結果ではありません。計算は無限クラッド＋遠方の人工吸収であり、描画上のガラス外径125 µmでのガラス空気反射は解いていません。

## 保存契約

ケース名は `linear`、`weak_reference`、`intermediate_kerr`、`educational_kerr` です。

| ファイル | キーと軸順 |
| --- | --- |
| `<case>_sections.npz` | `x_um, y_um, z_um`、`central_psi=(z,x)`、`cross_z_um`、`cross_psi=(snapshot,y,x)`、ψはcomplex128 |
| 同NPZの診断 | `diagnostic_z_um` と各コア内/全領域の量。診断z間隔5 µm、数値指標は全計算格子で算出 |
| `<case>_flows.npz` | `position_um=(seed,z,3)` の最後軸はXYZ、`valid=(seed,z)`、`z_um`、`seed_quantiles`、`seed_position_um`、`intensity`、`velocity_xy_um_per_um` |
| 主2ケースの `<case>_iso_0p1/0p3/0p6.npz` | `vertices_um=(vertex,3)` XYZ、`faces=(triangle,3)` int32、`normals_xyz`、`level` |

XYZは物理的な横x・横y・伝搬z、長さ単位はµmです。中央縦断面と横断面のx座標は同じです。保存ROIは各軸約±50 µm、実座標は−49.875…49.625 µm、保存横間隔0.5 µm、保存z間隔2.5 µmです。元格子0.25 µmから固定stride=2で点を間引いています。y=0の中央場だけは元格子の対称な近接2行の複素場を線形補間しました。光の移動時刻を示す配列ではありません。

流線は `J_perp=Im(conj(psi) grad_perp(psi))/beta0`、`dr_perp/dz=J_perp/I` です。固定した5×5の入射等パワー分位25個から全線を積分します。IとJの補間後に比を取り、RK4のz刻み0.5 µmで積分し、途中または候補終点の `I<1e-4` で打ち切ります。打切り後に再開しません。`metrics.json` に全打切り理由・比較を保持し、無効点は位置NaN＋valid=Falseです。コア壁での反射点を手で指定していません。

等値面は全ケース/距離で入射ピークに対する固定閾値0.1、0.3、0.6を使います。scikit-imageの[公開marching_cubes API](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.marching_cubes)に物理spacingを渡し、ZYXの頂点をXYZへ変換しました。面の巻き方向と物理法線を通常テストで確認しています。断面図は連続の共通色スケール、等値面は閾値ごとの形状です。両者を同じ種類の輝度表現とは読み替えません。

大きなI/Jx/Jy volumeは `.work/spatial-kerr-production-bulk/` のみです。ψ計算と指標はcomplex128/float64、保存用volumeのみ明示的なfloat32です。疎な保存横断面で、保存横格子のNyquistを超えるψパワー割合は最大2.66e-8でした。これは弱い参照条件の絶対誤差や全節の流線精度を保証する量ではありません。

## 指標の定義と結果

幅・重心は `r<25 µm` の強度に条件づけた平均です。強度差・位相差もコアのマスク内を使いますが、各ケースのコア内パワーを1へ再正規化していません。コア外の強度をすべて漏洩損失と呼ばず、人工吸収へ到達した量と分けています。

`compare_results(A,B)` の相対量は **Bのノルムを分母**とします。

```text
complex_relative_l2 = ||psi_A - psi_B||_core / ||psi_B||_core
intensity_relative_l2 = ||I_A - I_B||_core / ||I_B||_core
```

したがって `cases.<case>.difference_vs_linear` はA=linear、B=記載ケースであり、**記載ケースのコア内場/強度が分母**です。距離刻み・横格子・窓比較ではBが細かい格子/広い窓/座標整列した参照です。結果に合わせた位相・振幅補正はしていません。半セル・45°の比較だけ、既知の物理座標の平行移動/回転を戻しています。

| κ | 保存最大ピーク | 終端コア内量 / 入射 | 終端コア内半径RMS (µm) | 終端コア内x重心 (µm) | 終端強度相対差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.75108 | 0.925211 | 16.31476 | 0.251806 | 0 |
| 1.6e-11 | 1.75108 | 0.925211 | 16.31476 | 0.251806 | 1.229e-8 |
| 1e-4 | 1.88701 | 0.932999 | 16.30393 | 0.724588 | 8.072% |
| 3e-4 | 2.24585 | 0.946715 | 16.20280 | 2.164055 | 26.615% |

主な結果は、反射付近の明るい領域・位相・後半の干渉分布が変わることです。幅の縮小は単調でなく、全領域RMSを自己収束の主結論に使っていません。弱い参照は共通色スケールでは見分けられず、絶対的な微小差の精度も認定していません。

## 通常検証と未確認

- 2D自由Gaussianの複素場誤差1.13e-12、一様Kerr/吸収の誤差最大2.26e-12。
- 独立Bessel境界LP01を800 µmで比較。複素場誤差0.0465%、強度誤差0.0612%。近軸固有位相 `(beta_mode²-beta0²)/(2 beta0)` を使っています。
- 3 mmのdz=0.125/0.0625/0.03125 µm比較は誤差比4.121、最後の強度差2.45e-6。
- dx=0.25→0.125 µmは0…800 µmのみ比較し、強度差0.0979%。最初の反射を含みますが、3 mm全体の横格子収束ではありません。
- 半セルの円＋入力平行移動は800 µmで強度差1.237%、45°の向きは0.1127%。物理座標を整列した比較であり、Kerrの違いではありません。
- 窓±125→±150 µmは両主ケースで3 mmを比較し、コア内強度差は線形0.0155%、Kerr0.00813%。
- 全主ケースの残存＋吸収収支誤差は8.37e-12以下。完全なMaxwell・3 mmの追加横格子・材料損傷/崩壊は未確認です。

表示する強いケースの26.6%の終端強度差は、上記の限定した数値比較差より大きい値ですが、長さと比較対象の違いを無視して完全収束と称していません。点標本の円は階段状近似です。自己収束のバルクTownes比約0.276を、円形導波路や材料の崩壊しない保証には使いません。

主2ケースの25本は全て3 mmまで到達しました。中間ケースは1本をz≈1.685 mmで閾値打切りしました。積分刻み0.5→0.25 µmの差は最大0.0191 µm（線形）、0.0355 µm（Kerr）。一方、線形/局所三次の補間差は最大2.03 µm、0.777 µmで、**流線の細部位置は未資格**です。全線と差を残し、偽のジグザグや滑らかな代替線へ置換していません。近軸パワー流の近似表示として読み、精密な光子軌道とは扱いません。

等値面の保存z=2.5→5 µm比較は、面積差最大0.0105%、bounds差最大0.000462 µmでした。これは保存形状のz依存の確認であり、元の場の横格子誤差を解消する検証ではありません。

`tests.txt` は通常テスト11件（独立解析限界・Gaussian流線・マスク・XYZ/面方向・CPU/GPU parity・破損検出）の実行記録です。`cli_reproducibility.json` は別の10 µm/粗いCPU問題でCLIを2回実行し、全ケースの断面/流線/meshを再現した記録で、3 mm GPU結果を再計算した記録ではありません。独立validatorは0回です。

## 実行と保存量

```bash
uv sync --extra gpu --extra spatial --group dev
.venv/bin/python scripts/simulate_spatial_kerr.py --config configs/spatial_kerr.json --output .work/spatial-reproduction --scratch .work/spatial-reproduction-bulk
.venv/bin/python scripts/simulate_spatial_kerr.py --output artifacts/spatial_kerr --verify-only
.venv/bin/python -m pytest tests/test_spatial_kerr.py -q
```

既存出力・scratchを上書きしません。完全な本計算・比較・抽出は35.8分、主ケースは各144…149秒でした。RTX 5060 Tiのcomplex128、最大CuPyプール使用約532 MB/確保約921 MBです。FFT計画・CUDAコンテキストや他プロセスは別にメモリを消費します。

生成データは総約109 MB、最大単体16.2 MBです。float32 bulk約2.3 GBはGit対象外です。`manifest.json` は生成データ、入力、ソース、実効バックエンドと精度を固定します。後から付けた本README/テスト記録は `normal_verification.json` にハッシュを保存する補足であり、凍結済み生成manifestを変更していません。

実行時ソースと同じ内容の保存commitは `aca53be` です。数値runはこのcommitの作成前に実行され、そのファイル内容のSHA256をmanifestに記録しています。
