# 反射する光と Kerr 効果

[説明ページ](index.html)で、3本の動画と4条件の比較を順に読める。
教育用の円形コア径 50 µm、
長さ 3 mm の連続光の計算であり、
仮の強い係数を含む。1600 km の単一モード通信・学習研究とは別の計算。

仮の強条件では、反射後の位置と幅の差が、今回調べた計算条件の差より大きく現れた。弱い参照条件は線形の場合とほぼ重なり、その差は今回の数値確認の目安より小さい。

| 動画 | 見るところ | MP4 | Blender |
| --- | --- | --- | --- |
| 01 | 境界で、進む向きが戻る。 | [01_reflection.mp4](01_reflection.mp4) | [01_reflection.blend](01_reflection.blend) |
| 02 | 光の強さが、その先の分布を変える。 | [02_kerr_comparison.mp4](02_kerr_comparison.mp4) | [02_kerr_comparison.blend](02_kerr_comparison.blend) |
| 03 | 波の山谷のずれが、重なり方を変える。 | [03_phase_change.mp4](03_phase_change.mp4) | [03_phase_change.blend](03_phase_change.blend) |

線は場のエネルギーの流れ、面は同じ強度の場所をつないだ表示。
カメラ移動や距離の走査は観察の時間配分で、光の飛行時間ではない。
冒頭の [入口画像](entrance_01.png) は動画01の最初のフレームのコピー。
画像と動画のハッシュ、および元フレームの来歴をマニフェストに記録する。

4条件のボタンは、保存済みの別々の計算画像を同じ座標・色尺度で切り替える。
係数の間を補間しない。[4条件の静的比較](figures/four_case_intensity.png)も保存し、
印刷時にはこの比較図を表示する。図の SVG は文字を輪郭で保存している。

## 再生成

リポジトリ直下で実行する。保存済みの計算データ、3本の MP4、Blender シーン、
ポスター、両方のマニフェストが必要。NumPy、Matplotlib、日本語フォント、
FFprobe を使う。標準では Noto Sans CJK JP、IPAexGothic、IPAGothic の順に探す。

```bash
.venv/bin/python scripts/build_spatial_kerr_report.py --physics-run artifacts/spatial_kerr --renders reports/animations/spatial_kerr
```

このコマンドは図の PNG/SVG、`index.html`、この README、
`explanation_manifest.json` を再生成する。物理伝搬や Blender の描画は行わない。
入力不足・ハッシュ不一致では終了し、完成ページを出力しない。
別の出力先は `--output-dir .work/spatial-kerr-report-reproduction`、
日本語フォントは `--font /path/to/font.ttf` で指定できる。

元の動画、図、データと相対パスを保持して `index.html` を開く。
ブラウザの標準操作で再生・シーク・全画面表示を行い、各動画の再生速度を選べる。
動画にフォーカスしたときは Space と左右キーをブラウザの標準操作として使える。
K でも再生・一時停止を切り替えられる。キーによる移動幅はブラウザの標準に従う。
再生を始めると他の動画は一時停止する。外部サービス、外部フォント、配信 CDN は使わない。

ローカルファイルを直接開いた場合、保存リンクの扱いはブラウザによって異なる。
動画の標準メニューから保存できる。保存リンクをHTTPで使う場合は、リポジトリ直下で
次のローカルサーバーを起動し、表示されたURLの末尾へ
`reports/animations/spatial_kerr/index.html` を付けて開く。外部サービスへの接続は不要。

```bash
.venv/bin/python -m http.server 8000 --bind 127.0.0.1
```

[方法と出典](../../../docs/spatial-kerr.md)、
[原データ](../../../artifacts/spatial_kerr/README.md)、
[数値・検証記録](../../../artifacts/spatial_kerr/metrics.json)、
[映像の記録](render_manifest.json)を参照できる。
`explanation_manifest.json` に入力・出力・フォント・コードのハッシュ、生成条件、
通常の確認を記録する。自己参照を避け、同マニフェスト自身のハッシュは内包しない。
この改訂の独立 validator は指定0・試行0・完了0。元研究の1回を今回の評価に流用していない。
