# レポートと図

[日本語レポート](final_report.md)と[オフラインHTML](final_report.html)は、専門外の読者に向けて実際の結果を図表中心に説明する。HTMLはPNGを埋め込み、表示時にネット接続や日本語フォントの取得を要しない。ブラウザの印刷でPDF化できる。説明用の模式図と、保存済みデータから描く結果図を区別する。本文は結論・測定内容・結果の意味・限界を中心とし、候補選択・数値条件・実装依存の資源比較・再現記録は4つの折り畳み補足に保存する。HTMLとMarkdownの両方で開閉でき、印刷には必要な補足を開いてから進む。

| 図 | 内容・保存元 |
| --- | --- |
| [link_overview](figures/link_overview.png) | 通信路と4方式の説明図。測定値ではない |
| [learning_workflow](figures/learning_workflow.png) | 学習・候補選択・評価の分離。測定値ではない |
| [result_ber](figures/result_ber.png) | 全7電力、全4方式の直接計数BER、95%計数区間、独立フレームのばらつき |
| [result_constellations](figures/result_constellations.png) | 事前指定電力で同じテストフレーム・同じ点・同じ軸を比較 |
| [result_validation](figures/result_validation.png) | 全検証候補と選択結果。読出しは一括計算なのでepoch曲線を作らない |
| [result_resources](figures/result_resources.png) | 共通EDC後の追加処理時間と保存モデル配列量。装置と測定範囲を明示 |
| [result_qualification](figures/result_qualification.png) | モデル評価とは別の、数値条件と境界の確認 |

PNGはMarkdown/HTML用、SVGは拡大表示用。結果図には必要に応じて `--format svg png pdf` を指定できる。0誤りのBERは対数軸上で実測ゼロ点に置き換えず、計数区間の上限から下向き矢印を描く。図に都合のよい電力・点を選別したり、結果の補間や平滑化をしない。

リポジトリ直下で実行する。今回の最終runは `final-1600km-20261012`。

```bash
.venv/bin/python scripts/visualize_concepts.py
.venv/bin/python scripts/visualize_results.py --run artifacts/final-1600km-20261012 --qualification artifacts/numerical-qualification
.venv/bin/python scripts/build_report.py --run artifacts/final-1600km-20261012 --qualification artifacts/numerical-qualification --verification-record artifacts/verification/ordinary_checks.json --review-record artifacts/verification/independent_review.json
```

生成前に、完成artifactのハッシュ、集計とフレームごとの誤り数・評価数の一致、比較するデータ数、検証データだけで選択した候補を確認する。最終runは7電力すべてが完成している必要がある。pilotは `--allow-pilot --output-dir .work/pilot-figures` を使い、最終成果物へ書き込めない。

技術補足4は、連続2-sps診断波形が存在するrunでだけ、既存テストフレームのseedと入力・出力ハッシュを照合してそのマニフェストへリンクする。保存手順はルートREADMEと `scripts/export_waveform_diagnostics.py` を参照する。独立評価の回数・原判定・指摘への対応は親の記録から表示し、`validator_result.md`の原判定を保持する。評価後の修正と通常検証を、独立再評価の実施済み表示に置き換えない。

[concepts_manifest.json](figures/concepts_manifest.json)、[results_manifest.json](figures/results_manifest.json)、[report_manifest.json](report_manifest.json)に、入力・実行コード・出力ハッシュ、環境、フォント、正確な再生成コマンドを保存する。通常検証と独立評価は親が作った別JSONを `--verification-record`、`--review-record` で指定する。記録を指定しない場合は「記録待ち」と表示し、評価を実施済みとは称しない。公開版と同じ添付を再現するときは `report_manifest.json` のコマンドを使う。

Matplotlibと日本語フォントが必要。`Noto Sans CJK JP`、`IPAexGothic`、`IPAGothic` の順に利用可能なフォントを選び、ない場合は明示する。フォント本体はリポジトリに複製しない。

```bash
.venv/bin/python scripts/visualize_results.py --run artifacts/final-1600km-20261012 --font /path/to/Japanese-font.ttf
```

SVGは文字を輪郭で保存し、PDFはフォントを埋め込む。方式の色はEDC `#6B7280`、線形補償 `#2677B8`、ESN `#D97706`、DBP `#25846C`。色に加えて名称・マーカーでも方式を識別する。変更後は生成PNGを目視確認する。

このディレクトリは読者向けの文章と図を保存する。生成処理は `scripts/`、再利用する数値処理は `src/optical_comm/`、測定・学習のcompact artifactは `artifacts/` に置く。中間ログや一時的な描画確認は `.work/` に分離する。

[光ファイバ内部の3本の物理映像と解説](animations/kerr_physics/index.html)は、80 kmを伝搬する同じガウスパルスから、自己位相変調、波長分散、両者の同時作用を比べる。計算した二偏波の電場をBlender/Cyclesで接写映像にし、各12秒・1080pのMP4、編集できる `.blend`、ポスターを保存した。解説は動画を中心に、見るところ、変化の理由、測定したRMS幅の比を示す。3条件に共通する損失と、描画上の拡大・色・パルス内の時間座標を明示する。HTMLと動画のフォルダ構成を保持すればオフラインで読める。

[解説のREADME](animations/kerr_physics/README.md)と[マニフェスト](animations/kerr_physics/explanation_manifest.json)に、入力・出力のハッシュと再生成コマンドを保存する。説明用の単独パルス実験であり、1600 kmの学習結果やBERの追加評価ではない。生成CLIは `scripts/build_kerr_explanation.py`、物理計算は `scripts/simulate_kerr_films.py`、動画は `scripts/render_kerr_films.py`。詳細は[物理モデル](../docs/kerr-physics.md)と[描画手順](../docs/kerr-rendering.md)を参照する。

旧版の[30秒の模式アニメーション](animations/kerr/README.md)は、当時のMP4、Blenderシーン、ポスター、来歴とフォントライセンスを保持している。単一偏波・無損失・無分散の解析例で、生成CLIは `scripts/animate_kerr.py`。今回の3本の動画の入力には使っていない。
