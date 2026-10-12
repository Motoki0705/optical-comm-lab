# レポートと図

このディレクトリには、専門外の読者に向けた日本語の最終レポートと、その説明図・実験結果の図を保存する。図は軸・単位・不確かさ・比較条件を示し、説明用の模式図と測定結果を区別する。最終レポートは、実際に得られた結果と限界を表・図を中心に説明する。

現在の説明図は、[通信路の流れ](figures/link_overview.png)と[学習・評価の分離](figures/learning_workflow.png)。いずれも研究設計の模式図であり、補償効果を測った結果ではない。PNG は Markdown 用、SVG は拡大表示用、PDF は印刷用。図の説明・入力文書のハッシュ・生成環境・出力ハッシュは [concepts_manifest.json](figures/concepts_manifest.json) に記録する。

リポジトリ直下で次を実行すると、２図とマニフェストを再生成する。

```bash
.venv/bin/python scripts/visualize_concepts.py
```

Matplotlib と日本語フォントが必要。生成スクリプトは `Noto Sans CJK JP`、`IPAexGothic`、`IPAGothic` の順に利用可能なフォントを選ぶ。Debian/Ubuntu では `fonts-noto-cjk` または `fonts-ipaexfont-gothic` をシステムのパッケージ管理で導入できる。管理者権限を使わない場合は、正規に取得した日本語 TTF/OTF/TTC ファイルを次のように指定する。フォント本体はリポジトリに複製しない。

```bash
.venv/bin/python scripts/visualize_concepts.py --font .work/fonts/IPAexGothic.ttf
```

SVG は文字を輪郭として保存し、PDF は利用する文字のフォントを埋め込むため、閲覧側の日本語フォントには依存しない。同じフォントと生成環境を使うと同じ出力を再現できる。見出し・ボックスの文字が収まることは生成時に検査し、追加・変更後は PNG を目視確認する。

比較方式の色は、EDC `#6B7280`、線形補償 `#2677B8`、ESN `#D97706`、DBP `#25846C` とする。色だけに意味を持たせず、名称・凡例でも方式を識別する。実験結果の図を追加するときは、対応する生成コマンドと入力 artifact を記録し、生成処理は `scripts/`、再利用する数値処理は `src/optical_comm/` に置く。
