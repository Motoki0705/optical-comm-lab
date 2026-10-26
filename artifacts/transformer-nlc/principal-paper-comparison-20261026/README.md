# 論文メトリクス比較を中心にしたレポート改訂

[Issue #62](https://github.com/Motoki0705/optical-comm-lab/issues/62) /
[PR #63](https://github.com/Motoki0705/optical-comm-lab/pull/63) の通常検証記録。
対象は[主2条件レポート](../../../reports/transformer-nlc/principal-20261025/report.md)の説明・図表であり、追加学習やtestによる再選択は行っていない。

生成器は commit `db63638579853297a13efe49c44887a3f7059813` の
`experiments/transformer_nlc/report.py`。公開値は arXiv:2304.13119v3 の
IV-C / Table II と IV-D を参照した。64QAMの最良値8.17 dBは探索全体の
参考値であり、今回の候補・maskごとの対応値として代入していない。

## 記録と来歴

- [verification.json](verification.json): 現行manifestのhash、生成コマンド、入力・出力整合性、保存seedからの差分再計算、通常試験、表示確認の範囲。
- [preserved-artifact-hashes.json](preserved-artifact-hashes.json): 改訂前から存在する234成果物のhash。checkpoint 88件を含め、改訂後も全件一致した。
- [previous-report-manifest.json](previous-report-manifest.json): 改訂直前の描画manifest。元の入力91件と数値sourceを現行manifestと照合した。
- [pytest.log](pytest.log): 統合後の通常試験。269 passed、2 skipped（任意のCuPyがこのPython環境にない）、61 warnings。独立validatorの指定／試行／完了は0／0／0。

旧 `principal-report-verification-20261025/` の検証・索引・patch・historyは
当時の成果物を説明する履歴として変更せず残す。そこに記録された出力hashは
旧レポートのものであり、現行レポートの検証として再利用しない。
旧描画は当該manifest、保存済みpatch、Git履歴の `b7897d44aa67b5b4104d62c88ba92ee1e1d7dd34`
に対応する。科学データのproducer SHA、絶対path、選択freezeは書き換えていない。

## 再生成と検証の範囲

[実験READMEのコマンド](../../../experiments/transformer_nlc/README.md#主2条件の保存結果2026-10-26)
または `verification.json` の `generation_command` を使う。入力は元のproducerの
絶対パス配置を前提とする。新cloneへ任意に移動しても動くという検証ではない。
生成器は学習・推論をせず、保存compact入力とhashを検査する。

別の出力先にも再生成し、PNG/SVG 14件と数値表4件がbyte単位で一致した。
Markdown/HTMLは表示する実行コマンド以外が一致した。元のPNG/SVG 10件と
counts.csv・summary.json・run-inventory.jsonは改訂前後でbyte単位に一致する。
新しい比較表は全12候補の平均・標本SD・同一seed通常DSPとの差、6組のmask差、
16QAMの公開Q/RMPSとの差、固定された64QAM primaryの参考値との差を保存する。
これらを保存metricsの42 replicateから通常検証で再計算した。

新しい比較PNG 2枚の日本語、軸、誤差棒、凡例を確認し、HTMLの相対リンク12件の
存在を検査した。ブラウザ描画を確認したとは主張しない。
