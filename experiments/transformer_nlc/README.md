# Transformer nonlinear-compensation reproduction

[主2条件の再現仕様](docs/reproduction-spec.md)に、論文の出典、採用した仮定、データ/モデル契約、候補、予算、実行ゲート、受入条件を保存する。基準は[arXiv:2304.13119v3](https://arxiv.org/abs/2304.13119v3)。現在は仕様段階で、モデル、生成/学習CLI、測定結果はまだない。

今回の実行範囲はDP-16QAM 32 GBd / 40×80 kmとDP-64QAM 42 GBd / 12×80 kmの学習・独立評価・checkpoint・図表・日本語レポートまで。mask有無と通常DSPを比較する。初期計画は6構造×2 mask×3 seedの36学習run上限で、原論文の数百モデル探索に対する**部分再現**とする。4 learning pilotsに加え、最大構造A/highのmask有無を実batchで資源benchmarkし、数値精度、時間、VRAM、保存量を測定してから本番設定を固定する。通常の数値refinementや回復可能OOMへのmicrobatch調整は、親が範囲内で記録して継続する。

進行は[親#40](https://github.com/Motoki0705/optical-comm-lab/issues/40)で追跡する。

1. [#42](https://github.com/Motoki0705/optical-comm-lab/issues/42)：実験workspaceの分離・既存成果物保全・回帰。
2. [#43](https://github.com/Motoki0705/optical-comm-lab/issues/43)：出典/仮定・主2条件・段階計画（本仕様）。
3. [#44](https://github.com/Motoki0705/optical-comm-lab/issues/44) / [#45](https://github.com/Motoki0705/optical-comm-lab/issues/45)：物理/DSP/独立データとTransformer/mask。
4. [#46](https://github.com/Motoki0705/optical-comm-lab/issues/46)：学習・再開・validation選択・固定test。
5. [#47](https://github.com/Motoki0705/optical-comm-lab/issues/47)：pilot/数値資格後の本学習・評価・保存・報告。

共通の物理/DSP/指標は`optical_comm`から使い、このworkspaceがモデル、学習、実験設定、報告を所有する。[拡張手順](../README.md)に従い、ESNは独立したpeerとして保持する。将来の入口は`python -m experiments.transformer_nlc`、成果物は`artifacts/transformer-nlc/`、レポートは`reports/transformer-nlc/`を予定する（いずれも現段階では未実装）。

PMD・レート変更、CNN-LSTM/DBP拡張比較、広範なアブレーションは実行前に相談し、未完の親#40を開いたままにする。今回の独立validatorは0回。過去の結果・生成元SHA・別タスクのmerge許可は引き継がない。
