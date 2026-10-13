# optical-comm-lab

長距離の光ファイバ伝送を数値シミュレーションで再現し、ソフトウェアのリザバー（ESN）が受信品質を改善できるかを、未学習データのビット誤り率（BER）で調べる研究リポジトリです。1600 km、単一波長・DP-16QAM・32 GBd、理想的な送受信器を対象にします。各送信電力で別モデルを学習し、通常補償、線形学習、ESN、数値逆伝搬（DBP）を同じフレームで比較します。改善が得られない結果もそのまま報告します。

結果と図は [日本語レポート](reports/final_report.md)、オフラインで読める [画像埋め込みHTML](reports/final_report.html) に保存します。HTMLはブラウザの印刷機能でPDFにできます。

## 環境

Python 3.12 と [uv](https://docs.astral.sh/uv/) が必要です。`.python-version` と `uv.lock` を用いて環境を固定します。CPUで全処理を実行できます。図の再生成には日本語フォント（推奨 `Noto Sans CJK JP`）が必要で、Ubuntu/Debianでは `fonts-noto-cjk` を導入します。フォントを導入できない場合は各描画CLIの `--font /path/to/font.ttf` を使えます。

```bash
uv sync --python 3.12 --locked --group dev
.venv/bin/python -m pytest
```

GPUは通信路の伝搬とDBPを高速化する任意の機能です。線形学習とESNはNumPy/SciPy/NumbaでCPU実行します。GPUを使う場合はNVIDIAドライバとCUDA 13環境が必要です。

```bash
uv sync --python 3.12 --locked --extra gpu --group dev
```

CUDA toolkitを備えずドライバだけの環境では、別途CUDA 13 toolkitを導入するか、ロックで選ばれたCuPy 14.2.0に対応する実行時コンポーネントを入れます。同時に複数のCuPy種類を導入しないでください。

```bash
uv pip install 'cupy-cuda13x[ctk]==14.2.0'
```

この追加コンポーネントは通常のロック同期とは別のセットアップです。現在確認した環境はWSL/Linux、RTX 5060 Ti 16 GB、CUDA toolkit 13.0です。詳細は [通信路の環境記録](docs/channel-methods.md) と [公式CuPy導入ガイド](https://docs.cupy.dev/en/stable/install.html) を参照してください。

## 小さな動作確認と本評価

リポジトリ直下で実行します。小さなpilotは探索・動作確認で、最終BERに合算しません。CPUのみなら次のように明示します。

```bash
.venv/bin/python scripts/run_experiment.py --config configs/pilot.json --run-id pilot-cpu --powers 0 --backend numpy
```

GPUでのpilotと、設定した評価数を表示する確認は次のとおりです。

```bash
.venv/bin/python scripts/run_experiment.py --config configs/pilot.json --run-id pilot-check --powers 0
.venv/bin/python scripts/run_experiment.py --config configs/full.json --run-id final-qualified --dry-run
```

本評価は、[数値条件の確認](docs/numerical-qualification.md)で伝搬刻み・サンプル数・フレーム長・境界の扱いを確定してから行います。`configs/full.json` はその研究用設定です。生の伝搬・学習計算からやり直すコマンドは次のとおりです。

```bash
.venv/bin/python scripts/run_experiment.py --config configs/full.json --run-id final-reproduction
.venv/bin/python scripts/export_waveform_diagnostics.py --run artifacts/final-reproduction
```

CPUで本評価するなら `--backend numpy` を付け、新しいrun IDを使います。長い実行を再開する場合は同じ設定とrun IDで `--resume` を付けます。設定・実行ソース・環境・完成artifactのハッシュが一致する条件だけを再利用します。`--powers -6 -4` で条件を分けて実行できますが、最終図とレポートの生成には7電力すべてが必要です。結果は `artifacts/<run-id>/` に保存します。

[連続診断波形の保存CLI](scripts/export_waveform_diagnostics.py)は、本評価完了後に同じ既存テストフレームのseedから短いXYの2-sps波形（1シンボル当たり2サンプル）を保存します。既定は各電力の最初のテストフレームから512シンボルの連続区間で、送信bits/symbols、mask、開始位置とフレーム境界を含みます。`waveform_diagnostics/manifest.json`に来歴とハッシュを記録し、新しいBER評価や再学習は加えません。既存出力を上書きしないため、保存済みrunから再生成するときは空の出力先を指定します。

```bash
.venv/bin/python scripts/export_waveform_diagnostics.py --run artifacts/final-1600km-20261012 --output-dir .work/waveform-diagnostics-reproduction
```

## 図とレポートの再生成

保存済みの最終runを指定すれば、学習をやり直さずに図とレポートを再生成できます。今回の実行IDは `final-1600km-20261012` です。

```bash
.venv/bin/python scripts/visualize_concepts.py
.venv/bin/python scripts/visualize_results.py --run artifacts/final-1600km-20261012 --qualification artifacts/numerical-qualification
.venv/bin/python scripts/build_report.py --run artifacts/final-1600km-20261012 --qualification artifacts/numerical-qualification --verification-record artifacts/verification/ordinary_checks.json --review-record artifacts/verification/independent_review.json
```

正確なコマンド、フォント・環境、入力と出力のハッシュは [結果図のマニフェスト](reports/figures/results_manifest.json) と [レポートのマニフェスト](reports/report_manifest.json) に保存します。検証・独立評価の集約記録はレポート生成の追加入力です。元の公開版と同じ検証記録を含めるには、レポートマニフェストの `regeneration_command` を使います。pilotの描画確認は `--allow-pilot --output-dir .work/pilot-figures` のように、最終図から分離します。

独立評価の原判定と指摘は [validator_result.md](artifacts/verification/validator_result.md)、親の採否・修正状況・通常検証は [independent_review.json](artifacts/verification/independent_review.json) に保存します。評価後の修正を独立評価済みとは扱いません。

## 内容と開発

- [要件](docs/requirements.md)と[実験計画](docs/experiment-plan.md)：採用条件、比較、完了基準。
- [通信路](docs/channel-methods.md)、[学習・評価](docs/learning-methods.md)、[数値条件の確認](docs/numerical-qualification.md)：実装方法と測定の限界。
- [scripts](scripts/README.md)、[configs](configs/README.md)、[artifacts](artifacts/README.md)、[reports](reports/README.md)：再実行入口と保存物の役割。

作業は[ルートのAGENTS.md](AGENTS.md)に従い、Issueを先に作り、feature branchと[管理PR #7](https://github.com/Motoki0705/optical-comm-lab/pull/7)で管理します。実機での性能、多波長伝送、未学習条件への汎化やFEC後の性能は今回の結論に含めません。開発開始・構成変更・PRレビュー前には[scoutで構造を点検](docs/repository-maintenance.md)し、根拠を伴う問題を重複確認してIssue化します。この保守点検は、成果物への1回の独立validator評価とは別です。
