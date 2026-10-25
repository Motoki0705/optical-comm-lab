# optical-comm-lab

長距離の光ファイバ伝送を数値シミュレーションで再現し、ソフトウェアのリザバー（ESN）が受信品質を改善できるかを、未学習データのビット誤り率（BER）で調べる研究リポジトリです。1600 km、単一波長・DP-16QAM・32 GBd、理想的な送受信器を対象にします。各送信電力で別モデルを学習し、通常補償、線形学習、ESN、数値逆伝搬（DBP）を同じフレームで比較します。改善が得られない結果もそのまま報告します。

共通の物理・DSP・指標は `src/optical_comm/`、研究ごとの実装・設定・方法は
[`experiments/`](experiments/README.md) に分けます。既存研究は
[`esn_longhaul`](experiments/esn_longhaul/README.md)、次の Transformer 再現は
[`transformer_nlc`](experiments/transformer_nlc/README.md) で独立に管理します。
Transformer の科学的実装はこの構成変更に含めません。

結果と図は [日本語レポート](reports/final_report.md)、オフラインで読める [画像埋め込みHTML](reports/final_report.html) に保存します。HTMLはブラウザの印刷機能でPDFにできます。

光ファイバ内部の変化は、[3本の物理シミュレーション映像と日本語解説](reports/animations/kerr_physics/index.html)で見られます。自己位相変調、波長分散、両者の同時作用を、計算した電場からBlenderで映像化しました。各12秒・1080pのMP4と編集できるシーン、[再生成記録](reports/animations/kerr_physics/README.md)を保存しています。80 kmの単独パルスによる説明用実験で、1600 kmの学習評価とは別のデータです。

円形コア内の空間的な折り返しは、各6秒・1080pの
[反射](reports/animations/spatial_kerr/01_reflection.mp4)、
[Kerr OFF/ON比較](reports/animations/spatial_kerr/02_kerr_comparison.mp4)、
[位相と明暗](reports/animations/spatial_kerr/03_phase_change.mp4)で見られます。
[映像を中心にした日本語解説](reports/animations/spatial_kerr/index.html)では、入口の構造、
反射、Kerrによる分布差、位相の違いを順に読み、4条件を同じ尺度で切り替えて比較できます。
図と解説の再生成は [解説のREADME](reports/animations/spatial_kerr/README.md)を参照してください。
計算された3D等強度面、パワー流線、位相断面を、定常場のカメラ観察として描きました。
強い仮係数と表示の発光・色による強調を含む教育用モデルで、通信SMFの実パワー・材料n₂や
時間的に進む光パルスの映像ではありません。入力・表示尺度・補修来歴・再生成手順は
[空間Kerrの描画ガイド](docs/spatial-kerr-rendering.md)へ保存しています。

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
.venv/bin/python -m experiments.esn_longhaul --config experiments/esn_longhaul/configs/pilot.json --run-id pilot-cpu --powers 0 --backend numpy
```

GPUでのpilotと、設定した評価数を表示する確認は次のとおりです。

```bash
.venv/bin/python -m experiments.esn_longhaul --config experiments/esn_longhaul/configs/pilot.json --run-id pilot-check --powers 0
.venv/bin/python -m experiments.esn_longhaul --config experiments/esn_longhaul/configs/full.json --run-id final-qualified --dry-run
```

本評価は、[数値条件の確認](docs/numerical-qualification.md)で伝搬刻み・サンプル数・フレーム長・境界の扱いを確定してから行います。[`full.json`](experiments/esn_longhaul/configs/full.json) はその研究用設定です。生の伝搬・学習計算からやり直すコマンドは次のとおりです。

```bash
.venv/bin/python -m experiments.esn_longhaul --config experiments/esn_longhaul/configs/full.json --run-id final-reproduction
.venv/bin/python -m experiments.esn_longhaul.export_waveform_diagnostics --run artifacts/final-reproduction
```

CPUで本評価するなら `--backend numpy` を付け、新しいrun IDを使います。長い実行を再開する場合は同じ設定とrun IDで `--resume` を付けます。設定・実行ソース・環境・完成artifactのハッシュが一致する条件だけを再利用します。`--powers -6 -4` で条件を分けて実行できますが、最終図とレポートの生成には7電力すべてが必要です。結果は `artifacts/<run-id>/` に保存します。

[連続診断波形の保存CLI](experiments/esn_longhaul/export_waveform_diagnostics.py)は、本評価完了後に同じ既存テストフレームのseedから短いXYの2-sps波形（1シンボル当たり2サンプル）を保存します。既定は各電力の最初のテストフレームから512シンボルの連続区間で、送信bits/symbols、mask、開始位置とフレーム境界を含みます。`waveform_diagnostics/manifest.json`に来歴とハッシュを記録し、新しいBER評価や再学習は加えません。既存出力を上書きしないため、保存済みrunから再生成するときは空の出力先を指定します。

```bash
.venv/bin/python -m experiments.esn_longhaul.export_waveform_diagnostics --run artifacts/final-reproduction --output-dir .work/waveform-diagnostics-reproduction
```

## 図とレポートの再生成

保存済みの最終run `final-1600km-20261012` と公開レポートは変更しません。
ソース移動後は実際の新しいパス・内容をハッシュ化するため、旧runの厳密な監査・resumeは
新しいソースでは拒否します。旧成果物の監査・波形・図・レポートの再生成には
[元版の隔離worktree手順](experiments/esn_longhaul/README.md#historical-source-provenance)
で固定した `8a1475620fa5fa61e843f03bdbce48e1ce54419a` を使い、出力は `.work/` に分離します。
Kerrの厳密なソース検証も同じ元版を使います。

新しいソースから得たrunには、対応するモジュール入口を使えます。

```bash
.venv/bin/python -m experiments.esn_longhaul.visualize_concepts --output-dir .work/new-concepts
.venv/bin/python -m experiments.esn_longhaul.visualize_results --run artifacts/final-reproduction --qualification artifacts/numerical-qualification --output-dir .work/new-results
.venv/bin/python -m experiments.esn_longhaul.build_report --run artifacts/final-reproduction --figures-dir .work/new-results --output-dir .work/new-report
```

空間Kerrの解説は、保存済みの数値場と完成した3動画から図・HTMLを再生成します。
[生成CLI](scripts/build_spatial_kerr_report.py)と
[解説のマニフェスト](reports/animations/spatial_kerr/explanation_manifest.json)に、入力・出力と
この版のブラウザ通常確認を記録しています。

正確なコマンド、フォント・環境、入力と出力のハッシュは [結果図のマニフェスト](reports/figures/results_manifest.json) と [レポートのマニフェスト](reports/report_manifest.json) に保存します。検証・独立評価の集約記録はレポート生成の追加入力です。元の公開版と同じ検証記録を含めるには、レポートマニフェストの `regeneration_command` を使います。pilotの描画確認は `--allow-pilot --output-dir .work/pilot-figures` のように、最終図から分離します。

独立評価の原判定と指摘は [validator_result.md](artifacts/verification/validator_result.md)、親の採否・修正状況・通常検証は [independent_review.json](artifacts/verification/independent_review.json) に保存します。評価後の修正を独立評価済みとは扱いません。

## 内容と開発

- [要件](docs/requirements.md)と[実験計画](docs/experiment-plan.md)：採用条件、比較、完了基準。
- [通信路](docs/channel-methods.md)、[学習・評価](docs/learning-methods.md)、[数値条件の確認](docs/numerical-qualification.md)：実装方法と測定の限界。
- [experiments](experiments/README.md)：実験ごとの責任と第三の実験の追加手順。
- [scripts](scripts/README.md)、[configs](configs/README.md)、[artifacts](artifacts/README.md)、[reports](reports/README.md)：再実行入口と保存物の役割。

作業は[ルートのAGENTS.md](AGENTS.md)に従い、Issueを先に作り、feature branchと機能ごとのPRで管理します。元のESN研究は[PR #7](https://github.com/Motoki0705/optical-comm-lab/pull/7)、並列実験構成への移行は[Issue #42](https://github.com/Motoki0705/optical-comm-lab/issues/42)に記録します。実機での性能、多波長伝送、未学習条件への汎化やFEC後の性能は今回の結論に含めません。開発開始・構成変更・PRレビュー前には[scoutで構造を点検](docs/repository-maintenance.md)し、根拠を伴う問題を重複確認してIssue化します。この保守点検は独立validator評価とは別です。元のESN研究の1回は消費済みで、構成変更の評価指定は0回です。
