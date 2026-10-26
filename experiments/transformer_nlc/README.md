# Transformer nonlinear-compensation reproduction

[主2条件の再現仕様](docs/reproduction-spec.md)に、論文の出典、採用した仮定、データ/モデル契約、候補、予算、実行ゲート、受入条件を保存する。基準は[arXiv:2304.13119v3](https://arxiv.org/abs/2304.13119v3)。物理/DSP・データ生成、モデル・物理マスク・成分別RMPS、学習・epoch再開・validation選択固定・直接BER評価の実装を含む。主条件の本学習・held-out結果・読者向け報告の完了は、数値qualificationと実行記録で別に判断する。実装の存在は論文性能の再現結果を意味しない。

## 導入とモデルの合成入力例

リポジトリ直下で次を実行する。

```bash
uv sync --locked --extra transformer --group dev
```

PyTorchは任意の`transformer` extraで、既存のCPU実験の必須依存には追加しない。config・NumPy mask・理論RMPSの利用にはPyTorchを必要としない。[pyproject.toml](../../pyproject.toml)のuv source設定と[uv.lock](../../uv.lock)は、Linux/WindowsでPyTorch 2.7.1のCUDA 12.8 wheelを選ぶ。CUDA 12.8でのBlackwell対応は[公式リリース説明](https://pytorch.org/blog/pytorch-2-7/)を参照する。このsource選択はuv固有であり、`pip install '.[transformer]'`では同じCUDA buildを保証しない。上の手順はこのリポジトリの`.venv`を使用する。

次はCPU上のshape確認だけを行う。通信路データ、学習済み重み、BER/Qの研究結果は含まない。

```bash
.venv/bin/python - <<'PY'
import torch
from experiments.transformer_nlc import (
    ModelConfig, TransformerNLC, predict_dual_polarization,
)
from experiments.transformer_nlc.complexity import (
    complexity_breakdown, count_parameters,
)

torch.manual_seed(0)
config = ModelConfig.candidate("main16", "C", masked=True)
model = TransformerNLC(config).eval()
features = torch.zeros(2, config.input_length, 4)  # synthetic XI,XQ,YI,YQ
with torch.inference_mode():
    residual_x = model(features)
    residual_xy = predict_dual_polarization(model, features)
print(residual_x.shape)   # [2,128,2]: X residual, IQ
print(residual_xy.shape)  # [2,128,2,2]: time, XY, IQ after batch
print(count_parameters(model))  # actual trainable parameters
print(complexity_breakdown(config).to_dict())
PY
```

## 実装済みAPI

| モジュール | 契約 |
| --- | --- |
| [model_config.py](model_config.py) | `ModelConfig.candidate(condition, name, masked=True)`は`main16/A,B,C`と`main64/high,mid,low-adopted`の6候補。`masked=False`は同じ構造のmaskなし候補。`to_dict()` / `from_dict()`で設定を保存・復元し、未知のfieldや無効なshapeを拒否する |
| [model.py](model.py) | `TransformerNLC(config)`の`forward(features, mask=None)`は`[B,b+2t,4]`から片偏波の受信残差`[B,b,2]`を出力。valid CNN、sinusoidal PE、Post-LN、独立したattention射影全次元、近傍MLPを使用する |
| 同上 | `swap_xy(features)`は`XI,XQ,YI,YQ`を`YI,YQ,XI,XQ`へ交換。`predict_dual_polarization(model, features, mask=None)`は同じ重みを2回使い、`[B,b,2,2]`（batch、時刻、XY、IQ）を返す。補償の減算・power倍率は呼出側の責務 |
| [mask.py](mask.py) | `individual_mask(t, rho)`と`block_mask(t, b, rho)`は許可位置をTrueとするNumPy配列。`additive_mask(allowed)`は許可0・禁止−∞へ変換する |
| [complexity.py](complexity.py) | `complexity_breakdown(config)`は成分別RMPS、片偏波symbol/DP時刻の分母、疎計算を仮定した値とdense backendの推定値を分けて返す。`analytic_parameter_count(config)`の構造由来計数と`count_parameters(model)`の実パラメータ計数は別API |

`forward(mask=None)`は**configのmaskを使用する**。完全なmaskなし経路には`mask_rho=None`のconfigを使うか、全許可のbool maskを明示する。明示maskはConv後の`[N,N]`で、bool許可配列または0/−∞の加算配列を受け取る。`return_attention=True`は`(prediction, tuple(layer_attention))`を返す診断経路で、各配列は`[B,h,N,N]`。モデルには保持せず、保存も自動では行わない。

maskの中心行はEq. 8の積形式とFig. 4を根拠に許可し、Algorithm 1の文字どおりの実装との差を[固定仕様](docs/reproduction-spec.md)に記す。Fig. 4bとの密度不一致も未解決のまま保持する。通常経路のPyTorch SDPAはdense attentionであり、maskの理論RMPS削減を実速度向上と扱わない。RMPSのsoftmax/LN等の算入も採用規約であり、原実装との完全同定は主張しない。

モデル・crop/XY交換・mask・独立dense参照と勾配・計算量は[モデル試験](../../tests/test_transformer_model.py)、[mask試験](../../tests/test_transformer_mask.py)、[計算量試験](../../tests/test_transformer_complexity.py)で検証する。モデル試験は任意PyTorchが未導入の場合のみskipする。

## 学習・選択固定・評価の入口

統合CLIは`python -m experiments.transformer_nlc --help`で確認する。`generate`はPyTorchをimportしない。`train`はTrainingConfig JSON、`generate`はDataConfig JSONを受け取り、未知fieldは拒否する。`--seed`は0/1/2の独立replicate keyであり、master entropyを変更する引数ではない。trainではdatasetのreplicate一致を確認する。

次は専用pilotを生成し、最小構造のCPU smoke（2 optimizer steps）を実行する例。生成する4096 DP時刻は32 target blocksであり、本番train `2^19`の代替ではない。

```bash
.venv/bin/python -m experiments.transformer_nlc generate \
  --condition 16 --pilot --seed 0 --run-id data-pilot16 --dry-run
.venv/bin/python -m experiments.transformer_nlc generate \
  --condition 16 --pilot --seed 0 --run-id data-pilot16
.venv/bin/python -m experiments.transformer_nlc train \
  --config experiments/transformer_nlc/configs/smoke-main16-masked.json \
  --data-manifest artifacts/transformer-nlc/data-pilot16/manifest.json \
  --run-id smoke-main16-masked --dry-run
```

`--dry-run`を外すと表示された設定で実行する。64QAMには`--condition 64`と`smoke-main64-*.json`を使い、maskなしには`*-unmasked.json`を使う。各条件で同じdatasetをmaskペアに共有する。4つの`learning-pilot-*.json`はCUDA・最大5 epochsで、生成されたvalidation4096の先頭2048 DP時刻を明示して使う。生成件数・使用件数・元symbol index範囲をrunに保存する。production validationを削る機能はない。

```bash
.venv/bin/python -m experiments.transformer_nlc train \
  --config experiments/transformer_nlc/configs/learning-pilot-main16-masked.json \
  --data-manifest artifacts/transformer-nlc/data-pilot16/manifest.json \
  --run-id learning-pilot-main16-masked
.venv/bin/python -m experiments.transformer_nlc benchmark \
  --condition 16 --masked --run-id resource-main16-A-masked --dry-run
```

`benchmark`は最大構造A/high、合成入力、有効batch512、実際のAdam更新1回を測定する。mask有無×2条件の4ケースを使い、重みは学習trialへ転用しない。default microbatchは512、資源不足時は`--microbatch-blocks 64`で有効batchを維持する。GPU実行は共通`.work/orchestration/gpu.lock`を非blocking取得し、他jobが占有中ならbusyを返す。trainにも同じロックを適用する。

production生成には一致した合格qualification artifactを`--qualification`で渡す。qualificationの計画・実行入口は`python -m experiments.transformer_nlc.data_cli qualify --help`。未資格をpilot扱いで通してproductionと称さない。production `sweep`は両条件×3replicateの6 manifestを要求し、固定36 trialsを全保存する。

物理/DSPの契約、numerical-source fingerprint、実import位置の照合、専用qualificationの詳細は[data pipeline](docs/data-pipeline.md)を参照する。学習側はこの検査結果を再利用し、model/training/evaluationの実際にimportされたファイルが同じstudyに属することも照合する。wheelでも所在と内容hashを記録し、checkoutが混在したimportを拒否する。

```bash
.venv/bin/python -m experiments.transformer_nlc sweep \
  --data-manifest artifacts/transformer-nlc/data-main16-r{0,1,2}/manifest.json \
                  artifacts/transformer-nlc/data-main64-r{0,1,2}/manifest.json \
  --device cuda --run-id principal-sweep --dry-run
.venv/bin/python -m experiments.transformer_nlc freeze \
  --ledger artifacts/transformer-nlc/principal-sweep/ledger.json \
  --run-id principal-freeze
.venv/bin/python -m experiments.transformer_nlc evaluate \
  --freeze artifacts/transformer-nlc/principal-freeze/selection-freeze.json \
  --device cuda --run-id principal-test
.venv/bin/python -m experiments.transformer_nlc recount \
  --metrics artifacts/transformer-nlc/principal-test/metrics.json
```

sweepの`--dry-run`を外した実行が完了した後にfreezeへ進む。freezeは36のbest-validation checkpointすべてとconfig/data/code hashを固定し、条件×maskごとに3replicateの平均validation MSEからprimaryを決める。同率は低RMPS、さらに同率なら候補表の下側。test配列はfreezeまで開かない。testではprimary以外も固定候補を全評価し、no-NLCは同一DSP・同一frameで一度だけ計数する。pilotの明示的なfreezeは`--allow-pilot`を要求し、本番の選択とは区別する。

trainはX残差の中央b×2 MSE、validationは同一重みのXY swapによる両偏波平均MSE。有効batch512、Adam、10-epoch warmup後inverse-square-root、strict改善、patience100、production最大500 epochsを実効設定へ記録する。cap到達を収束とは呼ばない。failure/OOM/NaN・全attempt・途中epochの位置を残す。`--stop-after-epochs N`は設定のcapを変えずepoch境界で運用上のpauseを行い、`--resume`はlast completed epochへ戻る。途中epochの再実行はresume eventへ記録する。設定・data・実行コード・数値runtimeが不一致のresumeは拒否し、Adam/scheduler/PCG64/PyTorch RNGを復元する。

記録済みOOM後に有効batchを保ってmicrobatchだけ小さくする場合、親が変更理由を決め、変更したTrainingConfigと`train --resume --microbatch-change-reason '理由'`を明示する。他の設定変更や理由なし変更は拒否する。元のbest checkpoint設定と変更履歴を別に保ち、sweep再開は手動回復済みの完了runを台帳へ照合できる。モデル初期化前にcheckpointを作れなかった失敗は`initialization-failures.json`を保存し、新run IDでの記録付き再試行とする。

保存先は`artifacts/transformer-nlc/<run-id>/`。trainは`run.json`、`history.json`、`best.pt`、`latest.pt`、sweepは失敗を含む`ledger.json`、freezeは不変の`selection-freeze.json`、evaluateは`metrics.json`/`metrics.csv`とignored `bulk/`内の再計数用予測を保存する。入力source/data/checkpointと重なる出力、同runへの重複job、非resume上書きを拒否する。GPU許容差は設定に先に保存し、実測した再開検証と未検証を分ける。通常試験には小さな合成pilotでのCPU実学習対再開・test非読込・checkpoint reload・保存予測の独立Gray BER再計数を含める。

BERはX/Y/合算のerrors/bit数と95% Clopper–Pearson区間、BER由来Qを保存する。ゼロ誤りのQは点推定を持たず下限を示す。独立bit計数区間とframe/replicate変動を区別し、MSE/EVM/SNRは補助値とする。追加のgain/phase fitはtestラベルから行わない。Eq.2の`r-alpha*e_hat`はunit-Es座標に適用する。保存結果の`report`生成には、以下に記録した内容の生成器を使う。

## 主2条件の保存結果（2026-10-26）

主16QAM 3200 km / 32 GBd / +2 dBm、主64QAM 960 km / 42 GBd / +3 dBmについて、各18件、計36件の学習・validationだけによる選択固定・held-out評価が完了した。
通常DSPは6つのcondition/seed基準にまとめ、全手法で同じ4 test frame/seedを計数した。自動・CLIの独立再計数が合格し、全36 checkpointの再読み込み評価で168 frameの指標が完全一致した。18件はpatience停止、18件は500 epoch上限で収束未確認。失敗・OOMはなかった。独立validatorは指定/試行/完了=0/0/0。

| 条件 / validation選択primary | 通常DSPのseed平均Q dB | primaryのseed平均Q dB | 同一seed通常DSPからの平均ΔQ dB |
| --- | ---: | ---: | ---: |
| 16QAM C / maskなし | 6.772690 | 6.771079 | −0.001611 |
| 16QAM C / maskあり | 6.772690 | 6.771247 | −0.001443 |
| 64QAM high / maskなし | 6.725544 | 6.727675 | ＋0.002131 |
| 64QAM high / maskあり | 6.725544 | 6.727181 | ＋0.001638 |

各値は3独立seedの記述値で、有意な改善の宣言ではない。差はseed間変動より小さく、論文の大きな改善を再現したとは扱わない。全36件・全seed/frame、計数区間と変動、負の結果、停止理由、理論RMPSと実測資源の範囲は[日本語レポート](../../reports/transformer-nlc/principal-20261025/report.md)と[オフラインHTML](../../reports/transformer-nlc/principal-20261025/report.html)に保存した。論文メトリクスとの比較では、16QAM A・maskなしは論文8.82 dBに対して今回6.7707 dB、64QAM primary・maskなしは論文の探索最良8.17 dBに対して今回6.7277 dBだった。補償による品質改善は再現できていない。16QAMの理論RMPSは公開値より0.30〜1.86%低いが、計数規約と丸めの差があり、速度向上の実証ではない。

[実行証拠索引](../../artifacts/transformer-nlc/principal-learning-20261025/ready-report-inputs.json)は、6データmanifest、qualification、全36 run/history、freeze、主評価とreload評価、再計数、資源資格、環境、入力整合性を指す。[現行レポート生成manifest](../../reports/transformer-nlc/principal-20261025/report-manifest.json)は入力・生成コード・生成物のhashを残す。[論文メトリクス比較の改訂検証](../../artifacts/transformer-nlc/principal-paper-comparison-20261026/verification.json)では、元の成果物234件（checkpoint 88件を含む）と数値表3件の不変性、図表の再生成、公開値との対応計算を検証する。

学習・評価の生成元は`5705e7b9fe8db08b26e829979d39e2279f1739dd`、学習数値source hashは`c67441678e6720387a830e9db781849f817e16e99920c8611fdade1cbe591142`のまま。現行の比較中心の生成器は`db63638579853297a13efe49c44887a3f7059813`にある。改訂前の[生成検証・索引](../../artifacts/transformer-nlc/principal-report-verification-20261025/report-verification.json)と、その配下の過去render記録は当時の成果物の履歴として保全した。現行レポートに対する検証とは区別する。旧manifestと改訂の対応は[来歴の説明](../../artifacts/transformer-nlc/principal-paper-comparison-20261026/README.md)を参照。元の科学manifestの生成元SHAやpathは変更していない。

元の配置にある保存済みcompact入力から、現行版を次のように別directoryへ再生成できる。`REPORT_CODE`は上記commitの生成器を持つcheckoutを指定する。

```bash
REPORT_CODE=/home/kamimura/projects/optical-comm-lab/.work/worktrees/paper-metric-reproduction
REPORT_RESULTS=/home/kamimura/projects/optical-comm-lab/.work/worktrees/transformer-results
cd "$REPORT_CODE"
PYTHONPATH="$REPORT_CODE/src:$REPORT_CODE" \
  /home/kamimura/projects/optical-comm-lab/.work/worktrees/transformer-model/.venv/bin/python \
  -m experiments.transformer_nlc report \
  --metrics "$REPORT_RESULTS/artifacts/transformer-nlc/principal-evaluation-20261025/metrics.json" \
  --ledger "$REPORT_RESULTS/artifacts/transformer-nlc/principal-sweep-20261025/ledger.json" \
  --resource-proof \
    "$REPORT_RESULTS/artifacts/transformer-nlc/resource-main16-masked-micro512-20261025/resource-benchmark.json" \
    "$REPORT_RESULTS/artifacts/transformer-nlc/resource-main16-unmasked-micro512-20261025/resource-benchmark.json" \
    "$REPORT_RESULTS/artifacts/transformer-nlc/resource-main64-masked-micro512-20261025/resource-benchmark.json" \
    "$REPORT_RESULTS/artifacts/transformer-nlc/resource-main64-unmasked-micro512-20261025/resource-benchmark.json" \
  --output-root "$REPORT_RESULTS/.work/rmps-label-fix-20261026/regenerated" \
  --run-id principal-20261025
```

この再生成先が既に存在する場合は、新しい出力先を選ぶか、未変更のreport所有directoryに限って`--regenerate`を明示する。既定の生成はcheckpointやbulk配列を開かない。図・表のhashは一致し、Markdown/HTMLの差は出力先を含むコマンドだけ、manifestの生成時刻は再生成ごとに変わる。

freeze/run/checkpoint/data manifestは元のproducerの絶対パスを保存している。再読み込み評価とcompact再生成はその元配置で検証済みで、任意の新cloneへそのまま移せるとは主張しない。元配置を再現するか、保存された生成recipeから新runを作る必要がある。元metadataを移動先へ書き換えた手順は検証していない。

## 保存結果から日本語レポートを生成する

`report`は学習・モデル推論をやり直さず、評価時に固定したfreeze、`run.json`/history、data manifest内のconfig/qualification、保存metricsからMarkdown、offline HTML、PNG/SVG図、CSV/JSON表、入力/出力/生成コードhashを生成する。Matplotlibは実際の図生成時だけimportする。元論文図を測定結果として転用しない。

```bash
.venv/bin/python -m experiments.transformer_nlc report \
  --metrics artifacts/transformer-nlc/principal-test/metrics.json \
  --ledger artifacts/transformer-nlc/principal-sweep/ledger.json \
  --run-id principal --dry-run
.venv/bin/python -m experiments.transformer_nlc report \
  --metrics artifacts/transformer-nlc/principal-test/metrics.json \
  --ledger artifacts/transformer-nlc/principal-sweep/ledger.json \
  --resource-proof artifacts/transformer-nlc/resource-main16-A-masked/resource-benchmark.json \
                   artifacts/transformer-nlc/resource-main16-A-unmasked/resource-benchmark.json \
                   artifacts/transformer-nlc/resource-main64-high-masked/resource-benchmark.json \
                   artifacts/transformer-nlc/resource-main64-high-unmasked/resource-benchmark.json \
  --run-id principal
```

出力は`reports/transformer-nlc/principal/report.md`と`report.html`。HTMLの図表リンクは相対パスで、出力directoryをまとめて移せばnetworkなしで読める。外部リンクは原論文への任意の参照だけ。日本語fontのfamily/元file hashをmanifestへ記録し、SVGはglyphをpathとして保存する。browserでの見た目の確認を自動実施したという意味ではない。

既定でcompact入力のhash・状態・全trial/seed/frame・bit分母・計数区間・合算・既存groupを照合する。checkpointはfreezeに保存したhash識別子を照合し、model bytesをloadしない。bulk配列も不要なので、bulkを別storageへ移した後もcompact入力を元pathへ揃えれば再生成できる。`--recount`を明示すると既存の独立再計数APIでbulk予測をCPU検査する。`--dry-run --recount`は拒否する。

本番freezeは全36学習runを必須とし、完了評価の欠測・hash不一致は拒否する。専用pilotの報告には`--allow-pilot`が必要で、必ずPILOTと表示する。開発fixtureの出力は`--output-root .work/transformer-report-fixtures`などignored scratchに置き、本番結果として保存しない。`failed`/`interrupted`/`in_progress`の評価を記述する場合は`--allow-partial`も明示し、欠測対象と失敗を表示する。これはfreeze前の未完了学習だけからの報告には対応しない。

全36予定runを表に残し、通常DSPをdataset/condition/seed/frameごとに一度だけ計数する。完全一致するbaseline重複だけを除外し、異なる計数は拒否する。Q合算とseed Q平均、独立bit計数区間とframe/seed変動、疎実装を仮定した理論RMPSとdense backendの実測資源を分ける。epoch capは未収束、未記録のmemory/latencyは未測定と表示する。resource proofは任意で、未入力を合格に読み替えない。

同じrun-idへの通常出力は拒否する。`--regenerate`は、そのdirectoryがこのreportのmanifestで所有され、全出力hashが一致し、追加file/directoryがない場合だけ置き換える。保存先run directory自体がsymlinkの場合は、CLI/APIともに`--regenerate`指定でも拒否する。変更済みの図や利用者の追加fileは削除しない。元入力とsource、既存ESN/Kerr reportへの出力は拒否する。CLIの`report`には変更すべき学習configやseedがないため、`--config`/`--seed`は受け取らない。再実行commandと図表のhashは`report-manifest.json`に残る。

## 今後の実行範囲と進行

今回の実行範囲はDP-16QAM 32 GBd / 40×80 kmとDP-64QAM 42 GBd / 12×80 kmの学習・独立評価・checkpoint・図表・日本語レポートまで。mask有無と通常DSPを比較する。初期計画は6構造×2 mask×3 seedの36学習run上限で、原論文の数百モデル探索に対する**部分再現**とする。4 learning pilotsに加え、最大構造A/highのmask有無を実batchで資源benchmarkし、数値精度、時間、VRAM、保存量を測定してから本番設定を固定する。通常の数値refinementや回復可能OOMへのmicrobatch調整は、親が範囲内で記録して継続する。

進行は[親#40](https://github.com/Motoki0705/optical-comm-lab/issues/40)で追跡する。

1. [#42](https://github.com/Motoki0705/optical-comm-lab/issues/42)：実験workspaceの分離・既存成果物保全・回帰。
2. [#43](https://github.com/Motoki0705/optical-comm-lab/issues/43)：出典/仮定・主2条件・段階計画（本仕様）。
3. [#44](https://github.com/Motoki0705/optical-comm-lab/issues/44) / [#45](https://github.com/Motoki0705/optical-comm-lab/issues/45)：物理/DSP/独立データとTransformer/mask。
4. [#46](https://github.com/Motoki0705/optical-comm-lab/issues/46)：学習・再開・validation選択・固定test。
5. [#47](https://github.com/Motoki0705/optical-comm-lab/issues/47)：pilot/数値資格後の本学習・評価・保存・報告。

共通の物理/DSP/指標は`optical_comm`から使い、このworkspaceがモデル、学習、実験設定、報告を所有する。[拡張手順](../README.md)に従い、ESNは独立したpeerとして保持する。研究成果物は`artifacts/transformer-nlc/`、読者向けレポートは`reports/transformer-nlc/`。通常試験・pilot・数値資格・production・held-out testの状態を、それぞれ保存物で判定する。

PMD・レート変更、CNN-LSTM/DBP拡張比較、広範なアブレーションは実行前に相談し、未完の親#40を開いたままにする。今回の独立validatorは0回。過去の結果・生成元SHA・別タスクのmerge許可は引き継がない。
