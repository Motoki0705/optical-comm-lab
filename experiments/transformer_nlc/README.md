# Transformer nonlinear-compensation reproduction

[主2条件の再現仕様](docs/reproduction-spec.md)に、論文の出典、採用した仮定、データ/モデル契約、候補、予算、実行ゲート、受入条件を保存する。基準は[arXiv:2304.13119v3](https://arxiv.org/abs/2304.13119v3)。このブランチにはモデル・物理マスク・成分別RMPSと通常テストがある（[#45](https://github.com/Motoki0705/optical-comm-lab/issues/45)、[PR #52](https://github.com/Motoki0705/optical-comm-lab/pull/52)）。物理/DSP・データ生成は別の[PR #51](https://github.com/Motoki0705/optical-comm-lab/pull/51)で進行中であり、このブランチへの統合済みとは扱わない。統合CLI、学習・評価・報告の実装と主条件の研究結果は今後の作業である。

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

## 今後の実行範囲と進行

今回の実行範囲はDP-16QAM 32 GBd / 40×80 kmとDP-64QAM 42 GBd / 12×80 kmの学習・独立評価・checkpoint・図表・日本語レポートまで。mask有無と通常DSPを比較する。初期計画は6構造×2 mask×3 seedの36学習run上限で、原論文の数百モデル探索に対する**部分再現**とする。4 learning pilotsに加え、最大構造A/highのmask有無を実batchで資源benchmarkし、数値精度、時間、VRAM、保存量を測定してから本番設定を固定する。通常の数値refinementや回復可能OOMへのmicrobatch調整は、親が範囲内で記録して継続する。

進行は[親#40](https://github.com/Motoki0705/optical-comm-lab/issues/40)で追跡する。

1. [#42](https://github.com/Motoki0705/optical-comm-lab/issues/42)：実験workspaceの分離・既存成果物保全・回帰。
2. [#43](https://github.com/Motoki0705/optical-comm-lab/issues/43)：出典/仮定・主2条件・段階計画（本仕様）。
3. [#44](https://github.com/Motoki0705/optical-comm-lab/issues/44) / [#45](https://github.com/Motoki0705/optical-comm-lab/issues/45)：物理/DSP/独立データとTransformer/mask。
4. [#46](https://github.com/Motoki0705/optical-comm-lab/issues/46)：学習・再開・validation選択・固定test。
5. [#47](https://github.com/Motoki0705/optical-comm-lab/issues/47)：pilot/数値資格後の本学習・評価・保存・報告。

共通の物理/DSP/指標は`optical_comm`から使い、このworkspaceがモデル、学習、実験設定、報告を所有する。[拡張手順](../README.md)に従い、ESNは独立したpeerとして保持する。将来の統合入口は`python -m experiments.transformer_nlc {generate,train,evaluate,sweep,report}`、研究成果物は`artifacts/transformer-nlc/`、レポートは`reports/transformer-nlc/`を予定する。モデルAPIが存在しても、これらの統合CLI・学習・報告が実装済みであることは意味しない。

PMD・レート変更、CNN-LSTM/DBP拡張比較、広範なアブレーションは実行前に相談し、未完の親#40を開いたままにする。今回の独立validatorは0回。過去の結果・生成元SHA・別タスクのmerge許可は引き継がない。
