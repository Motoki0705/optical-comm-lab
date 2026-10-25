# Transformer-NLC 主2条件の再現仕様

この文書は [#43](https://github.com/Motoki0705/optical-comm-lab/issues/43) の仕様固定物であり、[#44–47](https://github.com/Motoki0705/optical-comm-lab/issues/40#issuecomment-6396441791) の実装・実行契約とする。2026-10-25時点ではデータ生成、モデル、学習、測定結果はまだ存在しない。数値は出典値またはこれから検証する採用値であり、実測結果ではない。

## 要求と今回の境界

親 [#40](https://github.com/Motoki0705/optical-comm-lab/issues/40) に保存した原要求は次のとおり。

> Transformer NLCを再現する実験を行いたいです。再現する項目としては、データセット、モデル、評価などの主要な項目すべてです。

既存の[ユーザー要求](../../../docs/requirements.md)にある「モデルの学習まで行い。最終レポートまで作成してください」「レポートは不要な詳細を除き、必要な可視化を中心に説明する」も成果物の方針として保持する。[確定した実行順序・選択](https://github.com/Motoki0705/optical-comm-lab/issues/40#issuecomment-6396391018)は以下。

- まず共通 `src/optical_comm` と並列の `experiments/esn_longhaul` / `experiments/transformer_nlc` へ分離し、[#42](https://github.com/Motoki0705/optical-comm-lab/issues/42) の通常検証後に本実装を開始する。
- 今回は主16QAM・64QAMのデータ生成、学習、独立test、保存checkpoint、図表、日本語レポート、再実行手順まで行う。通常DSP、maskあり・なしの比較と計算量を含む。
- PMD・レート変更、CNN-LSTM/DBPとの拡張比較、広範なアブレーションは実行前に相談する。親#40の未完項目として残し、今回の終了で親を閉じない。
- 実装・実験・図表の担当は GPT-6.1-SOL / xhigh。親が統合、Git/GitHub、通常検証、約30分ごとの進行管理を担当する（[#41](https://github.com/Motoki0705/optical-comm-lab/issues/41)）。follow-upの登録・受理・受信は別々に確認し、科学的停止条件をtimerで解除しない。
- 今回の独立validatorは指定0回。通常テストと数値検証は行う。今回のmerge許可はなく、過去の別タスクの許可・validatorを流用しない。

`artifacts/final-1600km-20261012`、既存レポート、Kerrの計算・映像・URL・生成元SHAを保全する。旧ESN設定をTransformer条件として読み替えない。

## 出典と証拠の強さ

基準は Behinaein Hamgini et al., *Application of Transformers for Nonlinear Channel Compensation in Optical Systems*, **arXiv:2304.13119v3, 2024-08-01**。[HTML](https://arxiv.org/html/2304.13119v3)・[PDF](https://arxiv.org/pdf/2304.13119v3)・[公式source archive](https://arxiv.org/src/2304.13119v3)を参照する。以下の区別をconfig/manifest/reportにも残す。

| 状態 | 意味 |
| --- | --- |
| `published` | v3の表示本文・式・表・図にある記載 |
| `author_tex_commented` | 公式TeX内のコメント。公開PDFの記載値・実行コードの証拠には昇格しない |
| `adopted` | 再現側で明示して採用した未記載条件・補完 |
| `unresolved` | 原実装との同一性を確定できない点。検証必須か、結論の限界かを個別に記す |
| `deferred` | 今回は実行せず、拡張前に相談する項目 |

2026-10-25に公式sourceを確認。archive SHA256は `e9a1ef49f6af2a4dfff16fc082cd6ac31c0efddf072d035623b9d4c23028877a`、`Arxiv_Transformer_aug_2024_one_column.tex` は `51b548bd4d89f2633279143dd82388db843eb266ac7bb22233de90a2f3140326`。配布内容はTeX、参考文献、図PDFで、学習コード・データ・checkpointは含まれない。公式の学習コード・データの公開先は今回の調査では確認できていない。不存在やOSSライセンスは断定しない。新たな配布物が見つかったらURL・ライセンス・版・hash・利用範囲を記録し、test開封前に影響を判断する。元論文の図を再現結果として転用しない。

### 物理・データの仕様台帳

| 出典の箇所 | 記載値・事項 | 今回の採用値 | 状態・理由／限界 |
| --- | --- | --- | --- |
| IV-A、Figs. 7–8 | 単一チャネル、DP-16QAM 32 GBd 40×80 km / DP-64QAM 42 GBd 12×80 km | `main16`: 3200 km、+2 dBm学習・評価 / `main64`: 960 km、+3 dBm学習・評価 | `published`。主実験の評価点をまずこの2点に限定 |
| IV-A | EDFA NF=6 dB、RRC roll-off=1/16 | 同じ。各span後に損失を補うEDFA、span別・偏波別ASE | NF/roll-offは`published`。有限gainのPSD等は下記`adopted` |
| IV-A | 総分散の半分をTxで事前補償 | Tx逆分散L/2 + Rx逆分散L/2 | `published`。全量Rx補償と区別 |
| IV-A | adaptive SSFM、最大非線形位相0.05度 | 0.05° = 8.72664626e-4 rad。対称Manakov SSFM、下記の刻み決定 | 閾値は`published`、最大電力の評価位置・刻み決定は`adopted` |
| IV-A | SSMF、理想電気系/MZM/DAC/ADC、量子化・clippingなし | loss=0.2 dB/km、D=17 ps/(nm km)、gamma=1.3 /(W km)、lambda=1550 nm | 物性4値は`adopted`。既存値と同じでも論文記載とは呼ばない |
| IV-A、Fig. 6 | resampling/synchronization/frequency correction/CDC/matched filter/linear equalizer/CRの後にNLC | 完全clock・frequency、PMD=0、laser linewidth=0。線形等化・CRは下記契約 | 理想化の詳細は`adopted`。CRはKerr由来位相も扱うため省略しない |
| IV-B | train 2^19、evaluation 2^18 symbols、PCG64、異なるseed | 1 symbol = 1 DP時刻。train 2^19、**追加**validation 2^17、test 2^18、各条件・反復ごと | `published`＋`adopted`の数え方/validation。学習数からvalidationを差し引かない |
| 論文未記載 | powerの偏波規約、sampling、ASE band、filter/frame/guard長 | XY合計dBm、Es=1/偏波、初期4 sps、固定ASE両側band 128 GHz、RRC span128 symbols、下記frame契約 | 全て`adopted`。sampling/guard等はqualification後に固定 |
| II-A Eqs. 1–2 | 歪み推定を減算、power比alphaを乗算 | `e=r-s`、`s_hat=r-alpha*e_hat`、`alpha=10**((P_eval-P_train)/10)` | `published`。単位Esへの正規化後に適用。残差にはASEも含む |

### モデル・学習・評価の仕様台帳

| 出典の箇所 | 記載値・事項 | 今回の採用値 | 状態・理由／限界 |
| --- | --- | --- | --- |
| II-B Eqs. 3–5、III-A | encoder、sinusoidal位置符号、MHA、FFN/ReLU、Post-LN | 下記shape契約、biasあり線形層、LayerNorm eps=1e-5、embeddingの追加sqrt(d_model)倍率なし | 中核は`published`、bias/eps/追加倍率なしは`adopted` |
| III-B | CNN、kernel9、stride1、LeakyReLU(0.2) | valid Conv1D 4→d_model、paddingなし | `published`。長さをsame paddingで変えない |
| III-A、III-D | 一偏波歪み、近傍W、MLP W*d_model→2→10→2 | 中間LeakyReLU(0.2)、最後linear、同一重みをXY swapでも使用 | `published`。IV-Fの両偏波出力とは別 |
| III-A | d_k=d_v=d/h、d≠d_modelも探索 | `attention_projection_dim=d`、`head_dim=d/h`、d%h=0 | 関係式は`published`。「key size」の表欄=dという解釈は`adopted`。RMPSとの整合は支持材料、コードでの確認なし |
| III-C Eq. 8、Algs. 1–2、Figs. 4–5 | 双曲線個別maskとblock union | rho=2.6 / maskなし、中心行を許可する補完、下記定義 | `published`＋`adopted`。Alg. 1のm=0の欠落を無言で修正しない |
| IV-B、IV-C | Adam、MSE、batch512、warm-up、100 epochs改善なし停止、b=128 | 下記のLR/epoch上限/validation監視、b=128 | 記載値は`published`。LR数値・停止対象等は`adopted` |
| Table I、IV-B | 2段階探索、数百モデル、探索範囲 | 6構造×mask有無×3seed=36学習runを初期上限とする | `adopted`の縮小探索。原論文の全面Pareto探索は未実施 |
| Table II、公式TeX755–775行 | A/B/CのQ/RMPSは表示、対応しそうな構造はコメントのみ | 次表main16 A/B/C | Q/RMPSは`published`、構造対応は`author_tex_commented`。値が一致するが確定した実行configではない |
| 公式TeX825–855行、Table I | 64QAM region1/2/3の構造・数値がコメント、低域t=8は表示範囲[16:64]と不一致 | 高/中のコメント構造と、**t=16に変更した**低域候補 | `author_tex_commented`＋`adopted`。低域を原論文の同一モデルと称さない |
| Table IV、IV-F | rate/PMD比較のModel1/2 | 必要時の小shape確認の参考に留め、主学習候補にはしない | `deferred`。Table II代表点とは別、両偏波同時出力の節 |
| Figs. 7–8 | 主2条件のQ対RMPS | 同じ学習電力の比較＋論文値との差、BER実測 | `published`に対応。launch-power曲線・最適powerは今回未測定 |
| Fig. 8 caption対本文/legend | maskあり/なしの色説明が逆転 | 本文/legendの青=maskあり、赤=なしを採用し、不一致を記載 | `unresolved`。出典の色だけで学習runを分類しない |
| 対象論文のQ定義 | 数式による定義を確認できない | BER由来 `20*log10(sqrt(2)*erfcinv(2*BER))` | `adopted`。[引用[10]の論文Sec. IV-A、p.8](https://arxiv.org/pdf/2103.08212)を根拠にする。対象論文と同一定義とは未確定 |

## 物理・通常DSPの実装契約

再利用する[共通物理の単位・ASE規約](../../../docs/channel-methods.md)を基礎とし、Transformer用configですべて上書き・保存する。従来のESNデフォルトを変更しない。

- 一様Gray square QAM。16QAMは軸{-3,-1,1,3}/sqrt(10)、64QAMは{-7,-5,-3,-1,1,3,5,7}/sqrt(42)、各軸は振幅昇順のbinary-reflected Gray順。TX ensemble平均複素Es=1/偏波。物理fieldはsqrt(W)、`P_X=P_Y=1e-3*10**(P_dBm/10)/2`。フレーム実測powerで再正規化しない。
- 現共通DSPの符号で線形fiber応答を `H(L)=exp(i*beta2*omega**2*L/2)` と定義する。TxにH(-L_total/2)、fiberにH(+L_total)、RxにH(-L_total/2)。gamma=0・ASEなしの往復identityで符号と適用回数を確認する。
- 各SSFM候補刻みdzで、最初の半分散後の実fieldの `max_t(|Ex|²+|Ey|²)` を評価し、`(8/9)*gamma*P_max*L_eff(dz) <= phi_max` を満たすよう再試行する。損失積分は `L_eff=(1-exp(-alpha*dz))/alpha`。初期最大dz=1 km、最小dz=1e-6 km、span端を正確に切る。最小刻みでも超過なら失敗とする。刻み分布・最大実位相・再試行数を保存する。このアルゴリズムは論文の原実装と同定できていない。
- EDFA `G=10**(loss_dB_per_km*span_km/10)`、`F=10**(NF_dB/10)`、片偏波・両側複素baseband PSD `h*nu*(G*F-1)/2` W/Hz。各実quadratureの分散は複素分散の半分。ASEは全後段spanを伝搬し、既知noiseの減算はしない。物理bandを128 GHzで固定し、4→8 spsで雑音量を倍増させない。band/PSD/FFT bin/総分散を記録する。
- Tx整形→Tx事前CDC→fiber/EDFA→理想resampling・sync・frequency→Rx CDC→matched RRC→線形等化→CR→1-sps NLC→Gray判定の順。2-sps診断とraw受信fieldは別stage名で保持する。完全clock/frequencyでidentityになる工程もmanifestに記す。
- 線形等化の初期案は1-spsの2×2複素FIR、11 taps（±5）、trainの専用pilotを集約したridge LS、ridge=`1e-6*trace(R)/dim(R)`。データ領域の正解は使わず、重みはvalidation/test前に固定する。PMD=0でもidentityへの置換は、独立pilot/validationで残留ISI・偏波漏れ・BERを確認して理由を残す場合だけ認める。
- CRの初期案はdecision-directed blind phase search（square-QAM判定、64位相候補/π/2、中心窓65 symbols、位相unwrap）。独立pilotのみでπ/2 ambiguityと定数gain/phaseを校正する。train/validationで窓33/65/129の**3候補まで**比較し、合算validation BER、同率ならMSEの順で固定する。frameごとのpilot校正は全手法で共通。testのscored TX bits/symbolsをCR・equalizer・gain推定へ渡さない。

線形等化/CRの方式・pilot overheadは原論文の未記載を補う仮定である。無NLC基準はここまでを含める。CDC-only、train-label oracle、test-label最適回転を基準にしてΔQを大きくしない。CR未実装・不安定、power/ASE/CDCの不整合、未達の数値qualificationがあれば本生成・本学習へ進めない。

## データ・乱数・窓の契約

### 独立系列と保存schema

各条件・replicateのscored領域を次に固定する。pilot、guard、NLCのcontextはこの件数に含めない。

| split | frame数 | scored DP時刻/frame | 合計DP時刻 |
| --- | ---: | ---: | ---: |
| train | 8 | 2^16 | 2^19 |
| validation | 2 | 2^16 | 2^17（追加） |
| held-out test | 4 | 2^16 | 2^18 |

frame内の初期配置は `guard(g) / pilot(2048) / context(96) / scored(65536) / context(96) / pilot(2048) / guard(g)`。g=4096 symbols/sideをqualificationの起点とする。最大tap=96、線形FIR/CR窓、RRCの影響を含めてguardを検証する。periodic FFTの端は実データとして跨がず、前後の独立乱数列を十分確保する。scored境界のcontextは同一frameに属するが、別split/frameから借りない。RRC span/guardの変更でscored数を減らさない。

独立seedの規約は再現側の採用値として、master entropy=`2026102543`、`SeedSequence(entropy, spawn_key=(condition, replicate, split, frame, purpose))` を使う。condition=16/64、replicate=0/1/2、split=train:0/validation:1/test:2/pilot:3、purpose=bits:0/ASE:1/model:2/shuffle:3。データ生成・順序用engineはNumPy PCG64、span noiseはASE系列から独立spawnする。bitsは上記frame配置の7領域ごとの子系列（左からregion=0…6）に分け、guard変更でscored列が変わらないようにする。model/shuffleはsplit=train、frame=0のkeyから導出し、model seedを使用frameworkのnative RNGへ設定する。engineとRNG状態も保存し、Pythonのprocess依存hashでseedを作らない。

同一条件・replicate内では全候補・mask有無で同じデータを共有する。同一構造のmaskペアは初期重みseedとshuffleも揃える。replicate間はbits/ASE/model/shuffleを独立にする。pilot専用streamはproductionに転用しない。testのseed・recipeは生成時に固定してよいが、選択固定までtest指標を計算・参照しない。

配列とstageをschemaに明記する。

| 名前 | shape / dtype | 意味 |
| --- | --- | --- |
| tx_bits | `[T,2,log2(M)]`, uint8 | pilot/context/guardを含む独立TX bits |
| tx_symbols | `[T,2]`, complex128 | unit-EsのXY alphabet、TX indexと対応 |
| rx_field | `[T*sps,2]`, complex128 | fiber後、受信DSP前、sqrt(W)。bulk保存または再生成recipe |
| rx_2sps | `[2*T,2]`, complex128 | 診断用。CDC/matched/CR前後のどのstageかを明記 |
| rx_nlc | `[T,2]`, complex128 | 線形等化/CR/正規化後、NLC直前 |
| index / masks | frame ID、symbol index、bool `[T]` | pilot、DSP-valid、context-valid、scoringを別々に持つ |
| model features | `[B,b+2t,4]`, float32 | `XI,XQ,YI,YQ`。変換前のcomplex128も検証用に保持 |

bulkをGitへ無制限に追加しない。省略するraw配列は生成config・seed・実コマンド・依存版・schema・dtype・サイズ・SHA256・実在する保存先、または再生成recipeを残す。無い外部保存先を記載しない。manifestはsplit件数、stage、mask、normalizer、pilot fit、code/config/input hashesを含む。

### 正規化・教師・blockの対応

物理powerの既知倍率でEs=1へ戻し、残るgain/phaseはtrain pilotで学習した固定処理と、許可されたframe pilotだけで校正する。test scored配列から分散・平均・位相を推定しない。追加feature標準化は初期設定では行わない。実装する場合もtrainでfitし、rとsを同じ尺度・座標系へ移した後に残差を定義する。

`r=rx_nlc`、`s=tx_symbols`、教師はX偏波の `e_x=r_x-s_x`。MSEは中央b時刻×2実成分の平均で、trainはX出力を学習する（双偏波出力・XY augmentationは今回追加しない）。validationはX推論とXY swapによるY推論のMSEを等重み平均する。推論は同じ重みを2回呼び出してXYを補償し、両偏波を同じscoring maskで評価する。`e`は受信残差であり、純粋なKerr歪み真値とは呼ばない。

初期b=128、target block stride=b、scored開始位置から切り出す。contextの入力重複は許すがtargetはepoch内・評価内で重複しない。65536はbで割り切れる。各frameから512 blocks、trainは4096 blocks/epoch。windowのshuffleはblock順だけで、時刻順を内部で壊さない。frameやsplitを跨ぐwindowを禁止する。

入力blockの0-based target iは元入力index `t+i`。k=9のvalid Conv1D後は `ell=2*t-k+1`、`N=b+ell`、targetに対応する表現中心は `c_i=t-(k-1)/2+i`。出力headは表現 `[c_i-w : c_i+w+1]` を連結し、`W=2*w+1`。`t >= (k-1)/2+w` を満たさないconfigは拒否する。crop・impulse/tagged-index・先頭/末尾・XY swap・teacher符号・alpha=1および±3 dBの倍率を通常テストで確認する。

## モデル・mask・候補の固定

各encoder layerは `Y=LN(X+MHA(X))`、`Z=LN(Y+FFN(Y))`。Q/K/Vはd_model→d、head分割後d/h、連結後d→d_model。softmaxはhead_dimのsqrtでscaleし、FFNはd_model→d_ff→d_model、間にReLU。sinusoidal PEはConv後の位置0…N-1に加える。dropout=0、Xavier-uniform線形/Conv重み、bias=0、LayerNorm gain=1/bias=0を初期採用とし、frameworkの既定値に依存させない。

### maskの数学的定義と既知の差

今回のk=9ではellは偶数、q=ell/2。個別maskの整数座標m,n=-q…qについて、`allowed_indv[m+q,n+q] = (abs(m*n) <= rho*q)` と定義する。m=0行・n=0列を許可するこの積形式はEq. 8のm≠0領域と一致し、Fig. 4の中央の赤領域を補完根拠とする。Alg. 1の文字通りのm≠0条件では中心行が全maskとなるため、そのままの実装ではない。

block maskはbool `[N,N]` をfalseで初期化し、i=0…b-1について `allowed[i:i+ell+1, i:i+ell+1] |= allowed_indv`。許可0、禁止−∞をsoftmax前に加える。全mask行、NaN/Inf、端、b=1、対称性、dense参照と出力/gradient一致を確認する。将来kを変更する場合、ellの偶奇と座標定義を再指定し、黙って丸めない。

この定義による参照密度は t=32,b=1,rho=2.6で0.308095（Fig. 4aの0.31）、t=64,b=128でrho=2.6: 0.342709 / rho=0.4: 0.314061（Fig. 5の0.34/0.31）。t=64,b=1,rho=2.6は0.178198となり、Fig. 4bの0.19と一致しない。丸めだけでは説明できず`unresolved`として残し、図に合わせてmaskを改変しない。

### 今回の代表候補

全候補にk=9,b=128,h=4、single-polarization headを使い、maskなし / rho=2.6の両方を学習する。`d`はattention射影全体であり、headごとの次元ではない。

| condition / candidate | t | d_model | d | L | d_ff | W | 根拠 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| main16 / A | 96 | 96 | 64 | 3 | 64 | 15 | 公式TeXコメントregion1、Table II Aとの対応候補 |
| main16 / B | 64 | 64 | 32 | 2 | 32 | 7 | 同region2、Table II Bとの対応候補 |
| main16 / C | 16 | 16 | 16 | 2 | 32 | 7 | 同region3、Table II Cとの対応候補 |
| main64 / high | 64 | 96 | 32 | 3 | 64 | 11 | 公式TeXコメントregion1 |
| main64 / mid | 32 | 64 | 32 | 1 | 64 | 11 | 同region2 |
| main64 / low-adopted | 16 | 16 | 16 | 1 | 64 | 7 | コメントregion3のt=8を表示Table Iの下限16へ変更した追加候補 |

Table Iの表示探索範囲は16QAM: t/hidden∈[8:96], key∈[8:64], h∈{1,2,4}, L∈{1,2,3}, FFN∈{32,64}, W∈{1,7,15}; 64QAM: t∈[16:64], hidden∈[16:96], key∈[16:64], h∈{1,4}, L∈{1,2,3}, FFN=64, W∈{1,7,11}。区間内の刻みは表示値だけでは確定できない。今回は上表だけを実行し、全格子探索やそのPareto最適性は主張しない。

Table IV Model1=(t32,d_model48,key16,h4,L3,FFN32,W7)、Model2=(t12,d_model16,key16,h4,L1,FFN32,W7)、rho=2.6は別のrate/PMD比較用モデルである。この2モデルだけを主条件で学習して主実験再現完了とはしない。両偏波出力の変更も今回は採用しない。

## 学習、選択、有限な初期予算

未記載のoptimizer数値は**再現側の仮定**として固定する。Adam betas=(0.9,0.999)、eps=1e-8、weight_decay=0、最大LR=1e-3、最初の10 epochsを線形warm-up、その後inverse-square-root decay。optimizer step s、warm-up steps s_w=10×steps_per_epochとして `lr=1e-3*min(s/s_w,sqrt(s_w/s))`（s≥1）。batchは**512 blocks**、端数batchを捨てず、batch平均MSEで更新する。標準では8 optimizer steps/epoch。microbatch64×gradient accumulation8を許すが、full-batchと一致するloss重みで蓄積し、実際の設定・step数を保存する。

各run最大500 epochs、validation MSEを毎epoch測定し、strictな改善（min_delta=0）が100 epochsない場合停止する。best-validation checkpointを採用し、最良epoch・停止理由（patience / epoch cap / failure / interruption）を区別する。500到達は収束の証拠ではない。NaN、Inf、OOM、異常gradientの発生と失敗attemptを残し、testを見て救済設定を追加しない。回復可能なOOMでは有効batch512を保ったmicrobatch/accumulationの変更と同一runの再開を親の判断で行い、変更理由・設定・再開位置を記録する。これは候補の追加ではない。自動LR探索・追加アーキテクチャ・seedの良いものだけの再実行は初期予算に含めない。

checkpointには重み、optimizer/scheduler、epoch/optimizer step/microbatch位置、全RNG状態、sampler順・cursor、best score、config/data/code hashを保存する。中断再開はepoch境界を標準とし、途中epochの再開対応がなければ最後の完了epochへ戻ったことを明示する。CPU deterministic smokeでreload・連続学習対resumeの等価性、GPUではdtype/backendを固定した許容差を記録する。

| 段階 | 初期予算・内容 | 本番への扱い |
| --- | --- | --- |
| CPU smoke | 各条件64 target blocks以下、2 optimizer steps、最小構造、mask有無 | 機能検証のみ。成果指標に使わない |
| GPU learning pilot | 条件ごと最小構造×mask有無、専用train4096/validation2048 DP時刻、最大5 epochs（4 run） | 時間/VRAM/保存/有限lossを測定。productionの代わりに数えない |
| GPU resource benchmark | 最大構造main16/A・main64/highの各mask有無、計4ケース。実際の有効batch512とmicrobatch/accumulationでforward/backwardを行い、optimizer更新まで最低1回完了 | optimizer状態を含むpeak VRAM・所要時間を保存。学習trial/選択結果とは区別 |
| DSP/numerical pilot | 下記qualificationの初期6設定×独立3frame×2条件=36 full-link伝搬を見積もる。CR3窓候補まで。小analytic controlsは別 | 数値精度の実測に応じて必要な要因を追加refineし、理由・新見積もりを残す |
| production | 2条件×3構造×2 mask×3独立seed = **36 run**、各最大500 epochs | 初期探索の全量。全runを台帳へ残す |
| test | 固定した各候補・各seedのbest-validation checkpointを1回評価 | 同じseedのmaskペア・無NLCを共通frameで比較 |

primary選択は条件×maskごとに、3 replicateのvalidation MSE平均が最小の構造とする。同率（記録精度内）なら低RMPS、さらに同率なら表の下側を選ぶ。replicateは選別せず3つとも報告する。低・中・高の全候補も、test前にcheckpoint一覧を固定してからすべて評価し、固定候補集合の性能–計算量図を作る。testを用いてprimaryを変更したり、良い点だけで曲線を作ったりしない。

ローカル資源の初期前提はRTX 5060 Ti 16 GB、空きdisk約199 GB。最小構造のlearning pilotだけではA/highの本番memoryを保証できないため、上記4ケースのresource benchmarkを必須とする。専用pilotデータまたはshapeを保った合成入力/教師を使い、testラベルは使わない。実際のdtype/backend、有効batch、microbatch/accumulation、peak VRAM、時間、OOMと再試行設定を記録し、重みを選択結果やproduction checkpointへ転用しない。

pilot/benchmarkでsimulation時間/frame、学習時間/epoch、peak VRAM、checkpoint/trace容量を実測し、固定36学習runの必要時間・容量を見積もる。学習と物理生成のGPUジョブを重複起動しない。親は既存ローカル資源と主2条件の範囲内で、通常の数値refinement、microbatch調整、保存・実行順序の改善を継続し、理由と新見積もりを記録する。初期36伝搬の超過や回復可能OOMだけでは追加のユーザー承認を求めない。相談するのは範囲の拡張、外部有料資源、未解決の重要な科学的選択、または範囲内の解決経路がない資源制約に限る。500epoch上限で未収束ならその結果を明示する。科学的条件・学習量・seedを黙って削らず、実測前に終了時刻を保証しない。

## 段階ゲートと実行順序

| gate | 終了条件・保存する証拠 | 失敗時 |
| --- | --- | --- |
| 0: 仕様/基盤 | #42回帰、#43台帳・候補・seed・操作規約、shared/experiment所有範囲が確定 | 本実装の依存を解消 |
| 1: 小規模pilot | CPU smoke、4 learning pilots、A/high×mask有無の実batch資源benchmark、CR/no-NLC処理、時間/VRAM/storageの実測 | 原因をIssue化し、回復可能な資源設定は調整。production未開始 |
| 2: numerical qualification | 両主条件のB2B/AWGN・Gray/BER、power、分散/逆分散、Kerr/loss、ASE/NF/偏波/seed、CPU/GPU確認。下記独立refinementが通る | **本生成・本学習を停止**。未資格を成功扱いしない |
| 3: production固定 | pilot/benchmark結果、全config/seed、data schema/hash、CR/normalizer、36run台帳・予算・環境が保存 | 範囲内の調整は親が判断し理由を記録。上記相談条件に当たる場合のみ相談 |
| 4: 学習/validation選択 | 全候補の終了状態・曲線・checkpointが揃い、validationだけでprimary/全test対象を固定 | 失敗/未収束を残す。未実施runを隠さない |
| 5: selection freeze | commit/dirty内容hash、環境lock、dataset hash、checkpoint hash、対象一覧、判定法、参考値の取得法を固定 | testを開封しない |
| 6: held-out test | 2^18 scored DP時刻/条件/seed、全手法同一maskで直接BER/Q、独立再計数、checkpoint reload評価 | bugは記録し、汚染testを再選択に使わない。新testは別versionで扱う |
| 7: 報告 | 保存metricsから全図表再生成、再実行手順、主2条件の負結果を含む報告、未実施範囲の一覧 | #47/親#40の状態を実際の達成範囲に揃える |

qualificationは同じ物理系で、一度に一要因を変える。専用frameのscored長2^16、初期g=4096、4sps/0.05°/RRC128から始め、(a) phase bound 0.025°、(b) 8spsで固定ASE band、(c) scored frame長2^17、(d) g=8192、(e) RRC span256をそれぞれ比較する。初期設定と5変更の計6設定を各条件の独立3frameで測り、同じ物理時間・bandでは共通bits/ASE実現で差も確認する。frame/guard長が変わる比較では一致するscored indexを明示する。収束不十分なら親の判断で対応要因をさらにrefineし、追加の設定・実行数・理由・資源見積もりを記録する。初期36伝搬は見積もりであり、範囲内の数値qualificationを打ち切る上限ではない。gamma=0/ASEなしのanalytic limitsと全linkのbaselineを分ける。参考の数値目安はeffective-SNR差≤0.1 dB、BERが計数不確かさ/独立frameばらつきと整合すること。区間の重なりだけで物理的同一性を断定せず、差・件数・フレーム変動を保存する。この0.1 dBは**数値収束**の目安であり、論文性能再現の合格幅ではない。

## 評価・RMPS・再現判定

主比較は無NLC通常DSP、Transformerなしmask、rho=2.6。testは16QAM +2 dBmと64QAM +3 dBmのみを初期固定し、launch-power曲線・最適power・zero-shot rate/PMD汎化は未評価とする。Eq. 2の実装は合成テストと専用pilotの倍率検証を行う。他電力の性能sweepを追加する場合はtest開封前にpowerリスト・データ予算を別途固定する。

BERは復調したbitを直接比較し、X/Y別と合算のerrors/n_bits、mask/除外数、95% Clopper–Pearson計数区間を保存する。1 seedあたりの合算test分母は16QAM `2^18*2*4=2097152 bits`、64QAM `2^18*2*6=3145728 bits`。独立frameごとの値と3replicateの全点・平均・標準偏差を示し、非線形誤りの相関を独立bit区間だけで説明しない。ゼロ誤りは非零のBER上限と対応するQ下限、測定限界を示す。BER≥0.5など変換が有効でない領域はQを未定義としてBERを報告する。MSE/EVM/SNRは補助指標。

RMPSは出力**片偏波symbolあたり**の値と、XYを2回推論した**DP時刻あたり**の値（原則2倍）を併記し、論文比較の分母を明記する。N=b+ell、許可attention要素E（denseならN²）について、次の内訳を保存する。

| 成分 | 片偏波symbolあたりの初期計数規約 |
| --- | --- |
| CNN | `N*4*k*d_model/b` |
| Q/K/V + output projection / layer | `4*N*d_model*d/b` |
| attention / layer | Eq. 6準拠の推定 `(2*d+3*h)*E/b`。softmax/scaleの換算3*hを含む |
| FFN / layer | `2*N*d_model*d_ff/b` |
| LayerNorm / layer | 2回のLNにつき `N*(6*d_model+4)/b`（平均/分散の定数乗算、二乗、正規化、affine） |
| head | `2*W*d_model+2*10+10*2` |

add/bias、ReLU/LeakyReLU、事前計算PE、rsqrt、メモリ移動はこの乗算計数から除外する。LayerNorm等の算入とsoftmax換算は原論文全体の実装規約と未同定なので、**この規約での推定RMPS**と表記し、Eq. 6部分・線形部分・追加項を別保存する。Table IIとの照合に都合よく係数を後付けしない。maskありのE置換は疎実装を仮定した理論値。dense attention後にmaskを掛けるbackendの実コストはdenseのままであり、実測latency/throughput/peak VRAM/学習時間とは分ける。実測にはhardware、framework/library、dtype、backend、batch、warm-up、device同期、計測範囲、pol swapを含むかを添える。

論文参考値はTable IIの16QAM baseline6.67 dB、A/B/Cのunmasked Q={8.82,8.54,7.31} / masked Q={8.76,8.51,7.68} dB、RMPS={608,129,22}K / {404,81,9}K。これは測定前の参考値であり、モデル構造の根拠はコメントTeXである。64QAMの表示本文はbaseline約6.5→最良約8.1 dBで、上表の縮小候補の到達値を保証しない。図から数値を取得する場合は原図版・軸・座標・読取手法・誤差を別列に残す。

今回の研究分類は探索縮小と未記載物理/DSP条件のため、Qが一致しても**主2条件の部分再現**とする。全体の厳密再現成功とは別に、(1)実装・データ/学習/評価/保存の完了、(2)数値資格、(3)通信性能の結果、(4)出典条件の一致度を判定する。test前に固定する性能比較は絶対Q差、同一baselineからのΔQ差、3seed/4frameのばらつき、数値誤差・読取誤差であり、数値許容幅の根拠がない場合は「論文値との一致の合否は未確定」と報告する。改善必須や恣意的な±0.1 dBを合格条件にしない。負結果でも学習・評価・報告の要件は完了できる。

## 配置、入口、成果物、子Issueの受入

これは公開APIの責任分割案であり、新しいregistry/frameworkを前提にしない。

| 所有場所 | 責任・推奨インターフェース | Issue |
| --- | --- | --- |
| `src/optical_comm/{config,channel,dsp,metrics}.py` | 64QAM/物理/対称CDC/adaptive SSFM/通常DSP/BER。field `(time,2)`、symbol `(time,2)` とstage metadataを返す既存APIを互換維持で拡張 | [#44](https://github.com/Motoki0705/optical-comm-lab/issues/44) |
| `experiments/transformer_nlc/` のdata処理 | `generate(config, split, seed) -> manifest`、frame/block dataset、train-only校正、schema/hash | #44 |
| 同workspaceのmodel/mask/complexity処理 | `forward(features, mask) -> [B,b,2]`、mask生成、shape検証、成分別RMPS | [#45](https://github.com/Motoki0705/optical-comm-lab/issues/45) |
| 同workspaceのtraining/evaluation処理 | `train(..., resume)`、checkpoint、validation選択、freeze対象一覧、metrics計数、資源計測 | [#46](https://github.com/Motoki0705/optical-comm-lab/issues/46) |
| 同workspaceのnamed `configs/`、report処理 | 上表の2条件・6候補、pilot/productionの差、保存metricsから図表/報告 | [#47](https://github.com/Motoki0705/optical-comm-lab/issues/47) |

実装順は `#42/#43 → (#44 と #45を所有ファイル分離で並行) → #46 → pilot/qualification → #47`。各機能のIssue/PRを分離し、sharedのinterface変更は親が調整する。ESN用互換adapterへTransformerを追加しない。READMEや必要な安定責務の文書を更新し、checkpointごとのscout保守点検はroot規約に従う。

予定入口は `python -m experiments.transformer_nlc {generate,train,evaluate,sweep,report}`。**この仕様段階では未実装**。`--config --seed --run-id --dry-run` を共通にし、generate/trainは `--resume`、evaluateはfreeze manifest/checkpointを要求する。sweepは有限の候補台帳を実行する役割で、暗黙の探索をしない。dry-runは実効設定・入出力・件数・必要段階を示しGPU学習を開始しない。同じrun-idへの重複job/上書きを拒否し、resumeはconfig/data/sourceの一致を検査する。

保存先は `artifacts/transformer-nlc/<run-id>/`、読者向けは `reports/transformer-nlc/`。実効config、source/dirty/依存lock/環境、split/seeds/index/masks/data hash、qualification、pilot実測、全trial ledger（失敗含む）、best/latest checkpoint、history、selection freeze、frame/seed/条件別metrics、集約CSV/JSON、図生成commandを残す。再現可能なコンパクト表・診断・適切なサイズのcheckpoint・図・manifestはまとめて保存する。bulk/cacheはignoredとし、生成元を新しいSHAへ付け替えない。

最終レポートは非専門家向けの日本語で、主2条件/距離、無NLCとのBER/Q比較、学習曲線、全候補の性能–RMPS、mask形状と密度、seed/frame変動、資源量、出典対採用条件の表を中心にする。保存checkpointの再評価と、学習し直さずmetricsからの図表再生成を通常検証に含める。元論文図と自分の実測図を取り違えない。

### 仕様完了時点で残す限界

本番開始を阻む未検証事項は、通常DSP/CRの妥当性、power/ASE/対称CDCの単位・符号、SSFM/sps/frame/guard/RRC収束、CPU/GPU・reload/resume、pilotの時間/VRAM/容量である。#44–46で証拠を揃える。論文の未記載物性・CR詳細、key size解釈、コメント構造の対応、mask Fig. 4b密度、Fig. 8色、Q定義、完全なRMPS算入規則は原実装との一致の限界であり、採用規約を開示した部分再現は進められる。

[#43](https://github.com/Motoki0705/optical-comm-lab/issues/43) の完了は、この台帳・データ契約・候補/予算・依存/ゲートが揃い、リンクと内部整合性を確認すること。#44は物理/DSP/生成の数値資格とmanifest、#45はモデル/mask/RMPSの通常検証、#46は選択/再開/固定評価の検証、#47は主2条件の学習・独立評価・保存物・報告までを各受入条件で確認する。最終報告に独立validatorの指定/試行/完了=0/0/0、通常検証結果、全失敗と残課題を含める。拡張実験と親#40の完了は引き続き保留する。
