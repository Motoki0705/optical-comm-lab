# 独立validatorの原評価（第1回／指定1回）

以下は返却された評価の原文です。対象は修正前の固定版です。scratch内のパスは当該評価時の一時検証を指し、保守対象の再実行CLIとは区別します。指摘の採否・修正後の通常検証は `independent_review.json` に記録します。

---

1. **判定：CHANGES_REQUIRED**

対象は `f8e60469740e672036c2f6b71a2a8fea9355388d`。開始・終了とも237ファイルすべてのSHA-256が固定記録と一致し、working treeはcleanでした。固定記録のSHA-256は `fc7b94d70d05ed810db9f676a506b9fb784f3b02b5f465602ecd1fade70d6e98` です。

科学計算・保存結果・レポートについて、下記の検証で結果の不整合は確認されませんでした。ただし、合意した診断データの保存に要件漏れが1件あります。

指定1回／試行1回／今回の完了1回。自身と親のローカル`session_meta`・`turn_context`を読み、両者とも **OpenAI／gpt-6-astra／max** と確認しました。role設定にモデル等の上書きはありません。実効権限は強制read-onlyではありませんが、成果物・環境・Git/GitHub状態は変更せず、出力は指定scratch内に限定しました。

2. **自分で実施した検査と証拠**

原要件と実験計画から検査を組み立て、コード・保存物・既存テストを調査してから、親の補助資料と照合しました。

| 要件・観点 | 独自検証と結果 |
| --- | --- |
| 物理処理・解析極限・データ分離 | 指定のキャッシュ隔離環境で `pytest -p no:cacheprovider --basetemp=… -q` を実行。**52 passed、9.34秒**。CPU/GPU一致、分散・Kerr/loss、ASE、ラベル改変、状態リセット、保存復元等を確認。物理式・ASE規約も[Manakov公式資料](https://opticommpy.readthedocs.io/en/latest/generated/optic.models.channels.manakovSSF.html)、[EDFA公式資料](https://opticommpy.readthedocs.io/en/latest/generated/optic.models.devices.edfa.html)と照合。 |
| 実際の学習とBER | 独自スクリプト [reproduce.py](/home/kamimura/projects/optical-comm-lab/.work/validator-scratch/reproduce.py)で全7電力の第1テストフレームと0/+6 dBmの第5フレームを再生成。パッケージの判定関数を使わず、16点との距離から再計数した**36行の誤り数・分母・EVMが保存値と一致**。7電力の診断点も一致。 |
| 学習済みモデルの実在・再現性 | 0 dBmの学習2フレームを再生成し、選択済み線形モデルとESNを再学習。各126,704学習標本、校正・正規化・readoutが保存値と**完全一致**。独立した密行列による再帰計算も、無効区間後のリセットを含め一致。 |
| 数値条件の確認 | [numerics.py](/home/kamimura/projects/optical-comm-lab/.work/validator-scratch/numerics.py)で刻み・サンプリング56比較、長さ・境界60比較の判定を再計算。最大SNR差はそれぞれ **0.000308503 dB／0.00228724 dB**。+6 dBm・ASEありの4 sps/1 kmと8 sps/0.25 kmを再生成し、生波形ハッシュも一致。 |
| AWGN対照 | `qualify_numerics.py --stage controls --output .work/validator-scratch/numerical-controls` を実行。9フレーム・3集計の計数結果が保存値と完全一致。0/8/16 dBの誤り数は449,264／152,886／2,747、各1,560,576 bit。 |
| 保存物・来歴 | `audit_run.py --run artifacts/final-1600km-20261012 --output .work/validator-scratch/audit-final.json` が成功。28集計行、140フレーム別行、126個の異なるbit/ASE seed、候補選択・checkpoint・ハッシュを確認。 |
| 図表・日本語レポート | 全7図を目視確認。図生成CLIをscratchへ実行し、**16個のPNG/SVG/PDFすべてが元ファイルとバイト一致**。レポートも再生成し、出力先依存のリンク・コマンド以外の本文・数値が一致。HTMLの7埋込画像、代替テキスト、ローカルリンクを確認。156件の生成物／入力ハッシュに不一致なし。 |
| GitHub・担当分担・保守 | `gh pr view 7`、Issue一覧・#8/#9コメント、Git履歴、実際の`spawn_agent`引数を確認。[PR #7](https://github.com/Motoki0705/optical-comm-lab/pull/7)は対象HEADのdraft、CI成功。実装・可視化担当は指定モデル/max、調査はscout、全起動は`fork_turns="none"`。保守点検3記録と#9の修正を確認。 |

レポートの「ESNによる改善は確認できなかった」という結論、EVMとBERの区別、0誤りの測定限界は保存結果に対応しています。2,534,080 bit・0誤りの95%上限も独自計算で `1.4557065e-6` と一致しました。

3. **要修正事項**

**V1／P2：合意した2-spsの連続診断波形が保存されていない**

- **要件**：[experiment-plan.md:15](/home/kamimura/projects/optical-comm-lab/docs/experiment-plan.md:15)は、2-sps post-DSP診断波形とmask/frame境界の保存を要求しています。
- **実装の根拠**：[experiment.py:348](/home/kamimura/projects/optical-comm-lab/src/optical_comm/experiment.py:348)は等間隔に選んだ1-spsの星座点だけを保存します。[qualify_numerics.py:287](/home/kamimura/projects/optical-comm-lab/scripts/qualify_numerics.py:287)の診断配列も1-spsです。`artifacts/**/*.npz`全47ファイルのキーを調べ、2-sps波形を含む保存物がないことを確認しました。`rx_2sps`自体はAPI内で生成されています。
- **影響**：保存された診断点から、シンボル間の波形や連続区間の対応を調べられません。BER値の誤りではありませんが、合意した保存成果物が不足しています。
- **改善案**：同一seedから再生成した連続区間について、XYの`rx_2sps`、対応するTX bits/symbols、開始位置・frame識別・有効／採点maskをコンパクトに保存し、再生成コマンドとハッシュを付けてください。既存測定値・checkpointの来歴は維持してください。

その他の要修正事項は確認しませんでした。

4. **未検証事項・限界**

- 全35テストフレームと全196候補の再学習・再評価はしていません。独自再計数は9テストフレーム、再学習は0 dBmの選択済み2モデルです。
- 数値境界実験の全伝搬は再実行せず、保存比較の再計算・コード・テストを確認しました。
- HTMLのブラウザ別表示・印刷は未検証です。図の視認性、埋込画像、リンク、再生成は確認済みです。
- 初回保守点検のdirtyファイルハッシュ未取得はIssue #8で明示されています。その後の固定版点検を確認しており、追加の要修正事項とはしていません。
- 固定版の独立評価記録が未実施表示なのは、今回の返却後に更新する手順と整合します。返却後の修正・記録更新は、この評価で検証済みとは扱えません。

上記の未検証範囲を保証するものではありません。今回の判定理由は、確認済みの保存要件漏れV1です。
