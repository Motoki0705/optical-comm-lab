# 研究仕様と方法

このディレクトリは、研究要件、実験計画、モデル・数値手法、出典、研究上の判断を説明する。ユーザーの要求は [requirements.md](requirements.md)、採用した条件・データ分離・受入条件は [experiment-plan.md](experiment-plan.md) に集約する。関連する調査の結論は要件・計画・方法の文書に統合し、作業担当ごとの中間メモを並べない。

新しい機構は、その仕様と根拠を既存の関連文書へ追記する。新しい文書やディレクトリは、継続して担う責任と読む人が明確な場合に設ける。共通の物理・DSP・指標は `src/optical_comm/`、実験固有の実装・設定・方法は [`experiments/`](../experiments/README.md)、独立した数値確認とKerrの条件は `configs/`、読者向けの結果と図は `reports/`、再現用のデータとモデルは `artifacts/` が担当する。共通の作業規則はリポジトリ直下の `AGENTS.md` に置き、下位ディレクトリに複製しない。

[repository-maintenance.md](repository-maintenance.md) は、開発の節目で scout が配置・利用経路・一時コード・再現情報を点検し、根拠を重複確認して Issue に残す手順と依頼文を担当する。各回の結果は管理 PR または保守 Issue に記録する。

[kerr-physics.md](kerr-physics.md) は、Blender映像用の孤立パルスの物理条件、
SSFM計算、数値検証、電場・位相・チャープ・スペクトルのデータ形式を説明する。
1600 km通信研究とは独立した説明用計算として、その範囲と来歴を保持する。

[kerr-rendering.md](kerr-rendering.md) は、計算済みの電場を使った接写の Blender
動画、表示倍率・位相色・模式的な搬送波、ネイティブシーンと MP4 の再生成、
素材ライセンスと描画の検証を説明する。

[spatial-kerr.md](spatial-kerr.md) は、円形コア内の空間場と横方向の流れから
反射と Kerr 効果を描く追加モデルの条件・方法・保存結果を担当する。
時間包絡線の映像との違い、選択された教育用条件、正規化、再生成・監査コマンドを説明する。
全3 mmの空間収束を保証しない限定的な数値確認と、流線の補間による位置の差を明記し、
数値場の検証と描画・説明ページの確認を区別する。

[spatial-kerr-rendering.md](spatial-kerr-rendering.md) は、円形コアの保存済み場・等強度面・
流線を使う3本のBlender動画の座標、表示、カメラ、補修来歴、再生成を説明する。
読者向けの入口は [映像中心の日本語解説](../reports/animations/spatial_kerr/index.html)、
図とHTMLの再生成は [解説のREADME](../reports/animations/spatial_kerr/README.md)と
[builder CLI](../scripts/build_spatial_kerr_report.py)を参照する。

既存ESN研究の計画・学習方法は
[`experiments/esn_longhaul/docs/`](../experiments/esn_longhaul/README.md) が正本です。
このディレクトリの旧URLは転送ページとして保持します。元のユーザー要求は
`requirements.md` にそのまま保存し、物理・DSP・指標と数値確認の文書は共通の参照としてここに残します。
新しい研究の追加手順と来歴互換性は [実験構成](../experiments/README.md) を参照してください。
