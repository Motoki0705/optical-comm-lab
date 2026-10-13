# 研究仕様と方法

このディレクトリは、研究要件、実験計画、モデル・数値手法、出典、研究上の判断を説明する。ユーザーの要求は [requirements.md](requirements.md)、採用した条件・データ分離・受入条件は [experiment-plan.md](experiment-plan.md) に集約する。関連する調査の結論は要件・計画・方法の文書に統合し、作業担当ごとの中間メモを並べない。

新しい機構は、その仕様と根拠を既存の関連文書へ追記する。新しい文書やディレクトリは、継続して担う責任と読む人が明確な場合に設ける。コードは `src/optical_comm/`、実験条件は `configs/`、読者向けの結果と図は `reports/`、再現用のデータとモデルは `artifacts/` が担当する。共通の作業規則はリポジトリ直下の `AGENTS.md` に置き、下位ディレクトリに複製しない。

[repository-maintenance.md](repository-maintenance.md) は、開発の節目で scout が配置・利用経路・一時コード・再現情報を点検し、根拠を重複確認して Issue に残す手順と依頼文を担当する。各回の結果は管理 PR または保守 Issue に記録する。

[kerr-physics.md](kerr-physics.md) は、Blender映像用の孤立パルスの物理条件、
SSFM計算、数値検証、電場・位相・チャープ・スペクトルのデータ形式を説明する。
1600 km通信研究とは独立した説明用計算として、その範囲と来歴を保持する。
