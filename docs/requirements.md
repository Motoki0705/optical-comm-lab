# User requirements

## Original research request

> このrepoでは、光通信における研究リポジトリを開発していきます。基本的な内容としては、リバーサルコンピューティングを用いて、光ファイバの長距離伝送におけるノイズ除去をシミュレーションで再現し、それをモデルに学習させることです。まずは、シミュレーションの仕様、学習戦略、モデルアーキを決めて、それを実装する方針にいます。シミュレーションの仕様を提案してください。

Clarifications supplied by the user:

> リザバーコンピューティングのことです

> まずはソフトウェア上で補償性能を検証する（推奨：伝送モデルを先に検証しやすい）

## Implementation request (verbatim)

> それでは、実装をしてください。モデルの学習まで行い。最終レポートまで作成してください。レポートは不要な詳細を除き、必要な可視化を中心に説明する形にしてください。表、図、実験結果の可視化等を積極的に作成してください。これらはサブエージェントを用いて項目ごとに可視化を作成する実装を行い、成果物の保存までさせてください。レポートはその専門を知らない人でも理解できる粒度を意識してください。開発の進行はgithubを用いてください。タスク、問題はissueに起票し、実装はPRでコントロールしてください。また、サブエージェントベースの進行を行い。調査はscout, 実装はgpt-6.1-sol-maxにしてください。独立したvalidatorは１回実施すること。サブエージェントも独自にissueを作成できます。scoutは様々な用途で使用できます。様々なディレクトリにそれを拡張する方針を記述してください。無造作な欄列を防ぐ目的であり、適切に設置して下さい。AGENTS.mdはrepo直下においてください。これはサブエージェントにも伝播します。

## Accepted simulation proposal

The user authorized implementing the preceding proposal. Its numerical settings, data contract, controls, comparisons, and completion criteria are preserved in `experiment-plan.md`. It is a simulation study of ideal terminals and a single wavelength; performance on physical hardware or WDM is not claimed. Training success is not defined as mandatory improvement: negative or inconclusive held-out results must be reported honestly.

## Repository maintenance follow-up (verbatim)

> scoutを用いて、構造化されていないファイル、使われていなモジュールやスクリプト、一時限りのコード等がないかを定期的にスキャンして、issue化するというワークフローをお願いします。

Confirmed cadence:

> 開発の節目ごとに実施（推奨：変更に追従でき、不要な定期実行を避けられる）

The workflow runs at development checkpoints (start of work, structural changes, and before PR review), rather than on a calendar. A read-only scout gathers evidence; an authorized parent or implementation worker creates or updates GitHub Issues after deduplication. This maintenance investigation is distinct from the single independent validator evaluation. The simulation, training, visualization, and final report requirements remain active.
