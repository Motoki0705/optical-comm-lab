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

## Blender Kerr animation follow-up (verbatim)

> blenderを用いて、kerr効果についてのアニメーションを作成してください。また、PRはマージしてください。今後の運用としてPRは機能ごとに出す形にしてください。また、issueもタスク、機能、問題単位で分けてください。解決済みはcloseすること。AGENTS.mdに記述して

Quality and physical-model request:

> 品質が低すぎます。ブレンダーで物理的なシミュレーションを作成しそれを高品質な描写で撮影してください。それらをもとに説明を組んでください。アニメーションは複数用意してください。

User's visual preference:

> 実物に近いファイバの接写＋内部の光の可視化（推奨）

Latest spatial-propagation request:

> 光の光線が反射しつつ、どのような変化が起きるかを表現してください。今の表現は物理シミュレーションをする必要がないレベルです。claude code opus にアドバイスをもらってください。

The latest request requires a spatial field model with a fiber boundary; the
existing 80 km temporal-envelope films do not establish reflecting spatial rays.
The new method and its scope are recorded in [spatial-kerr.md](spatial-kerr.md).
The completed 1600 km research and 80 km pulse calculation remain separate
studies. Claude Code Opus was consulted for design advice under
[Issue #23](https://github.com/Motoki0705/optical-comm-lab/issues/23); implementing
and checking the circular spatial model is tracked in
[Issue #24](https://github.com/Motoki0705/optical-comm-lab/issues/24).

The current working proposal uses a circular multimode core, continuous light,
and a clearly labeled hypothetical strong Kerr coefficient, together with a
linear control and a weak reference. The user has not yet selected between
educational visibility and realism at communications powers; this proposal is
not recorded as their accepted choice. Any later clarification must be applied
if it changes that scope.

Completion requires computed transverse fields and flow rather than manually
placed reflecting paths, ordinary numerical checks, multiple high-quality
Blender animations with an explanation based on their saved calculations, and
reproducible artifacts whose normalization and display scales are disclosed.
Flow lines represent the local field flow, not individual photon trajectories.
Exploratory slab runs do not qualify the circular model. Merge authorization is
present in the user's request; each coherent feature or problem has its own
Issue and PR, resolved Issues are closed, and partial work remains open as
specified in the root `AGENTS.md`. No independent-validator count was specified
for this follow-up, so its count is zero; the original research's single
evaluation is not reused.
