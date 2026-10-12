# 開発の節目に行うリポジトリ点検

この手順は、ファイルの置き場所、利用されていない可能性のあるコード、放置された一時コードなどを `scout` が調査し、根拠のある問題を GitHub Issue に残すためのもの。依頼と確認済みの頻度は [requirements.md](requirements.md) に保存する。共通の必須規則は直下の [AGENTS.md](../AGENTS.md) にある。カレンダー実行や GitHub Actions からの Codex 起動は設定しない。

調査担当は読み取り専用の `scout`。親エージェントまたは権限を与えられた実装担当が結果を判断して Issue を作成・更新する。削除、移動、統合などの修正は別の作業として Issue と管理 PR に紐づける。点検だけで自動削除しない。研究成果を保存することと、不要なソースを残すことを混同しない。

## 実施する節目

| 節目 | 親が行うこと |
| --- | --- |
| 開発開始時 | 現在の構成と進行中の担当範囲を渡して点検する。同じ対象と版を既に点検した記録があれば、それを再利用する。 |
| 構造を大きく変更した後 | 新しいディレクトリ・モジュール・スクリプト、移動・削除、公開 API・CLI・CI の入口、設定や artifact の参照先・保存構成を変えた作業が安定した時点で、影響範囲を点検する。 |
| PR をレビュー可能にする前／独立 validator の対象を固定する前 | 完成したコード・設定・図表・再現情報を点検し、記録と必要な Issue を揃える。同じ対象版なら、この二つの節目を一度の点検で満たせる。 |

「最近実施した」だけで再利用を判断しない。前回の対象範囲、commit、dirty ファイルの内容ハッシュ、除外・保留事項が現在も適用できることを確認し、参照する記録を示す。対象内で内容が変わった場合は差分を確認し、構造や利用経路に影響があれば再点検する。点検後の構造変更やレビュー修正で入口・参照先・配置が変わった場合も、変更した範囲を再点検する。

実装途中のファイルは担当者と完了予定を確認し、未完成であることだけを問題にしない。確かな不具合は追跡するが、不確かな未使用候補だけで開発を止めない。点検は最終成果物の独立評価ではなく、指定された validator 回数を消費・追加・代替しない。

## 対象版と調査範囲を渡す

親は開始前に、作業ディレクトリ、対象 commit、対象パス、管理 Issue/PR、各担当者の所有範囲と進行中の変更を揃える。`git status --short` と対象パスの一覧に加え、追跡済み差分および nonignored untracked ファイルの内容ハッシュを保存する。移動前後のパス、削除も一覧に残す。これは軽量な調査時点の記録であり、validator 用の全成果物固定を代替しない。

対象は原則として `git ls-files` の追跡ファイルと `git ls-files --others --exclude-standard` の未追跡ファイル。無視されたキャッシュや一時出力を網羅するために ignore 設定を外さない。`.git/`、`.venv/`、依存パッケージ、Python/各種ツールのキャッシュ、`.work/`、ignore 対象の raw/bulk/temporary traces を除外する。大量データやモデル配列の全読み込みは不要で、対応する README・設定・マニフェスト・参照経路を調べる。対象を絞った場合は、調べていない範囲も記録する。

並行作業中の対象が調査中に変わったら、親に知らせて影響範囲を保留するか、安定した版で読み直す。変更前後の観察を単一の確定した版として記録しない。

## 確認する事項と根拠

| category | 確認する事項 | 判定に必要な根拠の例 |
| --- | --- | --- |
| `placement` | 責任のない場所に散在したファイル、読者や責任が不明なディレクトリ | 直下 AGENTS と各 README の責任、実際の内容と利用先、移す先の責任 |
| `unused-source` | 未使用候補のモジュール・関数・スクリプト | import/呼び出し、公開 export/API、パッケージ入口、CLI、テスト、文書、CI、設定・動的ロードの経路を調べた範囲 |
| `one-off-source` | 放置された一時的な実験コード、hard-coded な作業専用スクリプト | 作成目的、継続利用・再現用の役割、既存 CLI との関係、担当者の作業状態 |
| `duplicate-mechanism` | 同じ数値処理・機構の重複 | 各実装の目的と利用者、独立対照・バックエンド差・科学的検証として分ける理由、差異と保守上の具体的影響 |
| `orphan-config` | 利用経路や目的を失った設定 | CLI/CI/文書の指定、設定ファイル参照、マニフェストの入力、実験名・目的、過去版との関係 |
| `artifact-provenance` | 生成物の再生成方法や由来がない、入力・出力参照が途切れたもの | 再生成コマンド、入力設定・seed・版・hash、manifest、README/レポートからの参照 |

`rg` で検索結果がないことは候補を見つける手掛かりに留める。パッケージの公開 API、モジュール実行、外部利用、設定に入ったパス、`importlib` などの動的ロード、文書中のコマンド、CI、artifact マニフェストも確認する。検索したパスと用語、存在する利用者、調べられない外部経路、意図を確認した根拠を示す。

`artifacts/` の学習済みモデル、計測表、compact diagnostics や、`reports/` の PNG/SVG/PDF は保存すべき研究成果になり得る。Python から import されないことや、複数の export 形式があることだけでは未使用・重複と判定しない。再現用の一度だけ使うスクリプト、独立した数値対照、解析極限のテストにも役割がある。実際の影響と継続責任を確認する。

scout は各候補について、パスと行番号、調査範囲、利用者・目的、影響、確度、推奨する限定的な対応、未確認事項を返す。根拠が不足した候補は「追加調査」で返し、欠陥を捏造して Issue にしない。

## Issue の重複確認と起票

1. 親または書込担当が、既存の **open と closed の両方** の Issue を読む。件数が多い場合は全ページを取得し、候補に関係する本文・コメント・関連 PR と履歴も確認する。今回の新規候補同士も比較する。
2. category、正規化した repository-relative path、root cause を照合する。パスは `/` 区切りにし、先頭の `./` を外す。大文字小文字を保持し、移動・名称変更があれば旧パスも検索する。category が異なっても同じ原因なら既存 Issue にまとめる。
3. 識別子を `hygiene:<category>:<path>` として本文に残す。例は `hygiene:orphan-config:configs/retired.json`。識別子の一致だけで判断せず、原因・履歴も読む。同じ原因の再発は新しい commit と証拠を既存 Issue にコメントし、未解決なら更新、解決後の再発なら再開する。
4. 確かな観察、具体的な影響、受入条件を定義できる発見だけを新規 Issue にする。未確認の候補は点検記録に留め、担当者確認または限定した追加 scout 調査を行う。
5. 新規または更新した Issue を管理 PR に関連付け、修正作業の所有者と範囲を決める。scout は GitHub 状態を書き換えない。

新規 Issue には以下を含める。GitHub の「リポジトリ保守」Issue form も同じ項目を入力できる。

```text
Observed snapshot: commit / dirty paths and hashes / checkpoint
Category: one of the categories above
Fingerprint: hygiene:<category>:<normalized repository-relative path>
Root cause: same cause / recurrence / new cause, with existing issue search result
Evidence: path:line, callers, references and search scope checked
Confidence and limits: supported conclusion and what remains unverified
Impact: concrete maintenance, execution or reproducibility problem
Proposal: bounded repair; no automatic deletion
Acceptance criteria: observable conditions for closing the issue
Related PR and issues: controlling PR, existing related issues, scan record
```

## 再利用できる scout 依頼文

以下を親が実際の情報で埋め、`fork_turns="none"` の `scout` に渡す。会話の履歴を参照しなければ分からない依頼にしない。root AGENTS と共通方針・scout role 設定を読むよう伝え、指定の role を起動できなければ親が制約を報告する。保守の点検担当は read-only とし、実装担当・validator への置き換えを行わない。

```text
目的: 開発の節目に配置・未使用候補・一時コード・再現情報の問題を調べる。
問い: [変更内容] により責任や利用経路が失われたもの、放置されたものがあるか。
作業ディレクトリ: [absolute path]
管理 Issue / PR: [URLs]
節目・対象版: [start / structural change / pre-review], [base commit]
dirty snapshot: [tracked diff, added/moved/deleted paths and content hashes]
対象範囲: [directories/files and changes; tracked + nonignored untracked]
除外範囲: [.git/.venv/dependencies/caches/.work/ignored bulk traces; other exclusions]
進行中の作業と所有者: [who owns which paths, unfinished work and completion checkpoint]
既知事項・入力: [requirements/plan/READMEs, prior scan link, manifests]
既存 Issue: [open + closed issue list, relevant bodies/comments and recurrence history]
責任: 読み取り専用。ファイル変更、重い実験、GitHub書き込み、削除、追加エージェントは禁止。
調査: docs/repository-maintenance.md の各categoryと利用経路を確認する。
      imports/exports/public API/CLI/tests/docs commands/CI/config paths/
      dynamic loading/artifact manifests を調べ、検索結果なしだけで未使用と断定しない。
      研究成果のモデル・計測値・図のexport形式とsource入口を区別する。
      現在の担当者の未完成ファイルはその旨を明示する。
終了条件: 渡した範囲を調べ、根拠のある発見と追加調査候補を区別して返す。
返却形式: 対象版・実際の範囲と除外、確定した発見（path:line、調査した参照、
          利用者/目的、具体的影響、確度、root cause、推奨対応、既存Issue候補）、
          反証/未確認事項、保留した進行中作業、指摘なしの場合もその結論。
          生ログ全量は不要。独立validator評価を行わない。
```

## 点検ごとの記録

親または書込担当が、**すべての点検** を管理 PR または保守 Issue のコメントに残す。指摘がない場合も記録する。各回専用の audit Markdown は作らず、同じスレッドで経過を追えるようにする。

```text
時刻・節目: [timezone付き日時 / checkpoint]
担当: [actual scout identity / role]
対象版: [commit / dirty snapshot hashes or attached snapshot reference]
調査範囲・除外: [tracked + nonignored untracked paths / excluded paths]
結果: [confirmed findings, or no supported new findings]
根拠: [key path:line references and limitations]
重複確認: [open/closed issue scope, matching issue or none]
Issue: [new/updated/reopened issue links, or none]
保留・次の節目: [in-flight work / unanswered questions / follow-up checkpoint]
```

再利用した点検には元のコメントへのリンクと、現在の版に適用できる理由を記す。保留した作業は完成後の節目で確認する。確かな問題を Issue に残すことは、修正済み・最終評価済みであることを意味しない。
