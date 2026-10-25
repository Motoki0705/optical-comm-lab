# Shared queue component

## Codex queueの共通部品

`codex_queue.py` supplies durable queue submission and rollout receipt checks for
the repository-local
[scheduled follow-up skill](../../skills/codex-scheduled-followup/SKILL.md).
It uses only the Python standard library and has no timer or agent launcher.
The adjacent `__init__.py` makes this directory the helper's `lib` package.

The caller fixes the thread ID, cwd, rollout, environment and prompt, holds the
task lock, and provides the state persistence and command functions.
`enqueue(task, marker, persist=..., command=...)` saves `uncertain` before one
queue submission and preserves an acknowledgement even after a timeout.
`reconcile(task, persist)` accepts markers only as user input in a matching turn,
records completion, and blocks replaced, truncated or locally rewritten history.
Recovery and permission decisions remain with the caller.

Run the skill's `scripts/test_cli_heartbeat.py` to exercise these shared paths with
temporary rollouts and mocked commands. Source hashes and the unchanged runtime
files are recorded in the skill's [provenance](../../skills/codex-scheduled-followup/README.md).
