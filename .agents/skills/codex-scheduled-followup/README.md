# Scheduled follow-up copy provenance

This is the repository-local copy for optical-comm-lab Issue #41 (parent #40).
Use [SKILL.md](SKILL.md) for operating decisions and the
[CLI reference](references/cli-heartbeat.md) for commands. This copy does not
install or overwrite the globally installed skill.

## Source version

- Source checkout: `/home/kamimura/projects/tennis-lab`.
- Source commit: `cfbd5e729a74f9e63cb3afd12c903415d6757590`.
- Copy date: 2026-10-25. The source paths were clean and their bytes were checked
  against that commit when copied.
- Paths are preserved below, relative to each repository root. SHA-256 values
  describe the original committed source bytes, including the adapted reference.

| Original source path / local destination | Source SHA-256 | Local copy |
| --- | --- | --- |
| `.agents/skills/codex-scheduled-followup/SKILL.md` | `d9644844c0c14909264be36e060cc631a678cbf96b9d7073205c7ed98f2d8d68` | Byte-identical |
| `.agents/skills/codex-scheduled-followup/agents/openai.yaml` | `ddc2aeecf90be3e73e6eb79975e764b712f489b5d7085c62da2f2f71c0b723ba` | Byte-identical |
| `.agents/skills/codex-scheduled-followup/references/cli-heartbeat.md` | `d9350145aa1d402fc76ee9470df9cf477009d79d569bc4b0d00c98765b86501c` | Documentation adapted below |
| `.agents/skills/codex-scheduled-followup/scripts/cli_heartbeat.py` | `ef81ccbc5b8ec7ffd40206776d7274eb4a86934e61f6f2e41ebe7a939d21a486` | Byte-identical |
| `.agents/skills/codex-scheduled-followup/scripts/adopt_rollout.py` | `0fdd4fc989b36aacc61fe93b5348b57dc3fa4d0fdede7e5127e1e17f98c9c69a` | Byte-identical |
| `.agents/skills/codex-scheduled-followup/scripts/test_cli_heartbeat.py` | `6d48702b58d1cc3e30c3713e27d7063da4bc04cc79609efa42c4ec4a6750a50e` | Byte-identical |
| `.agents/skills/codex-scheduled-followup/scripts/test_adopt_rollout.py` | `bfae7731d508db953f9b6043d754dd61fab15fbc76e2c4b26a5c581b72b57bb2` | Byte-identical |
| `.agents/ops/lib/__init__.py` | `e66858959f88ecf1c9320e80c3615410afd24741517eb620fe46095f7b4101f9` | Byte-identical |
| `.agents/ops/lib/codex_queue.py` | `3cb5bdb8b6044a6a0516735b4c8de22466aecd97fe88c1b54157c0a0f25848e0` | Byte-identical |

## Exact local adaptations

The only edited source file is `references/cli-heartbeat.md`:

1. Remove “also used by the parent board” from the shared queue description;
   this repository does not copy the tennis-lab board.
2. Replace the Tests paragraph's tennis-lab pytest wrapper path with direct,
   portable invocations of both included test scripts using `-B`; state that the
   helper suite also covers the shared queue and link to this provenance.

Its local SHA-256 is `4cccc2a87d1008d65030470e7f10af8ce572fff5abe723b363ba5e976fb44991`.

All Python files, `SKILL.md` and `agents/openai.yaml` remain byte-identical to the
source. In particular, the default prompt stays `これはハートビートです。`, an
omitted interval stays 60 minutes, and callers pass an explicitly requested
30-minute interval. Pause/resume, evidence-preserving recovery and verified
continuation-rollout adoption keep their original contracts.

The new files `.agents/README.md`, `.agents/ops/lib/README.md` and this README
explain the local extension point, the minimal dependency and this provenance.
The source lib README was not copied because its agent runners, board and other
operations are outside this task. No source caches, other agent infrastructure,
product code or live task state are included.

## Dependencies and verification

Python 3.11+ and the standard library suffice for imports and mocked tests. The
helper resolves `.agents/ops/lib/codex_queue.py` relative to its own location;
no import reads the tennis-lab checkout. Copy the entire
`.agents/skills/codex-scheduled-followup` directory together with `.agents/ops/lib/__init__.py`
and `.agents/ops/lib/codex_queue.py` when relocating it.

From the repository root, run:

```bash
python3 -B .agents/skills/codex-scheduled-followup/scripts/test_cli_heartbeat.py
python3 -B .agents/skills/codex-scheduled-followup/scripts/test_adopt_rollout.py
```

The helper suite has 43 tests, including the shared queue submission and receipt
paths; the rollout adoption suite has 7 tests. They create temporary rollouts and
mock external commands. The same suites pass after relocating the copied tree
to a temporary directory outside either checkout. No live queue delivery or timer
registration is part of this verification. Live operation additionally requires
the Linux/WSL, user-systemd and existing-runtime prerequisites in the reference.

For future updates, select an explicit source commit, compare the files and
record new source hashes and any adaptations here. Retain the minimal heartbeat
and the original resume, recovery and adoption contracts; run both portable
suites before handing the copy to the scheduling operator.
