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
| `.agents/skills/codex-scheduled-followup/scripts/cli_heartbeat.py` | `ef81ccbc5b8ec7ffd40206776d7274eb4a86934e61f6f2e41ebe7a939d21a486` | Recovery adapted below |
| `.agents/skills/codex-scheduled-followup/scripts/adopt_rollout.py` | `0fdd4fc989b36aacc61fe93b5348b57dc3fa4d0fdede7e5127e1e17f98c9c69a` | Byte-identical |
| `.agents/skills/codex-scheduled-followup/scripts/test_cli_heartbeat.py` | `6d48702b58d1cc3e30c3713e27d7063da4bc04cc79609efa42c4ec4a6750a50e` | Recovery regression tests added |
| `.agents/skills/codex-scheduled-followup/scripts/test_adopt_rollout.py` | `bfae7731d508db953f9b6043d754dd61fab15fbc76e2c4b26a5c581b72b57bb2` | Byte-identical |
| `.agents/ops/lib/__init__.py` | `e66858959f88ecf1c9320e80c3615410afd24741517eb620fe46095f7b4101f9` | Byte-identical |
| `.agents/ops/lib/codex_queue.py` | `3cb5bdb8b6044a6a0516735b4c8de22466aecd97fe88c1b54157c0a0f25848e0` | Superseded phase preserved below |

## Exact local adaptations

The initial copy adapted `references/cli-heartbeat.md`:

1. Remove “also used by the parent board” from the shared queue description;
   this repository does not copy the tennis-lab board.
2. Replace the Tests paragraph's tennis-lab pytest wrapper path with direct,
   portable invocations of both included test scripts using `-B`; state that the
   helper suite also covers the shared queue and link to this provenance.

Its initial local SHA-256 was `4cccc2a87d1008d65030470e7f10af8ce572fff5abe723b363ba5e976fb44991`.

[Issue #57](https://github.com/Motoki0705/optical-comm-lab/issues/57) adds the
following local recovery adaptations:

1. `scripts/cli_heartbeat.py`: accept recovery for active or paused tasks. Keep
   the reason, duplicate-risk acknowledgement, exact latest marker, unresolved
   phase and history/registration guards. Recovery preserves the paused state
   and neither starts a timer nor sends a message.
2. `.agents/ops/lib/codex_queue.py`: continue checking rollout history and anchors
   for a `superseded` delivery, while retaining the operator's recovery decision
   instead of replaying an old interrupted receipt over it. The original error,
   queue and turn identifiers, and recovery record remain available.
3. `scripts/test_cli_heartbeat.py`: add four regression tests for paused failed
   recovery followed by hourly resume, authorization/marker/phase guards,
   registration-state refusal, and retained history checks after recovery.
   Extend the changed-history recovery test to paused tasks.
4. `references/cli-heartbeat.md`: explain active/paused recovery and the explicit
   resume required to restart a paused clock.

Current local hashes of the adapted source files:

| Local path | Local SHA-256 |
| --- | --- |
| `references/cli-heartbeat.md` | `1aac41dd3270ab940569eb5c213b9b97d08876432470ead3f5bd7d80b39d79e6` |
| `scripts/cli_heartbeat.py` | `9e9852160e71fa78a7f79e1bbe2a224a1bb55f9b51b750dd22a2ac5552bad3d5` |
| `scripts/test_cli_heartbeat.py` | `0d7424c246bc08c62e61ab89aa47cb9d0183299fc003c53b13d3cf563c0668a5` |
| `.agents/ops/lib/codex_queue.py` (repository-relative) | `ecf7532c1ce7bb585e3be825b52652e76409a5310a6f93068a7e02ecffedacce` |

The other copied files remain byte-identical to the source. The default prompt
stays `これはハートビートです。`, an omitted interval stays 60 minutes, and explicit
intervals remain honored. Recovery retains evidence; resume separately registers
the requested timer. Verified continuation-rollout adoption keeps its original
contract.

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

The helper suite has 47 tests, including the shared queue submission and receipt
paths; the rollout adoption suite has 7 tests. They create temporary rollouts and
mock external commands. The same suites pass after relocating the copied tree
to a temporary directory outside either checkout. No live queue delivery or timer
registration is part of this verification. Live operation additionally requires
the Linux/WSL, user-systemd and existing-runtime prerequisites in the reference.

For future updates, select an explicit source commit, compare the files and
record new source hashes and any adaptations here. Retain the minimal heartbeat
and the original resume, recovery and adoption contracts; run both portable
suites before handing the copy to the scheduling operator.
