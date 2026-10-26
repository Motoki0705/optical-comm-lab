# Repository agent tools

This directory holds repository-local skills and their required support code.
It is separate from the globally installed Codex skills.

The [Codex scheduled follow-up skill](skills/codex-scheduled-followup/SKILL.md)
delivers scheduled prompts to an existing CLI session. Its
[provenance](skills/codex-scheduled-followup/README.md) records the source version,
file hashes and local adaptations. The [shared queue component](ops/lib/README.md)
is the only copied runtime dependency.

Keep reusable agent operations here; scientific code, experiments and research
outputs retain their existing repository directories. Runtime task state,
rollouts, timer registrations and local prompt files do not belong in this tree.
