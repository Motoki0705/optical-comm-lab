# Research repository rules

## Scope and scientific integrity

This repository studies software reservoir equalization of long-haul coherent optical links. Read `docs/requirements.md` and `docs/experiment-plan.md` before changing scientific assumptions. Report measured outcomes, including negative results. Never manufacture, smooth away, or select test results to claim an improvement. Distinguish exploratory runs, numerical verification, validation, and held-out testing.

## GitHub workflow

- Track work and material defects in GitHub Issues before implementing them. Sub-agents may create scoped issues; check existing issues to avoid duplicates. Link issues to the controlling PR.
- Work on a feature branch and use a draft PR while implementation is in progress. Do not push to the default branch or merge without the coordinating agent's explicit task authorization.
- In a shared checkout, only the coordinating agent changes branches, stages, commits, and pushes. Workers edit only their assigned files and must not revert others' changes.
- Keep generated, reproducible research artifacts together. Commit compact result tables, model checkpoints where reasonably sized, report figures, and provenance. Ignore environments, caches, raw bulk traces, and temporary work.

## Agent roles

- Spawn all sub-agents with `fork_turns="none"` and a self-contained brief: purpose, question, directory, ownership, exclusions, inputs, acceptance criteria, issue, and return format.
- Independent read-only investigations use the `scout` role. Suitable uses include literature checks, module/API tracing, numerical-method questions, experimental-design questions, and locating causes of a concrete failure. A scout does not edit files, execute heavy experiments, write GitHub state, or act as an uncounted final validator.
- Implementation, experiment execution, visualization code, and saved report artifacts use `gpt-6.1-sol` with `reasoning_effort="max"`. If that setting cannot be requested, report the limitation rather than silently substituting a model.
- Every worker is sharing the codebase. Honor file ownership, coordinate interface changes, and preserve other workers' edits. Do not spawn further agents unless the coordinating task explicitly permits it.
- Read `$CODEX_HOME/agents/common.md` and the applicable role settings. Research conclusions should return evidence and uncertainty, not full exploration logs.

## Directory and extension policy

Keep this `AGENTS.md` at the repository root; do not proliferate nested copies. Directory READMEs explain local responsibilities and extension points, without duplicating these rules.

| Directory | Responsibility and place for extensions |
| --- | --- |
| `src/optical_comm/` | Reusable channel, DSP, metrics, reservoir, and experiment modules. Put a new mechanism beside the owning module; separate it only when it has a distinct API. |
| `configs/` | Versioned experiment configurations. Name configurations by purpose; no unexplained parameter variants. |
| `scripts/` | Thin, reproducible command-line entry points and visualization generators. Put reusable numerical logic in `src`. |
| `tests/` | Analytic limits, data isolation, reproducibility, and substantive regression tests. Do not mirror implementation internals. |
| `docs/` | Requirements, architecture, methods, source references, and research decisions. Consolidate related investigation notes. |
| `reports/` | The reader-facing report and its figures. Explain axes, units, uncertainty, and the scope of each conclusion. |
| `artifacts/` | Named run directories containing configs, seeds, metrics, compact diagnostic samples, trained models, and manifests. |
| `.work/` | Ignored scratch files, temporary logs, downloaded references, and isolated validation outputs. |

Create a directory or document when it has a stable responsibility and a reader, not merely because a new subtask exists. Assign scouts to these responsibilities without scattering their intermediate output into the tree.

## Validation and final review

Run normal verification appropriate to the change before requesting independent evaluation. The user authorized exactly **one** independent validator evaluation for this implementation task; attempts consume the count. Do not add or replace it with scout reviews.

Before that attempt, confirm the validator inherits the parent's model, provider, and reasoning effort; the validator role must not override them. Report any unavailable effective-runtime verification. Freeze the commit, hashes of uncommitted/untracked deliverables, configurations, and input artifacts. Pass original requirements and acceptance criteria; put parent test logs in a separate supplement. No evaluated files may change during evaluation, and validator scratch output must be isolated.

Retain the original result and record the parent's evidence-based disposition of each finding. Run ordinary verification after fixes. Changes made after the sole evaluation are not independently re-evaluated. The final report must state the requested/attempted/completed review counts, findings and dispositions, test results, and limitations.
