# Reproducible entry points

Install the project environment according to the root README. Run from the
repository root with the installed package:

```bash
.venv/bin/python scripts/run_experiment.py --config configs/pilot.json --run-id pilot-check --powers 0
.venv/bin/python scripts/run_experiment.py --config configs/full.json --run-id final-qualified
```

The full sweep trains a separate model at every configured power. A subset can
be added to the same run later with `--powers -6 -4` and `--resume`. The runner
verifies unchanged configuration, source hashes, environment/backend, and saved
artifact hashes before reusing complete conditions. It regenerates incomplete
conditions with exactly the same seeds only when `--resume` is explicit.

```bash
.venv/bin/python scripts/run_experiment.py --config configs/pilot.json --run-id pilot-check --powers 2 --resume
.venv/bin/python scripts/run_experiment.py --config configs/full.json --run-id final-qualified --dry-run
.venv/bin/python scripts/run_experiment.py --config configs/pilot.json --run-id pilot-cpu --powers 0 --backend numpy
```

`--output-directory` defaults to `artifacts`; use `.work` for isolated smoke or
validation output. Names must start with the configuration's `pilot-` or
`final-` prefix. A source/config/environment change requires a new run name.
Run final experiments only after the separate channel numerical qualification
has fixed their settings. Do not tune candidates against held-out scores.

Saved outputs can be audited without propagating ASE, retraining, or running
model predictions:

```bash
.venv/bin/python scripts/audit_run.py --run artifacts/final-1600km-20261012 --output .work/final-audit.json
```

The audit checks frozen source/configuration hashes, all completion hashes,
CSV/JSON correspondence, frame/seed isolation, validation-only candidate
selection and its tie rule, checkpoint arrays/transforms, denominators, pooled
energies/counts, and declared constellation indices. `--output` is optional;
without it the JSON is printed. Saved run inputs remain read-only. The explicit
output may replace only `verification.json` inside a run, never its scientific
data. A nonzero exit status identifies failed integrity or consistency checks.

The JSON separates deterministic `semantic_checks` and their digest from fresh
audit timing/timestamps in `observations`. `recorded_runner_evidence_observed`
means that the original runner's prediction-reload record was inspected, not
that predictions were repeated. Original experiment duration and resume timing
remain in `historical_measurements`; clone file mtimes never reconstruct them.
The tracked CLI's actual SHA-256 is recorded. In an isolated source copy, set
`PYTHONPATH` to that copy's `src` and use the existing environment; the audit
rejects accidentally importing a different scientific source tree.

This directory owns thin CLI adapters and reproducible figure generators.
Reusable calibration, windowing, training, scoring, and artifact logic belong
in `src/optical_comm/`; a new CLI should call those modules rather than copy
numerical methods. Saved figures and report text belong in `reports/`, and
compact run data belong in `artifacts/`.
