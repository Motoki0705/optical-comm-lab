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

This directory owns thin CLI adapters and reproducible figure generators.
Reusable calibration, windowing, training, scoring, and artifact logic belong
in `src/optical_comm/`; a new CLI should call those modules rather than copy
numerical methods. Saved figures and report text belong in `reports/`, and
compact run data belong in `artifacts/`.
