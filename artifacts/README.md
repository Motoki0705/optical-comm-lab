# Compact reproducible runs

Each run is a `pilot-*` or `final-*` directory. Pilot scores are exploratory and
are not mixed into held-out final scores. Raw waveforms can be regenerated from
the resolved channel configuration and frame seeds; bulk traces and temporary
outputs belong in ignored `.work/` rather than this directory.

| File | Content |
| --- | --- |
| `config.json` | Fully resolved physical, frame, context, and candidate settings |
| `manifest.json` | Source/package hashes and versions, commit, seed contract, future context |
| `summary.csv`, `summary.json` | Pooled results from completed launch powers |
| `power_*/metrics.csv`, `metrics.json` | Per-method direct BER, bounds, counts, EVM/SNR, frame spread, timing, model size |
| `power_*/test_frames.csv`, `test_frames.json` | Every test seed, frame, and receiver result |
| `power_*/selection.json` | Validation candidates and frozen selected IDs; saved before test generation |
| `power_*/calibration.json` | Train-only EDC and DBP forward-gain inversion and input standardization |
| `power_*/seeds.json`, `generation.json` | Whole-frame independent bit/ASE recipes and measured generation times |
| `power_*/linear.npz`, `esn.npz` | Prediction-complete selected model checkpoints without pickle |
| `power_*/constellations.npz` | Same fixed first-test-frame indices and symbols for all receivers |
| `power_*/complete.json` | Compatibility record and hashes of per-power deliverables |

The power-directory format is a signed fixed-width number with `p`/`m` for the
sign and `p` for the decimal point, e.g. `power_m0006p000` and
`power_p0000p000`. Checkpoints load with `optical_comm.reservoir.load_equalizer`;
prediction takes a `SymbolFrame` of normalized received XY symbols and a valid
mask. Labels are optional for prediction and required only for metrics/training.

Clopper-Pearson BER intervals have an independent-bit assumption; use the saved
independent-frame spread to assess repeatability. No finite measurement proves
zero error risk. Effective SNR is label-normalized EVM, with `null` for a perfect
zero-error-energy realization. Receiver timing is additional processing after
shared generation/EDC, so it is not a claim of end-to-end device throughput.

Add a named run when it provides interpretable evidence. Keep configurations,
compact metrics, selected models, diagnostics, figures, and provenance together
through the controlling research PR. Do not save transient tuning notebooks or
unexplained variants as permanent run artifacts.

## Normal verification and independent review

`verification/` is a permanent supplement for the reader-facing report. It
contains `ordinary_checks.json` and its integration-test log, plus
`independent_review.json` with the requested/attempted/completed counts, the
original evaluation, the coordinator's disposition, and any changes afterward.
These records are report inputs and are committed so regeneration from a clone
does not depend on ignored `.work/` files. Independent evaluation is distinct
from analytic tests, numerical qualification, validation-data model selection,
and held-out testing. A pending record must not be described as completed.

The final run also keeps `execution-hardware.json`, a separately observed
CPU/GPU/environment supplement. Receiver times compare extra processing on
those devices, not a hardware throughput benchmark. The supplement does not
modify the immutable configuration, selected weights, or test metrics.

The final run's `verification.json` is reproducibly regenerated as a saved-data
consistency audit through the tracked CLI:

```bash
.venv/bin/python scripts/audit_run.py --run artifacts/final-1600km-20261012 --output .work/final-audit.json
```

`semantic_checks` contains the source/config/artifact hashes, seed and candidate
contracts, loaded-checkpoint fields, count/energy pools, and diagnostic checks.
Its digest is stable for the same inputs. `observations` contains fresh audit
time and environment information. Historical runner duration and its earlier
resume check remain explicitly under `historical_measurements`; they are not
recalculated from checkout timestamps. The audit reads existing evidence that
the runner compared predictions on checkpoint reload, while recording that
this audit performs no prediction, propagation, training, or new test-frame
generation. This audit is normal verification, distinct from the independent
validator. See `scripts/README.md` for output protection and isolated-copy use.

`pilot-800km-development/` records a separate 800 km development check at 0 dBm
using the existing pilot candidates and independent splits. It counts 145824
bits per receiver across three test frames. Its exploratory scores verify the
distance parameter and pipeline; they are not evidence of distance
generalization, receiver improvement, or 1600 km final performance. Scope,
commands, counts, and hash/resume checks are in its `verification.json`.
Regenerate with `.venv/bin/python scripts/run_experiment.py --config artifacts/pilot-800km-development/config.json --run-id pilot-800km-reproduction`;
verify existing outputs by replacing the run ID with `pilot-800km-development`
and adding `--resume`.

The final run's `waveform_diagnostics/` supplement stores one contiguous NPZ for
the first existing test frame at each of its seven powers. Each default slice
contains 512 symbols and 1024 complex XY samples at 2 sps, including the guard
boundary. Arrays retain uncalibrated EDC/matched-filter alphabet normalization,
aligned 1-sps RX/TX symbols and TX bits, absolute indices, frame identity/seeds,
and separate channel-valid and original common-score masks. `manifest.json`
records Rs/Fs, full-frame periodic boundaries, source/input/output hashes,
stored-constellation correspondence, and the export/regeneration commands.
These are reconstructions of previously counted frames, not additional test
evidence. Regenerate into a fresh directory with
`scripts/export_waveform_diagnostics.py` as described in `scripts/README.md`.

`kerr-physics/` contains a separate deterministic isolated-pulse demonstration
for the Blender films: compact complex XY fields at 81 distances, physical
observables, full-grid scalar metrics, analytic/numerical qualification, tests
and a source/config/artifact manifest. Spectra are pulse energy density in J/Hz;
raw data retain loss. The saved display data use qualified Fourier decimation
and complex64; the actual solver and all numerical checks use complex128.
Regenerate into a fresh directory with `scripts/simulate_kerr_films.py` as
documented in [kerr-physics.md](../docs/kerr-physics.md).
