# Transformer physical data pipeline

This reference describes the implementation for [Issue #44](https://github.com/Motoki0705/optical-comm-lab/issues/44). Scientific choices, assumptions, and gates remain in [reproduction-spec.md](reproduction-spec.md). The data entrypoint uses NumPy/SciPy and optional CuPy; it does not require PyTorch. A short pilot does not establish full-link convergence or satisfy the production data budget.

## Conditions and receiver

The named conditions are `main16` (DP-16QAM, 32 GBd, 40 × 80 km, +2 dBm XY-total launch power) and `main64` (DP-64QAM, 42 GBd, 12 × 80 km, +3 dBm). These settings do not alter the ESN defaults. Physical fields use complex128 and sqrt(W); alphabet-scale symbols have ensemble Es = 1 per polarization.

Processing is TX RRC → inverse half-link dispersion → adaptive symmetric Manakov propagation and per-span EDFA/ASE → inverse half-link dispersion → matched RRC → train-pilot 2×2 FIR → dedicated frame-pilot gain calibration → blind phase search → dedicated frame-pilot ambiguity/gain calibration → `rx_nlc`. Ideal timing, resampling and frequency correction are explicit identity stages. `rx_2sps`, when saved, is after receive CDC/matched RRC and before FIR/carrier recovery. Raw `rx_field` is after the final EDFA and before receive DSP.

The FIR uses dedicated train pilots only. Carrier-recovery window selection compares 33/65/129 on independent validation, by aggregate BER then MSE. Test scored labels are not inputs to receiver fitting or calibration; generation does not compute held-out test metrics. The no-NLC baseline is the resulting `rx_nlc`, including linear equalization and carrier recovery.

## Commands and staged execution

Run from the intended checkout and select its source explicitly. On the development machine:

```bash
PROJECT_ROOT=/home/kamimura/projects/optical-comm-lab
DATA_CHECKOUT="$PROJECT_ROOT/.work/worktrees/transformer-data"
PYTHON_EXEC="$PROJECT_ROOT/.venv/bin/python"
GPU_LOCK="$PROJECT_ROOT/.work/orchestration/gpu.lock"
cd "$DATA_CHECKOUT"
export PYTHONPATH="$DATA_CHECKOUT/src:$DATA_CHECKOUT"
```

Inspect production budgets or the numerical plan without propagation:

```bash
"$PYTHON_EXEC" -m experiments.transformer_nlc.data_cli generate --condition 16 --dry-run
"$PYTHON_EXEC" -m experiments.transformer_nlc.data_cli generate --condition 64 --dry-run
"$PYTHON_EXEC" -m experiments.transformer_nlc.data_cli qualify --condition 16 --dry-run
"$PYTHON_EXEC" -m experiments.transformer_nlc.data_cli qualify --condition 64 --dry-run
```

`qualify` also defaults to planning when `--execute` is absent. Its initial plan contains six settings per condition: baseline 4 sps / 0.05° / 65536 scored symbols / guard 4096 / RRC 128, then separate refinements to 0.025°, 8 sps, 131072 scored symbols, guard 8192, and RRC 256. Three independent dedicated pilot frames per setting give 18 full-link propagations per condition, 36 total. This is an initial estimate, not a cap on justified refinement.

A bounded GPU resource pilot measures timing/storage for two spans and a short frame:

```bash
flock -n "$GPU_LOCK" "$PYTHON_EXEC" -m experiments.transformer_nlc.data_cli benchmark \
  --condition 16 --backend cupy --scored-symbols 4096 --spans 2 \
  --run-id resource-main16 --output-root .work/transformer-nlc
```

Repeat with condition 64 and a distinct run ID. The printed FFT-NlogN/span extrapolation is a planning estimate: it does not model changed adaptive peaks, GPU occupancy, longer refinement records, or common-reference noise generation. It is not measured production runtime.

After the coordinating agent releases a frozen numerical snapshot, run the three baseline frames first for each condition:

```bash
flock -n "$GPU_LOCK" "$PYTHON_EXEC" -m experiments.transformer_nlc.data_cli qualify \
  --condition 16 --backend cupy --execute --cases baseline \
  --run-id qualification-main16-20261025 --output-root artifacts/transformer-nlc
flock -n "$GPU_LOCK" "$PYTHON_EXEC" -m experiments.transformer_nlc.data_cli qualify \
  --condition 64 --backend cupy --execute --cases baseline \
  --run-id qualification-main64-20261025 --output-root artifacts/transformer-nlc
```

Baseline-only output is incomplete qualification. Inspect analytic controls, CPU/GPU agreement, finiteness, actual elapsed time, storage and receiver behavior before continuing. Resume the remaining settings with the same source, numerical runtime, run ID, and output root:

```bash
flock -n "$GPU_LOCK" "$PYTHON_EXEC" -m experiments.transformer_nlc.data_cli qualify \
  --condition 16 --backend cupy --execute --resume --cases phase sps frame guard rrc \
  --run-id qualification-main16-20261025 --output-root artifacts/transformer-nlc
flock -n "$GPU_LOCK" "$PYTHON_EXEC" -m experiments.transformer_nlc.data_cli qualify \
  --condition 64 --backend cupy --execute --resume --cases phase sps frame guard rrc \
  --run-id qualification-main64-20261025 --output-root artifacts/transformer-nlc
```

The global GPU flock prevents overlapping GPU experiments. Each output directory also has its own nonblocking advisory flock. Its persistent `.generation.lock` inode is retained; the OS releases ownership after normal exit or process death. A recorded PID is informational. Do not unlink a lock file while another process may be using the run.

Production generation requires a matching `passed` qualification artifact. For example, after the numerical and production gates are released:

```bash
flock -n "$GPU_LOCK" "$PYTHON_EXEC" -m experiments.transformer_nlc.data_cli generate \
  --condition 16 --replicate 0 --backend cupy --run-id data-main16-r0 \
  --output-root artifacts/transformer-nlc \
  --qualification artifacts/transformer-nlc/qualification-main16-20261025/qualification.json
```

Use the corresponding condition 64 qualification and separate run IDs for replicates 0/1/2. `--seed` is an alias for the replicate key; it does not replace the fixed master entropy. Add `--resume` only to continue an existing matching run. Reusing a run ID without resume is rejected. Full generation or training must not bypass an incomplete or failed numerical gate.

For explicitly smaller functional data, use `generate --pilot --scored-symbols ...` or an explicit pilot-regime JSON supplied through `--config`. `--pilot` changes data budgets and streams; it does not shorten the named physical distance. A custom short-span CPU smoke therefore needs a pilot JSON with its physical settings stated. These fixtures must remain labeled pilot. Production scored/pilot/context/frame budgets are enforced separately.

## Frames, random streams and learning blocks

Production geometry is `guard 4096 / pilot 2048 / context 96 / scored 65536 / context 96 / pilot 2048 / guard 4096` at the qualification starting settings. Guards, pilots and context do not count toward the scored budget. Train has 8 frames (2^19 DP times), additional validation 2 frames (2^17), and test 4 frames (2^18), for each condition and independent replicate.

The production PCG64 seed key is `SeedSequence(2026102543, spawn_key=(condition, replicate, split, frame, purpose))`. Split IDs are train0/validation1/test2/pilot3, and purpose IDs are bits0/ASE1/model2/shuffle3. Seven bit-region children and independent span-noise children are recorded. Left-guard bits are anchored at the pilot-facing edge, so changing guard length preserves nearby bits and the scored series. Dedicated pilot/qualification streams use split3 and disjoint encoded frame IDs; they are not production streams.

All candidate architectures and both mask choices share a condition/replicate dataset. The manifest records model/shuffle seed recipes for paired initializations and ordering. `load_frame(path, expected_sha256=...)` returns one physical frame; `iter_blocks(frame, block_symbols=128, context=96)` yields complex128 receive inputs, TX targets, residuals, and indices. Targets are contiguous, nonoverlapping blocks within one frame; inputs may overlap in context. The iterator rejects invalid support, scoring holes, symbol-index discontinuities and frame/split crossings. Training owns conversion to float32 `XI,XQ,YI,YQ`. The residual target is `rx_nlc - tx_symbols`, not pure Kerr distortion.

## Manifests, source scope and runtime

New data manifests use `transformer-data-v2`; qualification reports use `transformer-qualification-v2`. Older evidence is retained with its original schema and generating source. It is not relabeled as v2 or assigned a newer hash. Older artifacts remain readable through the array loaders, but cannot silently satisfy a new resume/production gate.

The explicit `transformer-numerical-source-v1` fingerprint covers exactly:

- `src/optical_comm/config.py`, `channel.py`, `dsp.py`, `metrics.py`;
- `experiments/transformer_nlc/data_config.py`, `data.py`, `receiver.py`, `data_cli.py`, `qualification.py`.

The scope version and individual file hashes participate in the fingerprint. Model/training/report files, prose, and optional dependency lockfile changes do not invalidate this guard. Full observed source/lock provenance is stored separately. Actual installed numerical versions remain guarded even when a dependency lockfile change concerns an unrelated package.

Each logical key resolves the actual imported module's source file. Shared and study modules must belong to one coherent source checkout/tree or one installed package root. Mixed imports from different checkouts/installations are rejected before run output is created. `source_layout` records absolute executing module paths and distinguishes source checkout, exported source tree, and installed package. Git commit/status are recorded only for the verified checkout; they are `null` for a true installed wheel. Identical contributing source bytes can have identical numerical fingerprints in a checkout and a wheel, while their layout/provenance records differ.

An installed wheel can generate, reload and resume data from a directory outside the checkout without Git. Bulk protection checks the output location's actual repository if one exists, and rejects writes into installed package sources. It does not infer a repository by walking upward from `site-packages`. Use an external run directory for installed-package execution.

The versioned scientific contract binds the condition, XY power, complete physical LinkConfig, scored/pilot/context/guard geometry, maximum and minimum SSFM bounds, FIR taps/ridge, BPS candidates/phase count and pilot-fit policy. Qualification and production may legitimately differ in regime, split counts, replicate/seed and storage choices; those fields are excluded from this contract. Production checks the explicit contract and its hash before creating output paths.

Resumes compare `transformer-numerical-runtime-v1`: backend, Python/NumPy/SciPy versions, architecture and physical dtype, plus CuPy/CUDA runtime/driver/GPU identity for CUDA runs. Optional Torch is excluded. Source/config/runtime changes are rejected rather than combined silently into one run. Fixes that change a contributing numerical dependency require a new coordinated snapshot and new evidence.

## Saved files and regeneration

| Location | Contents |
| --- | --- |
| `artifacts/transformer-nlc/<data-run>/manifest.json` | Effective config, scientific contract, source/runtime, command, stage identities/units, seed states, per-frame paths/schema/hashes/mask counts, receiver calibration and failure records |
| `.../receiver.npz` | Frozen train-pilot FIR weights; order/BPS choice and validation selection are in the manifest |
| `.../bulk/*-physical.npz`, `*-nlc.npz` | Independent frame arrays; bulk is ignored by Git |
| `artifacts/transformer-nlc/<qualification-run>/qualification.json` | Full six-setting plan/configs, controls, source/runtime/contract, completed-frame hashes/audits/metrics, comparisons, failures and current status |
| `.../bulk/<case>-<frame>.npz` | Reproducible qualification frames, ignored by Git |
| `.work/qualification/` | Temporary orchestration logs, process state and monitoring output |

Frame arrays include `tx_bits` uint8 `[T,2,log2(M)]`, `tx_symbols`, `rx_matched` and `rx_nlc` complex128 `[T,2]`, symbol/frame/split indices, and separate pilot/DSP-valid/context-valid/scoring masks. Optional raw receive field and 2-sps diagnostics have their own stage, units, shape, dtype and SHA256. When omitted, metadata records their computed hashes and regeneration recipes. Regenerate from the recorded effective config, split/frame seed, exact source identity, runtime/backend and command; a new implementation must not be substituted as the original producer.

Qualification duration changes cannot share the same physical ASE merely by using the same seed on different FFT grids. Its common-noise recipe uses a longer fixed-bandwidth 8-sps Fourier reference, cropped at the scored origin and decimated for each case. The reference geometry, grid, seed and variance are recorded. Matching scored bits and the first 65536 scored relative indices are checked. This coupling is a numerical comparison method; production retains its separately recorded periodic-frame noise recipe.

Outputs are written atomically and checkpointed after each physical frame. Resume checks their recorded hashes and does not overwrite a different run. Initial config/runtime records precede long propagation. Bulk inside the checkout must actually match an ignored path such as `.work/` or `artifacts/**/bulk/`; compact manifests and reproducible diagnostic summaries belong with the named run.

## Interpreting qualification results

The report retains per-frame no-NLC BER counts/intervals, MSE/effective SNR, actual SSFM step statistics, maximum accepted nonlinear phase, retries, noise-grid details and physical elapsed time. A result becomes `passed` only after all 18 per-condition propagations, required analytic controls and CPU/GPU controls, and all five comparisons pass. A failed or partial result is preserved and blocks production. The reference criterion compares mean effective-SNR differences against 0.1 dB and BER differences against recorded counting/frame uncertainty; per-frame differences remain available for assessment.

The 0.1 dB value is a numerical-convergence guide, not an allowed error against the paper. Analytic controls, small CPU/GPU smoke, resource timing, full-link qualification, validation selection and held-out testing are distinct evidence. No held-out communications performance or paper reproduction success follows from completing this data pipeline. See each run's actual report for measured status; this document does not declare unexecuted qualification successful.
