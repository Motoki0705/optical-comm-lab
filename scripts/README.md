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

Continuous 2-sps waveform diagnostics are exported from existing saved test
recipes through a separate CLI:

```bash
.venv/bin/python scripts/export_waveform_diagnostics.py --run artifacts/final-1600km-20261012 --output-dir .work/waveform-diagnostics-reproduction
```

The exporter audits source/configuration/seed/artifact hashes, reconstructs the
whole original frame, and then extracts a contiguous slice. Defaults are test
frame 0, 512 symbols, and start `max(0, guard-16)`; `--frame-index`, `--symbols`,
and `--start-symbol` select another nonwrapping slice. `--backend numpy` permits
CPU reconstruction; the actual backend and comparison tolerances are recorded.
The output directory must be empty, and input scientific files are read-only.
Each NPZ contains uncalibrated EDC/matched-filter 2-sps XY samples, aligned
1-sps RX and TX symbols/bits, global indices, frame identity/seeds, and separate
channel-valid/common-score masks. The manifest records stage, time origin,
frame boundaries, code/input/output hashes and regeneration command. Frame 0
is checked against every original stored constellation point; later test frames
explicitly lack that reference because the original diagnostics saved frame 0.
This export reconstructs previously counted data and performs no fitting,
model prediction, selection, or additional BER evaluation.

This directory owns thin CLI adapters and reproducible figure generators.
Reusable calibration, windowing, training, scoring, and artifact logic belong
in `src/optical_comm/`; a new CLI should call those modules rather than copy
numerical methods. Saved figures and report text belong in `reports/`, and
compact run data belong in `artifacts/`.

`simulate_kerr_films.py` saves qualified 80 km dual-polarization Manakov pulse
trajectories for three controls: Kerr only, dispersion only, and both together.
`render_kerr_films.py` uses those complex fields for three native Blender/Cycles
films, and `build_kerr_explanation.py` creates the movie-centered offline
Japanese page from the completed numerical and render manifests.

```bash
# Rebuild the explanation from the saved three films; no rendering or propagation.
.venv/bin/python scripts/build_kerr_explanation.py

# Isolate a numerical reproduction from the delivered artifacts.
.venv/bin/python scripts/simulate_kerr_films.py --config configs/kerr_films.json --output .work/kerr-physics-reproduction

# Render three movies from that reproduction into scratch output.
.venv/bin/python scripts/render_kerr_films.py --mode render --data-dir .work/kerr-physics-reproduction --output-dir .work/kerr-films-reproduction --scratch-dir .work/kerr-films-reproduction-frames
.venv/bin/python scripts/build_kerr_explanation.py --physics-run .work/kerr-physics-reproduction --renders .work/kerr-films-reproduction
```

The simulation is a separate deterministic Gaussian-pulse experiment, without
EDFA/ASE, training, or BER scores. Rendering requires Blender/OptiX, FFmpeg,
FFprobe, Pillow, and a licensed Japanese font; use the project environment
explicitly. The render manifest's `python3` command assumes that environment
is active. See the [render guide](../docs/kerr-rendering.md) for preview, saved
scenes, backend requirements, and options. The explanation CLI uses only Python
standard-library modules, validates data and media hashes, and refuses
incomplete inputs. Delivered films and provenance are linked from the
[offline page](../reports/animations/kerr_physics/index.html) and its
[README](../reports/animations/kerr_physics/README.md).

The historical `animate_kerr.py` builds a native, editable Blender scene and a 30-second
Japanese MP4 explaining Kerr self-phase modulation. It renders an analytic
scalar teaching example, separate from the measured DP/Manakov link results.
Run it with ordinary Python plus Blender and FFmpeg; the project environment
and research runs are not needed. The two packed fonts and their licenses,
commands, parameters, checks, and output hashes are documented with the
[saved animation](../reports/animations/kerr/README.md).

```bash
python3 scripts/animate_kerr.py --blender blender --preview
python3 scripts/animate_kerr.py --blender blender
```

Use `--blender` with a Windows `.exe` path from WSL if necessary. `--font` and
`--latin-font` select Japanese and math fonts. Chapter stills and the two-second
preview go to `.work/kerr-render/`; the full MP4, compressed `.blend`, poster,
and manifest go to `reports/animations/kerr/`.

`simulate_spatial_kerr.py` is the thin entry point for the circular-core CW
field API. Install the optional mesh dependency with
`uv sync --extra spatial --extra gpu --group dev` on the CUDA 13 machine; the
normal CPU tests do not require CUDA. A backend override is explicit and saved
in provenance. The configured production problem is expensive on CPU.

```bash
.venv/bin/python scripts/simulate_spatial_kerr.py --config configs/spatial_kerr.json --output artifacts/spatial_kerr --scratch .work/spatial-kerr-production-bulk
.venv/bin/python scripts/simulate_spatial_kerr.py --config configs/spatial_kerr.json --output .work/spatial-reproduction --scratch .work/spatial-reproduction-bulk
.venv/bin/python scripts/simulate_spatial_kerr.py --output artifacts/spatial_kerr --verify-only
```

Output and bulk scratch must be empty; previous evidence is never overwritten.
In-repository bulk data must be inside ignored `.work`. `--skip-qualification`
is an explicitly marked exploratory export and does not claim qualification.
The CLI saves all four registered conditions, compact complex sections, all
seeded/masked power-flow lines, and fixed iso-intensity meshes for the primary
two cases. Mesh extraction uses the optional scikit-image public API and
physical spacing; Blender does not solve Kerr physics. The NPZ coordinate
contract is in the run manifest and `artifacts/spatial_kerr/README.md`.

`render_spatial_kerr.py` reads those completed numerical artifacts without
propagating a new field. It draws fixed 0.1/0.3/0.6 iso-intensity surfaces,
all 25 saved power-flow seeds, and joint-masked principal Kerr phase slices.
The three delivered six-second, 24 fps, 1080p films and packed scenes are in
`reports/animations/spatial_kerr/`. They observe a stationary educational CW
field with hypothetical coefficients and illustrative emission/colors, not a
physical-power SMF pulse. The final entry point uses target-facing intro keys,
explicit active scene evaluation, and runtime OptiX checks.

```bash
.venv/bin/python scripts/render_spatial_kerr.py --mode inspect
.venv/bin/python scripts/render_spatial_kerr.py --mode posters
.venv/bin/python scripts/render_spatial_kerr.py --mode render
.venv/bin/python scripts/render_spatial_kerr.py --mode verify-scene
```

Use `--data-dir` for external numerical artifacts and `--output-dir` for an
isolated media reproduction. The saved scenes pack their Japanese font and
float textures. See the [spatial rendering guide](../docs/spatial-kerr-rendering.md)
for coordinate units, camera observation, fixed scales, and scientific limits.
The guide also documents `repair-cameras`, a fixed-scope repair of the known
original camera bugs, with original/current provenance and all frame hashes.
It is separate from normal full reproduction and does not implement general
resume. Encoder binary paths/versions/hashes are saved for the run where they
were queried. The superseded short preview is kept only in ignored history.

[`build_spatial_kerr_report.py`](build_spatial_kerr_report.py) builds the
[offline Japanese explanation](../reports/animations/spatial_kerr/index.html)
from the saved circular-core fields and three completed films. It generates
the four-case intensity comparison, endpoint intensity/phase figures, mobile
variants, HTML, and provenance. NumPy, Matplotlib, a local Japanese font, and
FFprobe are required.

```bash
.venv/bin/python scripts/build_spatial_kerr_report.py --physics-run artifacts/spatial_kerr --renders reports/animations/spatial_kerr
.venv/bin/python scripts/build_spatial_kerr_report.py --figures-only --physics-run artifacts/spatial_kerr --output-dir .work/spatial-figures-reproduction
```

The full mode requires production movies and final media hashes; previews are
rejected. `--figures-only` reads the saved numerical fields without requiring
movies. See the [explanation README](../reports/animations/spatial_kerr/README.md)
for output paths, fixed display scales, offline viewing, and local HTTP saving.
The [explanation manifest](../reports/animations/spatial_kerr/explanation_manifest.json)
retains ordinary browser checks for the delivered revision. Regeneration does
not run those browser checks automatically; `--browser-record` accepts only
a successful record for the same final HTML, movies, and render manifest.
