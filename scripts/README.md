# Reproducible entry points

Study implementations live in [experiments/esn_longhaul](../experiments/esn_longhaul/README.md).
The root `run_experiment.py`, `audit_run.py`, `export_waveform_diagnostics.py`,
`visualize_concepts.py`, `visualize_results.py`, and `build_report.py` remain thin
compatibility adapters. Prefer canonical package entry points for new work:

```bash
python -m experiments.esn_longhaul --run-id pilot-check --dry-run
python -m experiments.esn_longhaul.audit_run --run /path/to/new-source-run
python -m experiments.esn_longhaul.export_waveform_diagnostics --run /path/to/new-source-run --output-dir /empty/output
```

The default pilot config ships in the wheel. `--config configs/pilot.json` and
`configs/full.json` still work in a checkout via compatibility symlinks. See the
[canonical config documentation](../experiments/esn_longhaul/configs/README.md)
for counting budgets and the study README for resume/source integrity, report
inputs, and historical `8a14756` checkout commands. Source moves require new run
IDs; strict audit of delivered ESN or Kerr source provenance uses that old tree.

`qualify_numerics.py` stays here as an independent qualification of the shared
channel, with `configs/numerical.json` and `docs/numerical-qualification.md`.
It does not depend on ESN training and is reusable by another communication study.
The following Kerr scripts keep their separate numerical/rendering scopes.

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
