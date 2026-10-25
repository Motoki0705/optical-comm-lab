# ESN long-haul equalization study

This workspace owns the existing software reservoir experiment. `experiment.py`
owns independent frame/seed orchestration, validation-only candidate selection,
held-out scoring and run integrity. `reservoir.py` owns preprocessing, linear and
ESN equalizers and their JSON/NPZ checkpoints. Physical simulation, DSP, alphabet
mapping, and metrics remain in `optical_comm`.

The original [experiment plan](docs/experiment-plan.md) and
[learning methods](docs/learning-methods.md) retain the scientific contract.
The repository-wide [requirements](../../docs/requirements.md) preserve the
original user text. This refactor changes ownership and source provenance, not
the numerical settings, model arrays, seeds, counted denominators, or results.

From a checkout or an installed wheel:

```bash
python -m experiments.esn_longhaul --run-id pilot-check --dry-run
python -m experiments.esn_longhaul --config experiments/esn_longhaul/configs/full.json --run-id final-new-source --dry-run
python -m experiments.esn_longhaul.audit_run --run /path/to/new-source-run
python -m experiments.esn_longhaul.export_waveform_diagnostics --run /path/to/new-source-run --output-dir /empty/output
python -m experiments.esn_longhaul.visualize_concepts --help
python -m experiments.esn_longhaul.visualize_results --help
python -m experiments.esn_longhaul.build_report --help
```

The default packaged [pilot configuration](configs/pilot.json) is exploratory;
the [full configuration](configs/full.json) remains the qualified study setting.
Outside a checkout, pass an absolute configuration path or use the packaged
pilot default. Artifact/output paths are explicit or relative to the working
directory. Report builders additionally need the documented external report,
qualification, verification, and font inputs; the wheel does not bundle saved
research results.

`optical_comm.experiment` and `optical_comm.reservoir` are explicit compatibility
module aliases to these implementations, including the existing checkpoint API.
The six same-named root `scripts/` files are thin compatibility CLI adapters.
`configs/pilot.json` and `configs/full.json` are filesystem symlinks to the single
canonical JSON files here; they open normally in a checkout. GitHub displays the
symlink target, so documentation links point to the canonical files. The old
`docs/experiment-plan.md` and `docs/learning-methods.md` URLs remain redirect pages.

## Historical source provenance

New runs hash the actual shared and study source files, with their new paths.
Moving code changes the manifest, so strict audit and resume reject runs from
the previous source layout. No old hash is synthesized and no historical
manifest, model, output, denominator, report, or generation SHA is rewritten.
JSON/NPZ checkpoints contain arrays and metadata, not pickled Python import paths.

For strict audit or regeneration of `artifacts/final-1600km-20261012`, use the
pre-refactor checkout below. Its eight recorded experiment source hashes match
the original execution `daa52bf810f213b057fada90e8c1b903509c766c`, while it also
contains the later saved-data audit and waveform exporter:

```bash
git worktree add --detach .work/historical-esn 8a1475620fa5fa61e843f03bdbce48e1ce54419a
cd .work/historical-esn
# Use the existing absolute interpreter; force imports from this isolated tree.
PYTHONPATH="$PWD/src" /home/kamimura/projects/optical-comm-lab/.venv/bin/python scripts/audit_run.py --run artifacts/final-1600km-20261012 --output .work/final-audit.json
PYTHONPATH="$PWD/src" /home/kamimura/projects/optical-comm-lab/.venv/bin/python scripts/export_waveform_diagnostics.py --run artifacts/final-1600km-20261012 --output-dir .work/waveform-diagnostics-reproduction
```

Reproduction into fresh directories uses the original config/seeds and requires
the recorded compatible backend/environment. The strict saved-data audit does
not rerun propagation or training. Full reproduction is a new measurement and
must not replace the delivered evidence. Run historical report generators from
that same checkout with explicit isolated output paths. Spatial Kerr manifests
also hash `pyproject.toml` and `uv.lock`; strict verification of historical Kerr
sources therefore uses this checkout as well, even though those scientific
modules and their settings were not moved.
