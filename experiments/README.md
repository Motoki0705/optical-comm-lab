# Independent experiment workspaces

Each study owns its orchestration, learning methods, named configurations, and
method documents. The importable `experiments` package ships in the project
wheel alongside `optical_comm`; it is not a set of checkout-only scripts.

| Workspace | Responsibility | Status |
| --- | --- | --- |
| [esn_longhaul](esn_longhaul/README.md) | Frozen-weight ESN/linear training, launch-power sweeps, audit, waveform export, and study report | Existing study, settings and saved evidence preserved |
| [transformer_nlc](transformer_nlc/README.md) | Next Transformer nonlinear-compensation reproduction | Reserved; no scientific implementation yet |

The shared library in `src/optical_comm` owns physical configuration, propagation,
DSP, metrics, and the existing independent Kerr visualization models. Independent
numerical qualification remains in `scripts/qualify_numerics.py`: it checks the
shared channel and must precede any study claiming qualified channel settings.
The Kerr pulse/spatial configurations, scripts, and documents keep their existing
paths and separate scientific scope.

To add a third study:

1. Record its own Issue, original requirements, data/seed contract, and acceptance
   criteria. Reuse the shared numerical APIs before adding another mechanism.
2. Create `experiments/<study>/` with `__init__.py`, a README, named `configs/`,
   and method `docs/` as needed. Own its orchestration and model code there;
   promote a mechanism to `optical_comm` only when it has a reusable API.
3. Expose an explicit `python -m experiments.<study>` entry point with a dry-run.
   Use package-relative imports and package-relative default configuration paths.
   There is no registry or automatic discovery protocol.
4. Add analytic/data-isolation tests and a small CPU smoke run. Build a wheel and
   test it outside the checkout; editable installs alone cannot prove packaging.
5. Use a distinct named artifact directory and retain real source/config/input
   hashes. Do not migrate another study's saved evidence or claim its validator.
   Keep generated results in `artifacts/` and reader-facing results in `reports/`.
6. Update this table and arrange the required structural maintenance scan.

Existing compatibility paths are listed in the ESN README. New studies import
their own modules directly, rather than extending the old ESN adapters.
