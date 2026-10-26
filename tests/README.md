# Scientific and isolation tests

Run `uv sync --group dev` and `uv run python -m pytest`. Enable the optional GPU checks with `uv sync --extra gpu --group dev` on a CUDA 13 machine. GPU parity is skipped when CuPy or an accessible device is absent; CPU verification remains available.

`test_channel.py` compares analytic loss, Kerr and dispersion limits; checks RRC power and alignment; verifies ASE PSD, complex/quadrature/polarization variance, independent seeds, and fixed physical bandwidth across sps; checks full waveform AWGN against independently derived BER; and uses a separately finer forward/inverse solver to avoid relying only on cancellation of identical discretizations. `test_metrics.py` checks uniform Gray mapping, counted bits/masks, exact AWGN limits, and nonzero uncertainty after zero errors.

Add substantive limits or regressions that check externally observable scientific behavior. Do not treat these small numerical-verification problems as proof of full-link convergence. Long sampling/step/frame/guard studies belong in recorded experiment artifacts, with acceptance criteria in [experiment-plan.md](../docs/experiment-plan.md).

ESN experiment/reservoir tests retain the old public imports to exercise their
compatibility aliases; implementation ownership is now in
`experiments/esn_longhaul`. Audit tests explicitly distinguish synthetic,
current-source fixtures from immutable historical evidence. The historical run
must reject new-source audit/resume without writes. Isolated-copy checks include
both packages, reject mixed checkouts, and detect changed canonical study code.
Waveform output guards protect `experiments/` as well as shared source/configs.
Wheel verification must additionally run outside the checkout, including the
packaged-default dry-run and a tiny CPU run/audit/resume; editable tests alone
do not verify package inclusion.

`test_spatial_kerr.py` verifies the spatial CW solver using independent complex
2D Gaussian and Kerr/loss limits, Bessel LP01 boundary/phase, analytic free
Gaussian streamlines, low-intensity/domain termination, and physical mesh XYZ
coordinates/winding. It also audits small deterministic saved cases and
detects corruption. GPU parity skips without CUDA; mesh/export tests skip if
the optional `spatial` extra is absent. These tests do not establish full
3 mm circular-grid or Maxwell convergence. Measured long GPU refinements and
limitations belong in `artifacts/spatial_kerr/metrics.json`.
