# Scientific and isolation tests

Run `uv sync --group dev` and `uv run python -m pytest`. Enable the optional GPU checks with `uv sync --extra gpu --group dev` on a CUDA 13 machine. GPU parity is skipped when CuPy or an accessible device is absent; CPU verification remains available.

`test_channel.py` compares analytic loss, Kerr and dispersion limits; checks RRC power and alignment; verifies ASE PSD, complex/quadrature/polarization variance, independent seeds, and fixed physical bandwidth across sps; checks full waveform AWGN against independently derived BER; and uses a separately finer forward/inverse solver to avoid relying only on cancellation of identical discretizations. `test_metrics.py` checks uniform Gray mapping, counted bits/masks, exact AWGN limits, and nonzero uncertainty after zero errors.

Add substantive limits or regressions that check externally observable scientific behavior. Do not treat these small numerical-verification problems as proof of full-link convergence. Long sampling/step/frame/guard studies belong in recorded experiment artifacts, with acceptance criteria in [experiment-plan.md](../docs/experiment-plan.md).
