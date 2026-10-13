# Physical channel, units, and numerical conventions

This document describes the implementation controlled by [Issue #2](https://github.com/Motoki0705/optical-comm-lab/issues/2). Accepted scientific settings and the convergence acceptance rule remain in [experiment-plan.md](experiment-plan.md). Short analytic tests verify numerical building blocks; they do not establish convergence or held-out performance of the 1600 km study.

## Signal and power

`LinkConfig` defaults to a single wavelength, 32 GBd, 1550 nm, two polarizations, 20 × 80 km SSMF, loss 0.2 dB/km, D = 17 ps/(nm km), and gamma = 1.3 /(W km). Terminals are ideal: exact timing, no PMD, laser phase noise, frequency offset, or quantization.

Each polarization uses uniform Gray 16QAM. Four bits map the real and imaginary 4PAM coordinates using `00 → -3`, `01 → -1`, `11 → +1`, `10 → +3`, divided by sqrt(10). The alphabet has ensemble mean complex-symbol energy 1. Launch dBm denotes the **XY total**, equally divided between polarizations: `P_X = P_Y = 10^(dBm/10) × 10^-3 / 2` W. Fields are in sqrt(W); `abs(E_X)^2 + abs(E_Y)^2` is the instantaneous total power. Random finite frames fluctuate around this nominal power; the implementation does not fit a separate measured-power normalization to each frame.

Transmit RRC taps have roll-off 0.1, finite support of 64 symbol periods by default, and unit discrete energy. An impulse train multiplied by sqrt(sps) gives ensemble mean waveform power 1 before physical scaling. Taps are centered, evaluated analytically at their removable singularities, and applied by periodic FFT convolution. Receive matched filtering uses the conjugate response. Dividing its result by `sqrt(P_X × sps)` returns the original alphabet scale. Sampling starts at the known symbol positions, and the 2-sps diagnostic array has its even entries exactly aligned to the 1-sps output. No transmitted labels are used to fit gain or phase in channel/DSP code.

Finite RRC truncation leaves small residual ISI. Back-to-back tests explicitly tolerate and measure this floor instead of pretending that a truncated filter is exactly Nyquist. Calibration of carrier phase, complex gain, and learned normalization belongs to training/pilot data in the learning layer.

## Manakov propagation

The implemented equation is

$$\partial_z A_{x,y}=-\frac{\alpha}{2}A_{x,y}-i\frac{\beta_2}{2}\partial_t^2 A_{x,y}+i\frac{8}{9}\gamma(|A_x|^2+|A_y|^2)A_{x,y}.$$

Distance is in km, time in s, alpha is the **power** attenuation coefficient in 1/km, beta2 is s²/km, and gamma is 1/(W km). The conversion is `alpha = loss_dB_per_km × ln(10)/10` and `beta2 = -(D × 10^-3) × wavelength_m²/(2πc)`, giving beta2 ≈ -21.6826 ps²/km at the defaults. The 8/9 factor represents averaged polarization nonlinear coupling; setting PMD to zero does not turn this into a scalar Kerr model. The equation and symmetric SSFM convention agree with the [OptiCommPy Manakov documentation](https://opticommpy.readthedocs.io/en/latest/generated/optic.models.channels.manakovSSF.html).

One symmetric step is half dispersion, exact local loss-plus-Kerr, half dispersion. A dispersion distance h has Fourier response `exp(i × beta2 × omega² × h/2)` with omega = 2πf. The middle operator attenuates field amplitude by `exp(-alpha × h/2)` and rotates both polarizations by `(8/9) × gamma × P_XY_at_step_start × L_eff`, where `L_eff = (1-exp(-alpha × h))/alpha`, or h when alpha = 0. This integrated operator gives the exact constant-envelope lossy Kerr limit at every step size. The dispersive/nonlinear interaction remains a second-order splitting approximation. Adjacent dispersion half steps are combined for efficiency. A non-divisible span ends with the remaining fractional step, and inverse propagation reverses this ordering.

Initial settings are 4 samples/symbol (Fs = 128 GHz) and 1 km steps. The API supports 8 samples/symbol and 0.5 km, as well as finer settings. Float64/complex128 are used on CPU and GPU.

## Amplifier ASE and sampling bandwidth

An EDFA follows **each** span, multiplies the field by sqrt(G), and adds independent circular complex ASE in each polarization. Its power gain is derived from span loss, so the default G = 10^(16/10). With noise figure F = 10^(5/10), the per-polarization, two-sided complex-envelope PSD is

$$N_{ASE}=n_{sp}h\nu(G-1)=\frac{h\nu(GF-1)}{2},\qquad n_{sp}=\frac{GF-1}{2(G-1)}.$$

This uses the finite-gain noise-figure convention in the [OptiCommPy EDFA documentation](https://opticommpy.readthedocs.io/en/latest/generated/optic.models.devices.edfa.html). It is **per polarization**: complex noise variance is N_ASE × B per polarization, each real quadrature has half that variance, and XY-total ASE is twice it. A zero-loss span has G = 1 and is modeled without an amplifier or ASE. Every earlier noise realization subsequently propagates through fiber Kerr/dispersion/loss and later amplifiers.

The default **physical two-sided ASE bandwidth B is fixed at 128 GHz**, the rectangular complex-baseband band `[-64 GHz, +64 GHz)`. At 4 sps this occupies the entire sampled Nyquist band; at 8 sps it occupies half of it. Raising sps therefore does not silently double the injected noise power or change the amplifier bandwidth. This finite rectangular noise band is a declared simulation assumption, not a model of a particular EDFA/filter passband.

For period T = N_samples/Fs, FFT spacing is df = 1/T. Active frequency bins receive independent complex normal coefficients of variance `N_samples × Fs × N_ASE`, and inverse FFT normalization produces time-domain variance `N_ASE × df × n_active_bins`. When B/df is integral this equals N_ASE × B. Integer Fourier mode indices handle band endpoints without floating-point boundary ambiguity. Physical-frequency ordering and CPU NumPy RNG permit the same seed and same frame duration at different sps to represent exactly the same sampled bandlimited ASE realization. A separate `SeedSequence` child stream for each span prevents accidental reuse; bit and noise seeds are independent arguments.

`ase_bandwidth_hz=None` deliberately injects full-Nyquist-band noise, whose variance is N_ASE × Fs. This mode changes the physical band when sps changes and must not be mixed into a numerical refinement comparison. Recorded `LinkConfig.to_dict()` includes the bandwidth. At the defaults N_ASE ≈ 8.00 × 10^-18 W/Hz per polarization and full-band per-span variance ≈ 1.02 × 10^-6 W per polarization. The matched-filter AWGN symbol variance is approximately `N_ASE × symbol_rate_hz / P_X` for one amplifier and this wide noise band; finite RRC tails are checked numerically.

## Receiver, DBP, and boundaries

Baseline DSP applies exact inverse accumulated linear dispersion, matched RRC filtering, and known symbol timing. The remaining deterministic nonlinear phase is retained. A train-only fitted constant phase/gain is allowed in the experiment layer and should be applied consistently to comparisons.

`dbp_symbols(frame, step_km=..., backend=...)` uses only the raw receive waveform, physical config, and nominal launch power. It divides out each EDFA gain in reverse order, propagates through the deterministic inverse fiber using negative step lengths, and matched-filters without applying EDC a second time. It does **not** subtract known simulated noise or recover ASE. With ASE present, DBP transforms noisy fields and signal-noise nonlinear interactions; it is a numerical reference, not a guaranteed upper bound. This limitation follows the inverse-channel discussion in [Ip and Kahn, JLT 2008, section III](https://www-ee.stanford.edu/~jmk/pubs/digital.backpropagation.jlt.10-08.pdf).

Waveforms and transmit filtering use a **periodic record** of N symbols. There is no hidden zero padding or repeated pseudorandom short sequence inside a frame. `valid_mask` excludes `guard_symbols` at each end, and frame boundaries must be respected before any overlapping learning windows are extracted. The default exclusion is 256 symbols per side; this is an initial convention, not a qualification that it is sufficient. Vary frame length and exclusion independently (for example 256/512/1024 on a 2^16-symbol frame) along with step size and sps. Periodic copies can affect dispersive/nonlinear interactions beyond an assumed guard. The full-link qualification must target <=0.1 dB effective-SNR change and statistically consistent BER, as stated in the accepted plan.

## Public API and measured metrics

- `LinkConfig.to_dict()/from_dict()` serializes the physical inputs; unknown keys are rejected.
- `simulate_frame(config, launch_power_dbm, n_symbols, bit_seed, noise_seed, backend="numpy")` returns `SimulatedFrame`: `tx_symbols (N,2)`, `tx_bits (N,2,4)`, `tx_indices (N,2)`, `rx_symbols (N,2)`, `rx_2sps (2N,2)`, `rx_waveform (N*sps,2)` in sqrt(W), `valid_mask (N,)`, config, seeds, launch power, and backend. The count N includes excluded symbols and must exceed twice the guard.
- `propagate_fiber()` is a one-span field API without EDFA; `propagate_link()` includes per-span gain and noise. Public arrays are NumPy even when the internal propagation backend is CuPy.
- `compute_metrics(tx_symbols, rx_symbols, tx_bits=None, mask=None)` counts Gray-decision bit errors and bits and returns BER, its counting interval, MSE, EVM, and `effective_snr_db = 10log10(mean|TX|²/MSE)`. It does not fit a calibration.
- `ber_interval(errors,n_bits,confidence=.95)` gives the exact two-sided Clopper-Pearson binomial counting interval. Nonlinear error correlation is not covered by this conditional independent-bit uncertainty; independent frame replication remains necessary. Zero counted errors have a nonzero upper measurement bound.

For circular AWGN with variance N0 per complex symbol, theoretical SNR is Es/N0. Independently summing the Gray 4PAM decision-interval probabilities weighted by bit Hamming distances gives the exact uniform 16QAM expression `Pb = (3/4)Q(a) + (1/2)Q(3a) - (1/4)Q(5a)`, `a = sqrt((Es/N0)/5)`. This includes multi-threshold errors and approaches 1/2 at vanishing SNR; a nearest-neighbor approximation alone is not used for the AWGN control.

## Environment, verification, and timing

CPU reproduction: `uv sync --group dev`, then `uv run python -m pytest`. `.python-version` selects Python 3.12 and `uv.lock` records dependency versions. GPU reproduction with an available CUDA 13 toolkit: `uv sync --extra gpu --group dev`. The optional extra installs only `cupy-cuda13x`; it is not required for CPU work. On a **driver-only** system, additionally run `uv pip install 'cupy-cuda13x[ctk]==14.2.0'` to install NVIDIA runtime components, or install the corresponding CUDA 13 toolkit before enabling the extra. Do not install multiple CuPy variants. These choices follow the [official CuPy installation guide](https://docs.cupy.dev/en/stable/install.html), checked on 2026-10-12. Driver-only components are an alternative environment setup and are not part of this machine's lockfile installation.

The verified machine is Ubuntu/WSL, Python 3.12.3, RTX 5060 Ti 16 GB, CuPy 14.2.0, local toolkit 13.0, driver 13.2, NumPy 2.5.4, and SciPy 1.18.1. CuPy reported one accessible GPU and successfully ran float64 FFTs and the CPU/GPU parity test. `pytest tests/test_channel.py tests/test_metrics.py -q` passed 19 tests, including Gray mapping, back-to-back power/timing, lossy Kerr, analytic dispersion, ASE variance/quadratures/polarization/seeding, fixed-band sps equivalence, full waveform AWGN BER, no-ASE independent step refinement, DBP without a label/noise oracle, and backend parity.

An **exploratory timing-only** run at the default 1600 km link, 0 dBm, bit seed 3101 and noise seed 3109 measured these wall times after GPU warmup. It used the initial numerical settings and did not qualify their convergence or estimate final communications performance.

| Backend | Symbols/polarization | Forward including DSP | DBP including DSP |
| --- | ---: | ---: | ---: |
| NumPy/SciPy CPU | 4096 | 2.66 s | 2.62 s |
| CuPy GPU | 4096 | 0.46 s | 0.39 s |
| CuPy GPU | 65536 | 1.58 s | 1.50 s |

CPU-generated noise, transmit filtering, final receive DSP, and transfers are included. Timing depends on hardware/load and excludes ESN fitting and report generation. The short step-refinement tests are numerical verification of a small deterministic problem; separate recorded experiments must establish the step/sampling/frame/boundary qualification for the full link before conclusions are presented.
