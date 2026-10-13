# Kerr film model and data contract

The revised films use actual distance-resolved solutions of the repository's
lossy dual-polarization Manakov model. This illustrative **isolated pulse** is
a separate calculation from the accepted 1600 km, average-power DP-16QAM study.
It has no symbols, learning, receiver, EDFA, or ASE, and produces no new BER
measurement. [Issue #15](https://github.com/Motoki0705/optical-comm-lab/issues/15)
controls the physical model/data; scene appearance and movie production have
their own scope.

## Fixed physical input

| Quantity | Value and meaning |
| --- | --- |
| Input envelopes | `Ax=Ay=sqrt(Ppeak/2)*exp(-t²/(2*T0²))` |
| XY peak power | 50 mW instantaneous total; 25 mW per polarization at t=0 |
| T0 | 20 ps intensity 1/e half-width; intensity RMS width is T0/sqrt(2) |
| Wavelength | 1550 nm |
| Fiber length and loss | 80 km; 0.2 dB/km, hence 16 dB total power loss |
| Dispersion | D=17 ps/(nm km), beta2 approximately -21.68 ps²/km |
| Kerr coefficient | gamma=1.3 /(W km); XY Manakov factor 8/9 |
| Full solver grid | 8192 samples, dt=0.244140625 ps, 2 ns periodic window |
| Longitudinal solver step | 0.25 km; snapshots z=0,1,...,80 km |
| Amplifier/noise | None; only `propagate_fiber`, never `propagate_link` |

`symbol_rate_hz=32e9` and `sps=128` in `LinkConfig` define the sampling rate
4.096 THz here. They do not imply a 32 GBd pulse train. RRC, guard-symbol,
EDFA/noise-figure fields are inactive when this isolated input is passed to
`propagate_fiber`. The pulse envelope is in the retarded time coordinate moving
with the group velocity; a film's longitudinal motion, fiber dimensions and
enlarged carrier period are explanatory geometry rather than measured scales.
PMD, higher dispersion orders, Raman response, self-steepening, terminal noise,
and laser fluctuations are omitted.

Three cases share the input, loss, sampling, distance and view limits:

| Case ID | D [ps/(nm km)] | gamma [1/(W km)] | Comparison |
| --- | ---: | ---: | --- |
| `pure_spm` | 0 | 1.3 | Loss plus nonlinear phase without dispersion |
| `dispersion_only` | 17 | 0 | Loss plus linear temporal spreading |
| `combined` | 17 | 1.3 | Interaction of dispersion and instantaneous Kerr |

The equation, attenuation conversion and symmetric split are those in
[channel-methods.md](channel-methods.md). No extra hand-authored phase, pulse
width, chirp, or spectral deformation is used to make an effect visible. The
input power/width were selected before numerical generation; refinement probes
do not tune physical parameters against the desired image.

## Analytic limits and measured verification

For D=0 the exact field is
`A(z,t)=A(0,t)*exp(-alpha*z/2 + i*(8/9)*gamma*Pxy(0,t)*Leff(z))`, where
`Leff=(1-exp(-alpha*z))/alpha`. This input predicts a center phase about 1.22 rad
at 80 km. Attenuation reduces power while preserving the normalized temporal
envelope in this case. Nonlinear phase accumulated early in the span remains
after the instantaneous intensity has decayed.

For gamma=0 the implemented Fourier multiplier is
`exp(+i*beta2*omega²*z/2)`. Its infinite-domain Gaussian solution is
`A=sqrt(Ppeak/2)/sqrt(q)*exp(-t²/(2*T0²*q)-alpha*z/2)`, with
`q=1-i*beta2*z/T0²`. The intensity 1/e half-width is
`T0*sqrt(1+(beta2*z/T0²)²)`, about 89 ps at 80 km. The comparison checks this
**complex** field, including phase sign, rather than fitting an output intensity.

`qualification.json` records the actual outcomes, including a failed status if
a threshold is exceeded. The criteria are fixed in the versioned source:

- Exact lossy Kerr and complex Gaussian dispersion fields: relative RMS <1e-10.
- Expected exponential energy loss and Parseval: maximum relative error <1e-10.
- Segmented snapshots versus one whole-span call at the same integration step:
  relative field RMS <1e-10.
- Combined case at 0.5, 0.25 and 0.125 km: production/fine field RMS <1e-4;
  successive-difference ratio in (3.5,4.5), consistent with second order.
- Independent doubled time rate at fixed 2 ns period and doubled period at
  fixed time rate: common-field RMS <1e-8; expanded-window energy outside the
  original window <1e-10.
- Saved display data: discarded Fourier energy <1e-10; complex64 field RMS
  <1e-6; stored-field energy error <1e-6; valid phase error <1e-5 rad;
  valid chirp error <1e-4 GHz.

No fitted phase or amplitude alignment is applied to numerical comparisons.
The Richardson continuum estimate is explicitly an estimate under the observed
second-order behavior, not an additional finest-grid solution. The doubled-grid
and period probes are new propagations of the same physical input. They are not
interpolations of the baseline output or matched forward/inverse cancellation.
These ordinary checks verify this deterministic pulse and do not prove the
continuum limit, hardware accuracy, or communications receiver performance.
The independent-validator count for this follow-up is zero.

## Saved film data

Each `artifacts/kerr-physics/{case}.npz` loads with `allow_pickle=False`. The full
2 ns time window is retained; Fourier truncation before a factor-four reduction
prevents aliasing. The display fields use complex64 and their observables use
float32 after high-precision calculation. Axis, energy and attenuation values
are float64, masks are bool. The measured export errors are retained per case.
Full complex128 trajectories are reproducible from the config and are not
committed as bulk traces; their digests and scalar metrics are saved.

| Key | Shape | Unit/meaning |
| --- | --- | --- |
| `z_km` | (81,) | km |
| `t_ps` | (2048,) | ps; -1000 to 999.0234375, dt=0.9765625 ps |
| `A_xy_sqrtW` | (81,2048,2) | Complex envelope, sqrt(W), polarization order X,Y |
| `power_xy_W` | (81,2048,2) | Per-polarization instantaneous W |
| `phase_xy_rad` | (81,2048,2) | Unwrapped envelope phase, rad; masked values NaN |
| `delta_omega_rad_s` | (81,2048,2) | Instantaneous optical frequency offset, rad/s; NaN in masked tails |
| `phase_valid` | (81,2048,2) | Explicit power-threshold validity mask |
| `frequency_offset_Hz` | (2048,) | Increasing physical optical frequency-offset axis |
| `spectrum_xy_J_per_Hz` | (81,2048,2) | Pulse spectral energy density per polarization |
| `energy_xy_J` | (81,2) | Per-polarization pulse energy, J |
| `attenuation_power_factor` | (81,) | Expected power loss `exp(-alpha*z)` |

For the physical carrier convention
`E=Re[A(t)*exp(i*(k*z-omega0*t))]`, the chirp is
`delta_omega=-d(arg(A))/dt`. The spectral optical offset is therefore the
negative of NumPy's FFT frequency. Chirp is calculated with a spectral field
derivative as `-Im(conj(A)*dA/dt)/|A|²`, avoiding differentiation of wrapped
phase. At each z, phase/chirp are masked where per-polarization power is at most
`1e-6*max_t(Pxy)/2`. Only contiguous valid lobes are unwrapped; the central
lobe is anchored at t=0 with phase continuous across z. Unrelated lobes across
a masked zero have local phase anchors. Power and spectrum remain unmasked.

The spectral density is `|dt*FFT(A_pol)|²`: W*s² = **J/Hz**. Its frequency
integral equals pulse energy. Calling it W/Hz would require an explicitly
defined averaging period, which these pulse views do not use. Orthogonal XY
powers and spectra are added separately; `FFT(Ax+Ay)` would incorrectly treat
orthogonal polarizations as coherently interfering.

Raw arrays retain loss. A plot or scene may divide power/spectrum by the saved
attenuation factor to compare shapes only if it displays a loss-compensated
label and uses a common input-based normalization. No curve is independently
rescaled to its own maximum. The suggested common views are ±300 ps and ±60 GHz;
`derived_metrics.json` gives the energy fraction outside each view at every z.
Those limits do not crop the saved data. Between the 1 km snapshots, animation
interpolation is a rendering choice and must not be described as an additional
SSFM observation.

`derived_metrics.json` contains `cases->{case}` with `z_km`, `rms_time_ps`,
`rms_frequency_GHz`, `peak_xy_power_W`, `center_phase_rad`, `energy_xy_J`,
`attenuation_power_factor`, both view-tail fractions, and the full solver-field
digest. Time/frequency widths are energy-weighted standard deviations. CSV
retains those trajectories; the generated artifact README provides measured
80 km endpoints. The manifest hashes all numerical source/configuration inputs
and all output files except itself; movie/explanation manifests should hash the
physical manifest externally. Runtime and environment are observations.

## Regeneration and sources

```bash
.venv/bin/python scripts/simulate_kerr_films.py --config configs/kerr_films.json --output .work/kerr-physics-reproduction
.venv/bin/python -m pytest tests/test_kerr_pulses.py -q
```

The output directory must be empty. The CLI checks sources did not change during
generation, retains qualification and test failures, and refuses to label a
failed run complete. The main experiment modules and historical results are
inputs only; the new visualization subpackage has its own scoped provenance.

SPM frequency broadening is established in
[Stolen and Lin, Physical Review A 17, 1448 (1978)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.17.1448),
DOI 10.1103/PhysRevA.17.1448. Averaged polarization coupling follows
[Marcuse, Menyuk and Wai, JLT 15, 1735 (1997)](https://doi.org/10.1109/50.622902),
DOI 10.1109/50.622902, with the existing repository's 8/9 convention. Nonlinear
interaction with anomalous dispersion underlies
[Hasegawa and Tappert, Applied Physics Letters 23, 142 (1973)](https://doi.org/10.1063/1.1654836),
DOI 10.1063/1.1654836; this lossy Gaussian is not claimed to be a stationary
soliton. Numerical step selection and SSFM accuracy are discussed by
[Sinkin et al., JLT 21, 61 (2003)](https://umbc.edu/photonics/members/publications/PdfPapers/PAJ167.pdf),
DOI 10.1109/JLT.2003.808628. These sources support model foundations; the saved
qualification supplies the evidence for this particular input and grid.
