# Numerical qualification

These runs qualify the numerical receiver and discretization. They are separate
from model validation, held-out BER testing, and the one independent validator
evaluation. The generation settings and seeds are in
[`configs/numerical.json`](../configs/numerical.json); compact measurements and
provenance are in [`artifacts/numerical-qualification`](../artifacts/numerical-qualification).

## Receiver controls

The full pulse-shaping, one-span loss/gain, ASE injection, matched-filter and
sampling chain is exercised with dispersion and Kerr disabled. The launch power
is calculated from an independently evaluated EDFA PSD and the requested
symbol-energy-to-noise ratio, rather than adjusted to the observed errors. An
independent 4PAM decision-bin integration enumerates all Gray labels and Hamming
distances; the control does not call the package's closed-form BER function.
Terminal normalization uses the known launch power and no fitted calibration.

Each AWGN point uses three independently generated 65,536-symbol XY frames,
excluding 256 symbols from each end: 1,560,576 counted bits per point. The
measured results are:

| Es/N0 (dB) | Measured BER | Independent theory | Errors / bits | Measured SNR (dB) |
| --- | --- | --- | --- | --- |
| 0 | 0.287883 | 0.287280 | 449,264 / 1,560,576 | -0.02114 |
| 8 | 0.097968 | 0.098171 | 152,886 / 1,560,576 | 8.01096 |
| 16 | 0.001760 | 0.001791 | 2,747 / 1,560,576 | 16.01086 |

All three theoretical values lie inside the predeclared 99.9% exact counting
intervals, and all SNR deviations are below 0.1 dB. The 95% intervals and every
replicate are retained in `controls.json` and CSV files. Counting intervals use a
binomial independent-bit assumption; they do not account for all within-symbol
or nonlinear-channel dependence. Replicates are retained rather than treating
pooled bits as proof of independent errors.

The back-to-back control counted 0 errors in 520,192 bits, with EVM 0.001043
(finite RRC truncation) and a nonzero 95% BER upper counting limit of 0.00000709.
The physical-module analytic tests are recorded separately.
`physical_tests.json` is the physical implementation worker's reported
19-test result, with its command, software versions and methods-document path.
It does not certify full-link convergence. See
[`channel-methods.md`](channel-methods.md) for physical units, analytic limits,
ASE convention and the tested CPU/GPU correspondence.

## Independent numerical refinement

The 1600 km link is measured at 0 and +6 dBm XY-total launch power, with and
without ASE. Step refinement is 1, 0.5 and 0.25 km at 4 samples/symbol; sampling
refinement compares 4 and 8 samples/symbol at fixed 0.5 and 0.25 km. The accepted
initial/reference comparison, 4 sps / 1 km versus 8 sps / 0.5 km, is also retained.
There is one deterministic probe and three independent noisy probes per power,
each with 65,536 symbols/polarization. All use complex128.

Each configuration has a separate independently generated calibration frame.
Per-polarization forward least squares fits
`h = sum(RX * conj(TX)) / sum(abs(TX)**2)` on its valid calibration symbols;
evaluation symbols are divided by this fixed `h`. This corrects a deterministic
mean phase/gain using training data and avoids the noise-dependent contraction
from inverse/MMSE fitting. No evaluation or final test labels are used in fitting.
Raw uncalibrated SNR and BER are retained so this convention is visible.

Within each refinement pair, TX bits and ASE seeds are identical. ASE occupies
the same fixed physical two-sided 128 GHz band. Ordered spectral-frequency
draws generate the same continuous noise realization at shared physical sampling
times. Changing samples/symbol therefore does not accidentally change total ASE
power. Raw received complex waveforms are compared on common time instants,
without phase alignment fitted to evaluation labels; calibrated symbol-pair EVM
is also recorded.

The primary numerical target is an effective-SNR change of at most 0.1 dB in
every replicate, plus overlapping 99.9% counting BER intervals. Paired bit-error
discordance and its exact symmetric binomial test are retained as a more sensitive
diagnostic; their independence assumption also limits interpretation. Qualification
does not rest on matched-grid forward/inverse cancellation: no-ASE DBP additionally
uses an independently finer 0.125 km inverse, and forward waveform differences
remain available regardless of inverse behavior.

This refinement approach follows the practical recommendation to reduce the SSFM
step until results stabilize in
[Serena et al., JLT 2020](https://www.tlc.unipr.it/bononi/ricerca/SLMB_jlt_20.pdf).
That paper addresses wider systems than this single-wavelength study; its broad
modeling conclusions are not a validation of the present implementation.

All 28 condition/pair summaries passed, including every individual noisy
replicate and the retained paired-bit diagnostic. There were 56 paired
comparisons in total. The initial 4 sps / 1 km setting compared with the finest
8 sps / 0.25 km reference gave:

| Power (dBm) | ASE | Maximum absolute SNR change (dB) | Mean raw waveform RMS difference | Mean symbol-pair EVM |
| --- | --- | --- | --- | --- |
| 0 | off | 0.000213 | 0.014855% | 0.0000930 |
| 0 | on | 0.000092 | 0.202355% | 0.0001852 |
| +6 | off | 0.000238 | 0.019907% | 0.0001122 |
| +6 | on | 0.000309 | 0.405385% | 0.0002077 |

The largest SNR drift across **all** refinement pairs was 0.000308503 dB.
The largest retained raw-waveform discrepancy is nonzero; qualification
establishes stability of the scored receiver metrics at these settings rather
than identity of broadband fields. Further 16-sps refinement was not required
by the predeclared receiver-metric criterion. No step/sampling candidate failed.
The independent 0.125 km inverse returned no-ASE EVM near 0.00104–0.00105,
where finite RRC truncation also contributes; this is a supplementary check,
not the only accuracy estimate.

Measured refinement-stage wall time was 400.24 s on the recorded RTX 5060 Ti.
Per-frame forward timings span roughly 1.5 s for 4 sps / 1 km to 12.4 s for
8 sps / 0.25 km. Exact calibration/evaluation/inverse timings are retained.

## Frame length and boundary exclusion

The FFT solver propagates a periodic record. A scoring guard excludes symbols;
it does not add a physical guard or remove periodic propagation. Frame lengths
32,768, 65,536 and 131,072 symbols correspond to periods 1.024, 2.048 and 4.096 µs.
The shorter TX records are exact prefixes of the longest record. Each shorter
result is compared with the same TX positions in the longest result, rather than
with unrelated data.

For noisy length comparisons, each span's ASE is generated on the longest period
and the identical temporal prefix is injected into shorter records. The end of
that crop is periodized, introducing a discontinuity. This is a deliberate
boundary perturbation, rather than a fresh shorter-period stationary spectral
ASE realization. At a higher sample rate its periodization can also cause spectral
leakage outside the original ASE band. The recipe, longest noise period and seeds
are explicit in the artifact. Noisy checks use two independent common-prefix
realizations; no-ASE checks use one.

Exclusions 0, 256, 512, 1024 and 2048 symbols are evaluated on the same propagated
frames. One separate training calibration gain is shared across all lengths and
exclusions for a power/ASE condition, so refitting cannot hide a boundary mismatch.
Edge-distance profiles record the received-symbol difference from the longer
reference. SNR differences always compare exactly the same scored TX region in
both records; a change in the composition of scored symbols is not called a
numerical error.

All 40 length/exclusion summaries passed the predeclared aggregate target and
their paired-bit diagnostics. This includes the unguarded cases: a small
aggregate SNR drift can coexist with a substantial perturbation in a few edge
positions. At 65,536 symbols compared with the same prefix of 131,072 symbols,
the maxima across the measured powers/ASE conditions were:

| Exclusion at each end (symbols) | Maximum absolute SNR drift (dB) | Largest mean symbol-pair EVM | Largest mean raw waveform RMS difference |
| --- | --- | --- | --- |
| 0 | 0.00228724 | 0.00935962 | 4.26953% |
| 256 | 0.000000690 | 0.000006878 | 0.261700% |
| 512 | 0.0000000666 | 0.000000141 | 0.005449% |
| 1024 | 0.0000000367 | 0.0000000619 | 0.002366% |
| 2048 | 0.0000000370 | 0.0000000311 | 0.001242% |

The initial 256-symbol exclusion passed; it is not classified as a failed
candidate. The production recommendation retains **4 sps, 1 km and 65,536
symbols**, and uses the predeclared **1024-symbol exclusion per end** for extra
margin against the localized periodic perturbation. The 32,768-symbol probes
also passed, but the full frame target is retained for learning and BER counting.
No tested candidate failed; 16 sps and periods beyond 131,072 symbols were not
measured. Length/exclusion-stage wall time was 53.04 s.

In the first +6 dBm noisy 65,536-symbol probe, the symbol difference from the
longer record was concentrated at the edge: RMS 0.205 in positions 0–63 from
either end, 0.0756 at 64–127, 0.0152 at 128–255, and 0.0000774 at 256–511.
At 512–1023 it was 0.000000667. `boundary_edge_profiles.csv` retains both noisy
replicates and all conditions. These are differences between two periodic
simulations, rather than error bars on physical hardware.

## Reproduction and scope

The recorded execution used `.venv/bin/python scripts/qualify_numerics.py --stage
controls`, `--stage refinement`, and `--stage boundaries`, with the default output
`artifacts/numerical-qualification`. For a fresh reproduction, run
`.venv/bin/python scripts/qualify_numerics.py --stage all --output
.work/numerical-reproduction`. Keep regenerated outputs separate from the saved
decision and its completed artifact-hash snapshot. A pre-existing output rejects different
configuration/source hashes, preventing silent mixing of numerical versions.
Manifests retain Python, packages, GPU/driver/runtime, source hashes, settings,
seeds, UTC times, measured runtimes and compact central diagnostics. Raw bulk
waveforms are reproducibly regenerated rather than committed.

The control-stage wall time was 1.24 s. A representative full receiver control
was regenerated from its stored configuration and seeds, with exact equality
of its counted errors, SNR and energy metrics (`reproducibility_check.json`).
The qualification-specific CPU tests passed **9/9**; their separate record is
`qualification_tests.json`. The methods/recommendation are normal verification,
not an independent validator evaluation; this worker used zero validator calls.
`production_recommendation.json` records the actual decision, retained initial
settings, empty failed-candidate set, limitations and input-result hashes.

The CPU CI workflow installs the package and runs substantive analytic,
data-isolation and regression tests. CuPy is optional and is not installed by
CI, so the GPU parity test skips there. No full-link experiment runs in CI.

The qualification measures two launch-power anchors and finite numerical
references. It does not prove an infinite-domain continuum limit or hardware
accuracy. Short development pilots remain exploratory even if a numerical metric
is stable; final learning comparisons use independently generated frames and
their own held-out metrics.
