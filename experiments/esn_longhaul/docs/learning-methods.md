# Learning and held-out comparison

The main study trains a separate receiver at each launch power. It assesses
compensation within that power condition; it does not establish generalization
to a power, distance, modulation, terminal impairment, or hardware device absent
from training. The final result can be improvement, no improvement, or worse BER.

## Frames and train-only transformations

`experiment.frame_seeds` assigns independent bit and ASE streams to each whole
training, validation, and test frame. The namespace includes root seed, launch
power, split, frame index, and stream. Increasing a training-frame count does
not change test seeds. The recipe, resolved physical configuration, environment,
and source hashes are saved. Frames are split **before** extracting windows.

Each polarization is calibrated using a scalar forward channel estimate from
training symbols only:

\[
h_p=\frac{\sum_{\mathrm{train}}r_{t,p}s^*_{t,p}}
           {\sum_{\mathrm{train}}|s_{t,p}|^2},\qquad
\tilde r_{t,p}=r_{t,p}/h_p.
\]

This convention avoids the noise-dependent constellation shrinkage introduced
by fitting an inverse least-squares regression to AWGN samples. It removes a
mean phase/gain, not symbol-specific phase errors. `test_reservoir.py` checks
the calibrated conventional receiver against exact Gray-16QAM AWGN BER.
The two complex coefficients are frozen before validation/testing. DBP gets
its own forward-gain calibration on the same training frames. Metrics never fit
a gain or phase to held-out labels.

The four real components `[XI, XQ, YI, YQ]` are standardized using means and
standard deviations of calibrated valid **training** symbols. The same frozen
transformation feeds both learned receivers. A context window is accepted only
when every sample is valid and belongs to its own frame. The reservoir starts
at zero for every frame and every disconnected valid segment. Context-invalid
samples and the first declared washout samples of each segment are excluded
from fitting and scoring. All four receivers use the same final mask.

## Frozen reservoir and fitted readout

For standardized context input \(u_t\), the software echo-state network is

\[
x_t=(1-\ell)x_{t-1}+\ell\tanh(W_{\rm in}u_t+W_{\rm rec}x_{t-1}+b).
\]

Input weights, sparse recurrent weights, and biases are generated once and
remain fixed. Bernoulli sparse connections have uniform signed weights;
the recurrent matrix is scaled using its measured eigenvalue spectral radius.
Input scale, leak, radius, units, density, bias, and seed are saved. Spectral
radius alone is not claimed to prove the echo-state property. Frame reset and
washout are explicit even for slowly decaying state.

The readout features are `[1, u_t, x_t]`. The four-dimensional target is the
real/imaginary residual \(s_t-\tilde r_t\). Only the linear readout \(A\) is fit:

\[
\min_A \frac{1}{n}\|Y-ZA\|_F^2+
\lambda\|A_{\mathrm{nonconstant}}\|_F^2.
\]

The delivered output is \(\tilde r_t+\operatorname{complex}(z_tA)\). The direct
input features allow a small linear correction without requiring the reservoir
to reproduce it. The comparator uses `[1, u_t]`, the same residual targets,
scales, context, ridge objective, and scoring mask. Its ridge is also selected
on validation data. A lower EVM or training loss does not establish a BER gain.
An independent synthetic nonlinear-memory task verifies that the fitted ESN
can learn a nonlinear residual, separately from optical-link performance.

The full configuration starts with 128 units, 5% recurrent density, fixed bias
scale 0.2, and six combinations of radius/leak/input scale. Four ridge values
are tried for each. Train sufficient statistics and validation design features
are reused across ridge candidates, so a ridge sweep does not repeatedly
compute reservoir states. The sparse recurrence uses Numba on the CPU; channel
propagation and DBP can use CuPy. This is a software feasibility study, without
a hardware energy or throughput claim.

## Selection, latency, and test scope

The selection rule is fixed: lowest pooled **direct-count validation BER**,
then lowest validation EVM, smallest stored-array byte count, and lexicographic
candidate ID. The candidate table and chosen IDs are written before generating
test frames. Linear and ESN checkpoints include all transforms, fixed weights,
and readouts; the runner checks exact prediction equality after reload.
No retraining on validation labels is performed before testing.

The pilot uses two past and two future symbols; the full configuration
uses four past and four future symbols. The latter needs four future symbols,
or 125 ps at 32 GBd, before a symbol prediction can be released. Reservoir
memory also summarizes past windows. That context delay excludes block EDC,
frame buffering, numerical processing, and device implementation latency.
The same explicit future context is available to the ridge linear comparator.

The conventional receiver is EDC/matched-filter symbol sampling followed by
train-calibrated gain/phase. The additional ridge linear receiver controls for
explicit memory. ESN and numerical DBP are evaluated on the same held-out
symbols. DBP includes backward propagation of the noisy waveform with the
configured inverse steps; it is a computational reference, not a guaranteed
upper bound. Numerical qualification of the forward channel is a separate
task; the training runner does not label its settings converged. The retained
full settings are 4 samples/symbol, 1 km steps, 65,536 symbols/frame, and 1024
excluded symbols at each edge, as qualified in
[numerical-qualification.md](../../../docs/numerical-qualification.md). Together with context
and washout, five held-out frames count 2,534,080 bits per receiver and power.

BER is counted from Gray-demapped bits with explicit `bit_errors` and `n_bits`.
Clopper-Pearson bounds quantify finite counts under an independent-bit model.
Nonlinear errors may correlate, so every independent test-frame BER and its
spread are also retained. Zero measured errors has a nonzero upper confidence
limit. EVM uses the actual transmitted-label energy in the scoring mask;
effective SNR is `-20 log10(EVM)`. Energy is pooled exactly across frames.

Timing records distinguish channel generation, model selection, DBP training
calibration, and additional test receiver processing. The conventional EDC
operation already occurs in shared channel generation, so extra receiver times
are not end-to-end comparisons. Disk checkpoint bytes and in-memory model-array
bytes are saved separately. Fixed first-frame, evenly spaced valid indices
supply all constellation examples, avoiding receiver-dependent sample choice.

## Reproduction and artifact boundaries

Use the commands in `scripts/README.md` and versioned configurations in
`configs/`. Run names start with `pilot-` or `final-`; a pilot is exploratory and
is never pooled into final BER. Resuming verifies resolved config, executable
source, Python/package versions, backend, and the saved per-power file hashes.
Changed settings require a new run ID. A completed power is not regenerated.
An explicitly resumed partial power is regenerated with identical recipes.
Raw bulk traces are reproducible from those recipes and are not committed.

## Sources

- H. Jaeger, *A Tutorial on Training Recurrent Neural Networks, Covering BPTT,
  RTRL, EKF and the “Echo State Network” Approach*:
  https://www.ai.rug.nl/minds/uploads/ESNTutorialRev.pdf — frozen reservoir,
  washout, leak/radius/input-scale tuning, and linear readouts.
- P. J. Freire et al., *Neural Networks-based Equalizers for Coherent Optical
  Transmission: Caveats and Pitfalls*, https://arxiv.org/html/2109.14942 —
  optical equalization, data isolation, metrics, and complexity cautions.
- E. Ip and J. M. Kahn, *Compensation of Dispersion and Nonlinear Impairments
  Using Digital Backpropagation*:
  https://www-ee.stanford.edu/~jmk/pubs/digital.backpropagation.jlt.10-08.pdf.

The precise grids, forward-calibration convention, tie rule, and context budget
above are this study's declared choices, rather than claims of an optimal ESN.
