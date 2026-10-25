# Experiment plan and acceptance criteria

## Accepted channel specification

- Single wavelength, dual polarization, coherent ideal transmitter/receiver; uniform Gray-mapped 16QAM; 32 GBd; gross 256 Gbit/s before FEC; 1550 nm.
- SSMF: 80 km/span, 20 spans (1600 km) for primary evaluation; 10 spans (800 km) for development checks. Loss 0.2 dB/km, D=17 ps/(nm km), gamma=1.3 /(W km).
- Each span ends in EDFA: 16 dB gain, 5 dB noise figure, independent circular complex ASE in each polarization. ASE co-propagates through subsequent spans. Define PSD, sampling bandwidth, complex variance, and polarization convention explicitly.
- Launch power sweep: -6, -4, -2, 0, 2, 4, 6 dBm per wavelength, XY total, equally split between polarizations.
- RRC transmit/matched receive filters, roll-off 0.1. Manakov equation solved with symmetric split-step Fourier propagation. Start at 4 samples/symbol and 1 km steps; qualify against 8 samples/symbol and 0.5 km, refining further if necessary.
- Initial terminals have perfect synchronization, no laser phase noise/frequency offset, no PMD, no ADC quantization. This isolates fiber and amplifier impairments.
- Receiver: dispersion compensation, matched filtering, aligned samples, then equalizer and Gray demapping. Preserve complex XY waveforms and symbol correspondence.

## Data and learning contract

- Save independently generated transmitted bits and symbols, raw receive waveforms or reproducible generation recipes, 2-sps post-DSP diagnostic waveforms, and masks/frame boundaries. Large traces can remain ignored; commit compact diagnostics, trained checkpoints, configs, seeds, software versions, and hashes.
- Split by independently generated frames before extracting overlapping windows. Bit and ASE seeds must differ between training, validation, and testing. Calibration and normalization fit only training/pilot data; test labels are metrics-only.
- Preserve contiguous time series and mark frame boundaries/washout; do not carry reservoir state from training into validation/testing.
- Train a software reservoir with frozen recurrent/input weights and a fitted readout. Select its hyperparameters on validation data before looking at final test scores. Record the selected configuration and candidates.
- Primary comparison: conventional linear receiver/equalizer, reservoir equalizer, and numerical digital backpropagation (DBP). DBP is a computational reference, not a guaranteed upper bound.
- Main endpoint is directly counted pre-FEC BER. Also report uncertainty, error counts/bit counts, EVM, constellations, runtime, and model size. A decrease in MSE alone does not establish communications improvement.

## Required controls and acceptance

1. Back-to-back and AWGN: validate power/variance conventions and Gray-16QAM BER against independent theory with sampling uncertainty.
2. Dispersion-only, no ASE: analytical dispersion and inverse compensation agree.
3. Dispersion plus Kerr, no ASE: verify deterministic propagation, analytic Kerr/loss limits, and independent numerical refinement/DBP consistency.
4. Full link: establish BER versus launch-power curves from independent test frames and ASE realizations.
5. Numerical qualification varies step, samples/symbol, frame length, and excluded boundary length separately. Target baseline effective-SNR change <=0.1 dB with statistically consistent BER. Record failures and any refined settings; do not silently treat unqualified settings as converged.
6. Initial generation frames may contain 2^16 symbols/polarization; shorter pilot runs are explicitly exploratory. Final BER evidence should use enough independent bits/errors; when errors are scarce report measurement limits rather than zero risk. Record every actual deviation and its reason.
7. Same fixed config, seeds, and software environment reproduce metrics/artifacts. Tests, CLI commands, a readable visual report, source references, and one independent validator evaluation are delivered through a GitHub PR.

## Source foundations

- OptiCommPy channel models: https://opticommpy.readthedocs.io/en/latest/models.html
- OptiCommPy EDFA convention: https://opticommpy.readthedocs.io/en/latest/generated/optic.models.devices.edfa.html
- Coherent transmission and spectral broadening: https://www.nature.com/articles/s41598-017-12614-x
- NN equalization caveats, data isolation, and metrics: https://arxiv.org/html/2109.14942
- Training sequence pitfalls: https://opg.optica.org/ol/abstract.cfm?uri=ol-45-18-5113
- DBP and ASE: https://www-ee.stanford.edu/~jmk/pubs/digital.backpropagation.jlt.10-08.pdf
- Numerical accuracy and periodic boundaries: https://www.tlc.unipr.it/bononi/ricerca/SLMB_jlt_20.pdf
