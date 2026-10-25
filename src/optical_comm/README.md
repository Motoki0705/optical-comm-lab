# Reusable scientific modules

`config.py` validates and serializes physical settings. `channel.py` owns single-wavelength Manakov propagation, span-local amplifier noise, and DBP. `dsp.py` owns RRC shaping, ideal linear dispersion compensation, and aligned deterministic sampling. `metrics.py` owns the normalized Gray alphabet, directly counted BER, AWGN theory, and EVM/SNR. `reservoir.py` and `experiment.py` are compatibility module aliases; training and orchestration live in [`experiments/esn_longhaul`](../../experiments/esn_longhaul/README.md).

Extend a mechanism beside its owning module. Keep field arrays `(time, 2)` in sqrt(W) inside physical propagation; public symbol arrays `(symbols, 2)` use the unit-mean-energy TX alphabet. APIs return NumPy arrays across backends. Preserve seeds, masks, frame isolation, and training-only calibration. Scientific formulas and periodic/filter/noise conventions are documented in [channel-methods.md](../../docs/channel-methods.md); study configurations belong in `experiments/<study>/configs/`, not hard-coded presets in shared modules. Independent channel qualification and Kerr configurations remain in `configs/`.

The [visualization subpackage](visualization/README.md) owns isolated-pulse field
trajectories and physical rendering observables. It reuses the channel solver
and has separate [Kerr-film provenance](../../docs/kerr-physics.md); it does not
alter the communication experiment's scientific settings or historical inputs.
