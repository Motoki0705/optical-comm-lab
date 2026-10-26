"""Lossy symmetric Manakov SSFM, independent per-span ASE, and deterministic DBP.

Fields have shape (time, polarization) and units sqrt(W), so abs(E)**2 is W.
Distance is km, beta2 is s^2/km, gamma is /(W km), and frequency is Hz.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import numpy as np
from scipy import fft as scipy_fft

from optical_comm.config import LinkConfig
from optical_comm.dsp import pulse_shape, receiver_dsp
from optical_comm.metrics import bits_to_symbols


@dataclass
class SimulatedFrame:
    """A contiguous periodic frame plus explicit boundary exclusions and seeds."""

    tx_symbols: np.ndarray
    tx_bits: np.ndarray
    tx_indices: np.ndarray
    rx_symbols: np.ndarray
    rx_2sps: np.ndarray
    rx_waveform: np.ndarray
    valid_mask: np.ndarray
    config: LinkConfig
    launch_power_dbm: float
    bit_seed: int
    noise_seed: int
    backend: str

    @property
    def valid_slice(self) -> slice:
        guard = self.config.guard_symbols
        return slice(guard, len(self.tx_symbols) - guard)


def _backend(backend: str) -> tuple[Any, Any]:
    if backend == "numpy":
        return np, scipy_fft
    if backend == "cupy":
        try:
            import cupy as cp
        except ImportError as exc:
            raise RuntimeError("CuPy is optional: install with uv sync --extra gpu") from exc
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("CuPy backend requires an accessible CUDA GPU")
        return cp, cp.fft
    raise ValueError("backend must be 'numpy' or 'cupy'")


def _to_numpy(array: Any, xp: Any) -> np.ndarray:
    return np.asarray(array) if xp is np else xp.asnumpy(array)


def edfa_ase_psd(config: LinkConfig) -> float:
    """Two-sided complex-baseband ASE PSD [W/Hz], per polarization.

    Exact finite-gain NF convention: n_sp=(G*F-1)/(2*(G-1));
    N_ASE=n_sp*h*nu*(G-1)=h*nu*(G*F-1)/2.
    With gain=1 the span amplifier is absent and adds no ASE in this model.
    """
    gain = config.edfa_gain
    if gain == 1:
        return 0.0
    planck_j_s = 6.626_070_15e-34
    photon_energy = planck_j_s * 299_792_458 / config.wavelength_m
    noise_factor = 10 ** (config.edfa_noise_figure_db / 10)
    return photon_energy * (gain * noise_factor - 1) / 2


def ase_noise(
    n_samples: int, config: LinkConfig, rng: np.random.Generator
) -> np.ndarray:
    """Independent circular complex XY noise in the fixed physical ASE band.

    A rectangular two-sided band [-B/2,B/2) is placed on the periodic FFT grid.
    Variance is N_ASE*df*n_active_bins per polarization (N_ASE*B when aligned).
    Ordered physical-frequency draws make sps refinements of the same duration
    share the same continuous bandlimited realization with the same RNG state.
    CPU RNG is used for both propagation backends for exact seed correspondence.
    """
    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")
    sample_rate = config.sample_rate_hz
    bandwidth = sample_rate if config.ase_bandwidth_hz is None else config.ase_bandwidth_hz
    # Integer modes avoid floating-point fftfreq boundary round-off accidentally
    # dropping -B/2 or including +B/2 at only one of two sps settings.
    modes = np.arange(n_samples, dtype=np.int64)
    modes[modes >= (n_samples + 1) // 2] -= n_samples
    half_band_modes = bandwidth * n_samples / (2 * sample_rate)
    active = np.flatnonzero((modes >= -half_band_modes) & (modes < half_band_modes))
    active = active[np.argsort(modes[active])]
    spectrum = np.zeros((n_samples, 2), dtype=np.complex128)
    normal = (rng.standard_normal((len(active), 2)) + 1j * rng.standard_normal((len(active), 2))) / np.sqrt(2)
    spectrum[active] = normal * np.sqrt(n_samples * sample_rate * edfa_ase_psd(config))
    return np.fft.ifft(spectrum, axis=0)


def _step_lengths(span_km: float, step_km: float) -> list[float]:
    n_full = int(math.floor(span_km / step_km + 1e-12))
    remainder = span_km - n_full * step_km
    result = [step_km] * n_full
    if remainder > 1e-10 * max(1.0, span_km):
        result.append(remainder)
    return result


def _fiber_on_backend(
    field: Any,
    config: LinkConfig,
    step_km: float,
    inverse: bool,
    xp: Any,
    fft: Any,
) -> Any:
    """Strang split: half dispersion, exact loss+Kerr, half dispersion.

    The integrated nonlinear/loss operator makes the constant-envelope lossy
    Kerr limit exact at every step size, rather than approximating effective L.
    """
    lengths = _step_lengths(config.span_km, step_km)
    if inverse:
        lengths = [-length for length in reversed(lengths)]
    omega = xp.asarray(2 * np.pi * np.fft.fftfreq(len(field), d=1 / config.sample_rate_hz))
    operators: dict[float, tuple[float, float]] = {}
    for dz in set(lengths):
        amplitude_loss = math.exp(-config.alpha_per_km * dz / 2)
        effective_length = (
            dz if config.alpha_per_km == 0
            else -math.expm1(-config.alpha_per_km * dz) / config.alpha_per_km
        )
        operators[dz] = (amplitude_loss, effective_length)
    dispersion_cache: dict[float, Any] = {}

    def disperse(value: Any, distance: float) -> Any:
        if config.beta2_s2_per_km == 0:
            return value
        if distance not in dispersion_cache:
            dispersion_cache[distance] = xp.exp(0.5j * config.beta2_s2_per_km * omega**2 * distance)
        return fft.ifft(fft.fft(value, axis=0) * dispersion_cache[distance][:, None], axis=0)

    # Consecutive half-dispersion operations commute and are combined. This
    # preserves the symmetric scheme while requiring one FFT pair per step.
    field = disperse(field, lengths[0] / 2)
    for i, dz in enumerate(lengths):
        loss, effective_length = operators[dz]
        if config.gamma_per_w_km != 0:
            xy_power = xp.sum(xp.abs(field) ** 2, axis=1)
            phase = (8 / 9) * config.gamma_per_w_km * xy_power * effective_length
            field = field * xp.exp(1j * phase[:, None])
        field = field * loss
        next_half = 0.0 if i == len(lengths) - 1 else lengths[i + 1] / 2
        field = disperse(field, dz / 2 + next_half)
    return field


def propagate_fiber(
    field: np.ndarray,
    config: LinkConfig,
    *,
    step_km: float | None = None,
    inverse: bool = False,
    backend: str = "numpy",
) -> np.ndarray:
    """One fiber span, without EDFA. Public input and output are always NumPy."""
    field = np.asarray(field, dtype=np.complex128)
    if field.ndim != 2 or field.shape[1] != 2 or len(field) == 0:
        raise ValueError("field must be a nonempty (time,2) complex array")
    step = config.step_km if step_km is None else step_km
    if not math.isfinite(step) or step <= 0:
        raise ValueError("step_km must be finite and positive")
    xp, fft = _backend(backend)
    result = _fiber_on_backend(xp.asarray(field), config, step, inverse, xp, fft)
    return _to_numpy(result, xp)


def propagate_link(
    field: np.ndarray, config: LinkConfig, noise_seed: int, *, backend: str = "numpy"
) -> np.ndarray:
    """Forward propagation; ASE is added *after each* amplifier and co-propagates."""
    field = np.asarray(field, dtype=np.complex128)
    if field.ndim != 2 or field.shape[1] != 2 or len(field) == 0:
        raise ValueError("field must be a nonempty (time,2) complex array")
    xp, fft = _backend(backend)
    result = xp.asarray(field)
    # Separate seed stream per span; adding spans preserves existing earlier ASE.
    span_sequences = np.random.SeedSequence(noise_seed).spawn(config.n_spans)
    for span in range(config.n_spans):
        result = _fiber_on_backend(result, config, config.step_km, False, xp, fft)
        result = result * math.sqrt(config.edfa_gain)
        if config.ase_enabled:
            rng = np.random.default_rng(span_sequences[span])
            result = result + xp.asarray(ase_noise(len(field), config, rng))
    return _to_numpy(result, xp)


def simulate_frame(
    config: LinkConfig,
    launch_power_dbm: float,
    n_symbols: int,
    bit_seed: int,
    noise_seed: int,
    backend: str = "numpy",
) -> SimulatedFrame:
    """Generate uniform Gray DP16QAM and a reproducible physically scaled link.

    Launch dBm is nominal ensemble XY-total power, equally split between XY;
    no finite-frame or test-label gain/phase calibration is performed.
    """
    if not isinstance(n_symbols, int) or n_symbols <= 2 * config.guard_symbols:
        raise ValueError("n_symbols must be an integer greater than 2*guard_symbols")
    if not math.isfinite(launch_power_dbm):
        raise ValueError("launch_power_dbm must be finite")
    rng = np.random.default_rng(bit_seed)
    bits = rng.integers(0, 2, size=(n_symbols, 2, 4), dtype=np.uint8)
    symbols = bits_to_symbols(bits)
    indices = np.sum(bits * np.array([8, 4, 2, 1], dtype=np.uint8), axis=-1, dtype=np.uint8)
    power_watts = 1e-3 * 10 ** (launch_power_dbm / 10)
    tx_waveform = pulse_shape(symbols, config) * np.sqrt(power_watts / 2)
    waveform = propagate_link(tx_waveform, config, noise_seed, backend=backend)
    rx_symbols, rx_2sps = receiver_dsp(waveform, config, launch_power_dbm)
    mask = np.ones(n_symbols, dtype=bool)
    guard = config.guard_symbols
    if guard:
        mask[:guard] = False
        mask[-guard:] = False
    return SimulatedFrame(
        tx_symbols=symbols, tx_bits=bits, tx_indices=indices,
        rx_symbols=rx_symbols, rx_2sps=rx_2sps, rx_waveform=waveform,
        valid_mask=mask, config=config, launch_power_dbm=launch_power_dbm,
        bit_seed=bit_seed, noise_seed=noise_seed, backend=backend,
    )


def dbp_symbols(
    frame: SimulatedFrame, step_km: float | None = None, backend: str | None = None
) -> np.ndarray:
    """Reverse deterministic gain/fiber operators, then matched-filter and align.

    The raw receive field includes ASE. No realized ASE is subtracted or claimed
    to be recovered; DBP can modify signal-noise interactions and noise itself.
    """
    config = frame.config
    step = config.step_km if step_km is None else step_km
    if not math.isfinite(step) or step <= 0:
        raise ValueError("step_km must be finite and positive")
    xp, fft = _backend(frame.backend if backend is None else backend)
    field = xp.asarray(frame.rx_waveform)
    for _span in range(config.n_spans - 1, -1, -1):
        field = field / math.sqrt(config.edfa_gain)
        field = _fiber_on_backend(field, config, step, True, xp, fft)
    symbols, _ = receiver_dsp(
        _to_numpy(field, xp), config, frame.launch_power_dbm, compensate_dispersion=False
    )
    return symbols


@dataclass(frozen=True)
class SpanStepAudit:
    """Actual accepted Strang steps, peak local phase, and retry accounting."""

    span: int
    count: int
    distance_km: float
    min_km: float
    max_km: float
    mean_km: float
    quantiles_km: tuple[float, ...]
    max_phase_rad: float
    retries: int


@dataclass(frozen=True)
class PropagationAudit:
    phase_bound_rad: float
    max_step_km: float
    min_step_km: float
    spans: tuple[SpanStepAudit, ...]


def _adaptive_span(
    field: Any, config: LinkConfig, phase_bound: float, min_step: float,
    xp: Any, fft: Any, span: int,
) -> tuple[Any, SpanStepAudit]:
    """Audit power *after* trial half dispersion; reject unsafe trial lengths.

    Each accepted step is symmetric. A rejected trial is recalculated from the
    unchanged step-start field. No prediction of the peak is trusted as a bound.
    """
    omega = xp.asarray(2 * np.pi * np.fft.fftfreq(len(field), d=1 / config.sample_rate_hz))

    def disperse(value: Any, distance: float) -> Any:
        if config.beta2_s2_per_km == 0:
            return value
        response = xp.exp(0.5j * config.beta2_s2_per_km * omega**2 * distance)
        return fft.ifft(fft.fft(value, axis=0) * response[:, None], axis=0)

    position, trial_length, retries, peak = 0.0, min(config.step_km, config.span_km), 0, 0.0
    lengths: list[float] = []
    while position < config.span_km:
        remaining = config.span_km - position
        dz = min(trial_length, config.step_km, remaining)
        while True:
            half = disperse(field, dz / 2)
            power = xp.sum(xp.abs(half) ** 2, axis=1)
            max_power = float(xp.max(power).item())
            effective = dz if config.alpha_per_km == 0 else -math.expm1(-config.alpha_per_km * dz) / config.alpha_per_km
            maximum_phase = (8 / 9) * config.gamma_per_w_km * max_power * effective
            if not math.isfinite(maximum_phase):
                raise FloatingPointError("Nonfinite nonlinear phase during adaptive propagation")
            if maximum_phase <= phase_bound:
                break
            if dz <= min_step:
                raise RuntimeError("Minimum SSFM step still exceeds nonlinear phase bound")
            retries += 1
            dz = max(min_step, dz * min(0.5, 0.95 * phase_bound / maximum_phase))
            dz = min(dz, remaining)
        phase = (8 / 9) * config.gamma_per_w_km * power * effective
        field = disperse(half * xp.exp(1j * phase[:, None]) * math.exp(-config.alpha_per_km * dz / 2), dz / 2)
        position += dz
        # Absorb only floating-point endpoint residue; physical distance is kept.
        if config.span_km - position < 1e-12 * config.span_km:
            position = config.span_km
        lengths.append(dz)
        peak = max(peak, maximum_phase)
        trial_length = min(config.step_km, dz * (1.2 if maximum_phase == 0 else min(1.2, 0.98 * phase_bound / maximum_phase)))
    values = np.asarray(lengths)
    return field, SpanStepAudit(span, len(lengths), float(values.sum()), float(values.min()),
        float(values.max()), float(values.mean()), tuple(float(x) for x in np.quantile(values, [0, .1, .25, .5, .75, .9, 1])), peak, retries)


def propagate_link_adaptive(
    field: np.ndarray,
    config: LinkConfig,
    noise_seed: int | np.random.SeedSequence,
    *,
    phase_max_deg: float = 0.05,
    min_step_km: float = 1e-6,
    backend: str = "numpy",
    ase_source: Callable[[int, int, LinkConfig], np.ndarray] | None = None,
) -> tuple[np.ndarray, PropagationAudit]:
    """Forward link with a checked maximum local Manakov phase per step.

    The legacy fixed-step path is unchanged. SeedSequence is cloned before
    spawning so callers can reuse a recipe without mutating its RNG state.
    """
    field = np.asarray(field, dtype=np.complex128)
    if field.ndim != 2 or field.shape[1] != 2 or len(field) == 0 or not np.all(np.isfinite(field)):
        raise ValueError("field must be a finite nonempty (time,2) complex array")
    if not math.isfinite(phase_max_deg) or phase_max_deg <= 0:
        raise ValueError("phase_max_deg must be finite and positive")
    if not math.isfinite(min_step_km) or not 0 < min_step_km <= config.step_km:
        raise ValueError("min_step_km must be positive and <= maximum step_km")
    phase_bound = math.radians(phase_max_deg)
    sequence = (np.random.SeedSequence(noise_seed.entropy, spawn_key=noise_seed.spawn_key, pool_size=noise_seed.pool_size)
                if isinstance(noise_seed, np.random.SeedSequence) else np.random.SeedSequence(noise_seed))
    span_sequences = sequence.spawn(config.n_spans)
    xp, fft = _backend(backend)
    result, audits = xp.asarray(field), []
    for span, seq in enumerate(span_sequences):
        result, audit = _adaptive_span(result, config, phase_bound, min_step_km, xp, fft, span)
        result *= math.sqrt(config.edfa_gain)
        if config.ase_enabled:
            noise = (ase_noise(len(field), config, np.random.Generator(np.random.PCG64(seq)))
                     if ase_source is None else np.asarray(ase_source(span, len(field), config), dtype=np.complex128))
            if noise.shape != field.shape or not np.all(np.isfinite(noise)):
                raise ValueError("ASE source returned invalid field shape or nonfinite values")
            result += xp.asarray(noise)
        audits.append(audit)
    return _to_numpy(result, xp), PropagationAudit(phase_bound, config.step_km, min_step_km, tuple(audits))
