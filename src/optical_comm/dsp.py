"""Periodic RRC pulse shaping, ideal EDC, and deterministic matched sampling."""

from __future__ import annotations

import numpy as np

from optical_comm.config import LinkConfig


def rrc_taps(sps: int, rolloff: float = 0.1, span_symbols: int = 64) -> np.ndarray:
    """Real, symmetric, finite-support root raised cosine; unit discrete energy.

    Time is measured in symbol periods. Singularities are evaluated analytically.
    The returned odd-length impulse is centered at len(taps)//2.
    """
    if sps < 2 or not isinstance(sps, int) or not 0 <= rolloff <= 1:
        raise ValueError("Require integer sps>=2 and 0<=rolloff<=1")
    if span_symbols < 2 or span_symbols % 2:
        raise ValueError("span_symbols must be positive and even")
    t = np.arange(-span_symbols * sps // 2, span_symbols * sps // 2 + 1) / sps
    if rolloff == 0:
        h = np.sinc(t)
    else:
        h = np.empty_like(t)
        zero = np.isclose(t, 0, atol=1e-14)
        special = np.isclose(np.abs(t), 1 / (4 * rolloff), atol=1e-14)
        normal = ~(zero | special)
        tn = t[normal]
        h[normal] = (
            np.sin(np.pi * tn * (1 - rolloff))
            + 4 * rolloff * tn * np.cos(np.pi * tn * (1 + rolloff))
        ) / (np.pi * tn * (1 - (4 * rolloff * tn) ** 2))
        h[zero] = 1 + rolloff * (4 / np.pi - 1)
        h[special] = rolloff / np.sqrt(2) * (
            (1 + 2 / np.pi) * np.sin(np.pi / (4 * rolloff))
            + (1 - 2 / np.pi) * np.cos(np.pi / (4 * rolloff))
        )
    return h / np.sqrt(np.sum(h**2))


def periodic_filter_response(taps: np.ndarray, n_samples: int) -> np.ndarray:
    """Centered taps folded onto a periodic record, with no hidden group delay."""
    taps = np.asarray(taps)
    impulse = np.zeros(n_samples, dtype=float)
    positions = (np.arange(len(taps)) - len(taps) // 2) % n_samples
    np.add.at(impulse, positions, taps)
    return np.fft.fft(impulse)


def pulse_shape(symbols: np.ndarray, config: LinkConfig) -> np.ndarray:
    """Unit mean-power pulse train for unit-mean-energy input symbols.

    The mean is an ensemble convention, not a normalization fitted to each frame.
    Waveform power fluctuates slightly with the random finite TX realization.
    """
    symbols = np.asarray(symbols, dtype=np.complex128)
    if symbols.ndim != 2 or symbols.shape[1] != 2 or len(symbols) == 0:
        raise ValueError("symbols must be a nonempty (N,2) dual-polarization array")
    impulse = np.zeros((len(symbols) * config.sps, 2), dtype=np.complex128)
    impulse[:: config.sps] = np.sqrt(config.sps) * symbols
    response = periodic_filter_response(
        rrc_taps(config.sps, config.rolloff, config.rrc_span_symbols), len(impulse)
    )
    return np.fft.ifft(np.fft.fft(impulse, axis=0) * response[:, None], axis=0)


def receiver_dsp(
    waveform: np.ndarray,
    config: LinkConfig,
    launch_power_dbm: float,
    *,
    compensate_dispersion: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned 1-sps and 2-sps alphabet-scaled outputs; no fitted calibration."""
    waveform = np.asarray(waveform, dtype=np.complex128)
    if waveform.ndim != 2 or waveform.shape[1] != 2 or len(waveform) % config.sps:
        raise ValueError("waveform must be (N*sps,2)")
    frequencies = np.fft.fftfreq(len(waveform), d=1 / config.sample_rate_hz)
    response = periodic_filter_response(
        rrc_taps(config.sps, config.rolloff, config.rrc_span_symbols), len(waveform)
    ).conj()
    if compensate_dispersion:
        response *= np.exp(
            -0.5j * config.beta2_s2_per_km * (2 * np.pi * frequencies) ** 2
            * config.total_length_km
        )
    filtered = np.fft.ifft(np.fft.fft(waveform, axis=0) * response[:, None], axis=0)
    total_watts = 1e-3 * 10 ** (launch_power_dbm / 10)
    scale = np.sqrt(total_watts * config.sps / 2)
    filtered /= scale
    return filtered[:: config.sps].copy(), filtered[:: config.sps // 2].copy()


def apply_dispersion(field: np.ndarray, config: LinkConfig, distance_km: float) -> np.ndarray:
    """Apply H(L)=exp(+i beta2 omega² L/2); negative L is inverse CDC."""
    field = np.asarray(field, dtype=np.complex128)
    if field.ndim != 2 or field.shape[1] != 2 or not len(field) or not np.isfinite(distance_km):
        raise ValueError("Require nonempty (time,2) field and finite distance")
    omega = 2 * np.pi * np.fft.fftfreq(len(field), d=1 / config.sample_rate_hz)
    return np.fft.ifft(np.fft.fft(field, axis=0) * np.exp(.5j * config.beta2_s2_per_km * omega**2 * distance_km)[:, None], axis=0)


def fir_design(symbols: np.ndarray, taps: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """Nonperiodic complex 2×2 FIR design; edge rows are explicitly invalid."""
    symbols = np.asarray(symbols, dtype=np.complex128)
    if symbols.ndim != 2 or symbols.shape[1] != 2 or not isinstance(taps, int) or taps < 1 or taps % 2 == 0 or len(symbols) < taps:
        raise ValueError("Require (time,2), odd positive taps, and at least taps rows")
    half = taps // 2
    padded = np.pad(symbols, ((half, half), (0, 0)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, taps, axis=0)
    matrix = windows.transpose(0, 2, 1).reshape(len(symbols), 2 * taps)
    valid = np.ones(len(symbols), dtype=bool)
    if half:
        valid[:half] = valid[-half:] = False
    return matrix, valid


def fit_pilot_fir(
    inputs: list[np.ndarray], pilot_targets: list[np.ndarray], pilot_masks: list[np.ndarray],
    *, taps: int = 11, ridge_relative: float = 1e-6,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Aggregate dedicated pilots only; targets must be zero outside pilot masks."""
    if not inputs or len(inputs) != len(pilot_targets) or len(inputs) != len(pilot_masks):
        raise ValueError("Mismatched or empty pilot-frame lists")
    gram = np.zeros((2 * taps, 2 * taps), dtype=np.complex128)
    cross = np.zeros((2 * taps, 2), dtype=np.complex128)
    count = 0
    for rx, targets, mask in zip(inputs, pilot_targets, pilot_masks, strict=True):
        matrix, valid = fir_design(rx, taps)
        targets, mask = np.asarray(targets), np.asarray(mask)
        if mask.dtype != bool or mask.shape != valid.shape or targets.shape != rx.shape or np.any(targets[~mask] != 0):
            raise ValueError("Pilot-only targets/mask required; scored labels are forbidden")
        # Regressors must stay within a dedicated pilot region, including taps.
        safe = np.convolve(mask.astype(int), np.ones(taps, dtype=int), mode="same") == taps
        selected = safe & valid
        x, y = matrix[selected], targets[selected]
        gram += x.conj().T @ x
        cross += x.conj().T @ y
        count += len(x)
    if count < 2 * taps or ridge_relative < 0 or not np.isfinite(ridge_relative):
        raise ValueError("Insufficient independent pilot rows or invalid ridge")
    ridge = ridge_relative * float(np.trace(gram).real) / len(gram)
    weights = np.linalg.solve(gram + ridge * np.eye(len(gram)), cross)
    return weights, {"pilot_rows": count, "ridge": ridge, "ridge_relative": ridge_relative, "taps": taps}


def apply_fir(symbols: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(weights, dtype=np.complex128)
    if weights.ndim != 2 or weights.shape[1] != 2 or weights.shape[0] % 2:
        raise ValueError("FIR weights must have shape (2*taps,2)")
    design, valid = fir_design(symbols, weights.shape[0] // 2)
    return design @ weights, valid


def pilot_gain_calibration(
    symbols: np.ndarray, pilot_targets: np.ndarray, pilot_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-polarization constant complex LS gain, using dedicated pilots only."""
    symbols, targets, mask = np.asarray(symbols), np.asarray(pilot_targets), np.asarray(pilot_mask)
    if symbols.shape != targets.shape or mask.dtype != bool or mask.shape != (len(symbols),) or not np.any(mask) or np.any(targets[~mask] != 0):
        raise ValueError("Require pilot-only targets and a nonempty pilot mask")
    denominator = np.sum(np.abs(symbols[mask])**2, axis=0)
    if np.any(denominator <= 0):
        raise ValueError("Degenerate pilot power")
    gain = np.sum(symbols[mask].conj() * targets[mask], axis=0) / denominator
    return symbols * gain, gain


def blind_phase_search(
    symbols: np.ndarray, order: int = 16, *, window: int = 65, n_phases: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decision-directed BPS over π/2, followed by period-π/2 phase unwrap.

    No transmitted labels enter this function. Finite centered windows never
    wrap around the record; edge outputs are marked invalid.
    """
    from optical_comm.metrics import qam_decisions
    symbols = np.asarray(symbols, dtype=np.complex128)
    if symbols.ndim != 2 or symbols.shape[1] != 2 or window < 1 or window % 2 == 0 or len(symbols) < window or n_phases < 2:
        raise ValueError("Require (time,2), odd centered window<=length and n_phases>=2")
    candidates = np.arange(n_phases) * (np.pi / 2 / n_phases) - np.pi / 4
    best_cost = np.full(symbols.shape, np.inf)
    best_phase = np.zeros(symbols.shape)
    half = window // 2
    for phase in candidates:
        rotated = symbols * np.exp(-1j * phase)
        distance = np.abs(rotated - qam_decisions(rotated, order))**2
        # Edge values are placeholders; validity excludes incomplete windows.
        padded = np.pad(distance, ((half, half), (0, 0)))
        cumulative = np.concatenate((np.zeros((1, 2)), np.cumsum(padded, axis=0)), axis=0)
        cost = cumulative[window:] - cumulative[:-window]
        better = cost < best_cost
        best_cost[better], best_phase[better] = cost[better], phase
    phase = np.unwrap(4 * best_phase, axis=0) / 4
    valid = np.ones(len(symbols), dtype=bool)
    if half:
        valid[:half] = valid[-half:] = False
    return symbols * np.exp(-1j * phase), phase, valid
