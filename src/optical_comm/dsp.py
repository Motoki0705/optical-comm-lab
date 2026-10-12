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
