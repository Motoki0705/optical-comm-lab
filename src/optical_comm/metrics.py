"""Gray 16QAM decisions and directly counted communication metrics.

All SNRs use Es/N0: one unit-energy complex symbol and circular complex noise
of variance N0 in *each* polarization. No gain or phase is fitted here.
"""

from __future__ import annotations

import math
import numpy as np
from scipy.special import ndtr
from scipy.stats import beta


_GRAY_LEVELS = np.array([-3.0, -1.0, 3.0, 1.0]) / np.sqrt(10.0)


def bits_to_symbols(bits: np.ndarray) -> np.ndarray:
    """Map the last four bits to Re/Im Gray 4PAM: 00,-3; 01,-1; 11,+1; 10,+3."""
    bits = np.asarray(bits)
    if bits.ndim < 1 or bits.shape[-1] != 4 or not np.all((bits == 0) | (bits == 1)):
        raise ValueError("bits must have a final axis of four binary values")
    indices_i = 2 * bits[..., 0].astype(np.int64) + bits[..., 1]
    indices_q = 2 * bits[..., 2].astype(np.int64) + bits[..., 3]
    return _GRAY_LEVELS[indices_i] + 1j * _GRAY_LEVELS[indices_q]


def symbols_to_bits(symbols: np.ndarray) -> np.ndarray:
    """Nearest-neighbor Gray decisions at fixed, normalized alphabet thresholds."""
    symbols = np.asarray(symbols)
    if not np.all(np.isfinite(symbols)):
        raise ValueError("symbols must be finite")
    thresholds = np.array([-2, 0, 2]) / np.sqrt(10.0)
    binary_labels = np.array([0, 1, 3, 2], dtype=np.uint8)
    i = binary_labels[np.searchsorted(thresholds, symbols.real, side="right")]
    q = binary_labels[np.searchsorted(thresholds, symbols.imag, side="right")]
    return np.stack((i >> 1, i & 1, q >> 1, q & 1), axis=-1)


def ber_interval(errors: int, n_bits: int, confidence: float = 0.95) -> tuple[float, float]:
    """Exact Clopper-Pearson interval under a binomial independent-bit model.

    This measures finite counting uncertainty; it is not a claim that nonlinear
    channel errors are independent. Independent frames must assess replication.
    """
    if not isinstance(errors, (int, np.integer)) or not isinstance(n_bits, (int, np.integer)):
        raise ValueError("errors and n_bits must be integers")
    if n_bits <= 0 or not 0 <= errors <= n_bits or not 0 < confidence < 1:
        raise ValueError("Require n_bits>0, 0<=errors<=n_bits, 0<confidence<1")
    tail = (1 - confidence) / 2
    low = 0.0 if errors == 0 else float(beta.ppf(tail, errors, n_bits - errors + 1))
    high = 1.0 if errors == n_bits else float(beta.ppf(1 - tail, errors + 1, n_bits - errors))
    return low, high


def bit_error_counts(
    tx_bits: np.ndarray, rx_symbols: np.ndarray, mask: np.ndarray | None = None
) -> tuple[int, int]:
    tx_bits = np.asarray(tx_bits)
    rx_symbols = np.asarray(rx_symbols)
    if tx_bits.shape != (*rx_symbols.shape, 4):
        raise ValueError("tx_bits must have shape rx_symbols.shape + (4,)")
    if not np.all((tx_bits == 0) | (tx_bits == 1)):
        raise ValueError("tx_bits must be binary")
    if mask is not None:
        mask = _validate_mask(mask, len(rx_symbols))
        tx_bits, rx_symbols = tx_bits[mask], rx_symbols[mask]
    if tx_bits.size == 0:
        raise ValueError("Cannot count BER on an empty array")
    decisions = symbols_to_bits(rx_symbols)
    return int(np.count_nonzero(tx_bits != decisions)), int(tx_bits.size)


def _validate_mask(mask: np.ndarray, n: int) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.dtype != np.bool_ or mask.shape != (n,):
        raise ValueError("mask must be a boolean vector matching the symbol time axis")
    return mask


def compute_metrics(
    tx_symbols: np.ndarray,
    rx_symbols: np.ndarray,
    tx_bits: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Count pre-FEC bit errors and measure EVM against the uncalibrated inputs."""
    tx_symbols, rx_symbols = np.asarray(tx_symbols), np.asarray(rx_symbols)
    if tx_symbols.shape != rx_symbols.shape or tx_symbols.ndim < 1:
        raise ValueError("TX and RX symbol arrays must have the same non-scalar shape")
    if not np.all(np.isfinite(tx_symbols)) or not np.all(np.isfinite(rx_symbols)):
        raise ValueError("TX and RX symbols must be finite")
    bits = symbols_to_bits(tx_symbols) if tx_bits is None else np.asarray(tx_bits)
    errors, n_bits = bit_error_counts(bits, rx_symbols, mask)
    if mask is not None:
        mask = _validate_mask(mask, len(tx_symbols))
        tx_symbols, rx_symbols = tx_symbols[mask], rx_symbols[mask]
    signal_power = float(np.mean(np.abs(tx_symbols) ** 2))
    mse = float(np.mean(np.abs(rx_symbols - tx_symbols) ** 2))
    if signal_power <= 0:
        raise ValueError("TX signal energy must be positive")
    evm = math.sqrt(mse / signal_power)
    snr_db = math.inf if mse == 0 else 10 * math.log10(signal_power / mse)
    low, high = ber_interval(errors, n_bits)
    return {
        "bit_errors": errors,
        "n_bits": n_bits,
        "ber": errors / n_bits,
        "ber_ci_low": low,
        "ber_ci_high": high,
        "signal_power": signal_power,
        "mse": mse,
        "evm": evm,
        "effective_snr_db": snr_db,
    }


def gray16qam_awgn_ber(snr_db: float | np.ndarray) -> float | np.ndarray:
    """Exact uniform Gray-square-16QAM BER in complex AWGN for Es/N0 [dB].

    Derived independently from the 4PAM decision intervals and Hamming distances:
    Pb = 3/4 Q(a) + 1/2 Q(3a) - 1/4 Q(5a), a=sqrt((Es/N0)/5).
    Unlike the usual nearest-neighbor approximation this remains exact at low SNR.
    """
    values = np.asarray(snr_db, dtype=float)
    a = np.sqrt(10 ** (values / 10) / 5)
    result = 0.75 * ndtr(-a) + 0.5 * ndtr(-3 * a) - 0.25 * ndtr(-5 * a)
    return float(result) if result.ndim == 0 else result
