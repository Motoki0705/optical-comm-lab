"""Gray 16QAM decisions and directly counted communication metrics.

All SNRs use Es/N0: one unit-energy complex symbol and circular complex noise
of variance N0 in *each* polarization. No gain or phase is fitted here.
"""

from __future__ import annotations

import math
import numpy as np
from scipy.special import ndtr
from scipy.stats import beta


def qam_axis(order: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Ascending normalized PAM levels and binary-reflected Gray labels."""
    if isinstance(order, bool) or not isinstance(order, int):
        raise ValueError("QAM order must be an integer square power of two")
    side = math.isqrt(order)
    if side < 2 or side * side != order or side & (side - 1):
        raise ValueError("QAM order must be a square power of two >= 4")
    positions = np.arange(side, dtype=np.int64)
    return (2 * positions - side + 1) / math.sqrt(2 * (order - 1) / 3), positions ^ (positions >> 1)


def bits_to_symbols(bits: np.ndarray, order: int = 16) -> np.ndarray:
    """Uniform square-QAM, Gray bits on I then Q; unit ensemble complex Es."""
    levels, gray = qam_axis(order)
    n_axis = int(math.log2(len(levels)))
    bits = np.asarray(bits)
    if bits.ndim < 1 or bits.shape[-1] != 2 * n_axis or not np.all((bits == 0) | (bits == 1)):
        raise ValueError(f"bits must have a final axis of {2 * n_axis} binary values")
    weights = 2 ** np.arange(n_axis - 1, -1, -1)
    inverse = np.argsort(gray)
    i = np.sum(bits[..., :n_axis] * weights, axis=-1, dtype=np.int64)
    q = np.sum(bits[..., n_axis:] * weights, axis=-1, dtype=np.int64)
    return levels[inverse[i]] + 1j * levels[inverse[q]]


def symbols_to_bits(symbols: np.ndarray, order: int = 16) -> np.ndarray:
    """Nearest-neighbor fixed-alphabet decisions without fitted calibration."""
    symbols = np.asarray(symbols)
    if not np.all(np.isfinite(symbols)):
        raise ValueError("symbols must be finite")
    levels, labels = qam_axis(order)
    n_axis = int(math.log2(len(levels)))
    thresholds = (levels[:-1] + levels[1:]) / 2
    i = labels[np.searchsorted(thresholds, symbols.real, side="right")]
    q = labels[np.searchsorted(thresholds, symbols.imag, side="right")]
    shifts = np.arange(n_axis - 1, -1, -1)
    return np.concatenate(((i[..., None] >> shifts) & 1, (q[..., None] >> shifts) & 1), axis=-1).astype(np.uint8)


def qam_decisions(symbols: np.ndarray, order: int = 16) -> np.ndarray:
    """Nearest normalized QAM symbols (used by blind receiver processing)."""
    levels, _ = qam_axis(order)
    thresholds = (levels[:-1] + levels[1:]) / 2
    return levels[np.searchsorted(thresholds, np.asarray(symbols).real, side="right")] + 1j * levels[np.searchsorted(thresholds, np.asarray(symbols).imag, side="right")]


def gray_qam_awgn_ber(snr_db: float | np.ndarray, order: int = 16) -> float | np.ndarray:
    """Exact uniform Gray square-QAM BER by all PAM decision intervals.

    Includes errors across several thresholds, not a nearest-neighbor formula.
    SNR is complex-symbol Es/N0, each real noise variance N0/2.
    """
    levels, labels = qam_axis(order)
    n_axis = int(math.log2(len(levels)))
    bounds = np.r_[-np.inf, (levels[:-1] + levels[1:]) / 2, np.inf]
    snr = np.asarray(snr_db, dtype=float)
    sigma = np.sqrt(10 ** (-snr / 10) / 2)
    result = np.zeros_like(snr)
    for i, level in enumerate(levels):
        for j, label in enumerate(labels):
            distance = int(int(labels[i]) ^ int(label)).bit_count()
            if distance:
                probability = ndtr((bounds[j + 1] - level) / sigma) - ndtr((bounds[j] - level) / sigma)
                result += distance * probability / (len(levels) * n_axis)
    return float(result) if result.ndim == 0 else result


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
    tx_bits: np.ndarray, rx_symbols: np.ndarray, mask: np.ndarray | None = None, *, order: int = 16
) -> tuple[int, int]:
    tx_bits = np.asarray(tx_bits)
    rx_symbols = np.asarray(rx_symbols)
    n_bits_symbol = int(math.log2(order))
    qam_axis(order)
    if tx_bits.shape != (*rx_symbols.shape, n_bits_symbol):
        raise ValueError(f"tx_bits must have shape rx_symbols.shape + ({n_bits_symbol},)")
    if not np.all((tx_bits == 0) | (tx_bits == 1)):
        raise ValueError("tx_bits must be binary")
    if mask is not None:
        mask = _validate_mask(mask, len(rx_symbols))
        tx_bits, rx_symbols = tx_bits[mask], rx_symbols[mask]
    if tx_bits.size == 0:
        raise ValueError("Cannot count BER on an empty array")
    decisions = symbols_to_bits(rx_symbols, order)
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
    *, order: int = 16,
) -> dict[str, float | int]:
    """Count pre-FEC bit errors and measure EVM against the uncalibrated inputs."""
    tx_symbols, rx_symbols = np.asarray(tx_symbols), np.asarray(rx_symbols)
    if tx_symbols.shape != rx_symbols.shape or tx_symbols.ndim < 1:
        raise ValueError("TX and RX symbol arrays must have the same non-scalar shape")
    if not np.all(np.isfinite(tx_symbols)) or not np.all(np.isfinite(rx_symbols)):
        raise ValueError("TX and RX symbols must be finite")
    bits = symbols_to_bits(tx_symbols, order) if tx_bits is None else np.asarray(tx_bits)
    errors, n_bits = bit_error_counts(bits, rx_symbols, mask, order=order)
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
