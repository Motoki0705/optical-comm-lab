"""Alphabet, exact AWGN theory, bit counting, and finite-sample uncertainty."""

import itertools

import numpy as np
import pytest

from optical_comm.metrics import (
    ber_interval,
    bits_to_symbols,
    compute_metrics,
    gray16qam_awgn_ber,
    symbols_to_bits,
)


def test_uniform_gray_alphabet_and_neighbor_labels():
    bits = np.array(list(itertools.product([0, 1], repeat=4)), dtype=np.uint8)
    symbols = bits_to_symbols(bits)
    np.testing.assert_array_equal(symbols_to_bits(symbols), bits)
    assert len(np.unique(symbols)) == 16
    assert np.mean(np.abs(symbols) ** 2) == pytest.approx(1.0)
    for first in range(16):
        for second in range(first + 1, 16):
            if np.isclose(abs(symbols[first] - symbols[second]), 2 / np.sqrt(10)):
                assert np.count_nonzero(bits[first] != bits[second]) == 1


@pytest.mark.parametrize("snr_db", [0.0, 8.0, 16.0])
def test_awgn_ber_matches_independent_exact_gray_theory(snr_db):
    rng_bits = np.random.default_rng(4101)
    bits = rng_bits.integers(0, 2, (100_000, 2, 4), dtype=np.uint8)
    tx = bits_to_symbols(bits)
    rng_noise = np.random.default_rng(7307)
    variance = 10 ** (-snr_db / 10)
    noise = np.sqrt(variance / 2) * (
        rng_noise.standard_normal(tx.shape) + 1j * rng_noise.standard_normal(tx.shape)
    )
    measured = compute_metrics(tx, tx + noise, bits)
    low, high = ber_interval(measured["bit_errors"], measured["n_bits"], confidence=0.999)
    assert low < gray16qam_awgn_ber(snr_db) < high
    assert measured["effective_snr_db"] == pytest.approx(snr_db, abs=0.04)


def test_exact_awgn_theory_has_random_guess_limit():
    assert gray16qam_awgn_ber(-200) == pytest.approx(0.5, abs=1e-8)
    assert gray16qam_awgn_ber(100) == 0


def test_bit_mask_counts_and_no_implicit_gain_calibration():
    bits = np.zeros((4, 2, 4), dtype=np.uint8)
    tx = bits_to_symbols(bits)
    rx = tx.copy()
    rx[0] = -tx[0]
    metrics = compute_metrics(tx, rx, bits, mask=np.array([False, True, True, True]))
    assert metrics["bit_errors"] == 0
    assert metrics["n_bits"] == 24
    assert metrics["ber_ci_high"] > 0
    doubled = compute_metrics(tx, 2 * tx, bits)
    assert doubled["evm"] == pytest.approx(1)
    assert doubled["effective_snr_db"] == pytest.approx(0)


def test_zero_error_interval_reports_nonzero_measurement_limit():
    low, high = ber_interval(0, 1000)
    assert low == 0
    assert high == pytest.approx(1 - 0.025 ** (1 / 1000))
    with pytest.raises(ValueError):
        ber_interval(2, 1)
