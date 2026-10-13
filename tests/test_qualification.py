"""Independent checks of qualification isolation and comparison semantics."""

from dataclasses import replace
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from optical_comm.channel import simulate_frame
from optical_comm.config import LinkConfig
from optical_comm.metrics import gray16qam_awgn_ber


spec = importlib.util.spec_from_file_location("qualify_numerics", Path(__file__).parents[1] / "scripts/qualify_numerics.py")
qualification = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualification)


def test_forward_calibration_uses_training_gain_without_mmse_contraction():
    rng = np.random.default_rng(501)
    tx = rng.standard_normal((30000, 2)) + 1j * rng.standard_normal((30000, 2))
    true_h = np.array([0.8 + 0.3j, 1.1 - 0.2j])
    noisy_training = tx * true_h + (rng.standard_normal(tx.shape) + 1j * rng.standard_normal(tx.shape)) * 0.3
    fitted = qualification.forward_training_gain(tx, noisy_training, np.ones(len(tx), bool))
    np.testing.assert_allclose(fitted, true_h, rtol=0.006)
    evaluation = tx[:100] * true_h
    np.testing.assert_allclose(evaluation / fitted, tx[:100], rtol=0.006)
    reverse_ls = np.sum(tx * noisy_training.conj(), axis=0) / np.sum(abs(noisy_training) ** 2, axis=0)
    assert np.all(abs(reverse_ls * true_h) < 0.96)


@pytest.mark.parametrize("snr_db", [-20, 0, 8, 16, 25])
def test_independent_decision_region_theory_agrees_with_closed_form(snr_db):
    assert qualification.independent_gray16qam_theory(snr_db) == pytest.approx(gray16qam_awgn_ber(snr_db), abs=2e-16)


def test_common_length_no_ase_prefix_and_guard_semantics():
    config = LinkConfig(n_spans=1, span_km=4, step_km=1, guard_symbols=16, ase_enabled=False)
    frames, duration = qualification.common_noise_length_frames(config, 6, [256, 512], 557, 563, "numpy")
    np.testing.assert_array_equal(frames[256].tx_bits, frames[512].tx_bits[:256])
    independent = simulate_frame(config, 6, 256, 557, 563)
    np.testing.assert_allclose(frames[256].rx_waveform, independent.rx_waveform, rtol=1e-12, atol=1e-15)
    assert duration[256] > 0
    mask = qualification.guarded_mask(256, 16)
    assert np.count_nonzero(mask) == 224
    with pytest.raises(ValueError):
        qualification.guarded_mask(256, 128)


def test_common_length_ase_is_the_same_physical_noise_prefix():
    config = LinkConfig(n_spans=2, span_km=4, step_km=4, guard_symbols=16,
                        dispersion_ps_nm_km=0, gamma_per_w_km=0)
    noisy, _ = qualification.common_noise_length_frames(config, 0, [512, 1024], 567, 569, "numpy")
    clean, _ = qualification.common_noise_length_frames(replace(config, ase_enabled=False), 0,
                                                       [512, 1024], 567, 569, "numpy")
    short_noise = noisy[512].rx_waveform - clean[512].rx_waveform
    long_noise = noisy[1024].rx_waveform - clean[1024].rx_waveform
    np.testing.assert_allclose(short_noise, long_noise[:len(short_noise)], rtol=2e-12, atol=2e-17)
    assert np.mean(abs(short_noise) ** 2) > 0


def test_sampling_pair_compares_same_physical_times_and_rejects_different_tx():
    config = LinkConfig(n_spans=0, guard_symbols=16)
    coarse = simulate_frame(config, 0, 512, 571, 577)
    fine = simulate_frame(replace(config, sps=8), 0, 512, 571, 577)
    result = qualification.compare_frames(coarse, fine, np.ones(2), np.ones(2), coarse.valid_mask, 0.999)
    assert result["raw_waveform_relative_rms_difference"] < 0.002
    assert result["n_bits"] == (512 - 32) * 8
    assert result["paired_ber_pvalue"] == 1
    wrong = simulate_frame(config, 0, 512, 579, 577)
    with pytest.raises(AssertionError):
        qualification.compare_frames(coarse, wrong, np.ones(2), np.ones(2), coarse.valid_mask, 0.999)
