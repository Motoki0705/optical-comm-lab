"""Independent physical limits and short numerical verification; no link claims."""

from dataclasses import replace
import importlib.util

import numpy as np
import pytest

from optical_comm.channel import (
    ase_noise,
    dbp_symbols,
    edfa_ase_psd,
    propagate_fiber,
    propagate_link,
    simulate_frame,
)
from optical_comm.config import LinkConfig
from optical_comm.dsp import pulse_shape, receiver_dsp, rrc_taps
from optical_comm.metrics import ber_interval, compute_metrics, gray16qam_awgn_ber


def test_physical_units_and_config_serialization():
    config = LinkConfig()
    assert config.beta2_s2_per_km / 1e-24 == pytest.approx(-21.6826, abs=1e-4)
    assert config.edfa_gain_db == pytest.approx(16)
    assert config.alpha_per_km == pytest.approx(np.log(10) * 0.2 / 10)
    assert config.total_length_km == 1600
    assert LinkConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError):
        replace(config, sps=3)
    with pytest.raises(ValueError):
        LinkConfig.from_dict({"unknown": 1})


def test_rrc_power_and_back_to_back_alignment():
    config = LinkConfig(n_spans=0, guard_symbols=64)
    frame = simulate_frame(config, 0, 8192, 11, 19)
    taps = rrc_taps(config.sps, config.rolloff, config.rrc_span_symbols)
    np.testing.assert_allclose(taps, taps[::-1], atol=1e-14)
    assert np.sum(taps**2) == pytest.approx(1)
    assert frame.rx_2sps.shape == (2 * len(frame.tx_symbols), 2)
    np.testing.assert_array_equal(frame.rx_2sps[::2], frame.rx_symbols)
    assert np.mean(np.sum(np.abs(frame.rx_waveform)**2, axis=1)) == pytest.approx(1e-3, rel=0.02)
    metrics = compute_metrics(frame.tx_symbols, frame.rx_symbols, frame.tx_bits, frame.valid_mask)
    assert metrics["bit_errors"] == 0
    assert metrics["evm"] < 0.002  # Finite RRC truncation is not asserted to be exact Nyquist.
    np.testing.assert_array_equal(dbp_symbols(frame), frame.rx_symbols)


def test_loss_only_matches_arbitrary_field_analytic_solution():
    config = LinkConfig(span_km=37, step_km=2.5, gamma_per_w_km=0,
                        dispersion_ps_nm_km=0, ase_enabled=False)
    rng = np.random.default_rng(21)
    field = (rng.normal(size=(256, 2)) + 1j * rng.normal(size=(256, 2))) * 0.02
    expected = field * 10 ** (-config.alpha_db_per_km * config.span_km / 20)
    np.testing.assert_allclose(propagate_fiber(field, config), expected, rtol=2e-14, atol=1e-16)


def test_constant_envelope_lossy_manakov_kerr_exact_phase():
    config = LinkConfig(span_km=37, step_km=2.5, dispersion_ps_nm_km=0, ase_enabled=False)
    field = np.tile([np.sqrt(0.7e-3), 1j * np.sqrt(0.3e-3)], (256, 1))
    effective_length = (1 - np.exp(-config.alpha_per_km * config.span_km)) / config.alpha_per_km
    phase = (8 / 9) * 1.3 * 1e-3 * effective_length
    expected = field * np.exp(-config.alpha_per_km * config.span_km / 2 + 1j * phase)
    np.testing.assert_allclose(propagate_fiber(field, config), expected, rtol=2e-14, atol=1e-16)


def test_linear_dispersion_analytic_spectral_phase_and_edc():
    config = LinkConfig(n_spans=2, span_km=13, step_km=2, gamma_per_w_km=0,
                        ase_enabled=False, guard_symbols=32)
    rng = np.random.default_rng(65)
    field = (rng.normal(size=(1024, 2)) + 1j * rng.normal(size=(1024, 2))) * 0.01
    omega = 2 * np.pi * np.fft.fftfreq(len(field), d=1 / config.sample_rate_hz)
    expected = np.fft.ifft(
        np.fft.fft(field, axis=0)
        * np.exp(0.5j * config.beta2_s2_per_km * omega**2 * config.total_length_km)[:, None],
        axis=0,
    )
    np.testing.assert_allclose(propagate_link(field, config, 17), expected, rtol=2e-12, atol=1e-14)
    linear = simulate_frame(config, 0, 1024, 37, 73)
    back_to_back = simulate_frame(replace(config, n_spans=0), 0, 1024, 37, 73)
    np.testing.assert_allclose(linear.rx_symbols, back_to_back.rx_symbols, rtol=2e-12, atol=2e-13)


def test_ase_psd_and_xy_circular_complex_variance():
    config = LinkConfig()
    gain, noise_factor = 10**1.6, 10**0.5
    photon_energy = 6.62607015e-34 * 299792458 / 1550e-9
    independently_derived_psd = (gain * noise_factor - 1) * photon_energy / 2
    assert edfa_ase_psd(config) == pytest.approx(independently_derived_psd)
    n = 32768
    noise = ase_noise(n, config, np.random.default_rng(101))
    per_pol_expected = independently_derived_psd * config.sample_rate_hz
    np.testing.assert_allclose(np.mean(np.abs(noise)**2, axis=0), per_pol_expected, rtol=0.025)
    np.testing.assert_allclose(np.var(noise.real, axis=0), per_pol_expected / 2, rtol=0.03)
    np.testing.assert_allclose(np.var(noise.imag, axis=0), per_pol_expected / 2, rtol=0.03)
    assert abs(np.mean(noise[:, 0] * noise[:, 1].conj())) < per_pol_expected * 0.025
    assert abs(np.mean(noise[:, 0]**2)) < per_pol_expected * 0.025


def test_fixed_ase_bandwidth_and_same_realization_under_sps_refinement():
    config = LinkConfig()
    coarse = ase_noise(8192, config, np.random.default_rng(207))
    fine_config = replace(config, sps=8)
    fine = ase_noise(16384, fine_config, np.random.default_rng(207))
    np.testing.assert_allclose(fine[::2], coarse, rtol=2e-12, atol=1e-16)
    expected_variance = edfa_ase_psd(config) * 128e9
    assert np.mean(np.abs(fine)**2) == pytest.approx(expected_variance, rel=0.04)
    spectrum = np.fft.fft(fine, axis=0)
    frequency = np.rint(np.fft.fftfreq(len(fine)) * len(fine)) * fine_config.sample_rate_hz / len(fine)
    outside = (frequency < -64e9) | (frequency >= 64e9)
    assert np.max(np.abs(spectrum[outside])) < 1e-12
    # Explicit full-Nyquist mode doubles noise power when Fs doubles.
    full_band = ase_noise(16384, replace(fine_config, ase_bandwidth_hz=None), np.random.default_rng(207))
    assert np.mean(np.abs(full_band)**2) == pytest.approx(expected_variance * 2, rel=0.04)


def test_linear_edfa_awgn_receiver_variance_and_ber():
    # A control power outside the research sweep deliberately creates countable
    # errors in one amplifier, and independently checks the full waveform DSP.
    config = LinkConfig(n_spans=1, step_km=80, dispersion_ps_nm_km=0,
                        gamma_per_w_km=0, guard_symbols=64)
    frame = simulate_frame(config, -20, 32768, 217, 223)
    power_per_pol = 1e-3 * 10**(-20 / 10) / 2
    expected_symbol_noise = edfa_ase_psd(config) * config.symbol_rate_hz / power_per_pol
    metrics = compute_metrics(frame.tx_symbols, frame.rx_symbols, frame.tx_bits, frame.valid_mask)
    assert metrics["mse"] == pytest.approx(expected_symbol_noise, rel=0.025)
    theory = gray16qam_awgn_ber(10 * np.log10(1 / expected_symbol_noise))
    low, high = ber_interval(metrics["bit_errors"], metrics["n_bits"], confidence=0.999)
    assert low < theory < high
    fine = simulate_frame(replace(config, sps=8), -20, 32768, 217, 223)
    fine_metrics = compute_metrics(fine.tx_symbols, fine.rx_symbols, fine.tx_bits, fine.valid_mask)
    assert fine_metrics["mse"] == pytest.approx(expected_symbol_noise, rel=0.025)
    assert fine_metrics["effective_snr_db"] == pytest.approx(metrics["effective_snr_db"], abs=0.01)


def test_independent_bits_noise_spans_and_reproducibility():
    config = LinkConfig(n_spans=2, span_km=5, step_km=1, guard_symbols=16)
    first = simulate_frame(config, 0, 512, 101, 201)
    repeat = simulate_frame(config, 0, 512, 101, 201)
    np.testing.assert_array_equal(first.rx_waveform, repeat.rx_waveform)
    changed_noise = simulate_frame(config, 0, 512, 101, 202)
    np.testing.assert_array_equal(first.tx_bits, changed_noise.tx_bits)
    assert not np.array_equal(first.rx_waveform, changed_noise.rx_waveform)
    changed_bits = simulate_frame(config, 0, 512, 102, 201)
    assert not np.array_equal(first.tx_bits, changed_bits.tx_bits)
    no_ase = replace(config, ase_enabled=False)
    np.testing.assert_array_equal(
        simulate_frame(no_ase, 0, 512, 101, 201).rx_waveform,
        simulate_frame(no_ase, 0, 512, 101, 202).rx_waveform,
    )
    # In a linear dispersion-free link each restored span adds one independent
    # contribution. Earlier ASE goes through fiber+gain, rather than being lost.
    linear = replace(config, n_spans=3, dispersion_ps_nm_km=0, gamma_per_w_km=0)
    noise = propagate_link(np.zeros((32768, 2), dtype=complex), linear, 1201)
    expected = 3 * edfa_ase_psd(linear) * linear.sample_rate_hz
    assert np.mean(np.abs(noise)**2) == pytest.approx(expected, rel=0.025)


def test_no_ase_independent_forward_and_inverse_step_refinement():
    config = LinkConfig(n_spans=1, span_km=20, step_km=2,
                        ase_enabled=False, guard_symbols=16)
    frame = simulate_frame(replace(config, n_spans=0), 6, 512, 151, 157)
    original = frame.rx_waveform
    reference = propagate_fiber(original, config, step_km=0.125)
    coarse = propagate_fiber(original, config, step_km=2)
    middle = propagate_fiber(original, config, step_km=1)
    fine = propagate_fiber(original, config, step_km=0.5)
    errors = [np.linalg.norm(output - reference) for output in (coarse, middle, fine)]
    assert errors[1] < errors[0] * 0.4
    assert errors[2] < errors[1] * 0.4
    # Use a separately finer inverse; passing cannot rely on identical solver
    # cancellation alone. Its error falls as the forward discretization improves.
    coarse_inverse = propagate_fiber(coarse, config, step_km=0.125, inverse=True)
    fine_inverse = propagate_fiber(fine, config, step_km=0.125, inverse=True)
    assert np.linalg.norm(fine_inverse - original) < np.linalg.norm(coarse_inverse - original) * 0.1
    # The matched-grid identity remains useful as an additional algebraic check.
    np.testing.assert_allclose(
        propagate_fiber(middle, config, step_km=1, inverse=True), original,
        rtol=2e-12, atol=1e-14,
    )


def test_dbp_alignment_no_ase_and_no_noise_oracle():
    config = LinkConfig(n_spans=2, span_km=10, step_km=0.5, ase_enabled=False,
                        guard_symbols=16)
    frame = simulate_frame(config, 6, 1024, 701, 709)
    back_to_back = simulate_frame(replace(config, n_spans=0), 6, 1024, 701, 709)
    np.testing.assert_allclose(dbp_symbols(frame), back_to_back.rx_symbols, rtol=2e-12, atol=1e-12)
    noisy_frame = simulate_frame(replace(config, ase_enabled=True), 6, 1024, 701, 709)
    noisy_dbp = dbp_symbols(noisy_frame)
    assert np.mean(np.abs(noisy_dbp - back_to_back.rx_symbols)**2) > 1e-7
    noisy_frame.tx_symbols[:] = 0
    noisy_frame.tx_bits[:] = 0
    np.testing.assert_array_equal(dbp_symbols(noisy_frame), noisy_dbp)


def test_cpu_gpu_parity_and_public_numpy_arrays():
    if importlib.util.find_spec("cupy") is None:
        pytest.skip("optional CuPy extra is not installed")
    import cupy as cp
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("No accessible CUDA GPU")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("No accessible CUDA driver")
    config = LinkConfig(n_spans=2, span_km=5, guard_symbols=16)
    cpu = simulate_frame(config, 2, 512, 1031, 1033)
    gpu = simulate_frame(config, 2, 512, 1031, 1033, backend="cupy")
    for array in (gpu.rx_waveform, gpu.rx_symbols, gpu.rx_2sps, dbp_symbols(gpu)):
        assert isinstance(array, np.ndarray)
    np.testing.assert_allclose(gpu.rx_waveform, cpu.rx_waveform, rtol=2e-11, atol=2e-13)
    np.testing.assert_allclose(dbp_symbols(gpu), dbp_symbols(cpu), rtol=2e-11, atol=2e-12)
