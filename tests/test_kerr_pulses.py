"""Independent pulse limits, polarization/spectral units and numerical accuracy."""

from dataclasses import replace

import numpy as np
import pytest

from optical_comm.channel import propagate_fiber
from optical_comm.config import LinkConfig
from optical_comm.visualization.pulses import (
    display_data,
    field_observables,
    gaussian_pulse,
    propagate_snapshots,
    time_grid,
)


def pulse_config(**overrides):
    return replace(LinkConfig(n_spans=1, sps=128, step_km=0.25, ase_enabled=False, guard_symbols=0), **overrides)


def test_gaussian_peak_power_energy_and_width_are_physical_not_symbol_normalization():
    config = pulse_config()
    times = time_grid(8192, config.sample_rate_hz)
    field = gaussian_pulse(times, 0.05, 20e-12)
    power = np.sum(np.abs(field) ** 2, axis=1)
    spacing = 1 / config.sample_rate_hz
    assert power[len(times) // 2] == pytest.approx(0.05)
    assert np.sum(power) * spacing == pytest.approx(0.05 * 20e-12 * np.sqrt(np.pi), rel=1e-13)
    rms_time = np.sqrt(np.sum(times**2 * power) / np.sum(power))
    assert rms_time == pytest.approx(20e-12 / np.sqrt(2), rel=1e-13)
    np.testing.assert_array_equal(field[:, 0], field[:, 1])


def test_lossy_pulse_kerr_matches_independent_exact_complex_solution():
    config = pulse_config(dispersion_ps_nm_km=0)
    times = time_grid(8192, config.sample_rate_hz)
    field = gaussian_pulse(times, 0.05, 20e-12)
    output = propagate_snapshots(field, config, np.array([0.0, 20.0, 80.0]))
    alpha = 0.2 * np.log(10) / 10
    power = np.sum(np.abs(field) ** 2, axis=1)
    for index, distance in enumerate([0.0, 20.0, 80.0]):
        phase = (8 / 9) * 1.3 * power * (1 - np.exp(-alpha * distance)) / alpha
        expected = field * np.exp(-alpha * distance / 2 + 1j * phase[:, None])
        np.testing.assert_allclose(output[index], expected, atol=2e-15, rtol=1e-12)
    peak_phase = np.angle(output[-1, len(times)//2, 0])
    assert 1.22 < peak_phase < 1.23


def test_linear_gaussian_matches_independent_complex_width_and_chirp_sign():
    config = pulse_config(gamma_per_w_km=0)
    times = time_grid(8192, config.sample_rate_hz)
    t0 = 20e-12
    output = propagate_fiber(gaussian_pulse(times, 0.05, t0), config)
    beta2 = -17e-3 * (1550e-9)**2 / (2 * np.pi * 299792458)
    q = 1 - 1j * beta2 * 80 / t0**2
    expected = np.sqrt(0.025) / np.sqrt(q) * np.exp(-times**2 / (2 * t0**2 * q)) * 10**(-16/20)
    np.testing.assert_allclose(output[:, 0], expected, atol=2e-15, rtol=2e-10)
    power = np.sum(np.abs(output) ** 2, axis=1)
    rms_width = np.sqrt(np.sum(times**2 * power) / np.sum(power))
    expected_width = t0 / np.sqrt(2) * np.sqrt(1 + (beta2*80/t0**2)**2)
    assert rms_width == pytest.approx(expected_width, rel=1e-12)
    observed = field_observables(output[None, ...], times)
    assert observed["delta_omega_rad_s"][0, np.searchsorted(times, 20e-12), 0] < 0
    assert observed["delta_omega_rad_s"][0, np.searchsorted(times, -20e-12), 0] > 0


def test_spm_chirp_is_minus_phase_derivative_and_tails_are_explicitly_masked():
    config = pulse_config(dispersion_ps_nm_km=0)
    times = time_grid(8192, config.sample_rate_hz)
    t0 = 20e-12
    output = propagate_fiber(gaussian_pulse(times, 0.05, t0), config)
    observed = field_observables(output[None, ...], times)
    peak_phase = (8/9) * 1.3 * 0.05 * (1 - 10**(-16/10)) / (0.2*np.log(10)/10)
    analytic_chirp = 2 * peak_phase * times / t0**2 * np.exp(-(times / t0)**2)
    visible = (np.abs(times) < 2 * t0)
    np.testing.assert_allclose(observed["delta_omega_rad_s"][0, visible, 0], analytic_chirp[visible], rtol=1e-9, atol=0.02)
    mask = observed["phase_valid"]
    assert np.all(np.isnan(observed["phase_xy_rad"][~mask]))
    assert np.all(np.isnan(observed["delta_omega_rad_s"][~mask]))
    assert np.all(np.isfinite(observed["power_xy_W"]))
    assert observed["delta_omega_rad_s"][0, np.searchsorted(times, -t0), 0] < 0
    assert observed["delta_omega_rad_s"][0, np.searchsorted(times, t0), 0] > 0


def test_spectral_energy_units_and_orthogonal_xy_cannot_cancel_coherently():
    times = time_grid(4096, 1e12)
    field = gaussian_pulse(times, 0.05, 20e-12)
    field[:, 1] *= -1  # A coherent sum would be identically zero.
    observed = field_observables(field[None, ...], times)
    df = observed["frequency_offset_Hz"][1] - observed["frequency_offset_Hz"][0]
    spectral_energy = np.sum(observed["spectrum_xy_J_per_Hz"], axis=1) * df
    np.testing.assert_allclose(spectral_energy, observed["energy_xy_J"], rtol=1e-12, atol=1e-25)
    assert np.sum(spectral_energy) == pytest.approx(0.05 * 20e-12 * np.sqrt(np.pi), rel=1e-12)
    assert np.max(observed["spectrum_xy_J_per_Hz"]) > 0
    frequency = observed["frequency_offset_Hz"]
    spectrum = np.sum(observed["spectrum_xy_J_per_Hz"][0], axis=1)
    rms_frequency = np.sqrt(np.sum(frequency**2 * spectrum) / np.sum(spectrum))
    assert rms_frequency == pytest.approx(1 / (2*np.pi*np.sqrt(2)*20e-12), rel=1e-12)


def test_optical_frequency_axis_uses_the_declared_negative_time_carrier_sign():
    times = time_grid(2048, 1e12)
    optical_offset_hz = 16 / (len(times) * (times[1] - times[0]))
    field = np.repeat((0.01*np.exp(-2j*np.pi*optical_offset_hz*times))[:, None], 2, axis=1)
    observed = field_observables(field[None, ...], times)
    peak = np.argmax(observed["spectrum_xy_J_per_Hz"][0, :, 0])
    assert observed["frequency_offset_Hz"][peak] == pytest.approx(optical_offset_hz, rel=1e-12)
    np.testing.assert_allclose(observed["delta_omega_rad_s"], 2*np.pi*optical_offset_hz, rtol=1e-11)


def test_saved_z_trajectory_is_same_integration_as_whole_fiber_without_gain():
    config = pulse_config(span_km=8)
    times = time_grid(4096, config.sample_rate_hz)
    field = gaussian_pulse(times, 0.05, 20e-12)
    snapshots = propagate_snapshots(field, config, np.arange(9, dtype=float))
    np.testing.assert_allclose(snapshots[-1], propagate_fiber(field, config), rtol=1e-12, atol=1e-15)
    initial_energy = np.sum(np.abs(field)**2)
    final_energy = np.sum(np.abs(snapshots[-1])**2)
    assert final_energy / initial_energy == pytest.approx(10**(-0.2*8/10), rel=1e-12)
    with pytest.raises(ValueError, match="explicitly disable ASE"):
        propagate_snapshots(field, replace(config, ase_enabled=True), np.arange(9, dtype=float))
    with pytest.raises(ValueError, match="integer multiples"):
        propagate_snapshots(field, config, np.array([0.0, 0.3]))


def test_combined_pulse_converges_second_order_without_phase_alignment():
    config = pulse_config()
    times = time_grid(8192, config.sample_rate_hz)
    field = gaussian_pulse(times, 0.05, 20e-12)
    outputs = [propagate_fiber(field, config, step_km=step) for step in (0.5, 0.25, 0.125)]
    differences = [np.linalg.norm(outputs[index] - outputs[index+1]) / np.linalg.norm(outputs[index+1]) for index in range(2)]
    assert 3.5 < differences[0] / differences[1] < 4.5
    assert differences[1] < 1e-4


def test_display_decimation_retains_full_window_and_is_qualified_in_physical_units():
    config = pulse_config(span_km=2)
    times = time_grid(8192, config.sample_rate_hz)
    positions = np.array([0.0, 1.0, 2.0])
    field = propagate_snapshots(gaussian_pulse(times, 0.05, 20e-12), config, positions)
    data, record = display_data(field, times, positions, config)
    assert record["passed"]
    assert data["A_xy_sqrtW"].dtype == np.complex64
    assert data["A_xy_sqrtW"].shape == (3, 2048, 2)
    np.testing.assert_array_equal(data["t_ps"], times[::4] * 1e12)
    assert data["t_ps"][0] == pytest.approx(-1000)
    stored_energy = np.sum(np.abs(data["A_xy_sqrtW"].astype(np.complex128))**2, axis=1) * 4 / config.sample_rate_hz
    np.testing.assert_allclose(stored_energy, data["energy_xy_J"], rtol=1e-6, atol=1e-24)


def test_display_fourier_filter_prevents_stride_aliasing_and_reports_lost_energy():
    config = pulse_config()
    times = time_grid(8192, config.sample_rate_hz)
    field = gaussian_pulse(times, 0.05, 20e-12)
    # An out-of-display-band tone would alias under a direct stride. It is
    # deliberately absent from the real Gaussian study, and must be rejected
    # by the display qualification instead of silently omitted.
    field += 0.002 * np.exp(2j*np.pi*(1024/2e-9)*times)[:, None]
    data, record = display_data(field[None, ...], times, np.array([0.0]), config)
    assert not record["passed"]
    assert record["max_discarded_spectral_energy_fraction"] > 0.001
    expected = gaussian_pulse(times[::4], 0.05, 20e-12)
    np.testing.assert_allclose(data["A_xy_sqrtW"][0], expected, atol=1e-8)
