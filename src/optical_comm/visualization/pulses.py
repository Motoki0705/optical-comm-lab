"""SSFM pulse trajectories and physically scaled observables for rendering.

The solver is the existing public ``propagate_fiber`` API, without amplifiers.
Arrays are (distance, time, polarization); fields are sqrt(W), time is seconds,
distance is km. The carrier convention is E=Re[A exp(i(kz-omega0*t))], hence
optical frequency offset is minus the usual NumPy FFT frequency.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import math
from typing import Any

import numpy as np
from scipy import fft
from scipy.signal import resample

from optical_comm.channel import propagate_fiber
from optical_comm.config import LinkConfig


def time_grid(n_samples: int, sample_rate_hz: float) -> np.ndarray:
    """A centered, even, uniformly sampled periodic time grid [s]."""
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 4 or n_samples % 2:
        raise ValueError("n_samples must be an even integer >= 4")
    if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be finite and positive")
    return (np.arange(n_samples, dtype=np.float64) - n_samples // 2) / sample_rate_hz


def _dt(t_s: np.ndarray) -> float:
    times = np.asarray(t_s, dtype=np.float64)
    if times.ndim != 1 or len(times) < 4 or not np.all(np.isfinite(times)):
        raise ValueError("t_s must be a finite one-dimensional uniform grid")
    spacing = float(times[1] - times[0])
    if spacing <= 0 or not np.allclose(np.diff(times), spacing, rtol=1e-10, atol=spacing * 1e-10):
        raise ValueError("t_s must be an increasing uniform grid")
    return spacing


def gaussian_pulse(t_s: np.ndarray, peak_xy_power_w: float, t0_s: float) -> np.ndarray:
    """Equal XY Gaussian fields; XY power=Ppeak*exp(-(t/T0)**2).

    T0 is the intensity 1/e half-width, not the intensity FWHM or RMS width.
    Ppeak is instantaneous XY-total peak power, not communications mean power.
    """
    _dt(t_s)
    if not math.isfinite(peak_xy_power_w) or peak_xy_power_w <= 0:
        raise ValueError("peak_xy_power_w must be finite and positive")
    if not math.isfinite(t0_s) or t0_s <= 0:
        raise ValueError("t0_s must be finite and positive")
    one_pol = np.sqrt(peak_xy_power_w / 2) * np.exp(-0.5 * (np.asarray(t_s) / t0_s) ** 2)
    return np.repeat(one_pol[:, None], 2, axis=1).astype(np.complex128)


def propagate_snapshots(
    field: np.ndarray,
    config: LinkConfig,
    z_km: np.ndarray,
    *,
    step_km: float | None = None,
) -> np.ndarray:
    """Propagate one isolated periodic pulse and retain requested z snapshots.

    Each successive interval is passed to the repository's loss/dispersion/
    Manakov SSFM. ``propagate_fiber`` never calls the amplifier/noise model.
    The requested snapshot interval must be a multiple of the solver step, so
    adding snapshots cannot change the integration discretization.
    """
    values = np.asarray(field, dtype=np.complex128)
    positions = np.asarray(z_km, dtype=np.float64)
    step = config.step_km if step_km is None else step_km
    if values.ndim != 2 or values.shape[1] != 2 or not np.all(np.isfinite(values)):
        raise ValueError("field must be a finite (time,2) complex array")
    if positions.ndim != 1 or len(positions) < 2 or not np.all(np.isfinite(positions)):
        raise ValueError("z_km must contain at least two finite positions")
    if positions[0] != 0 or not np.all(np.diff(positions) > 0):
        raise ValueError("z_km must start at zero and increase strictly")
    if not math.isfinite(step) or step <= 0:
        raise ValueError("step_km must be finite and positive")
    if not np.allclose(np.diff(positions) / step, np.rint(np.diff(positions) / step), atol=1e-9, rtol=0):
        raise ValueError("snapshot intervals must be integer multiples of step_km")
    if config.ase_enabled:
        raise ValueError("illustrative pulse configuration must explicitly disable ASE")
    trajectory = np.empty((len(positions), len(values), 2), dtype=np.complex128)
    trajectory[0] = values
    for index, interval in enumerate(np.diff(positions), 1):
        trajectory[index] = propagate_fiber(
            trajectory[index - 1], replace(config, span_km=float(interval)), step_km=step
        )
    return trajectory


def analytic_lossy_kerr(field: np.ndarray, config: LinkConfig, z_km: float) -> np.ndarray:
    """Exact D=0 solution for arbitrary finite XY input fields."""
    if config.dispersion_ps_nm_km != 0:
        raise ValueError("lossy Kerr analytic limit requires D=0")
    effective_length = z_km if config.alpha_per_km == 0 else -np.expm1(-config.alpha_per_km * z_km) / config.alpha_per_km
    power_xy = np.sum(np.abs(field) ** 2, axis=-1)
    phase = (8 / 9) * config.gamma_per_w_km * power_xy * effective_length
    return np.asarray(field) * np.exp(-config.alpha_per_km * z_km / 2 + 1j * phase[..., None])


def analytic_dispersion_gaussian(
    t_s: np.ndarray, peak_xy_power_w: float, t0_s: float, config: LinkConfig, z_km: float
) -> np.ndarray:
    """Infinite-domain complex Gaussian for the implemented +i beta2 omega²/2.

    Its complex width is q=1-i*beta2*z/T0². This checks both amplitude and phase,
    rather than fitting a Gaussian intensity after propagation.
    """
    if config.gamma_per_w_km != 0:
        raise ValueError("dispersion analytic limit requires gamma=0")
    q = 1 - 1j * config.beta2_s2_per_km * z_km / t0_s**2
    one_pol = np.sqrt(peak_xy_power_w / 2) / np.sqrt(q)
    one_pol = one_pol * np.exp(-0.5 * (t_s / t0_s) ** 2 / q - config.alpha_per_km * z_km / 2)
    return np.repeat(one_pol[:, None], 2, axis=1)


def _masked_phase(field: np.ndarray, valid: np.ndarray, center: int) -> np.ndarray:
    """Unwrap contiguous valid intervals, anchoring the center across distance."""
    phase = np.full(field.shape, np.nan, dtype=np.float64)
    center_phase = np.unwrap(np.angle(field[:, center, :]), axis=0)
    for snapshot in range(len(field)):
        for pol in range(2):
            indices = np.flatnonzero(valid[snapshot, :, pol])
            for segment in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1):
                if not len(segment):
                    continue
                values = np.unwrap(np.angle(field[snapshot, segment, pol]))
                if center in segment:
                    anchor = int(np.searchsorted(segment, center))
                    values += center_phase[snapshot, pol] - values[anchor]
                else:
                    # A disconnected lobe has no phase connection across the
                    # masked zero. Retain its local unwrap with a principal
                    # phase anchor at that lobe's strongest sample.
                    anchor = int(np.argmax(np.abs(field[snapshot, segment, pol])))
                    values += np.angle(field[snapshot, segment[anchor], pol]) - values[anchor]
                phase[snapshot, segment, pol] = values
    return phase


def field_observables(
    field: np.ndarray, t_s: np.ndarray, *, phase_relative_threshold: float = 1e-6
) -> dict[str, np.ndarray]:
    """Power, phase, instantaneous optical offset and spectral energy density.

    Spectra are |dt FFT(A_pol)|² [J/Hz], independently for each polarization.
    Their frequency integral equals pulse energy, not average/peak power. Phase
    and chirp are undefined in low-power tails; these values are NaN with an
    explicit validity mask. No power or spectrum is masked or renormalized.
    """
    values = np.asarray(field, dtype=np.complex128)
    spacing = _dt(t_s)
    if values.ndim != 3 or values.shape[1:] != (len(t_s), 2) or not np.all(np.isfinite(values)):
        raise ValueError("field must be finite with shape (distance,time,2)")
    if not 0 < phase_relative_threshold < 1:
        raise ValueError("phase_relative_threshold must be in (0,1)")
    power = np.abs(values) ** 2
    peak_xy = np.max(np.sum(power, axis=-1), axis=1)
    # The threshold is XY-total. Both identical input polarizations have the
    # same mask, without doubling the physical input power.
    valid = power > (phase_relative_threshold * peak_xy[:, None, None] / 2)
    center = int(np.argmin(np.abs(t_s)))
    phase = _masked_phase(values, valid, center)
    numpy_frequencies = fft.fftfreq(len(t_s), spacing)
    spectral_field = fft.fft(values, axis=1)
    derivative = fft.ifft(spectral_field * (2j * np.pi * numpy_frequencies)[None, :, None], axis=1)
    chirp = np.full(power.shape, np.nan)
    np.divide(-np.imag(np.conj(values) * derivative), power, out=chirp, where=valid)
    optical_order = np.argsort(-numpy_frequencies)
    optical_frequencies = -numpy_frequencies[optical_order]
    spectral_energy = np.abs(spectral_field[:, optical_order, :] * spacing) ** 2
    return {
        "power_xy_W": power,
        "phase_xy_rad": phase,
        "delta_omega_rad_s": chirp,
        "phase_valid": valid,
        "frequency_offset_Hz": optical_frequencies,
        "spectrum_xy_J_per_Hz": spectral_energy,
        "energy_xy_J": np.sum(power, axis=1) * spacing,
    }


def _relative_rms(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))


def _width_metrics(observables: dict[str, np.ndarray], t_s: np.ndarray) -> dict[str, np.ndarray]:
    power = np.sum(observables["power_xy_W"], axis=-1)
    spectrum = np.sum(observables["spectrum_xy_J_per_Hz"], axis=-1)
    frequencies = observables["frequency_offset_Hz"]
    mean_time = np.sum(power * t_s[None, :], axis=1) / np.sum(power, axis=1)
    mean_frequency = np.sum(spectrum * frequencies[None, :], axis=1) / np.sum(spectrum, axis=1)
    return {
        "rms_time_ps": np.sqrt(np.sum(power * (t_s[None, :] - mean_time[:, None]) ** 2, axis=1) / np.sum(power, axis=1)) * 1e12,
        "rms_frequency_GHz": np.sqrt(np.sum(spectrum * (frequencies[None, :] - mean_frequency[:, None]) ** 2, axis=1) / np.sum(spectrum, axis=1)) * 1e-9,
        "peak_xy_power_W": np.max(power, axis=1),
        "center_phase_rad": observables["phase_xy_rad"][:, int(np.argmin(np.abs(t_s))), 0],
    }


def trajectory_metrics(
    field: np.ndarray, t_s: np.ndarray, z_km: np.ndarray, config: LinkConfig,
    *, phase_relative_threshold: float = 1e-6, time_view_ps: float = 300,
    frequency_view_GHz: float = 60,
) -> dict[str, Any]:
    """Scalar diagnostic trajectories on the full solver grid."""
    observables = field_observables(field, t_s, phase_relative_threshold=phase_relative_threshold)
    result: dict[str, Any] = {key: value.tolist() for key, value in _width_metrics(observables, t_s).items()}
    total_power = np.sum(observables["power_xy_W"], axis=-1)
    total_spectrum = np.sum(observables["spectrum_xy_J_per_Hz"], axis=-1)
    result.update({
        "z_km": np.asarray(z_km).tolist(),
        "energy_xy_J": observables["energy_xy_J"].tolist(),
        "energy_outside_time_view_fraction": (np.sum(total_power[:, np.abs(t_s * 1e12) > time_view_ps], axis=1) / np.sum(total_power, axis=1)).tolist(),
        "energy_outside_frequency_view_fraction": (np.sum(total_spectrum[:, np.abs(observables["frequency_offset_Hz"]) > frequency_view_GHz * 1e9], axis=1) / np.sum(total_spectrum, axis=1)).tolist(),
        "attenuation_power_factor": np.exp(-config.alpha_per_km * z_km).tolist(),
        "field_dtype": str(field.dtype),
        "full_field_sha256": hashlib.sha256(np.ascontiguousarray(field).tobytes()).hexdigest(),
    })
    return result


def display_data(
    field: np.ndarray, t_s: np.ndarray, z_km: np.ndarray, config: LinkConfig,
    *, decimation: int = 4, phase_relative_threshold: float = 1e-6,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Fourier low-pass decimation followed by explicitly qualified float32.

    The full time window is retained. scipy.signal.resample truncates the
    periodic Fourier series before lowering Fs; a direct stride would alias.
    All spectral, power and chirp observables are computed in complex128 before
    their display-only float32 export. Validation never uses this saved field.
    """
    if isinstance(decimation, bool) or not isinstance(decimation, int) or decimation < 1 or field.shape[1] % decimation:
        raise ValueError("decimation must be a positive divisor of the time sample count")
    spacing = _dt(t_s)
    display_n = field.shape[1] // decimation
    reduced = resample(field, display_n, axis=1)
    times = np.asarray(t_s)[::decimation]
    observables = field_observables(reduced, times, phase_relative_threshold=phase_relative_threshold)
    exported = {
        "z_km": np.asarray(z_km, dtype=np.float64),
        "t_ps": times * 1e12,
        "A_xy_sqrtW": reduced.astype(np.complex64),
        **{key: value.astype(np.float32) if key not in {"phase_valid", "frequency_offset_Hz", "energy_xy_J"} else value for key, value in observables.items()},
        "attenuation_power_factor": np.exp(-config.alpha_per_km * z_km),
    }
    source_fft = fft.fft(field, axis=1)
    outside_band = np.abs(fft.fftfreq(field.shape[1], spacing)) >= 1 / (2 * spacing * decimation)
    discarded = np.sum(np.abs(source_fft[:, outside_band, :]) ** 2, axis=(1, 2)) / np.sum(np.abs(source_fft) ** 2, axis=(1, 2))
    original_common = field[:, ::decimation, :]
    lossless_reduced_error = _relative_rms(reduced, original_common)
    source_observables = field_observables(field, t_s, phase_relative_threshold=phase_relative_threshold)
    common_valid = exported["phase_valid"] & source_observables["phase_valid"][:, ::decimation, :]
    source_phase = source_observables["phase_xy_rad"][:, ::decimation, :]
    source_chirp = source_observables["delta_omega_rad_s"][:, ::decimation, :]
    relative_energy_error = np.max(np.abs(exported["energy_xy_J"] / source_observables["energy_xy_J"] - 1))
    # Energy computed from stored complex64, not merely the rounded cached
    # energy column, verifies the precision of the material actually rendered.
    stored_energy = np.sum(np.abs(exported["A_xy_sqrtW"].astype(np.complex128)) ** 2, axis=1) * spacing * decimation
    export_record = {
        "method": "periodic Fourier truncation before downsampling; full time window retained",
        "source_n_samples": int(field.shape[1]),
        "display_n_samples": display_n,
        "decimation": decimation,
        "source_dt_ps": spacing * 1e12,
        "display_dt_ps": spacing * decimation * 1e12,
        "display_field_dtype": "complex64",
        "display_observables_dtype": "float32 except axes, energy and attenuation=float64; mask=bool",
        "max_discarded_spectral_energy_fraction": float(np.max(discarded)),
        "lowpass_field_relative_rms_vs_common_source_samples": lossless_reduced_error,
        "stored_field_relative_rms_vs_common_source_samples": _relative_rms(exported["A_xy_sqrtW"], original_common),
        "max_energy_relative_error_before_storage": float(relative_energy_error),
        "max_stored_field_energy_relative_error": float(np.max(np.abs(stored_energy / source_observables["energy_xy_J"] - 1))),
        "max_valid_phase_absolute_error_rad": float(np.max(np.abs(exported["phase_xy_rad"][common_valid] - source_phase[common_valid]))),
        "max_valid_chirp_absolute_error_GHz": float(np.max(np.abs(exported["delta_omega_rad_s"][common_valid] - source_chirp[common_valid])) / (2 * np.pi * 1e9)),
        "thresholds": {"discarded_energy_fraction": 1e-10, "stored_field_relative_rms": 1e-6, "stored_energy_relative_error": 1e-6, "phase_error_rad": 1e-5, "chirp_error_GHz": 1e-4},
    }
    export_record["passed"] = bool(
        export_record["max_discarded_spectral_energy_fraction"] < 1e-10
        and export_record["stored_field_relative_rms_vs_common_source_samples"] < 1e-6
        and export_record["max_stored_field_energy_relative_error"] < 1e-6
        and export_record["max_valid_phase_absolute_error_rad"] < 1e-5
        and export_record["max_valid_chirp_absolute_error_GHz"] < 1e-4
    )
    return exported, export_record


def qualify_pulse_trajectories(
    trajectories: dict[str, np.ndarray], configs: dict[str, LinkConfig],
    t_s: np.ndarray, z_km: np.ndarray, peak_xy_power_w: float, t0_s: float,
    *, step_sizes_km: tuple[float, float, float] = (0.5, 0.25, 0.125),
    phase_relative_threshold: float = 1e-6,
) -> dict[str, Any]:
    """Analytic limits, energy, true refinement, and snapshot equivalence.

    Numerical verification of this isolated pulse is separate from the main
    link's qualification, model validation, BER testing, and independent review.
    All comparisons use complex128 solver fields, without fitted phase/gain.
    """
    spacing = _dt(t_s)
    input_field = gaussian_pulse(t_s, peak_xy_power_w, t0_s)
    checks: dict[str, Any] = {}
    nonlinear_config = configs["pure_spm"]
    analytic_kerr = np.stack([analytic_lossy_kerr(input_field, nonlinear_config, float(z)) for z in z_km])
    linear_config = configs["dispersion_only"]
    analytic_linear = np.stack([analytic_dispersion_gaussian(t_s, peak_xy_power_w, t0_s, linear_config, float(z)) for z in z_km])
    checks["analytic_limits"] = {
        "lossy_kerr_complex_field_relative_rms": _relative_rms(trajectories["pure_spm"], analytic_kerr),
        "complex_gaussian_dispersion_field_relative_rms": _relative_rms(trajectories["dispersion_only"], analytic_linear),
        "threshold_relative_rms": 1e-10,
    }
    checks["analytic_limits"]["passed"] = bool(
        checks["analytic_limits"]["lossy_kerr_complex_field_relative_rms"] < 1e-10
        and checks["analytic_limits"]["complex_gaussian_dispersion_field_relative_rms"] < 1e-10
    )
    energy_checks: dict[str, Any] = {}
    initial_energy = np.sum(np.abs(input_field) ** 2, axis=0) * spacing
    for name, field in trajectories.items():
        observables = field_observables(field, t_s, phase_relative_threshold=phase_relative_threshold)
        expected = initial_energy[None, :] * np.exp(-configs[name].alpha_per_km * z_km[:, None])
        df = 1 / (len(t_s) * spacing)
        spectral_integral = np.sum(observables["spectrum_xy_J_per_Hz"], axis=1) * df
        energy_checks[name] = {
            "max_expected_loss_energy_relative_error": float(np.max(np.abs(observables["energy_xy_J"] / expected - 1))),
            "max_parseval_relative_error": float(np.max(np.abs(spectral_integral / observables["energy_xy_J"] - 1))),
        }
    checks["energy"] = {"cases": energy_checks, "threshold_relative_error": 1e-10, "passed": bool(all(max(record.values()) < 1e-10 for record in energy_checks.values()))}
    whole_span: dict[str, float] = {}
    for name, field in trajectories.items():
        whole = propagate_fiber(input_field, replace(configs[name], span_km=float(z_km[-1])))
        whole_span[name] = _relative_rms(field[-1], whole)
    checks["segmented_vs_whole"] = {"field_relative_rms": whole_span, "threshold_relative_rms": 1e-10, "passed": bool(max(whole_span.values()) < 1e-10)}
    combined_config = configs["combined"]
    endpoints = [propagate_fiber(input_field, combined_config, step_km=step) for step in step_sizes_km]
    coarse_middle = _relative_rms(endpoints[0], endpoints[1])
    middle_fine = _relative_rms(endpoints[1], endpoints[2])
    ratio = coarse_middle / middle_fine
    fine_metrics = _width_metrics(field_observables(endpoints[2][None, ...], t_s), t_s)
    middle_metrics = _width_metrics(field_observables(endpoints[1][None, ...], t_s), t_s)
    metric_differences = {key: float(abs(middle_metrics[key][0] - value[0])) for key, value in fine_metrics.items()}
    checks["step_refinement"] = {
        "case": "combined", "step_sizes_km": list(step_sizes_km),
        "coarse_middle_field_relative_rms": coarse_middle,
        "middle_fine_field_relative_rms": middle_fine,
        "difference_ratio": ratio, "expected_second_order_ratio": 4,
        "production_to_continuum_relative_rms_richardson_estimate": middle_fine * 4 / 3,
        "production_vs_fine_metric_absolute_differences": metric_differences,
        "thresholds": {"production_vs_fine_field_relative_rms": 1e-4, "second_order_difference_ratio_min": 3.5, "second_order_difference_ratio_max": 4.5},
        "passed": bool(middle_fine < 1e-4 and 3.5 < ratio < 4.5),
    }
    # Double Fs at a fixed physical period, then double the period at fixed Fs.
    # These probes compare actual independently propagated fields, not an FFT
    # interpolation of the baseline output or an exactly cancelling inverse.
    fine_config = replace(combined_config, sps=combined_config.sps * 2)
    fine_times = time_grid(len(t_s) * 2, fine_config.sample_rate_hz)
    fine_input = gaussian_pulse(fine_times, peak_xy_power_w, t0_s)
    fine_output = propagate_fiber(fine_input, fine_config)
    window_times = time_grid(len(t_s) * 2, combined_config.sample_rate_hz)
    window_input = gaussian_pulse(window_times, peak_xy_power_w, t0_s)
    window_output = propagate_fiber(window_input, combined_config)
    central = slice(len(t_s) // 2, 3 * len(t_s) // 2)
    grid_error = _relative_rms(trajectories["combined"][-1], fine_output[::2])
    window_error = _relative_rms(trajectories["combined"][-1], window_output[central])
    outside_energy = np.sum(np.abs(np.concatenate((window_output[:len(t_s)//2], window_output[3*len(t_s)//2:]))) ** 2) / np.sum(np.abs(window_output) ** 2)
    checks["time_grid_and_window"] = {
        "baseline_n_samples": len(t_s), "baseline_period_ns": len(t_s) * spacing * 1e9,
        "baseline_dt_ps": spacing * 1e12,
        "refined_n_samples": len(fine_times), "refined_dt_ps": 1e12 / fine_config.sample_rate_hz,
        "expanded_period_ns": len(window_times) * spacing * 1e9,
        "time_grid_field_relative_rms": grid_error,
        "expanded_window_central_field_relative_rms": window_error,
        "expanded_window_energy_outside_original_fraction": float(outside_energy),
        "thresholds": {"field_relative_rms": 1e-8, "outside_original_energy_fraction": 1e-10},
        "passed": bool(grid_error < 1e-8 and window_error < 1e-8 and outside_energy < 1e-10),
    }
    checks["passed"] = all(record["passed"] for record in checks.values())
    checks["scope"] = "Normal numerical verification of an illustrative deterministic isolated pulse; no new communications scores or independent-validator evaluation."
    return checks
