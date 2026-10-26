"""Substantive short physical controls; no full-link convergence claim."""
from dataclasses import replace
import itertools
import numpy as np
import pytest

from optical_comm.channel import ase_noise, edfa_ase_psd, propagate_fiber, propagate_link_adaptive
from optical_comm.dsp import apply_dispersion, pulse_shape, receiver_dsp
from optical_comm.metrics import bits_to_symbols, symbols_to_bits, gray_qam_awgn_ber, gray16qam_awgn_ber, compute_metrics, ber_interval
from experiments.transformer_nlc.data_config import main16, main64


@pytest.mark.parametrize("order", [16, 64])
def test_square_gray_power_neighbors_and_exact_awgn(order):
    bits = np.array(list(itertools.product([0, 1], repeat=int(np.log2(order)))), dtype=np.uint8)
    symbols = bits_to_symbols(bits, order)
    np.testing.assert_array_equal(symbols_to_bits(symbols, order), bits)
    assert np.mean(abs(symbols)**2) == pytest.approx(1.)
    distances = abs(symbols[:, None] - symbols[None, :])
    minimum = distances[distances > 0].min()
    for i, j in zip(*np.where(np.isclose(distances, minimum)), strict=True):
        assert np.count_nonzero(bits[i] != bits[j]) == 1
    rng = np.random.Generator(np.random.PCG64(91))
    tx_bits = rng.integers(0, 2, (100_000, 2, int(np.log2(order))), dtype=np.uint8)
    tx = bits_to_symbols(tx_bits, order)
    snr = 14.
    noise = np.sqrt(10**(-snr/10)/2) * (rng.normal(size=tx.shape) + 1j * rng.normal(size=tx.shape))
    measured = compute_metrics(tx, tx + noise, tx_bits, order=order)
    lo, hi = ber_interval(measured["bit_errors"], measured["n_bits"], .999)
    assert lo < gray_qam_awgn_ber(snr, order) < hi
    assert gray_qam_awgn_ber(-200, order) == pytest.approx(.5, abs=1e-8)
    if order == 16:
        np.testing.assert_allclose(gray_qam_awgn_ber(np.array([0,8,16])), gray16qam_awgn_ber(np.array([0,8,16])), rtol=1e-12)


@pytest.mark.parametrize("factory", [main16, main64])
def test_symmetric_half_cdc_identity_and_b2b(factory):
    config = replace(factory().link, n_spans=2, span_km=2, gamma_per_w_km=0, ase_enabled=False)
    bits = np.random.default_rng(12).integers(0,2,(4096,2,int(np.log2(factory().condition))),dtype=np.uint8)
    symbols = bits_to_symbols(bits, factory().condition)
    field = pulse_shape(symbols, config) * np.sqrt(1e-3/2)
    tx = apply_dispersion(field, config, -config.total_length_km/2)
    fiber, audit = propagate_link_adaptive(tx, config, 4)
    recovered = apply_dispersion(fiber, config, -config.total_length_km/2)
    np.testing.assert_allclose(recovered, field, atol=1e-14, rtol=1e-12)
    rx, two = receiver_dsp(recovered, config, 0, compensate_dispersion=False)
    assert compute_metrics(symbols, rx, bits, order=factory().condition)["bit_errors"] == 0
    assert np.mean(abs(rx-symbols)**2) < 1e-5
    np.testing.assert_array_equal(two[::2], rx)
    assert all(span.max_phase_rad == 0 for span in audit.spans)


def test_adaptive_checked_phase_lossy_kerr_and_independent_refinement():
    config = replace(main16().link, n_spans=1, span_km=3.7, dispersion_ps_nm_km=0, ase_enabled=False)
    field = np.tile([np.sqrt(.03), 1j*np.sqrt(.01)], (128,1))
    got, audit = propagate_link_adaptive(field, config, 17)
    leff = -np.expm1(-config.alpha_per_km*config.span_km)/config.alpha_per_km
    expected = field*np.exp(1j*(8/9)*config.gamma_per_w_km*.04*leff)
    np.testing.assert_allclose(got, expected, rtol=2e-12, atol=1e-14) # EDFA restores loss
    span = audit.spans[0]
    assert span.count > 100 and span.retries > 0
    assert span.distance_km == pytest.approx(config.span_km, abs=1e-12)
    assert span.max_phase_rad <= np.deg2rad(.05)
    with pytest.raises(RuntimeError, match="Minimum"):
        propagate_link_adaptive(field, replace(config, step_km=1), 17, min_step_km=.5)
    rng = np.random.default_rng(67)
    signal = (rng.normal(size=(256,2))+1j*rng.normal(size=(256,2)))*.04
    config = replace(config, alpha_db_per_km=0, dispersion_ps_nm_km=17, span_km=.5)
    reference = propagate_fiber(signal, config, step_km=.0005)
    coarse, _ = propagate_link_adaptive(signal, config, 17, phase_max_deg=2.)
    fine, fine_audit = propagate_link_adaptive(signal, config, 17, phase_max_deg=.2)
    assert np.linalg.norm(fine-reference) < np.linalg.norm(coarse-reference)*.2
    assert fine_audit.spans[0].max_phase_rad <= np.deg2rad(.2)


@pytest.mark.parametrize("factory", [main16, main64])
def test_nf6_fixed_band_physical_noise_and_independent_span_seeds(factory):
    config = factory().link
    photon = 6.62607015e-34*299792458/config.wavelength_m
    expected = photon*(config.edfa_gain*10**.6-1)/2
    assert edfa_ase_psd(config) == pytest.approx(expected)
    coarse = ase_noise(32768, config, np.random.default_rng(13))
    fine = ase_noise(65536, replace(config,sps=8), np.random.default_rng(13))
    np.testing.assert_allclose(fine[::2], coarse, atol=1e-16, rtol=2e-12)
    np.testing.assert_allclose(np.mean(abs(coarse)**2, axis=0), expected*128e9, rtol=.03)
    np.testing.assert_allclose(np.var(coarse.real,axis=0), expected*128e9/2, rtol=.04)
    assert abs(np.mean(coarse[:,0]*coarse[:,1].conj())) < expected*128e9*.03


def test_64qam_full_waveform_edfa_awgn_and_seed_sequence_reuse():
    from experiments.transformer_nlc.data import simulate_physical_frame
    from experiments.transformer_nlc.data_config import pilot_config
    config = pilot_config(64,scored_symbols=32768)
    config = replace(config,launch_power_dbm=-20,link=replace(config.link,n_spans=1,step_km=80,
        dispersion_ps_nm_km=0,gamma_per_w_km=0))
    frame = simulate_physical_frame(config,"pilot",0)
    expected = edfa_ase_psd(config.link)*config.link.symbol_rate_hz/(1e-3*10**(-20/10)/2)
    metrics = compute_metrics(frame.arrays["tx_symbols"],frame.arrays["rx_matched"],frame.arrays["tx_bits"],frame.arrays["scoring_mask"],order=64)
    assert metrics["mse"] == pytest.approx(expected,rel=.025)
    theory = gray_qam_awgn_ber(10*np.log10(1/expected),64)
    lo,hi = ber_interval(metrics["bit_errors"],metrics["n_bits"],.9999)
    assert lo < theory < hi
    short = replace(config.link,n_spans=3,span_km=1,step_km=1)
    sequence = np.random.SeedSequence(72,spawn_key=(64,0,3,0,1))
    sequence.spawn(2) # Caller state mutation must not change the recipe.
    zeros = np.zeros((16384,2),complex)
    first,_ = propagate_link_adaptive(zeros,short,sequence)
    second,_ = propagate_link_adaptive(zeros,short,sequence)
    np.testing.assert_array_equal(first,second)
    expected_noise = sum(ase_noise(len(zeros),short,np.random.Generator(np.random.PCG64(child)))
        for child in np.random.SeedSequence(72,spawn_key=sequence.spawn_key).spawn(3))
    np.testing.assert_allclose(first,expected_noise,atol=1e-17,rtol=1e-12)
