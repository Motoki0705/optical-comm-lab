"""Receiver fits use dedicated pilots; scored/test labels cannot affect DSP."""
from dataclasses import replace
import numpy as np
import pytest
from optical_comm.dsp import fit_pilot_fir, blind_phase_search
from optical_comm.metrics import bits_to_symbols, compute_metrics
from experiments.transformer_nlc.data_config import pilot_config
from experiments.transformer_nlc.receiver import PilotFrame, fit_receiver, select_receiver_window


def fixture_frame(seed=3):
    rng = np.random.default_rng(seed)
    bits = rng.integers(0,2,(8192,2,4),dtype=np.uint8)
    tx = bits_to_symbols(bits)
    mixing = np.array([[.9+.15j,.16-.1j],[-.13+.05j,1.1-.08j]])
    rx = tx@mixing
    # A legitimate finite-memory residual, not an identity receiver fixture.
    rx[1:] += .06*tx[:-1]
    mask = np.zeros(len(rx),bool); mask[128:2176]=mask[-2176:-128]=True
    return PilotFrame(rx,np.where(mask[:,None],tx,0),mask),tx,bits


def test_train_pilot_2x2_fir_reduces_isi_and_polarization_leakage():
    config = pilot_config()
    train,_,_ = fixture_frame(1)
    frame,tx,bits = fixture_frame(2)
    fit = fit_receiver(config,[train])
    rx,valid,metadata = fit.apply(frame)
    scoring = valid & ~frame.pilot_mask
    before = compute_metrics(tx,frame.rx_matched,bits,scoring)
    after = compute_metrics(tx,rx,bits,scoring)
    assert after["mse"] < before["mse"]*.01
    assert after["bit_errors"] == 0
    assert fit.fit_metadata["pilot_rows"] < train.pilot_mask.sum()
    assert metadata["calibration_labels"] == "dedicated_frame_pilots_only"
    # Receiver accepts no scored labels: mutating unavailable ground truth has
    # no effect on either fitting or frame calibration.
    tx[~frame.pilot_mask] = 100j
    bits[~frame.pilot_mask] ^= 1
    np.testing.assert_array_equal(fit.apply(frame)[0],rx)
    with pytest.raises(ValueError, match="scored labels"):
        fit_pilot_fir([frame.rx_matched],[tx],[frame.pilot_mask])


def test_bps_tracks_blind_phase_and_pilot_resolves_ambiguity():
    config = pilot_config()
    train,_,_ = fixture_frame(1)
    fit = fit_receiver(config,[train])
    rng = np.random.default_rng(22)
    bits = rng.integers(0,2,(8192,2,4),dtype=np.uint8)
    tx = bits_to_symbols(bits)
    angle = .2*np.sin(np.arange(len(tx))/1500)[:,None]+np.array([1.8,-1.6])
    rx = tx*np.exp(1j*angle)
    corrected,_,valid = blind_phase_search(rx,window=65)
    # Raw BPS is ambiguous by π/2, as required for square QAM.
    assert compute_metrics(tx,corrected,bits,valid)["ber"] > .1
    mask = np.zeros(len(rx),bool); mask[128:2176]=mask[-2176:-128]=True
    # Identity FIR isolates the CR behavior rather than re-fitting scored labels.
    weights = np.zeros((22,2),complex); weights[10:12] = np.eye(2)
    fit = replace(fit,weights=weights)
    recovered,valid,_ = fit.apply(PilotFrame(rx,np.where(mask[:,None],tx,0),mask))
    metrics = compute_metrics(tx,recovered,bits,valid & ~mask)
    assert metrics["bit_errors"] == 0
    assert metrics["mse"] < .002


def test_validation_window_selection_uses_aggregate_ber_then_mse():
    config = pilot_config()
    frame,tx,bits = fixture_frame(5)
    fit = fit_receiver(config,[frame])
    mask = np.ones(len(tx),bool); mask[:128]=mask[-128:]=False
    selected = select_receiver_window(fit,config,[(frame,tx,bits,mask)])
    records = selected.validation_selection
    assert {r["window"] for r in records} == {33,65,129}
    expected = min(records,key=lambda r:(r["ber"],r["mse"],r["window"]))
    assert selected.bps_window == expected["window"]
