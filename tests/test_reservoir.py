from dataclasses import replace

import numpy as np
import pytest

from optical_comm.metrics import bits_to_symbols, compute_metrics, gray16qam_awgn_ber
from optical_comm.reservoir import (ContextConfig, ESNEqualizer, LinearEqualizer, ReservoirConfig,
                                    SymbolFrame, TrainPreprocessor, load_equalizer, real_to_complex)


def qam_frame(seed=1, n=800, noise=0.1):
    rng = np.random.default_rng(seed)
    symbols = bits_to_symbols(rng.integers(0, 2, (n, 2, 4), dtype=np.uint8))
    received = symbols + noise * (rng.normal(size=symbols.shape) + 1j * rng.normal(size=symbols.shape))
    return SymbolFrame(received, symbols)


def test_context_washout_and_invalid_gap_never_cross_boundaries():
    frame = qam_frame(n=30)
    valid = np.ones(30, dtype=bool)
    valid[12:15] = False
    frame = replace(frame, valid_mask=valid)
    preprocessor = TrainPreprocessor(ContextConfig(2, 2, 3)).fit([frame])
    prepared = preprocessor.prepare(frame)
    np.testing.assert_array_equal(prepared.indices, np.r_[np.arange(2, 10), np.arange(17, 28)])
    np.testing.assert_array_equal(np.flatnonzero(preprocessor.common_mask(frame)),
                                  np.r_[np.arange(5, 10), np.arange(20, 28)])
    model = ESNEqualizer(preprocessor, ReservoirConfig(units=24, density=0.2)).fit([frame])
    changed_received = frame.rx_symbols.copy()
    changed_received[~valid] = 1e6 + 1e6j
    changed = SymbolFrame(changed_received, frame.tx_symbols, valid)
    before, after = model.predict(frame), model.predict(changed)
    np.testing.assert_array_equal(before.symbols[before.valid_mask], after.symbols[after.valid_mask])


def test_train_only_transforms_and_prediction_ignore_other_frame_labels():
    train, held_out = qam_frame(1), qam_frame(2)
    preprocessor = TrainPreprocessor(ContextConfig(1, 1, 10)).fit([train])
    snapshot = tuple(array.copy() for array in (preprocessor.gain, preprocessor.mean, preprocessor.scale))
    model = ESNEqualizer(preprocessor, ReservoirConfig(units=32, density=0.2)).fit([train])
    original = model.predict(held_out)
    poisoned_labels = replace(held_out, tx_symbols=held_out.tx_symbols * (100 + 50j))
    poisoned_prediction = model.predict(poisoned_labels)
    np.testing.assert_array_equal(original.symbols, poisoned_prediction.symbols)
    model.predict(qam_frame(8))  # No state may carry through an intervening frame.
    np.testing.assert_array_equal(original.symbols, model.predict(held_out).symbols)
    for expected, actual in zip(snapshot, (preprocessor.gain, preprocessor.mean, preprocessor.scale)):
        np.testing.assert_array_equal(expected, actual)


@pytest.mark.parametrize("kind", ["linear", "esn"])
def test_checkpoint_is_prediction_complete_and_exact(tmp_path, kind):
    train, test = qam_frame(4), qam_frame(5)
    preprocessor = TrainPreprocessor(ContextConfig(1, 2, 12)).fit([train])
    if kind == "linear":
        model = LinearEqualizer(preprocessor, 1e-3)
    else:
        model = ESNEqualizer(preprocessor, ReservoirConfig(units=32, density=0.2, ridge=1e-3))
    model.fit([train])
    path = tmp_path / "model.npz"
    model.save(path)
    restored = load_equalizer(path)
    expected, actual = model.predict(test), restored.predict(SymbolFrame(test.rx_symbols))
    np.testing.assert_array_equal(expected.symbols, actual.symbols)
    np.testing.assert_array_equal(expected.valid_mask, actual.valid_mask)
    assert restored.training_samples == model.training_samples


def test_seeded_reservoir_is_fixed_during_readout_training():
    frame = qam_frame()
    preprocessor = TrainPreprocessor(ContextConfig(0, 0, 12)).fit([frame])
    configuration = ReservoirConfig(units=32, density=0.2, spectral_radius=0.7, seed=917)
    model, duplicate = ESNEqualizer(preprocessor, configuration), ESNEqualizer(preprocessor, configuration)
    for name in ("input_weights", "bias", "recurrent_data", "recurrent_indices", "recurrent_indptr"):
        np.testing.assert_array_equal(getattr(model, name), getattr(duplicate, name))
    weights = model.input_weights.copy(), model.recurrent_data.copy()
    fitted = model.fit_ridge_path([frame], [1e-4, 1e-2])
    np.testing.assert_array_equal(model.input_weights, weights[0])
    np.testing.assert_array_equal(model.recurrent_data, weights[1])
    assert fitted[0].recurrent_data is fitted[1].recurrent_data
    recurrent = np.zeros((configuration.units, configuration.units))
    for row in range(configuration.units):
        start, end = model.recurrent_indptr[row:row + 2]
        recurrent[row, model.recurrent_indices[start:end]] = model.recurrent_data[start:end]
    assert np.max(np.abs(np.linalg.eigvals(recurrent))) == pytest.approx(0.7, rel=1e-10)


def nonlinear_memory_frame(seed, n):
    rng = np.random.default_rng(seed)
    inputs = rng.uniform(-1, 1, (n, 4))
    target = inputs.copy()
    products = np.roll(inputs[:, 0], 1) * np.roll(inputs[:, 1], 1)
    target[:, 0] += 0.45 * products
    target[:, 3] -= 0.45 * products
    valid = np.ones(n, dtype=bool)
    valid[0] = False
    return SymbolFrame(real_to_complex(inputs), real_to_complex(target), valid)


def test_trained_esn_learns_independent_nonlinear_memory_task():
    train, test = nonlinear_memory_frame(44, 5000), nonlinear_memory_frame(45, 2000)
    preprocessor = TrainPreprocessor(ContextConfig(1, 0, 32)).fit([train])
    linear = LinearEqualizer(preprocessor, ridge=1e-5).fit([train])
    reservoir = ESNEqualizer(preprocessor, ReservoirConfig(units=96, density=0.1,
        spectral_radius=0.7, leak_rate=1, input_scale=0.7, bias_scale=0.8, ridge=1e-5, seed=24)).fit([train])
    baseline, learned = linear.predict(test), reservoir.predict(test)
    np.testing.assert_array_equal(baseline.valid_mask, learned.valid_mask)
    mask = learned.valid_mask
    linear_mse = np.mean(np.abs(baseline.symbols[mask] - test.tx_symbols[mask])**2)
    esn_mse = np.mean(np.abs(learned.symbols[mask] - test.tx_symbols[mask])**2)
    assert esn_mse < 0.7 * linear_mse


def test_forward_gain_calibration_preserves_awgn_ber_theory():
    snr_db, n_train, n_test = 12.0, 100_000, 160_000
    rng = np.random.default_rng(771)
    sigma = np.sqrt(10**(-snr_db / 10) / 2)
    frames = []
    for count in (n_train, n_test):
        bits = rng.integers(0, 2, (count, 2, 4), dtype=np.uint8)
        transmitted = bits_to_symbols(bits)
        received = transmitted + sigma * (rng.normal(size=transmitted.shape) + 1j * rng.normal(size=transmitted.shape))
        frames.append(SymbolFrame(received, transmitted))
    preprocessor = TrainPreprocessor(ContextConfig(0, 0, 0)).fit(frames[:1])
    assert np.max(np.abs(preprocessor.gain - 1)) < 0.005
    measured = compute_metrics(frames[1].tx_symbols, preprocessor.calibrate(frames[1].rx_symbols))
    theoretical = gray16qam_awgn_ber(snr_db)
    standard_error = np.sqrt(theoretical * (1 - theoretical) / measured["n_bits"])
    assert abs(measured["ber"] - theoretical) < 5 * standard_error
