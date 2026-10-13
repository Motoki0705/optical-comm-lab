"""Continuous timing, original masks, whole-frame reproduction and write safety."""

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from optical_comm.channel import simulate_frame
from optical_comm.config import LinkConfig
from optical_comm.experiment import ExperimentConfig, frame_seeds, run_experiment
from optical_comm.reservoir import ContextConfig, ESNEqualizer, LinearEqualizer, TrainPreprocessor


REPOSITORY = Path(__file__).resolve().parents[1]


@pytest.fixture
def exporter(monkeypatch):
    monkeypatch.syspath_prepend(str(REPOSITORY / "scripts"))
    spec = importlib.util.spec_from_file_location("waveform_export_test", REPOSITORY / "scripts/export_waveform_diagnostics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def saved_run(tmp_path_factory):
    config = ExperimentConfig(link=LinkConfig(n_spans=0, guard_symbols=24),
        launch_powers_dbm=(0.0,), root_seed=9173, n_symbols=256,
        train_frames=1, validation_frames=1, test_frames=2,
        context=ContextConfig(past_symbols=2, future_symbols=1, washout_symbols=5),
        linear_ridges=(1e-4,), esn_ridges=(1e-3,),
        esn_candidates=({"units": 8, "density": 0.4},), diagnostic_symbols=20)
    # An isolated tiny CPU fixture, never a scientific evidence run. Export tests
    # below prohibit all subsequent fit/predict calls on these frozen artifacts.
    run = run_experiment(config, tmp_path_factory.mktemp("waveform-input"), "pilot-waveform-test")
    return run, config


def test_contiguous_alignment_masks_and_full_frame_recipe(exporter, saved_run, tmp_path, monkeypatch):
    run, config = saved_run
    before = {str(p.relative_to(run)): exporter.file_hash(p) for p in run.rglob("*") if p.is_file()}
    calls = []
    original_simulate = exporter.simulate_frame

    def observed(*args, **kwargs):
        calls.append(args[2])
        return original_simulate(*args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("export attempted fitting or prediction")

    monkeypatch.setattr(exporter, "simulate_frame", observed)
    monkeypatch.setattr(TrainPreprocessor, "fit", forbidden)
    monkeypatch.setattr(LinearEqualizer, "predict", forbidden)
    monkeypatch.setattr(ESNEqualizer, "predict", forbidden)
    output = tmp_path / "diagnostics"
    manifest = exporter.export_diagnostics(run, output, symbols=128)
    assert calls == [256]  # The 128-symbol slice is never propagated alone.
    entry = manifest["files"][0]
    assert entry["saved_constellation_cross_check"]["all_stored_indices_checked"] == 20
    assert entry["saved_constellation_cross_check"]["rx_symbols_bitwise_equal"]
    seed = next(row for row in frame_seeds(config, 0) if row["split"] == "test" and row["frame_index"] == 0)
    reference = simulate_frame(config.link, 0, config.n_symbols, seed["bit_seed"], seed["noise_seed"])
    with np.load(output / entry["file"], allow_pickle=False) as saved:
        # Independently specified global sample positions, not merely a shape test.
        expected_symbols = np.arange(8, 136)
        np.testing.assert_array_equal(saved["symbol_indices"], expected_symbols)
        np.testing.assert_array_equal(saved["sample_indices_2sps"], np.arange(16, 272))
        np.testing.assert_array_equal(saved["rx_2sps"], reference.rx_2sps[16:272])
        np.testing.assert_array_equal(saved["rx_symbols"], reference.rx_symbols[8:136])
        np.testing.assert_array_equal(saved["rx_2sps"][::2], saved["rx_symbols"])
        np.testing.assert_array_equal(saved["tx_symbols"], reference.tx_symbols[8:136])
        np.testing.assert_array_equal(saved["tx_bits"], reference.tx_bits[8:136])
        # Original guard24, past2, future1, washout5: scored global indices
        # are [31,231), while the channel-valid interval is [24,232).
        np.testing.assert_array_equal(saved["frame_valid_mask"], (expected_symbols >= 24) & (expected_symbols < 232))
        np.testing.assert_array_equal(saved["score_mask"], (expected_symbols >= 31) & (expected_symbols < 231))
        assert saved["rx_2sps"].dtype == np.complex128
        assert saved["tx_bits"].shape == (128, 2, 4)
    assert manifest["evaluation_bits_added"] == manifest["new_independent_test_frames"] == 0
    assert manifest["diagnostic_frames_reconstructed"] == 1
    assert not manifest["fit_selection_prediction_or_training_executed"]
    assert manifest["input_files_sha256"] == before
    assert before == {str(p.relative_to(run)): exporter.file_hash(p) for p in run.rglob("*") if p.is_file()}
    assert manifest["command"].startswith(str(Path(sys.executable))) or manifest["command"].startswith(".venv/bin/python")


def test_same_seed_saved_arrays_reproduce(exporter, saved_run, tmp_path):
    run, _ = saved_run
    first = exporter.export_diagnostics(run, tmp_path / "first", start_symbol=0, symbols=80)
    second = exporter.export_diagnostics(run, tmp_path / "second", start_symbol=0, symbols=80)
    assert first["files"][0]["sha256"] == second["files"][0]["sha256"]
    assert first["input_files_sha256"] == second["input_files_sha256"]
    stored = json.loads((tmp_path / "second" / "manifest.json").read_text())
    assert stored["waveform_stage"] == exporter.STAGE
    assert stored["periodic_frame"] is True
    assert stored["frame_boundaries_symbol_indices"] == [0, 256]
    assert stored["diagnostic_sample_rate_hz"] == 64e9
    assert not stored["train_fitted_gain_or_feature_normalization_applied"]


def test_later_original_test_frame_marks_unavailable_constellation_reference(exporter, saved_run, tmp_path):
    run, config = saved_run
    manifest = exporter.export_diagnostics(run, tmp_path / "later", frame_index=1, symbols=64)
    entry = manifest["files"][0]
    seed = next(row for row in frame_seeds(config, 0) if row["split"] == "test" and row["frame_index"] == 1)
    assert entry["bit_seed"] == seed["bit_seed"]
    assert entry["noise_seed"] == seed["noise_seed"]
    assert entry["saved_constellation_cross_check"]["available_for_selected_frame"] is False
    assert manifest["new_independent_test_frames"] == 0


@pytest.mark.parametrize("options,match", [
    ({"symbols": 0}, "symbols"), ({"symbols": 257}, "extends"),
    ({"start_symbol": -1}, "start_symbol"), ({"start_symbol": 240, "symbols": 32}, "extends"),
    ({"frame_index": -1}, "frame_index"), ({"frame_index": 2}, "test split"),
    ({"backend": "other"}, "backend"),
])
def test_invalid_arguments_do_not_create_output(exporter, saved_run, tmp_path, options, match):
    run, _ = saved_run
    output = tmp_path / "invalid"
    arguments = {"symbols": 32}
    arguments.update(options)
    with pytest.raises(ValueError, match=match):
        exporter.export_diagnostics(run, output, **arguments)
    assert not output.exists()


def test_output_cannot_overwrite_input_or_existing_diagnostics(exporter, saved_run, tmp_path):
    run, _ = saved_run
    for output in (run, run / "power_p0000p000", REPOSITORY / "src/diagnostics"):
        with pytest.raises(ValueError, match="read-only|overwrite"):
            exporter.export_diagnostics(run, output, symbols=32)
    output = tmp_path / "existing"
    output.mkdir()
    (output / "preserve.txt").write_text("existing")
    with pytest.raises(ValueError, match="empty"):
        exporter.export_diagnostics(run, output, symbols=32)
    assert (output / "preserve.txt").read_text() == "existing"


def test_cli_failure_is_nonzero_and_does_not_write(saved_run, tmp_path):
    run, _ = saved_run
    output = tmp_path / "cli-invalid"
    result = subprocess.run([sys.executable, str(REPOSITORY / "scripts/export_waveform_diagnostics.py"),
                             "--run", str(run), "--output-dir", str(output), "--frame-index", "2"],
                            capture_output=True, text=True)
    assert result.returncode == 1
    assert "Waveform export failed" in result.stderr
    assert not output.exists()


def test_changed_input_hash_is_rejected_before_propagation(exporter, saved_run, tmp_path, monkeypatch):
    run, _ = saved_run
    changed = tmp_path / "changed-input"
    shutil.copytree(run, changed)
    artifact = changed / "power_p0000p000/constellations.npz"
    artifact.write_bytes(artifact.read_bytes() + b"changed")

    def forbidden(*args, **kwargs):
        pytest.fail("tampered input triggered propagation")

    monkeypatch.setattr(exporter, "simulate_frame", forbidden)
    output = tmp_path / "rejected-output"
    with pytest.raises(ValueError, match="saved file hash differs"):
        exporter.export_diagnostics(changed, output, symbols=32)
    assert not output.exists()
