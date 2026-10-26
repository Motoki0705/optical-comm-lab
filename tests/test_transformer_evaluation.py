"""Freeze-before-test, exact Gray counts, Eq2, reload and independent recount."""
from dataclasses import replace
import importlib.util
import json

import numpy as np
import pytest

if importlib.util.find_spec("torch") is None:
    pytest.skip("optional torch extra is absent", allow_module_level=True)

from optical_comm.metrics import bits_to_symbols
from experiments.transformer_nlc.evaluation import compensate, evaluate, freeze_selection, independent_recount, score_symbols
from experiments.transformer_nlc.training import train
from test_transformer_training import make_dataset, tiny_training


@pytest.mark.parametrize("delta", [0, 3, -3])
def test_eq2_sign_and_unit_es_scaling(delta):
    tx = np.array([[1 + 1j, -1 - 1j]], dtype=np.complex128)
    residual = np.array([[.1 + .2j, -.3 + .1j]])
    alpha = 10 ** (delta / 10)
    np.testing.assert_allclose(compensate(tx + alpha * residual, residual, train_power_dbm=2, eval_power_dbm=2 + delta), tx)


@pytest.mark.parametrize("condition", [16, 64])
def test_direct_gray_counts_and_zero_error_q_bound(condition):
    bits = np.zeros((7, 2, int(np.log2(condition))), dtype=np.uint8)
    tx = bits_to_symbols(bits, condition)
    prediction_bits = bits.copy()
    prediction_bits[0, 0, 0] = 1
    predicted = bits_to_symbols(prediction_bits, condition)
    metrics = score_symbols(tx, predicted, bits, condition)
    assert metrics["X"]["bit_errors"] == metrics["total"]["bit_errors"] == 1
    assert metrics["Y"]["bit_errors"] == 0
    assert metrics["total"]["n_bits"] == 7 * 2 * int(np.log2(condition))
    assert metrics["Y"]["ber_ci_high"] > 0 and metrics["Y"]["q_ber_db"] is None
    assert metrics["Y"]["q_ci_low_db"] is not None


def test_freeze_does_not_open_test_then_checkpoint_reload_and_recount(tmp_path):
    dataset = make_dataset(tmp_path / "data")
    config = tiny_training(max_epochs=1)
    train(config, dataset, tmp_path / "masked")
    unmasked = replace(config, masked=False, model=replace(config.model, mask_rho=None))
    train(unmasked, dataset, tmp_path / "unmasked")
    raw = (dataset.parent / "test.npz").read_bytes()
    (dataset.parent / "test.npz").unlink()
    with pytest.raises(ValueError, match="explicitly marked"):
        freeze_selection([tmp_path / "masked"], tmp_path / "forbidden.json")
    frozen = freeze_selection([tmp_path / "masked", tmp_path / "unmasked"], tmp_path / "freeze.json", allow_pilot=True)
    assert frozen["test_metrics_seen"] is False and frozen["regime"] == "pilot"
    assert len(frozen["trials"]) == len(frozen["primary"]) == 2
    (dataset.parent / "test.npz").write_bytes(raw)
    first = evaluate(tmp_path / "freeze.json", tmp_path / "evaluation")
    second = evaluate(tmp_path / "freeze.json", tmp_path / "reload")
    assert first["status"] == "complete" and len(first["replicates"]) == 3
    assert [r["metrics"] for r in first["replicates"]] == [r["metrics"] for r in second["replicates"]]
    assert first["independent_recount"]["status"] == "passed"
    assert independent_recount(tmp_path / "evaluation/metrics.json")["status"] == "passed"
    assert (tmp_path / "evaluation/metrics.csv").exists()
    assert all(r["metrics"]["total"]["n_bits"] == 40 * 8 for r in first["replicates"])
    with pytest.raises(FileExistsError):
        evaluate(tmp_path / "freeze.json", tmp_path / "evaluation")
    with pytest.raises(FileExistsError):
        freeze_selection([tmp_path / "masked"], tmp_path / "freeze.json", allow_pilot=True)
    diagnostic = tmp_path / "evaluation" / first["frames"][0]["diagnostic_path"]
    diagnostic.write_bytes(diagnostic.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Diagnostic hash"):
        independent_recount(tmp_path / "evaluation/metrics.json")


def test_freeze_rejects_paused_run_and_checkpoint_tampering(tmp_path):
    dataset = make_dataset(tmp_path / "data")
    config = tiny_training()
    train(config, dataset, tmp_path / "run", stop_after_epochs=1)
    with pytest.raises(ValueError, match="completed runs"):
        freeze_selection([tmp_path / "run"], tmp_path / "freeze.json", allow_pilot=True)
    train(config, dataset, tmp_path / "run", resume=True)
    freeze_selection([tmp_path / "run"], tmp_path / "freeze.json", allow_pilot=True)
    checkpoint = tmp_path / "run/best.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Checkpoint hash"):
        evaluate(tmp_path / "freeze.json", tmp_path / "evaluation")


def test_primary_uses_all_replicates_and_fixed_table_tie(tmp_path):
    directories = []
    scores = []
    for replicate in range(3):
        dataset = make_dataset(tmp_path / f"data-{replicate}", replicate=replicate)
        for candidate in ("A", "B", "C"):
            # Deliberately identical small pilot architecture/seed across names:
            # scores and RMPS tie exactly, testing the fixed lower-table rule.
            config = tiny_training(max_epochs=1, candidate=candidate)
            directory = tmp_path / f"{candidate}-{replicate}"
            record = train(config, dataset, directory)
            directories.append(directory)
            if candidate == "C":
                scores.append(record["best_validation_mse"])
    frozen = freeze_selection(directories, tmp_path / "freeze.json", allow_pilot=True)
    assert len(frozen["primary"]) == 1
    selected = frozen["primary"][0]
    assert selected["candidate"] == "C" and selected["replicates"] == [0, 1, 2]
    assert selected["mean_validation_mse"] == np.mean(scores)
