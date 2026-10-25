from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

import optical_comm.experiment as experiment
from optical_comm.config import LinkConfig
from optical_comm.metrics import bits_to_symbols
from optical_comm.reservoir import ContextConfig


def small_config():
    return experiment.ExperimentConfig(link=LinkConfig(n_spans=0, guard_symbols=4),
        launch_powers_dbm=(0.0,), n_symbols=256, train_frames=1, validation_frames=1, test_frames=2,
        context=ContextConfig(1, 1, 8), linear_ridges=(1e-4, 1e-2), esn_ridges=(1e-3, 1e-2),
        esn_candidates=({"units": 16, "density": 0.3},), diagnostic_symbols=20)


def fake_simulate(config, power, n, bit_seed, noise_seed, backend="numpy"):
    bits = np.random.default_rng(bit_seed).integers(0, 2, (n, 2, 4), dtype=np.uint8)
    transmitted = bits_to_symbols(bits)
    noise_rng = np.random.default_rng(noise_seed)
    received = transmitted + 0.15 * (noise_rng.normal(size=transmitted.shape) + 1j * noise_rng.normal(size=transmitted.shape))
    mask = np.ones(n, dtype=bool)
    mask[:config.guard_symbols] = mask[-config.guard_symbols:] = False
    return SimpleNamespace(tx_symbols=transmitted, rx_symbols=received, tx_bits=bits, valid_mask=mask)


def test_independent_seed_hierarchy_and_frame_count_changes_are_isolated():
    config = small_config()
    records = experiment.frame_seeds(config, 0)
    seeds = [record[key] for record in records for key in ("bit_seed", "noise_seed")]
    assert len(set(seeds)) == len(seeds)
    assert records == experiment.frame_seeds(config, 0)
    assert set(seeds).isdisjoint(record[key] for record in experiment.frame_seeds(config, 2)
                                  for key in ("bit_seed", "noise_seed"))
    expanded = experiment.frame_seeds(replace(config, train_frames=3), 0)
    assert [row for row in records if row["split"] == "test"] == [row for row in expanded if row["split"] == "test"]


def test_score_pooling_counts_same_mask_and_exact_energy():
    config = small_config()
    frame = fake_simulate(config.link, 0, 256, 12, 13)
    prediction = frame.tx_symbols.copy()
    prediction[~frame.valid_mask] = np.nan
    row = experiment.score_symbols(frame, prediction, frame.valid_mask)
    assert row["n_bits"] == np.count_nonzero(frame.valid_mask) * 8
    assert row["bit_errors"] == 0
    assert row["effective_snr_db"] is None
    other = experiment.score_symbols(frame, frame.rx_symbols, frame.valid_mask)
    pooled = experiment.pool_metrics([row, other])
    assert pooled["bit_errors"] == other["bit_errors"]
    assert pooled["n_bits"] == 2 * row["n_bits"]
    assert pooled["evm"] == pytest.approx(np.sqrt(other["error_energy"] / (2 * other["signal_energy"])))


def test_validation_tie_rule_is_fixed():
    candidates = [dict(ber=0.1, evm=evm, model_array_bytes=size, candidate_id=name)
                  for evm, size, name in [(0.5, 20, "a"), (0.4, 30, "b"), (0.4, 20, "d"), (0.4, 20, "c")]]
    assert min(candidates, key=experiment.selection_key)["candidate_id"] == "c"


def test_runner_locks_selection_before_test_and_resume_verifies_artifacts(tmp_path, monkeypatch):
    config = small_config()
    calls = []

    def simulate(*args, **kwargs):
        calls.append(args[3])
        if args[3] in {record["bit_seed"] for record in experiment.frame_seeds(config, 0) if record["split"] == "test"}:
            selection_path = next((tmp_path / "pilot-unit").glob("power_*/selection.json"))
            selection = json.loads(selection_path.read_text())
            assert selection["selection_split"] == "validation"
            assert not selection["test_labels_used_for_selection"]
        return fake_simulate(*args, **kwargs)

    monkeypatch.setattr(experiment, "simulate_frame", simulate)
    monkeypatch.setattr(experiment, "dbp_symbols", lambda frame, **kwargs: frame.rx_symbols.copy())
    output = experiment.run_experiment(config, tmp_path, "pilot-unit")
    assert len(calls) == 4
    rows = json.loads((output / "summary.json").read_text())["metrics"]
    assert {row["method"] for row in rows} == {"calibrated_edc", "linear", "esn", "calibrated_dbp"}
    expected_bits = (256 - 8 - 2 - 8) * 8 * 2
    assert all(row["n_bits"] == expected_bits for row in rows)
    selection = json.loads(next(output.glob("power_*/selection.json")).read_text())
    assert all(row["n_bits"] == expected_bits // 2 for row in selection["candidates"])
    assert experiment.run_experiment(config, tmp_path, "pilot-unit", resume=True) == output
    assert len(calls) == 4  # Completed conditions are never regenerated.
    with pytest.raises(ValueError, match="resume rejected"):
        experiment.run_experiment(replace(config, root_seed=42), tmp_path, "pilot-unit", resume=True)
    checkpoint = next(output.glob("power_*/esn.npz"))
    checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed artifact"):
        experiment.run_experiment(config, tmp_path, "pilot-unit", resume=True)


def test_final_counting_budget_and_pilot_names_are_explicit(tmp_path):
    with pytest.raises(ValueError, match="one million"):
        replace(small_config(), mode="final")
    with pytest.raises(ValueError, match="pilot-"):
        experiment.run_experiment(small_config(), tmp_path, "final-mislabelled")


def test_legacy_imports_resolve_to_study_implementations():
    from experiments.esn_longhaul import experiment as canonical
    from experiments.esn_longhaul import reservoir
    import optical_comm.reservoir as legacy_reservoir
    assert experiment is canonical
    assert legacy_reservoir is reservoir


def test_source_change_rejects_resume_before_outputs_change(tmp_path, monkeypatch):
    import hashlib
    config = small_config()
    monkeypatch.setattr(experiment, "simulate_frame", fake_simulate)
    monkeypatch.setattr(experiment, "dbp_symbols", lambda frame, **kwargs: frame.rx_symbols.copy())
    run = experiment.run_experiment(config, tmp_path, "pilot-source-integrity")
    before = {str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in run.rglob("*") if p.is_file()}
    actual = experiment.software_manifest()
    monkeypatch.setattr(experiment, "software_manifest", lambda: {**actual, "source_hash": "0" * 64})
    with pytest.raises(ValueError, match="resume rejected"):
        experiment.run_experiment(config, tmp_path, "pilot-source-integrity", resume=True)
    assert before == {str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in run.rglob("*") if p.is_file()}
