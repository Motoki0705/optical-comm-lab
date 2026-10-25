"""Actual CPU updates, strict data isolation and deterministic epoch restart."""
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json

import numpy as np
import pytest

if importlib.util.find_spec("torch") is None:
    pytest.skip("optional torch extra is absent", allow_module_level=True)
import torch

from optical_comm.metrics import bits_to_symbols
from experiments.transformer_nlc.data import (atomic_json, atomic_npz, canonical_hash, file_hash,
    frame_layout, learning_seed_recipe, seed_sequence)
from experiments.transformer_nlc.data_config import pilot_config
from experiments.transformer_nlc.model_config import ModelConfig
from experiments.transformer_nlc.training_config import TrainingConfig, fixed_trial_plan, learning_rate
from experiments.transformer_nlc.training import BlockDataset, read_manifest, train, training_plan
from experiments.transformer_nlc.checkpoint import load_checkpoint


def make_dataset(directory, *, condition=16, replicate=0, change_y_labels=False):
    """Synthetic unit-Es pilot fixture, explicitly not a physical qualification."""
    config = replace(pilot_config(condition, scored_symbols=40, guard_symbols=16, pilot_symbols=32), replicate=replicate)
    directory.mkdir(parents=True)
    manifest = {"schema": "transformer-data-v2", "status": "complete", "config": config.to_dict(),
        "config_sha256": canonical_hash(config.to_dict()), "source": {"fixture": "synthetic unit-Es pilot; no physics claim"},
        "receiver_fixed_before_test_metrics": True, "test_metrics": "not_computed", "qualification": None,
        "learning_seeds": learning_seed_recipe(config), "frames": []}
    for split_index, split in enumerate(("train", "validation", "test")):
        rng = np.random.Generator(np.random.PCG64(seed_sequence(config, split, 0, "bits")))
        bits = rng.integers(0, 2, (config.n_symbols, 2, int(np.log2(condition))), dtype=np.uint8)
        tx = bits_to_symbols(bits, condition)
        rx = (tx + .08 * tx * np.abs(tx) ** 2 + .01 * (rng.normal(size=tx.shape) + 1j * rng.normal(size=tx.shape))).astype(np.complex128)
        if change_y_labels:
            tx[:, 1] = -tx[:, 1]
        arrays = {**frame_layout(config), "tx_bits": bits, "tx_symbols": tx, "rx_nlc": rx,
                  "frame_id": np.full(config.n_symbols, split_index, dtype=np.int64),
                  "split_id": np.full(config.n_symbols, split_index, dtype=np.int64)}
        path = directory / f"{split}.npz"
        atomic_npz(path, arrays)
        manifest["frames"].append({"split": split, "frame": 0, "path": path.name,
                                    "file_sha256": file_hash(path), "metadata": {"fixture": "synthetic"}})
    atomic_json(directory / "manifest.json", manifest)
    return directory / "manifest.json"


def tiny_training(**overrides):
    values = dict(regime="pilot", max_epochs=2, effective_batch_blocks=3, microbatch_blocks=2,
        model=ModelConfig(t=8, b=8, d_model=4, attention_projection_dim=4, h=1, layers=1, d_ff=8, output_window=1))
    values.update(overrides)
    return TrainingConfig(**values)


def assert_nested_equal(first, second):
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_nested_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for x, y in zip(first, second):
            assert_nested_equal(x, y)
    else:
        assert first == second


def test_actual_cpu_training_restart_is_bitwise_continuous(tmp_path):
    dataset = make_dataset(tmp_path / "data")
    config = tiny_training()
    full = train(config, dataset, tmp_path / "full")
    partial = train(config, dataset, tmp_path / "resumed", stop_after_epochs=1)
    assert partial["status"] == "paused" and partial["optimizer_steps"] == 2
    resumed = train(config, dataset, tmp_path / "resumed", resume=True)
    assert resumed["stop_reason"] == "epoch_cap" and resumed["convergence"] == "not_established"
    assert resumed["completed_epochs"] == 2 and resumed["optimizer_steps"] == 4
    a, b = (load_checkpoint(tmp_path / name / "latest.pt") for name in ("full", "resumed"))
    for key in ("model", "optimizer", "scheduler", "rng", "sampler", "best_validation_mse", "patience_used"):
        assert_nested_equal(a[key], b[key])
    for h1, h2 in zip(full["history"], resumed["history"]):
        assert {k: v for k, v in h1.items() if k != "seconds"} == {k: v for k, v in h2.items() if k != "seconds"}
    assert a["scheduler"]["step"] == 4 and len(a["optimizer"]["state"]) > 0
    assert any(torch.count_nonzero(state["exp_avg"]).item() for state in a["optimizer"]["state"].values())
    assert (tmp_path / "full/best.pt").exists()
    with pytest.raises(FileExistsError):
        train(config, dataset, tmp_path / "full")
    with pytest.raises(ValueError, match="identity mismatch"):
        train(replace(config, microbatch_blocks=1), dataset, tmp_path / "full", resume=True)


def test_train_never_opens_test_files_and_x_labels_only(tmp_path):
    first = make_dataset(tmp_path / "first")
    second = make_dataset(tmp_path / "second", change_y_labels=True)
    # A missing test file is an intentional trap: metadata inspection is legal,
    # but any training/validation attempt to read its arrays must fail.
    (first.parent / "test.npz").unlink()
    (second.parent / "test.npz").unlink()
    config = tiny_training()
    train(config, first, tmp_path / "x")
    train(config, second, tmp_path / "y")
    a, b = (load_checkpoint(tmp_path / name / "latest.pt") for name in ("x", "y"))
    assert_nested_equal(a["model"], b["model"])
    assert a["history"][0]["validation_mse_XY_IQ"] != b["history"][0]["validation_mse_XY_IQ"]
    path, manifest, _ = read_manifest(first)
    with pytest.raises(ValueError, match="frozen selection"):
        BlockDataset(path, manifest, "test", config.model)


def test_partial_batch_accumulation_matches_one_effective_batch(tmp_path):
    dataset = make_dataset(tmp_path / "data")
    # Five blocks -> effective batches 3 and 2; microbatch 2 -> 2+1, then 2.
    config = tiny_training(max_epochs=1)
    from experiments.transformer_nlc.model import TransformerNLC
    path, manifest, _ = read_manifest(dataset)
    blocks = BlockDataset(path, manifest, "train", config.model)
    torch.manual_seed(123)
    direct = TransformerNLC(config.model)
    accumulated = TransformerNLC(config.model)
    accumulated.load_state_dict(direct.state_dict())
    features, targets = blocks.batch([0, 1, 2], "cpu")
    ((direct(features) - targets[..., 0, :]) ** 2).mean().backward()
    for indexes in ([0, 1], [2]):
        features, targets = blocks.batch(indexes, "cpu")
        (((accumulated(features) - targets[..., 0, :]) ** 2).mean() * len(indexes) / 3).backward()
    for first, second in zip(direct.parameters(), accumulated.parameters()):
        torch.testing.assert_close(first.grad, second.grad, atol=1e-6, rtol=1e-5)
    train(config, dataset, tmp_path / "micro")
    train(replace(config, microbatch_blocks=3), dataset, tmp_path / "whole")
    a, b = (load_checkpoint(tmp_path / name / "latest.pt") for name in ("micro", "whole"))
    for name in a["model"]:
        # Adam's division by sqrt(v)+eps amplifies rounding for analytically
        # near-zero attention-bias gradients. Batch-shape equality is numeric,
        # while restart with the same microbatch above is bitwise identical.
        torch.testing.assert_close(a["model"][name], b["model"][name], atol=1e-5, rtol=2e-5)
    assert a["optimizer_step"] == b["optimizer_step"] == 2


def test_failed_partial_epoch_is_retained_and_rolled_back(tmp_path, monkeypatch):
    import experiments.transformer_nlc.training as module
    dataset = make_dataset(tmp_path / "data")
    config = tiny_training()
    train(config, dataset, tmp_path / "full")
    original = module._train_epoch
    def failure(*args, **kwargs):
        original(*args, **kwargs)
        raise FloatingPointError("intentional nonfinite-gradient simulation")
    monkeypatch.setattr(module, "_train_epoch", failure)
    with pytest.raises(FloatingPointError):
        train(config, dataset, tmp_path / "failed")
    record = json.loads((tmp_path / "failed/run.json").read_text())
    assert record["status"] == "failed" and record["attempts"][0]["exception"] == "FloatingPointError"
    monkeypatch.setattr(module, "_train_epoch", original)
    resumed = train(config, dataset, tmp_path / "failed", resume=True)
    assert resumed["resume_events"][0]["discarded_partial_epoch"] == 1
    assert resumed["resume_events"][0]["discarded_optimizer_steps"] == 2
    a, b = (load_checkpoint(tmp_path / name / "latest.pt") for name in ("full", "failed"))
    assert_nested_equal(a["model"], b["model"])
    assert len(resumed["attempts"]) == 2 and resumed["attempts"][0]["status"] == "failed"


def test_fixed_config_budgets_scheduler_and_strict_unknowns(tmp_path):
    dataset = make_dataset(tmp_path / "data")
    plan = training_plan(tiny_training(), dataset)
    assert (plan["blocks_per_epoch"], plan["optimizer_steps_per_epoch"], plan["warmup_steps"]) == (5, 2, 20)
    assert learning_rate(1, warmup_steps=20) == 1e-3 / 20
    assert learning_rate(20, warmup_steps=20) == 1e-3
    assert learning_rate(80, warmup_steps=20) == 5e-4
    assert len(fixed_trial_plan()) == 36
    with pytest.raises(ValueError, match="Unknown training"):
        TrainingConfig.from_dict({"unknown": 1})
    with pytest.raises(ValueError, match="Pilot data"):
        training_plan(TrainingConfig(), dataset)
    with pytest.raises(ValueError, match="fixed candidate"):
        TrainingConfig(model=tiny_training().model)
