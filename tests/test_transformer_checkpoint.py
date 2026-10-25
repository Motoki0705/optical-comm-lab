"""Checkpoint integrity, OS-released lock and protected path contracts."""
import importlib.util
import multiprocessing
from pathlib import Path

import pytest

if importlib.util.find_spec("torch") is None:
    pytest.skip("optional torch extra is absent", allow_module_level=True)
import torch

from experiments.transformer_nlc.checkpoint import advisory_lock, code_identity, load_checkpoint, protect_output, save_checkpoint


def _hold_lock(path, ready):
    with advisory_lock(path):
        ready.set()
        ready.wait()
        # Parent terminates this process; no Python finally executes.
        import time
        time.sleep(30)


def test_checkpoint_atomic_save_reload_and_hash(tmp_path):
    path = tmp_path / "latest.pt"
    digest = save_checkpoint(path, {"schema": "transformer-checkpoint-v1", "model": {"weight": torch.arange(4)}})
    assert torch.equal(load_checkpoint(path, expected_sha256=digest)["model"]["weight"], torch.arange(4))
    assert not list(tmp_path.glob("*.tmp"))
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_checkpoint(path, expected_sha256=digest)


def test_lock_releases_after_process_death(tmp_path):
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    process = context.Process(target=_hold_lock, args=(tmp_path / "lock", ready))
    process.start()
    try:
        assert ready.wait(15)
        with pytest.raises(BlockingIOError, match="busy"):
            with advisory_lock(tmp_path / "lock"):
                pass
    finally:
        process.terminate()
        process.join(10)
    with advisory_lock(tmp_path / "lock"):
        pass


def test_output_cannot_overwrite_source_or_input(tmp_path):
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="source/report"):
        protect_output(root / "experiments/transformer_nlc", [])
    with pytest.raises(ValueError, match="protected input"):
        protect_output(tmp_path, [tmp_path / "data/manifest.json"])
    with pytest.raises(ValueError, match="protected input"):
        protect_output(tmp_path / "data/evaluation", [tmp_path / "data"])


def test_learning_source_identity_rejects_mixed_imported_model(tmp_path, monkeypatch):
    import experiments.transformer_nlc.model as model
    identity = code_identity()
    assert identity["source_layout"]["kind"] == "source_checkout"
    assert "experiments/transformer_nlc/evaluation.py" in identity["files"]
    foreign = tmp_path / "model.py"
    foreign.write_text(Path(model.__file__).read_text())
    monkeypatch.setattr(model, "__file__", str(foreign))
    with pytest.raises(ValueError, match="Mixed training imports"):
        code_identity()
