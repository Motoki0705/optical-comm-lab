"""Checkpoint integrity, OS-released lock and protected path contracts."""
import importlib.util
import multiprocessing
from pathlib import Path
import shutil
import subprocess
import sys

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


@pytest.mark.parametrize("installed", [False, True], ids=("checkout", "installed-layout"))
def test_public_source_output_guards_precede_mkdir_and_lock(tmp_path, installed):
    root = Path(__file__).resolve().parents[1]
    package_root = root
    if installed:
        package_root = tmp_path / "site-packages"
        ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
        shutil.copytree(root / "src/optical_comm", package_root / "optical_comm", ignore=ignore)
        (package_root / "experiments").mkdir()
        shutil.copyfile(root / "experiments/__init__.py", package_root / "experiments/__init__.py")
        shutil.copytree(root / "experiments/transformer_nlc", package_root / "experiments/transformer_nlc", ignore=ignore)
    script = r'''
import json, pathlib, runpy, sys
from unittest.mock import patch
sys.dont_write_bytecode = True
package_root, helper, scratch = map(pathlib.Path, sys.argv[1:4])
sys.path[:0] = [str(package_root), str(package_root / "src")]
from experiments.transformer_nlc.data import source_identity
from experiments.transformer_nlc.training import train, sweep, resource_benchmark
from experiments.transformer_nlc.evaluation import freeze_selection
helpers = runpy.run_path(str(helper))
dataset = helpers["make_dataset"](scratch / "data")
config = helpers["tiny_training"](max_epochs=1)
valid_run = scratch / "valid-run"
result = train(config, dataset, valid_run)
assert result["status"] == "complete" and (valid_run / "latest.pt").is_file()
freeze_selection([valid_run], scratch / "valid-freeze.json", allow_pilot=True)
layout = source_identity()["source_layout"]
shared = pathlib.Path(layout["module_paths"]["src/optical_comm/config.py"]).parent
study = pathlib.Path(layout["module_paths"]["experiments/transformer_nlc/data.py"]).parent
def snapshot():
    return {str(p): p.read_bytes() for directory in (shared, study) for p in directory.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts}
before = snapshot()
targets = {shared, study, shared / "new-run", study / "new-run", shared.parent, study.parent,
           pathlib.Path(layout["root"])}
operations = {
    "train": lambda target: train(config, dataset, target),
    "sweep": lambda target: sweep([dataset], target, regime="pilot", max_epochs=1),
    "benchmark": lambda target: resource_benchmark(target, condition=16, masked=True, device="cpu"),
    "freeze": lambda target: freeze_selection([valid_run], target, allow_pilot=True),
}
attempts = 0
with patch.object(pathlib.Path, "mkdir", side_effect=AssertionError("mkdir reached before rejection")):
    for operation, invoke in operations.items():
        for target in targets:
            try:
                invoke(target)
            except ValueError as exc:
                assert "Output overlaps" in str(exc), (operation, target, str(exc))
            else:
                raise AssertionError(("source output accepted", operation, target))
            attempts += 1
assert snapshot() == before
print(json.dumps({"layout": layout["kind"], "public_rejections": attempts, "valid_outside_run": result["status"]}))
'''
    completed = subprocess.run([sys.executable, "-I", "-c", script, str(package_root),
        str(root / "tests/test_transformer_training.py"), str(tmp_path / "outside-artifacts")],
        text=True, capture_output=True, timeout=90)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert f'"layout": "{"installed_package" if installed else "source_checkout"}"' in completed.stdout
    assert '"valid_outside_run": "complete"' in completed.stdout
