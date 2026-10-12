"""Saved-data integrity tests; no optical frames are propagated or trained."""

import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("audit_run_cli", ROOT / "scripts" / "audit_run.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def rehash(directory, filename):
    complete = json.loads((directory / "complete.json").read_text())
    complete["files_sha256"][filename] = hashlib.sha256((directory / filename).read_bytes()).hexdigest()
    write_json(directory / "complete.json", complete)


@pytest.fixture
def saved_run(tmp_path):
    """A one-condition copy of committed measurements, with its own config hash."""
    source = ROOT / "artifacts" / "final-1600km-20261012"
    run = tmp_path / "saved-run"
    run.mkdir()
    for name in ("manifest.json", "config.json", "verification.json"):
        shutil.copy2(source / name, run / name)
    dirname = "power_p0000p000"
    shutil.copytree(source / dirname, run / dirname)
    config = json.loads((run / "config.json").read_text())
    config["launch_powers_dbm"] = [0]
    write_json(run / "config.json", config)
    digest = audit.content_hash(audit.ExperimentConfig.from_dict(config).to_dict())
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["compatibility"]["config_hash"] = digest
    write_json(run / "manifest.json", manifest)
    complete = json.loads((run / dirname / "complete.json").read_text())
    complete["compatibility"]["config_hash"] = digest
    write_json(run / dirname / "complete.json", complete)
    rows = json.loads((source / "summary.json").read_text())["metrics"]
    rows = [row for row in rows if row["launch_power_dbm"] == 0]
    write_json(run / "summary.json", {"run_id": manifest["run_id"], "mode": "final", "metrics": rows})
    with (run / "summary.csv").open("w", newline="") as stream:
        csv_rows = [dict(run_id=manifest["run_id"], mode="final", **row) for row in rows]
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    return run


def data_snapshot(run):
    return {str(path.relative_to(run)): (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
            for path in run.rglob("*") if path.is_file()}


def test_read_only_audit_keeps_history_and_ignores_clone_mtimes(saved_run):
    before = data_snapshot(saved_run)
    original = audit.audit_run(saved_run)
    assert before == data_snapshot(saved_run)
    assert original["semantic_checks"]["pooled_rows"] == 4
    assert original["semantic_checks"]["expected_test_bits_per_receiver_power"] == 2_534_080
    assert original["semantic_checks"]["predictions_regenerated"] is False
    assert original["recorded_runner_evidence_observed"]["0"]["prediction_equality_reexecuted_by_this_audit"] is False
    measured = original["historical_measurements"]["runner_elapsed_manifest_to_last_complete_seconds"]
    for path in saved_run.rglob("*"):
        if path.is_file():
            os.utime(path, (1000, 1000))
    repeated = audit.audit_run(saved_run)
    assert repeated["semantic_checks"] == original["semantic_checks"]
    assert repeated["historical_measurements"]["runner_elapsed_manifest_to_last_complete_seconds"] == measured
    assert measured == pytest.approx(318.37985849380493)


@pytest.mark.parametrize("corruption", ["checkpoint_bytes", "config", "source_manifest", "summary"])
def test_saved_integrity_boundaries_reject_corruption(saved_run, corruption):
    if corruption == "checkpoint_bytes":
        path = saved_run / "power_p0000p000" / "esn.npz"
        path.write_bytes(path.read_bytes() + b"tampered checkpoint")
    elif corruption == "config":
        path = saved_run / "config.json"
        values = json.loads(path.read_text())
        values["root_seed"] += 1
        write_json(path, values)
    elif corruption == "source_manifest":
        path = saved_run / "manifest.json"
        values = json.loads(path.read_text())
        values["compatibility"]["source_hash"] = "0" * 64
        write_json(path, values)
    else:
        path = saved_run / "summary.json"
        values = json.loads(path.read_text())
        values["metrics"][0]["bit_errors"] += 1
        write_json(path, values)
    with pytest.raises(audit.AuditError):
        audit.audit_run(saved_run)


@pytest.mark.parametrize("corruption", ["test_seed", "denominator", "selected_candidate", "checkpoint_gain", "diagnostic_indices"])
def test_semantic_checks_reject_inconsistent_rehashed_artifacts(saved_run, corruption):
    """Refreshing one file's checksum does not hide a violated scientific contract."""
    directory = saved_run / "power_p0000p000"
    if corruption in {"test_seed", "denominator"}:
        filename = "test_frames.json"
        rows = json.loads((directory / filename).read_text())
        if corruption == "test_seed":
            rows[0]["bit_seed"] = json.loads((directory / "seeds.json").read_text())[0]["bit_seed"]
        else:
            rows[0]["n_bits"] += 8
        write_json(directory / filename, rows)
    elif corruption == "selected_candidate":
        filename = "selection.json"
        selection = json.loads((directory / filename).read_text())
        selection["selected"]["esn"] = next(row["candidate_id"] for row in selection["candidates"]
            if row["method"] == "esn" and row["candidate_id"] != selection["selected"]["esn"])
        write_json(directory / filename, selection)
    else:
        filename = "esn.npz" if corruption == "checkpoint_gain" else "constellations.npz"
        path = directory / filename
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        if corruption == "checkpoint_gain":
            arrays["gain"][0] *= 1.01
        else:
            arrays["indices"][1] += 1  # Unique, interior, but no longer the declared evenly spaced sample.
        np.savez_compressed(path, **arrays)
    rehash(directory, filename)
    with pytest.raises(audit.AuditError):
        audit.audit_run(saved_run)


def test_cli_runs_with_only_copied_source_and_data_and_identifies_actual_code(saved_run, tmp_path):
    clone = tmp_path / "clone"
    (clone / "scripts").mkdir(parents=True)
    shutil.copytree(ROOT / "src", clone / "src", ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("audit_run.py", "run_experiment.py"):
        shutil.copy2(ROOT / "scripts" / name, clone / "scripts" / name)
    target = clone / "artifacts" / "saved-run"
    shutil.copytree(saved_run, target)
    before = data_snapshot(target)
    env = dict(os.environ, PYTHONPATH=str(clone / "src"))
    result = subprocess.run([sys.executable, "scripts/audit_run.py", "--run", "artifacts/saved-run",
                             "--output", "audit.json"], cwd=clone, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    record = json.loads((clone / "audit.json").read_text())
    assert record["audit_code"]["sha256"] == hashlib.sha256((clone / "scripts/audit_run.py").read_bytes()).hexdigest()
    assert Path(record["observations"]["imported_scientific_source_root"]) == clone
    assert record["semantic_checks"]["input_files_sha256"]["config.json"] == hashlib.sha256((target / "config.json").read_bytes()).hexdigest()
    assert before == data_snapshot(target)
    assert record["semantic_checks_sha256"] == audit.audit_run(saved_run)["semantic_checks_sha256"]
    altered_source = clone / "src/optical_comm/reservoir.py"
    altered_source.write_text(altered_source.read_text() + "\n# changed copy\n")
    failed = subprocess.run([sys.executable, "scripts/audit_run.py", "--run", "artifacts/saved-run"],
                            cwd=clone, env=env, text=True, capture_output=True)
    assert failed.returncode == 1
    assert "source hash differs" in failed.stderr
    assert before == data_snapshot(target)


def test_cli_refuses_to_overwrite_science_input(saved_run):
    before = data_snapshot(saved_run)
    result = subprocess.run([sys.executable, str(ROOT / "scripts/audit_run.py"), "--run", str(saved_run),
                             "--output", str(saved_run / "config.json")], text=True, capture_output=True)
    assert result.returncode == 1
    assert "read-only" in result.stderr
    assert before == data_snapshot(saved_run)
