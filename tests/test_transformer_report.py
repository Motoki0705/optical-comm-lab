"""Saved synthetic PILOT evidence, never a physical or held-out research claim."""
from copy import deepcopy
from dataclasses import replace
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pytest

from experiments.transformer_nlc.complexity import complexity_breakdown
from experiments.transformer_nlc.data import atomic_json, canonical_hash, file_hash
from experiments.transformer_nlc.data_config import pilot_config
from experiments.transformer_nlc.evaluation import _aggregate_frames, _count_summary
from experiments.transformer_nlc.model_config import ModelConfig
from experiments.transformer_nlc.report import generate_report, inspect_inputs
from experiments.transformer_nlc.training_config import TrainingConfig

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def scratch():
    # All development plots stay under ignored .work, including test failures.
    root = ROOT / ".work" / "transformer-report-tests"
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as directory:
        yield Path(directory)


def _measure(errors, n):
    summary = _count_summary(errors, n)
    return {**summary, "mse": .1, "signal_power": 1., "evm": float(np.sqrt(.1)), "effective_snr_db": 10.}


def _frame_metrics(errors, n_bits):
    return {"X": _measure(errors, n_bits // 2), "Y": _measure(errors + 1, n_bits // 2), "total": _measure(errors * 2 + 1, n_bits)}


def synthetic_evidence(root):
    """Create compact examples only; model and test arrays intentionally absent."""
    root.mkdir()
    source = {"content_sha256": "0" * 64, "fixture": "SYNTHETIC PILOT, no measured channel or trained model"}
    runtime = {"device": "cpu", "platform": "synthetic fixture", "dtype": "float32", "attention_backend": "fixture dense (not executed)", "torch": "not executed"}
    trials, frames, replicates, ranking = [], [], [], []
    for condition, candidate in ((16, "C"), (64, "low-adopted")):
        for replicate in range(3):
            data_dir = root / f"data-{condition}-{replicate}"
            data_dir.mkdir()
            dc = replace(pilot_config(condition, scored_symbols=40, guard_symbols=16, pilot_symbols=32), replicate=replicate, test_frames=2)
            data_frames = [{"split": split, "frame": frame, "path": f"ABSENT-{split}-{frame}.npz", "file_sha256": canonical_hash([condition, replicate, split, frame])}
                           for split, count in dc.frame_counts.items() for frame in range(count)]
            # Compact receiver is hashed but never deserialized as JSON or used.
            receiver = data_dir / "receiver.npz"
            receiver.write_bytes(b"SYNTHETIC compact receiver fixture; not loaded")
            manifest = {"schema": "transformer-data-v2", "status": "complete", "config": dc.to_dict(), "config_sha256": canonical_hash(dc.to_dict()),
                        "source": source, "qualification": None, "receiver_fixed_before_test_metrics": True, "test_metrics": "not_computed", "frames": data_frames,
                        "receiver": {"path": receiver.name, "file_sha256": file_hash(receiver)}}
            data_path = data_dir / "manifest.json"
            atomic_json(data_path, manifest)
            for masked in (False, True):
                config = TrainingConfig(condition=condition, candidate=candidate, masked=masked, regime="pilot", max_epochs=2,
                    effective_batch_blocks=3, microbatch_blocks=2,
                    model=ModelConfig(t=8, b=8, d_model=4, attention_projection_dim=4, h=1, layers=1, d_ff=8, output_window=1, mask_rho=2.6 if masked else None))
                run_dir = root / f"run-{condition}-{masked}-{replicate}"
                run_dir.mkdir()
                checkpoint_hash = canonical_hash([condition, replicate, masked, "ABSENT synthetic checkpoint"])
                history = [{"epoch": i, "train_mse_X_IQ": .2 / i, "validation_mse_XY_IQ": .3 / i, "seconds": .02 * i, "optimizer_step": 2 * i,
                            "learning_rate": .001, "improved": True, "patience_used": 0} for i in (1, 2)]
                run = {"schema": "transformer-training-v1", "status": "complete", "config": config.to_dict(), "source": source, "runtime": runtime,
                       "data_manifest": str(data_path), "identities": {"config_sha256": canonical_hash(config.to_dict()), "data_manifest_sha256": file_hash(data_path)},
                       "history": history, "best_epoch": 2, "best_validation_mse": .15, "completed_epochs": 2, "stop_reason": "epoch_cap", "convergence": "not_established",
                       "peak_allocated_GPU_bytes": None, "checkpoints": {"best": {"path": "ABSENT-best.pt", "sha256": checkpoint_hash}},
                       "attempts": [{"status": "failed", "exception": "SyntheticOOM", "message": "<script>alert('fixture')</script>", "oom": True}, {"status": "complete"}],
                       "resume_events": [{"microbatch_change": {"from": 3, "to": 2, "reason": "SYNTHETIC example only"}}]}
                atomic_json(run_dir / "run.json", run)
                atomic_json(run_dir / "history.json", history)
                complexity = complexity_breakdown(config.model).to_dict()
                trial = {"condition": condition, "candidate": candidate, "masked": masked, "replicate": replicate, "regime": "pilot", "run_directory": str(run_dir),
                         "run_sha256": file_hash(run_dir / "run.json"), "checkpoint": str(run_dir / "ABSENT-best.pt"), "checkpoint_sha256": checkpoint_hash,
                         "config": config.to_dict(), "config_sha256": canonical_hash(config.to_dict()), "best_epoch": 2, "validation_mse": .15,
                         "data_manifest": str(data_path), "data_manifest_sha256": file_hash(data_path), "data_config_sha256": manifest["config_sha256"],
                         "data_source": source, "training_source": source, "training_runtime": runtime, "train_power_dbm": dc.launch_power_dbm, "complexity": complexity}
                trials.append(trial)
                method = f"main{condition}-{candidate}-{masked}-r{replicate}"
                methods = [(method, candidate, masked)] + ([(f"main{condition}-no-nlc-r{replicate}", "no-nlc", None)] if not masked else [])
                for method_id, name, mask in methods:
                    items = []
                    for frame in range(dc.test_frames):
                        item = {"method": method_id, "condition": condition, "candidate": name, "masked": mask, "replicate": replicate, "frame": frame,
                                "scored_DP_times": dc.scored_symbols, "excluded_DP_times": dc.n_symbols - dc.scored_symbols,
                                "data_frame_sha256": next(e["file_sha256"] for e in data_frames if (e["split"], e["frame"]) == ("test", frame)),
                                "diagnostic_path": f"ABSENT-bulk/{method_id}-{frame}.npz", "diagnostic_sha256": "f" * 64,
                                "metrics": _frame_metrics(1 + replicate + frame + (2 if name == "no-nlc" else int(bool(mask))), dc.scored_symbols * 2 * int(np.log2(condition)))}
                        items.append(item)
                        frames.append(item)
                    replicates.append({"method": method_id, "condition": condition, "candidate": name, "masked": mask, "replicate": replicate,
                                       "metrics": _aggregate_frames(items), "primary_selected_before_test": name != "no-nlc",
                                       "checkpoint_sha256": checkpoint_hash if name != "no-nlc" else None, "complexity": complexity if name != "no-nlc" else None,
                                       "evaluation_seconds_including_IO_scoring_XY": .5, "peak_allocated_GPU_bytes": None})
    for condition, candidate in ((16, "C"), (64, "low-adopted")):
        for masked in (False, True):
            ranking.append({"condition": condition, "candidate": candidate, "masked": masked, "replicates": [0, 1, 2], "mean_validation_mse": .15,
                            "rmps_per_polarization_symbol": next(t["complexity"]["total_per_output_polarization_symbol"] for t in trials if t["condition"] == condition and t["masked"] == masked)})
    freeze = {"schema": "transformer-selection-freeze-v1", "status": "frozen", "test_metrics_seen": False, "regime": "pilot", "source": source,
              "trials": trials, "ranking": ranking, "primary": ranking, "selection_rule": "synthetic fixture only"}
    freeze_path = root / "freeze.json"
    atomic_json(freeze_path, freeze)
    grouped = {}
    for record in replicates:
        grouped.setdefault((record["condition"], record["candidate"], record["masked"]), []).append(record)
    groups = []
    for (condition, candidate, masked), records in grouped.items():
        group = {"condition": condition, "candidate": candidate, "masked": masked, "replicates": [r["replicate"] for r in records], "n_replicates": len(records)}
        for name in ("ber", "q_ber_db", "mse", "effective_snr_db"):
            values = [r["metrics"]["total"][name] for r in records]
            group[name] = {"all_points": values, "mean": float(np.mean(values)), "sample_std": float(np.std(values, ddof=1))}
        groups.append(group)
    metrics = {"schema": "transformer-evaluation-v1", "status": "complete", "regime": "pilot", "source": source, "freeze_path": str(freeze_path), "freeze_sha256": file_hash(freeze_path),
               "frames": frames, "replicates": replicates, "groups": groups, "failures": [], "evaluation_runtime": runtime,
               "independent_recount": {"status": "passed", "frame_predictions": len(frames), "implementation": "SYNTHETIC saved evidence only; not a channel measurement"}}
    metrics_path = root / "metrics.json"
    atomic_json(metrics_path, metrics)
    return metrics_path


def _edit_metrics(path, change):
    metrics = json.loads(path.read_text())
    change(metrics)
    atomic_json(path, metrics)


def test_compact_only_counts_denominators_and_explicit_pilot(scratch):
    path = synthetic_evidence(scratch / "input")
    with pytest.raises(ValueError, match="allow-pilot"):
        inspect_inputs(path)
    model = inspect_inputs(path, allow_pilot=True)
    assert model["complete"] and len(model["trials"]) == 12 and len(model["summary"]) == 6
    assert model["label"].startswith("PILOT")
    baseline = next(s for s in model["summary"] if s["condition"] == 16 and s["candidate"] == "no-nlc")
    assert baseline["pooled_counting"]["n_bits"] == 40 * 2 * 8 * 3
    assert baseline["mean_seed_q_db"] != baseline["pooled_counting"]["q_ber_db"]
    output = scratch / "never-created"
    result = generate_report(path, output, allow_pilot=True, dry_run=True)
    assert not output.exists() and not result["training_or_inference"] and not result["bulk_access"]
    assert not list(path.parent.rglob("*.pt")) and not list(path.parent.rglob("ABSENT-*.npz"))


def test_matching_duplicate_baseline_is_counted_once_and_conflicts_fail(scratch):
    path = synthetic_evidence(scratch / "input")
    expected = inspect_inputs(path, allow_pilot=True)
    def duplicate(metrics):
        items = [deepcopy(f) for f in metrics["frames"] if f["candidate"] == "no-nlc"]
        reps = [deepcopy(r) for r in metrics["replicates"] if r["candidate"] == "no-nlc"]
        for item in [*items, *reps]:
            item["method"] += "-legacy-duplicate"
        metrics["frames"] += items
        metrics["replicates"] += reps
        metrics["independent_recount"]["frame_predictions"] = len(metrics["frames"])
    _edit_metrics(path, duplicate)
    model = inspect_inputs(path, allow_pilot=True)
    assert model["summary"] == expected["summary"]
    assert model["baseline_duplicates_removed"] == 12
    def conflict(metrics):
        metrics["frames"][-1]["excluded_DP_times"] += 1
    _edit_metrics(path, conflict)
    with pytest.raises(ValueError, match="Conflicting|denominator"):
        inspect_inputs(path, allow_pilot=True)


@pytest.mark.parametrize("mutation,match", [
    (lambda m: m["replicates"].pop(), "missing replicates"),
    (lambda m: m["frames"].pop(), "frame.*coverage"),
    (lambda m: m["frames"][0]["metrics"]["total"].update(n_bits=999), "saved total/"),
    (lambda m: m["frames"][0].update(scored_DP_times=39), "denominator"),
    (lambda m: m["frames"][0].update(data_frame_sha256="a" * 64), "frame identity"),
    (lambda m: m["groups"][0]["ber"].update(mean=.9), "group mean"),
    (lambda m: m["independent_recount"].update(status="not_run"), "independent recount"),
    (lambda m: m.update(regime="production"), "regime mismatch"),
])
def test_saved_measurement_corruption_is_rejected(scratch, mutation, match):
    path = synthetic_evidence(scratch / "input")
    _edit_metrics(path, mutation)
    with pytest.raises(ValueError, match=match):
        inspect_inputs(path, allow_pilot=True)


def test_hash_status_and_production_inventory_do_not_promote_pilot(scratch):
    path = synthetic_evidence(scratch / "input")
    data_path = next(path.parent.glob("data-*/manifest.json"))
    data_path.write_text(data_path.read_text() + " ")
    with pytest.raises(ValueError, match="hash mismatch"):
        inspect_inputs(path, allow_pilot=True)


def test_ledger_resource_proof_and_installed_shared_source_guard(scratch, monkeypatch):
    path = synthetic_evidence(scratch / "input")
    metrics = json.loads(path.read_text())
    freeze = json.loads(Path(metrics["freeze_path"]).read_text())
    ledger = {"schema": "transformer-sweep-v1", "regime": "pilot", "status": "complete",
              "trials": [{**{k: t[k] for k in ("condition", "candidate", "masked", "replicate", "run_sha256")}, "status": "complete", "attempts": [{"status": "failed"}, {"status": "complete"}]} for t in freeze["trials"]]}
    ledger_path = scratch / "ledger.json"
    atomic_json(ledger_path, ledger)
    resource_path = scratch / "resource.json"
    atomic_json(resource_path, {"schema": "transformer-model-resource-v1", "status": "failed", "config": freeze["trials"][0]["config"], "exception": "SyntheticOOM"})
    model = inspect_inputs(path, allow_pilot=True, ledger_path=ledger_path, resource_paths=[resource_path])
    assert model["ledger"]["trials"][0]["attempts"][0]["status"] == "failed"
    assert model["resources"][0]["status"] == "failed" and str(resource_path) in model["inputs"]
    import experiments.transformer_nlc.report as report
    monkeypatch.setattr(report, "source_identity", lambda: {"source_layout": {"module_paths": {"optical_comm/dsp.py": str(scratch / "installed" / "optical_comm" / "dsp.py")}}})
    with pytest.raises(ValueError, match="source"):
        report._destination(scratch / "installed" / "optical_comm" / "generated", [])
    ledger["trials"][0]["run_sha256"] = "a" * 64
    atomic_json(ledger_path, ledger)
    with pytest.raises(ValueError, match="ledger run hash"):
        inspect_inputs(path, allow_pilot=True, ledger_path=ledger_path)


def test_incomplete_evaluation_requires_flag_and_keeps_missing_runs(scratch):
    path = synthetic_evidence(scratch / "input")
    def partial(m):
        removed = m["replicates"].pop()
        m["frames"] = [f for f in m["frames"] if f["method"] != removed["method"]]
        m.update(status="failed", groups=[], failures=[{"exception": "SyntheticOOM", "message": "explicit synthetic interruption"}])
    _edit_metrics(path, partial)
    with pytest.raises(ValueError, match="allow-partial"):
        inspect_inputs(path, allow_pilot=True)
    model = inspect_inputs(path, allow_pilot=True, allow_partial=True)
    assert not model["complete"] and len(model["missing"]) == 1
    assert "未完了の部分実行" in model["label"]


def test_pilot_cannot_be_renamed_production_and_zero_error_q_is_bounded(scratch):
    path = synthetic_evidence(scratch / "input")
    metrics = json.loads(path.read_text())
    freeze_path = Path(metrics["freeze_path"])
    original = freeze_path.read_bytes()
    freeze = json.loads(original)
    freeze["regime"] = "production"
    atomic_json(freeze_path, freeze)
    metrics.update(regime="production", freeze_sha256=file_hash(freeze_path))
    atomic_json(path, metrics)
    with pytest.raises(ValueError, match="configuration/regime mismatch"):
        inspect_inputs(path, allow_pilot=True)
    freeze_path.write_bytes(original)
    # No finite or infinite point-Q is invented at the zero-error limit.
    assert _count_summary(0, 1000)["q_ber_db"] is None
    assert _count_summary(0, 1000)["q_ci_low_db"] is not None


def test_actual_small_cpu_evaluation_can_be_reported_without_bulk_or_checkpoints(scratch):
    pytest.importorskip("torch")
    from test_transformer_training import make_dataset, tiny_training
    from experiments.transformer_nlc.training import train
    from experiments.transformer_nlc.evaluation import evaluate, freeze_selection
    data = make_dataset(scratch / "data")
    masked = tiny_training(max_epochs=1)
    unmasked = replace(masked, masked=False, model=replace(masked.model, mask_rho=None))
    runs = [scratch / "masked", scratch / "unmasked"]
    train(masked, data, runs[0])
    train(unmasked, data, runs[1])
    frozen_path = scratch / "freeze.json"
    freeze_selection(runs, frozen_path, allow_pilot=True)
    evaluate(frozen_path, scratch / "evaluation")
    # Regeneration requires metadata, not model weights or diagnostic arrays.
    import shutil
    shutil.rmtree(scratch / "evaluation" / "bulk")
    for run in runs:
        for checkpoint in run.glob("*.pt"):
            checkpoint.unlink()
    model = inspect_inputs(scratch / "evaluation" / "metrics.json", allow_pilot=True)
    assert model["complete"] and len(model["replicates"]) == 3
    result = generate_report(scratch / "evaluation" / "metrics.json", scratch / "report", allow_pilot=True)
    assert Path(result["html"]).is_file()


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.values = []
    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name in ("href", "src"):
                self.values.append(value)


def test_offline_report_manifest_safe_regeneration_and_input_protection(scratch):
    path = synthetic_evidence(scratch / "input")
    output = scratch / "report"
    result = generate_report(path, output, allow_pilot=True, command=["SYNTHETIC", "--allow-pilot"])
    assert result["status"] == "complete"
    doc = (output / "report.html").read_text()
    markdown = (output / "report.md").read_text()
    assert "PILOT" in doc and "0 / 0 / 0" in doc and "epoch_cap" in doc
    assert "<script>" not in doc and "&lt;script&gt;" in doc
    assert "32 GBd" in doc and "42 GBd" in doc and "3145728" not in doc  # prose uses comma-separated real budgets
    assert "ゼロ誤り" in markdown and "低下から学習時間" in markdown
    links = Links()
    links.feed(doc)
    for link in links.values:
        if not link.startswith("https://"):
            assert not Path(link).is_absolute() and ".." not in Path(link).parts
            assert (output / link).is_file()
    manifest = json.loads((output / "report-manifest.json").read_text())
    assert manifest["validation"]["bulk_arrays"] == "not_read"
    assert manifest["font"]["SVG_text"] == "paths" and manifest["font"]["sha256"]
    for name, digest in manifest["outputs"].items():
        assert file_hash(output / name) == digest
    inventory = json.loads((output / "tables/run-inventory.json").read_text())
    assert len(inventory) == 36 and sum(i["training_status"] == "complete" for i in inventory) == 12
    with pytest.raises(ValueError, match="previous report"):
        generate_report(path, output / "nested-report", allow_pilot=True, dry_run=True)
    with pytest.raises(FileExistsError, match="exists"):
        generate_report(path, output, allow_pilot=True)
    # Deliberate unchanged regeneration works and leaves no external assets.
    generate_report(path, output, allow_pilot=True, regenerate=True)
    regenerated = json.loads((output / "report-manifest.json").read_text())
    assert {k: v for k, v in manifest["outputs"].items() if k.startswith("figures/")} == {k: v for k, v in regenerated["outputs"].items() if k.startswith("figures/")}
    user_file = output / "my-note.txt"
    user_file.write_text("keep this")
    with pytest.raises(FileExistsError, match="unowned"):
        generate_report(path, output, allow_pilot=True, regenerate=True)
    assert user_file.read_text() == "keep this"
    user_file.unlink()
    (output / "report.md").write_text("user edit")
    with pytest.raises(FileExistsError, match="modified"):
        generate_report(path, output, allow_pilot=True, regenerate=True)
    assert (output / "report.md").read_text() == "user edit"
    for forbidden in (path.parent / "new-report", ROOT / "src" / "new-report", ROOT / "reports" / "existing-esn"):
        with pytest.raises(ValueError, match="overlaps|Only a named"):
            generate_report(path, forbidden, allow_pilot=True, dry_run=True)


def test_cli_is_lazy_and_rejects_ignored_combinations(scratch):
    script = "from experiments.transformer_nlc.__main__ import main; import sys; main(['report','--help'])"
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0 and "--metrics" in result.stdout
    result = subprocess.run([sys.executable, "-m", "experiments.transformer_nlc", "report", "--metrics", "missing", "--run-id", "pilot", "--dry-run", "--recount"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 2 and "cannot be combined" in result.stderr
    script = "import sys; from experiments.transformer_nlc.__main__ import main; main(['generate','--pilot','--run-id','lazy-smoke','--dry-run']); assert 'matplotlib.pyplot' not in sys.modules; assert 'torch' not in sys.modules; assert 'experiments.transformer_nlc.report' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_public_report_cli_rejects_symlink_regeneration_and_preserves_target(scratch):
    path = synthetic_evidence(scratch / "input")
    output_root = scratch / "reports"
    output = output_root / "synthetic-pilot"
    command = [sys.executable, "-m", "experiments.transformer_nlc", "report", "--metrics", str(path),
               "--output-root", str(output_root.relative_to(ROOT)), "--allow-pilot"]
    for flags in ([], ["--regenerate"]):
        result = subprocess.run([*command, "--run-id", output.name, *flags], cwd=ROOT, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        plan = json.loads(result.stdout)
        assert plan["status"] == "complete" and plan["output"] == str(output)
        assert plan["label"].startswith("PILOT")
    assert "SYNTHETIC" in (output / "report.md").read_text()
    alias = output_root / "synthetic-pilot-alias"
    alias.symlink_to(output.name, target_is_directory=True)
    before = {str(p.relative_to(output)): file_hash(p) for p in output.rglob("*") if p.is_file()}
    entries = set(output_root.rglob("*"))

    # The direct API and public CLI must reject the same unchanged owned target.
    with pytest.raises(ValueError, match="destination cannot be a symlink"):
        generate_report(path, alias, allow_pilot=True, regenerate=True)
    result = subprocess.run([*command, "--run-id", alias.name, "--regenerate"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0 and "destination cannot be a symlink" in result.stderr
    assert alias.is_symlink() and alias.readlink() == Path(output.name)
    assert set(output_root.rglob("*")) == entries
    assert {str(p.relative_to(output)): file_hash(p) for p in output.rglob("*") if p.is_file()} == before
