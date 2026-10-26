"""Offline Japanese reports from sealed, compact measurements; no model loading.

The reporter checks saved counts with the evaluator's counting implementation.
It never fits data or selects a candidate from test performance. Bulk prediction
files and checkpoints are unnecessary unless independent recount is requested.
"""
from collections import defaultdict
from datetime import datetime, timezone
import csv
import html
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .data import canonical_hash, file_hash, source_identity
from .data_config import DataConfig
from .evaluation import _aggregate_frames, _count_summary, independent_recount
from .training_config import CANDIDATES, TrainingConfig, fixed_trial_plan

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "transformer-reader-report-v1"
PAPER = "https://arxiv.org/html/2304.13119v3"


def _key(item):
    return item["condition"], item["candidate"], item["masked"], item["replicate"]


def _method(item):
    return "通常DSP" if item["candidate"] == "no-nlc" else f"{item['candidate']} / {'maskあり' if item['masked'] else 'maskなし'}"


def _order(item):
    baseline = item["candidate"] == "no-nlc"
    candidate = -1 if baseline else CANDIDATES[item["condition"]].index(item["candidate"])
    return item["condition"], -1 if baseline else int(item["masked"]), candidate, item.get("replicate", -1), item.get("frame", -1)


def _path(value, relative_to):
    path = Path(value)
    return (path if path.is_absolute() else relative_to / path).resolve()


def _load(path, inputs, expected=None):
    path = Path(path).resolve()
    digest = file_hash(path)
    if expected is not None and digest != expected:
        raise ValueError(f"Input hash mismatch: {path}")
    inputs[str(path)] = digest
    return json.loads(path.read_text(encoding="utf-8"))


def _record_file(path, inputs, expected):
    path = Path(path).resolve()
    digest = file_hash(path)
    if digest != expected:
        raise ValueError(f"Input hash mismatch: {path}")
    inputs[str(path)] = digest


def _same(actual, expected, label):
    if isinstance(expected, (float, int)) and not isinstance(expected, bool):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-13):
            raise ValueError(f"Inconsistent saved {label}")
    elif actual != expected:
        raise ValueError(f"Inconsistent saved {label}")


def _check_counts(metrics):
    for pol in ("X", "Y", "total"):
        value = metrics[pol]
        errors, bits = value["bit_errors"], value["n_bits"]
        if type(errors) is not int or type(bits) is not int or not 0 <= errors <= bits or bits < 1:
            raise ValueError("Invalid saved bit counts/denominator")
        for name, expected in _count_summary(errors, bits).items():
            _same(value.get(name), expected, f"{pol}/{name}")
    for name in ("bit_errors", "n_bits"):
        _same(metrics["total"][name], metrics["X"][name] + metrics["Y"][name], name)


def inspect_inputs(metrics_path, *, allow_pilot=False, allow_partial=False, ledger_path=None,
                   resource_paths=(), recount=False):
    """Validate compact evidence and return a report model without opening arrays.

    Production always requires the actual immutable 36-checkpoint freeze. An
    interrupted evaluation can be described with allow_partial; it cannot gain
    a completed-test label. Checkpoint identities are copied from that freeze,
    not silently reloaded or presented as newly verified model bytes.
    """
    inputs = {}
    metrics_path = Path(metrics_path).resolve()
    metrics = _load(metrics_path, inputs)
    if metrics.get("schema") != "transformer-evaluation-v1":
        raise ValueError("Expected transformer-evaluation-v1 metrics")
    if metrics.get("status") not in ("complete", "failed", "interrupted", "in_progress"):
        raise ValueError("Unknown evaluation status")
    if metrics["status"] != "complete" and not allow_partial:
        raise ValueError("Incomplete evaluation requires --allow-partial")
    freeze_path = _path(metrics["freeze_path"], metrics_path.parent)
    freeze = _load(freeze_path, inputs, metrics["freeze_sha256"])
    if freeze.get("schema") != "transformer-selection-freeze-v1" or freeze.get("status") != "frozen" or freeze.get("test_metrics_seen") is not False:
        raise ValueError("Expected immutable pre-test selection freeze")
    regime = freeze.get("regime")
    if regime not in ("pilot", "production") or metrics.get("regime") != regime:
        raise ValueError("Freeze/evaluation regime mismatch")
    if regime == "pilot" and not allow_pilot:
        raise ValueError("Pilot report requires --allow-pilot")
    if metrics.get("source", {}).get("content_sha256") != freeze.get("source", {}).get("content_sha256"):
        raise ValueError("Freeze/evaluation numerical source mismatch")
    trials = {}
    runs, datasets, protected = {}, {}, [metrics_path.parent, freeze_path]
    for trial in freeze.get("trials", []):
        key = _key(trial)
        if key in trials:
            raise ValueError("Duplicate frozen trial")
        config = TrainingConfig.from_dict(trial["config"])
        if (config.condition, config.candidate, config.masked) != key[:3] or (config.regime == "production") != (regime == "production"):
            raise ValueError("Frozen trial configuration/regime mismatch")
        _same(canonical_hash(config.to_dict()), trial["config_sha256"], "training config hash")
        run_dir = _path(trial["run_directory"], freeze_path.parent)
        run = _load(run_dir / "run.json", inputs, trial["run_sha256"])
        if run.get("schema") != "transformer-training-v1" or run.get("status") != "complete":
            raise ValueError("Frozen trial requires a completed training run")
        _same(canonical_hash(run["config"]), trial["config_sha256"], "run config hash")
        _same(run["identities"]["config_sha256"], trial["config_sha256"], "run identity")
        _same(run["best_epoch"], trial["best_epoch"], "best epoch")
        _same(run["best_validation_mse"], trial["validation_mse"], "validation selection")
        _same(run["source"], trial["training_source"], "training source")
        _same(run["runtime"], trial["training_runtime"], "training runtime")
        _same(run["source"]["content_sha256"], freeze["source"]["content_sha256"], "frozen source")
        _same(run["checkpoints"]["best"]["sha256"], trial["checkpoint_sha256"], "checkpoint identity")
        history = run.get("history", [])
        if not history or [h["epoch"] for h in history] != list(range(1, run["completed_epochs"] + 1)):
            raise ValueError("Incomplete or duplicated training history")
        for row in history:
            for name in ("train_mse_X_IQ", "validation_mse_XY_IQ", "seconds"):
                if not isinstance(row.get(name), (int, float)) or not math.isfinite(row[name]) or row[name] < 0:
                    raise ValueError("Invalid training curve values")
        best = min(history, key=lambda h: h["validation_mse_XY_IQ"])
        _same(best["epoch"], trial["best_epoch"], "history best epoch")
        _same(best["validation_mse_XY_IQ"], trial["validation_mse"], "history best score")
        if run.get("stop_reason") not in ("patience", "epoch_cap"):
            raise ValueError("Unknown completed-run stop reason")
        if run["stop_reason"] == "epoch_cap" and run["completed_epochs"] != config.max_epochs:
            raise ValueError("Epoch cap/history mismatch")
        _same(run.get("convergence"), "not_established" if run["stop_reason"] == "epoch_cap" else "patience", "convergence scope")
        if run["stop_reason"] == "patience" and history[-1].get("patience_used", 0) < config.patience:
            raise ValueError("Patience stop/history mismatch")
        history_path = run_dir / "history.json"
        if history_path.exists():
            _same(_load(history_path, inputs), history, "separate history")
        initialization_path = run_dir / "initialization-failures.json"
        run["report_initialization_failures"] = _load(initialization_path, inputs) if initialization_path.exists() else []
        data_path = _path(trial["data_manifest"], freeze_path.parent)
        if data_path not in datasets:
            data = _load(data_path, inputs, trial["data_manifest_sha256"])
            data_config = DataConfig.from_dict(data["config"])
            if data.get("schema") != "transformer-data-v2" or data.get("status") != "complete":
                raise ValueError("Incomplete dataset manifest")
            _same(canonical_hash(data_config.to_dict()), data["config_sha256"], "data config hash")
            if data_config.regime != regime or not data.get("receiver_fixed_before_test_metrics") or data.get("test_metrics") != "not_computed":
                raise ValueError("Dataset regime/receiver/test protocol mismatch")
            expected_frames = {(s, i) for s, n in data_config.frame_counts.items() for i in range(n)}
            actual = [(f["split"], f["frame"]) for f in data["frames"]]
            if len(actual) != len(set(actual)) or set(actual) != expected_frames:
                raise ValueError("Dataset frame coverage mismatch")
            qualification = data.get("qualification")
            if regime == "production":
                from .data import validate_qualification
                validate_qualification(data_config, qualification or {}, data["source"])
            receiver = data.get("receiver", {})
            if "path" in receiver:
                receiver_path = _path(receiver["path"], data_path.parent)
                if not receiver_path.is_relative_to(data_path.parent):
                    raise ValueError("Receiver path escapes dataset")
                _record_file(receiver_path, inputs, receiver["file_sha256"])
            datasets[data_path] = (data, data_config)
            protected.append(data_path.parent)
        data, dc = datasets[data_path]
        _same(file_hash(data_path), trial["data_manifest_sha256"], "shared data hash")
        _same(data["config_sha256"], trial["data_config_sha256"], "frozen data config")
        _same(data["source"], trial["data_source"], "data source")
        _same(run["identities"]["data_manifest_sha256"], trial["data_manifest_sha256"], "run data identity")
        if (dc.condition, dc.replicate) != (key[0], key[3]):
            raise ValueError("Dataset condition/replicate mismatch")
        # Reuse the owning complexity implementation; these are theoretical
        # counts, never benchmark timings or sparse-backend measurements.
        from .complexity import complexity_breakdown
        _same(canonical_hash(trial["complexity"]), canonical_hash(complexity_breakdown(config.model).to_dict()), "RMPS convention")
        trials[key], runs[key] = trial, run
        protected.extend((run_dir, _path(trial["checkpoint"], freeze_path.parent)))
    if not trials:
        raise ValueError("Freeze contains no trials")
    expected = {_key(t) for t in fixed_trial_plan()}
    if regime == "production" and set(trials) != expected:
        raise ValueError("Production freeze requires full 36-trial inventory")
    common = defaultdict(set)
    for key, trial in trials.items():
        common[key[0], key[3]].add(trial["data_manifest_sha256"])
    if any(len(v) != 1 for v in common.values()):
        raise ValueError("Candidate/mask pairs must share the same dataset")
    primary = {(p["condition"], p["candidate"], p["masked"]) for p in freeze["primary"]}
    if any(not any(k[:3] == p for k in trials) for p in primary):
        raise ValueError("Primary candidate absent from freeze")

    def dataset_for(item):
        matches = [t for k, t in trials.items() if (k[0], k[3]) == (item["condition"], item["replicate"])]
        if not matches:
            raise ValueError("Metric outside frozen condition/replicate inventory")
        trial = matches[0]
        return trial, datasets[_path(trial["data_manifest"], freeze_path.parent)]

    # Baseline aliases from legacy duplicated exports are accepted only when
    # their complete count/metric/frame evidence is exactly identical.
    frames, frame_keys, aliases = [], {}, {}
    for frame in metrics.get("frames", []):
        candidate = frame["candidate"]
        if candidate != "no-nlc" and _key(frame) not in trials:
            raise ValueError("Test metric outside frozen trial inventory")
        if candidate == "no-nlc" and frame["masked"] is not None:
            raise ValueError("Baseline cannot carry a mask")
        trial, (data, dc) = dataset_for(frame)
        entries = {e["frame"]: e for e in data["frames"] if e["split"] == "test"}
        if frame["frame"] not in entries or frame["data_frame_sha256"] != entries[frame["frame"]]["file_sha256"]:
            raise ValueError("Test frame identity differs from frozen dataset")
        _check_counts(frame["metrics"])
        if frame["scored_DP_times"] != dc.scored_symbols:
            raise ValueError("Scored DP time denominator mismatch")
        if frame["excluded_DP_times"] != dc.n_symbols - dc.scored_symbols:
            raise ValueError("Excluded DP time denominator mismatch")
        for pol, factor in (("X", 1), ("Y", 1), ("total", 2)):
            _same(frame["metrics"][pol]["n_bits"], dc.scored_symbols * factor * int(math.log2(dc.condition)), "frame bit denominator")
        identity = (trial["data_manifest_sha256"], frame["condition"], frame["replicate"], frame["frame"], candidate, frame["masked"])
        if identity in frame_keys:
            old = frame_keys[identity]
            if candidate != "no-nlc" or any(frame.get(k) != old.get(k) for k in ("metrics", "scored_DP_times", "excluded_DP_times", "data_frame_sha256")):
                raise ValueError("Conflicting or duplicate test frame")
            aliases[frame["method"]] = old["method"]
            continue
        frame_keys[identity] = frame
        frames.append(frame)
    by_method = defaultdict(list)
    for frame in frames:
        by_method[frame["method"]].append(frame)
    replicates, rep_keys = [], {}
    for item in metrics.get("replicates", []):
        key = _key(item)
        if item["candidate"] != "no-nlc" and key not in trials:
            raise ValueError("Replicate outside frozen trials")
        _check_counts(item["metrics"])
        if key in rep_keys:
            old = rep_keys[key]
            if item["candidate"] != "no-nlc" or item["metrics"] != old["metrics"]:
                raise ValueError("Conflicting or duplicate replicate")
            continue
        trial, (data, dc) = dataset_for(item)
        method = aliases.get(item["method"], item["method"])
        selected_frames = by_method.get(method, [])
        if not selected_frames or any(_key(f) != key for f in selected_frames):
            raise ValueError("Replicate/frame method identity mismatch")
        if {f["frame"] for f in selected_frames} != set(range(dc.test_frames)):
            raise ValueError("Replicate test-frame coverage mismatch")
        for pol in ("X", "Y", "total"):
            for name in ("bit_errors", "n_bits"):
                _same(item["metrics"][pol][name], sum(f["metrics"][pol][name] for f in selected_frames), "frame aggregate")
        aggregate = _aggregate_frames(selected_frames)
        for pol in ("X", "Y", "total"):
            for name, value in aggregate[pol].items():
                _same(item["metrics"][pol][name], value, f"aggregate {pol}/{name}")
        if regime == "production":
            _same(item["metrics"]["total"]["n_bits"], 2**18 * 2 * int(math.log2(dc.condition)), "production bit denominator")
        if item["candidate"] != "no-nlc":
            own_trial = trials[key]
            _same(item["checkpoint_sha256"], own_trial["checkpoint_sha256"], "evaluated checkpoint")
            _same(canonical_hash(item["complexity"]), canonical_hash(own_trial["complexity"]), "evaluated RMPS")
            _same(item["primary_selected_before_test"], key[:3] in primary, "pre-test primary")
        rep_keys[key] = item
        replicates.append(item)
    required_reps = set(trials) | {(k[0], "no-nlc", None, k[3]) for k in trials}
    missing = sorted(required_reps - set(rep_keys), key=str)
    if metrics["status"] == "complete" and (missing or metrics.get("failures")):
        raise ValueError("Complete evaluation has missing replicates or failures")
    if metrics["status"] == "complete" and metrics.get("independent_recount", {}).get("status") != "passed":
        raise ValueError("Completed evaluation lacks passing independent recount")
    if metrics["status"] == "complete" and metrics["independent_recount"].get("frame_predictions") != len(metrics["frames"]):
        raise ValueError("Independent recount frame coverage mismatch")
    groups = defaultdict(list)
    for item in replicates:
        groups[_key(item)[:3]].append(item)
    for group in metrics.get("groups", []):
        key = group["condition"], group["candidate"], group["masked"]
        items = groups.get(key, [])
        # Group values are verified, never a source for weighting duplicate
        # no-NLC rows. Duplicate aliases may have inflated legacy group counts.
        if group["candidate"] == "no-nlc" and aliases:
            continue
        if group["replicates"] != [i["replicate"] for i in items] or group["n_replicates"] != len(items):
            raise ValueError("Saved group replicate coverage mismatch")
        for name in ("ber", "q_ber_db", "mse", "effective_snr_db"):
            values = [i["metrics"]["total"][name] for i in items]
            finite = bool(values) and all(v is not None and math.isfinite(v) for v in values)
            _same(group[name]["all_points"], values, "saved group points")
            _same(group[name]["mean"], float(np.mean(values)) if finite else None, "group mean")
            _same(group[name]["sample_std"], float(np.std(values, ddof=1)) if finite and len(values) > 1 else None, "group standard deviation")
    if metrics["status"] == "complete" and {(g["condition"], g["candidate"], g["masked"]) for g in metrics["groups"]} != set(groups):
        raise ValueError("Saved group coverage mismatch")
    ledger = None
    if ledger_path is not None:
        ledger_path = Path(ledger_path).resolve()
        ledger = _load(ledger_path, inputs)
        if ledger.get("schema") != "transformer-sweep-v1" or ledger.get("regime") != regime:
            raise ValueError("Ledger schema/regime mismatch")
        keys = [_key(t) for t in ledger["trials"]]
        if len(keys) != len(set(keys)) or any(k not in expected for k in keys) or (regime == "production" and set(keys) != expected):
            raise ValueError("Ledger inventory mismatch")
        for item in ledger["trials"]:
            if _key(item) in trials:
                _same(item["status"], "complete", "frozen ledger status")
                if item.get("run_sha256"):
                    _same(item["run_sha256"], trials[_key(item)]["run_sha256"], "ledger run hash")
        protected.append(ledger_path.parent)
    resources = []
    for path in resource_paths:
        resource = _load(path, inputs)
        if resource.get("schema") != "transformer-model-resource-v1":
            raise ValueError("Unknown resource evidence schema")
        if resource.get("status") not in ("passed", "failed", "interrupted", "in_progress"):
            raise ValueError("Unknown resource evidence status")
        TrainingConfig.from_dict(resource["config"])
        resource["report_source_matches_freeze"] = resource.get("source", {}).get("content_sha256") == freeze["source"]["content_sha256"]
        resources.append(resource)
        protected.append(Path(path).resolve())
    recount_result = independent_recount(metrics_path) if recount else None
    summary = []
    for key, items in sorted(groups.items(), key=lambda pair: str(pair[0])):
        points = [i["metrics"]["total"] for i in items]
        pooled = _count_summary(sum(p["bit_errors"] for p in points), sum(p["n_bits"] for p in points))
        qs = [p["q_ber_db"] for p in points]
        summary.append({"condition": key[0], "candidate": key[1], "masked": key[2],
                        "replicates": [i["replicate"] for i in items], "pooled_counting": pooled,
                        "mean_seed_q_db": float(np.mean(qs)) if all(q is not None for q in qs) else None,
                        "sample_std_seed_q_db": float(np.std(qs, ddof=1)) if len(qs) > 1 and all(q is not None for q in qs) else None,
                        "seed_q_points_db": qs, "seed_ber_points": [p["ber"] for p in points]})
    complete = metrics["status"] == "complete" and not missing
    return {"metrics": metrics, "freeze": freeze, "trials": trials, "runs": runs, "datasets": datasets,
            "frames": sorted(frames, key=_order), "replicates": sorted(replicates, key=_order), "summary": sorted(summary, key=_order), "inputs": inputs,
            "protected": protected, "ledger": ledger, "resources": resources, "missing": missing,
            "baseline_duplicates_removed": len(metrics["frames"]) - len(frames), "recount": recount_result,
            "regime": regime, "complete": complete,
            "label": ("PILOT / " + ("未完了の部分実行 / " if not complete else "") + "合成fixtureを含み得る機能確認（本番結果ではない）") if regime == "pilot" else
                     "production / 主2条件の部分再現" if complete else "production / 未完了の部分実行"}


def _destination(output, protected):
    raw = Path(output).absolute()
    output = raw.resolve()
    if raw.is_symlink():
        raise ValueError("Report destination cannot be a symlink")
    for item in protected:
        item = Path(item).resolve()
        if output.is_relative_to(item) or item.is_relative_to(output):
            raise ValueError(f"Report output overlaps protected input: {item}")
    layout = source_identity()["source_layout"]
    actual_source = {Path(p).resolve().parent for p in layout["module_paths"].values()}
    protected_source = actual_source | {ROOT / name for name in ("src", "experiments", "configs", "tests", "docs", ".git")}
    for source in protected_source:
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("Report output overlaps repository source")
    reports = ROOT / "reports"
    allowed = reports / "transformer-nlc"
    if output.is_relative_to(reports) and output.parent != allowed:
        raise ValueError("Only a named reports/transformer-nlc/ run may be written")
    if reports.is_relative_to(output):
        raise ValueError("Report output contains existing reports")
    if any((parent / "report-manifest.json").exists() for parent in output.parents):
        raise ValueError("Report output overlaps a previous report")
    return output


def _owned_output(output, regenerate):
    if not output.exists():
        return
    if not regenerate:
        raise FileExistsError("Report exists; choose a new run-id or explicitly --regenerate")
    manifest_path = output / "report-manifest.json"
    if not manifest_path.is_file():
        raise FileExistsError("Refusing regeneration of an unowned directory")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SCHEMA:
        raise FileExistsError("Unknown report ownership")
    expected = set(manifest["outputs"]) | {"report-manifest.json"}
    actual = {str(p.relative_to(output)) for p in output.rglob("*") if p.is_file() or p.is_symlink()}
    if actual != expected or any(p.is_symlink() for p in output.rglob("*")):
        raise FileExistsError("Report contains unowned files or symlinks; preserve them and use a new run-id")
    for relative, digest in manifest["outputs"].items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts or file_hash(output / relative) != digest:
            raise FileExistsError("Report output was modified; preserve it and use a new run-id")
    allowed_dirs = {str(Path(p).parent) for p in expected}
    if any(str(p.relative_to(output)) not in allowed_dirs for p in output.rglob("*") if p.is_dir()):
        raise FileExistsError("Report contains an unowned directory")


def _number(value, precision=4):
    return "未測定・未定義" if value is None else f"{value:.{precision}g}" if isinstance(value, float) else str(value)


def _interval(value, kind):
    low, high = (value["ber_ci_low"], value["ber_ci_high"]) if kind == "BER" else (value["q_ci_low_db"], value["q_ci_high_db"])
    if kind == "Q" and value["bit_errors"] == 0:
        return f"≥{_number(low)}（ゼロ誤りの下限）"
    point = value["ber"] if kind == "BER" else value["q_ber_db"]
    return f"{_number(point)} [{_number(low)}, {_number(high)}]"


def _figures(model, directory):
    # Matplotlib is imported only when the report is actually rendered. Static
    # scientific figures use saved values and standard plotting, never images
    # generated by AI. SVG uses paths, so Japanese glyphs remain portable.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from .mask import block_mask, individual_mask
    from .model_config import ModelConfig
    candidates = ("Noto Sans CJK JP", "IPAexGothic", "IPAGothic", "TakaoGothic", "Droid Sans Fallback", "DejaVu Sans")
    available = {f.name for f in font_manager.fontManager.ttflist}
    family = next(name for name in candidates if name in available)
    font_path = Path(font_manager.findfont(family)).resolve()
    font = {"family": family, "path": str(font_path), "sha256": file_hash(font_path),
            "Japanese_font_available": family != "DejaVu Sans", "SVG_text": "paths"}
    figures = []
    palette = {False: "#d55e00", True: "#0072b2", None: "#555555"}
    conditions = sorted({k[0] for k in model["trials"]})
    with plt.rc_context({"font.family": family, "font.size": 10, "axes.unicode_minus": False,
                         "svg.fonttype": "path", "svg.hashsalt": SCHEMA, "figure.dpi": 120}):
        def save(fig, name, caption):
            fig.suptitle(model["label"], fontsize=10, color="#8d341c")
            fig.tight_layout(rect=(0, 0, 1, .96))
            for extension in ("png", "svg"):
                metadata = {"Creator": "Transformer-NLC saved-metrics reporter"}
                if extension == "svg":
                    metadata["Date"] = None
                fig.savefig(directory / f"{name}.{extension}", metadata=metadata)
            plt.close(fig)
            figures.append((name, caption))

        fig, axes = plt.subplots(len(conditions), 2, figsize=(11, 3.4 * len(conditions)), squeeze=False)
        for row, condition in enumerate(conditions):
            records = [r for r in model["replicates"] if r["condition"] == condition]
            labels = list(dict.fromkeys(_method(r) for r in records))
            for column, kind in enumerate(("BER", "Q")):
                ax = axes[row, column]
                for record in records:
                    count = record["metrics"]["total"]
                    x = labels.index(_method(record)) + (record["replicate"] - 1) * .15
                    value = count["ber"] if kind == "BER" else count["q_ber_db"]
                    low = count["ber_ci_low"] if kind == "BER" else count["q_ci_low_db"]
                    high = count["ber_ci_high"] if kind == "BER" else count["q_ci_high_db"]
                    if kind == "Q" and value is None and count["bit_errors"] == 0 and low is not None:
                        ax.scatter(x, low, marker="^", color=palette[record["masked"]])
                    elif value is not None:
                        ax.plot(x, value, "o", color=palette[record["masked"]], markersize=4)
                        if low is not None and high is not None:
                            ax.vlines(x, low, high, color=palette[record["masked"]], linewidth=1)
                    if kind == "BER" and count["bit_errors"] == 0:
                        ax.scatter(x, count["ber_ci_high"], marker="v", color=palette[record["masked"]])
                ax.set_xticks(range(len(labels)), labels, rotation=28, ha="right")
                ax.set_ylabel("BER (誤りbit / 全bit)" if kind == "BER" else "BER由来 Q (dB)")
                ax.set_title(f"{condition}QAM / 点=独立seed、縦線=95%計数区間")
                ax.grid(axis="y", alpha=.25)
        save(fig, "ber-q", "全候補と通常DSP。各点はseed全frameの合算。縦線はClopper–Pearson 95%計数区間で、seed間の信頼区間ではない。ゼロ誤りではBER上限▽とQ下限△を示す。")

        fig, axes = plt.subplots(1, len(conditions), figsize=(6 * len(conditions), 5.4), squeeze=False)
        for ax, condition in zip(axes[0], conditions):
            candidate_markers = dict(zip(CANDIDATES[condition], ("o", "s", "D")))
            labeled = set()
            for record in model["replicates"]:
                if record["condition"] != condition or record["candidate"] == "no-nlc":
                    continue
                count = record["metrics"]["total"]
                q = count["q_ber_db"]
                q_bound = q is None and count["bit_errors"] == 0 and count["q_ci_low_db"] is not None
                if q_bound:
                    q = count["q_ci_low_db"]
                rmps = record["complexity"]["total_per_output_polarization_symbol"] / 1000
                if q is not None:
                    method = _method(record)
                    ax.scatter(rmps, q, color=palette[record["masked"]],
                               marker="^" if q_bound else candidate_markers[record["candidate"]], alpha=.75,
                               label=method if method not in labeled else "_nolegend_")
                    labeled.add(method)
            baseline = [r["metrics"]["total"]["q_ber_db"] for r in model["replicates"] if r["condition"] == condition and r["candidate"] == "no-nlc"]
            if baseline and all(q is not None for q in baseline):
                ax.axhline(np.mean(baseline), color="#555555", linestyle="--", label="通常DSP seed平均")
            if labeled or ax.lines:
                ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.22), ncol=2)
            ax.set(xlabel="理論 RMPS (千 / 出力片偏波symbol)", ylabel="各seedの BER由来 Q (dB)", title=f"{condition}QAM / 固定候補すべて")
            ax.grid(alpha=.25)
        save(fig, "performance-rmps", "理論乗算数と保存test性能。点=独立seed、形=候補、色=mask有無。ゼロ誤りはQ下限△。DP時刻の計数は横軸の2倍。maskは疎実装を仮定した計数であり、dense backendの実速度向上を表さない。論文参考点は重ねず別表に示す。")

        fig, axes = plt.subplots(len(conditions), 2, figsize=(11, 3.2 * len(conditions)), squeeze=False)
        for row, condition in enumerate(conditions):
            for col, masked in enumerate((False, True)):
                ax = axes[row, col]
                for key, run in sorted(model["runs"].items(), key=lambda p: str(p[0])):
                    if key[0] != condition or key[2] != masked:
                        continue
                    epochs = [h["epoch"] for h in run["history"]]
                    line, = ax.plot(epochs, [h["validation_mse_XY_IQ"] for h in run["history"]], label=f"{key[1]} r{key[3]} validation", linewidth=1)
                    ax.plot(epochs, [h["train_mse_X_IQ"] for h in run["history"]], linestyle=":", color=line.get_color(), linewidth=.8)
                    ax.scatter(run["best_epoch"], run["best_validation_mse"], color=line.get_color(), marker="x", s=22)
                ax.set(xlabel="完了 epoch", ylabel="残差MSE (unit-Es IQ)", title=f"{condition}QAM / {'maskあり' if masked else 'maskなし'}")
                if ax.lines:
                    ax.legend(fontsize=7, ncol=2)
                ax.grid(alpha=.25)
        save(fig, "learning", "全runの曲線。点線=train X、実線=validation XY、×=best-validation epoch。停止理由は全run表に保存。epoch capは収束の証拠ではない。")

        fig, axes = plt.subplots(1, len(conditions), figsize=(6 * len(conditions), 4), squeeze=False)
        for ax, condition in zip(axes[0], conditions):
            records = [f for f in model["frames"] if f["condition"] == condition]
            labels = list(dict.fromkeys(_method(f) for f in records))
            for item in records:
                x = labels.index(_method(item)) + (item["replicate"] - 1) * .18 + (item["frame"] - 1.5) * .025
                ax.scatter(x, item["metrics"]["total"]["ber"], s=16, color=palette[item["masked"]], marker=("o", "s", "^")[item["replicate"]], alpha=.7)
            ax.set_xticks(range(len(labels)), labels, rotation=28, ha="right")
            ax.set(ylabel="frame別 BER", title=f"{condition}QAM / ○ r0、□ r1、△ r2")
            ax.grid(axis="y", alpha=.25)
        save(fig, "frame-variation", "1点=独立frame。各seedのframeをすべて示し、相関誤りを含む変動を計数区間と分離する。小fixtureのframe数は本番4とは異なる。")

        configs = {}
        for trial in model["trials"].values():
            config = ModelConfig.from_dict(trial["config"]["model"])
            if config.mask_rho is not None:
                configs[(config.t, config.b, config.mask_rho)] = config
        if configs:
            fig, axes = plt.subplots(len(configs), 2, figsize=(8, 3 * len(configs)), squeeze=False)
            for row, config in enumerate(configs.values()):
                for col, allowed in enumerate((individual_mask(config.t, config.mask_rho), block_mask(config.t, config.b, config.mask_rho))):
                    axes[row, col].imshow(allowed, origin="lower", cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
                    axes[row, col].set(xlabel="key位置 (Conv後index)", ylabel="query位置 (Conv後index)",
                                       title=f"{'個別' if col == 0 else 'block union'} t={config.t}, b={1 if col == 0 else config.b}\n許可密度={allowed.mean():.6f}")
            save(fig, "mask", "青=attention許可、白=禁止。中心行・列を許可する再現側の補完を含む。t64/b1/rho2.6の密度0.178198は原図4bの0.19と未一致。図に合わせた改変はしていない。")
    return figures, font


def _write_tables(model, output):
    tables = output / "tables"
    tables.mkdir()
    counts = []
    for scope in ("replicates", "frames"):
        for record in model[scope]:
            for pol in ("X", "Y", "total"):
                counts.append({"scope": scope, **{k: record.get(k) for k in ("condition", "candidate", "masked", "replicate", "frame")},
                               "polarization": pol, **record["metrics"][pol]})
    fields = list(dict.fromkeys(k for row in counts for k in row))
    if not fields:
        fields = ["scope", "condition", "candidate", "masked", "replicate", "frame", "polarization", "bit_errors", "n_bits", "ber"]
    with (tables / "counts.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(counts)
    (tables / "summary.json").write_text(json.dumps(model["summary"], indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    inventory = []
    ledger = {_key(t): t for t in (model["ledger"] or {}).get("trials", [])}
    for item in fixed_trial_plan():
        key = _key(item)
        run, trial = model["runs"].get(key), model["trials"].get(key)
        inventory.append({**{k: item[k] for k in ("condition", "candidate", "masked", "replicate")},
                          "training_status": run["status"] if run else ledger.get(key, {}).get("status", "not_in_freeze"),
                          "test_status": "scored" if any(_key(r) == key for r in model["replicates"]) else "not_scored",
                          "best_epoch": run.get("best_epoch") if run else None,
                          "completed_epochs": run.get("completed_epochs") if run else None,
                          "stop_reason": run.get("stop_reason") if run else ledger.get(key, {}).get("stop_reason"),
                          "convergence": run.get("convergence", "not_recorded") if run else "not_established",
                          "attempts": run.get("attempts", []) if run else ledger.get(key, {}).get("attempts", []),
                          "ledger_record": ledger.get(key),
                          "initialization_failures": run.get("report_initialization_failures", []) if run else [],
                          "resume_events": run.get("resume_events", []) if run else [],
                          "runtime": run.get("runtime") if run else None,
                          "history": run.get("history", []) if run else [],
                          "peak_allocated_GPU_bytes": run.get("peak_allocated_GPU_bytes") if run else None,
                          "checkpoint_sha256": trial["checkpoint_sha256"] if trial else None,
                          "config": trial["config"] if trial else None,
                          "complexity": trial["complexity"] if trial else None})
    (tables / "run-inventory.json").write_text(json.dumps(inventory, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return inventory


def _documents(model, figures, inventory, output, command):
    # Build Markdown and HTML together from the same escaped content. No CDN,
    # script, remote image, Markdown renderer, or network access is needed.
    md, web = [], []
    def esc(value):
        return html.escape(str(value), quote=True)
    def heading(value, level=2):
        md.append("#" * level + " " + value + "\n")
        web.append(f"<h{level}>{esc(value)}</h{level}>")
    def paragraph(value):
        md.append(esc(value) + "\n")
        web.append(f"<p>{esc(value)}</p>")
    def table(headers, rows):
        rows = list(rows)
        md.append("| " + " | ".join(esc(h).replace("|", "\\|") for h in headers) + " |\n| " + " | ".join("---" for _ in headers) + " |\n" +
                  "\n".join("| " + " | ".join(esc(c).replace("|", "\\|").replace("\n", " ") for c in row) + " |" for row in rows) + "\n")
        web.append("<div class='table'><table><thead><tr>" + "".join(f"<th>{esc(h)}</th>" for h in headers) + "</tr></thead><tbody>" +
                   "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in row) + "</tr>" for row in rows) + "</tbody></table></div>")
    def link(title, relative):
        md.append(f"[{title}]({relative})\n")
        web.append(f"<p><a href='{esc(relative)}'>{esc(title)}</a></p>")
    def figure(name):
        found = next(((n, c) for n, c in figures if n == name), None)
        if found:
            _, caption = found
            md.append(f"![{caption}](figures/{name}.png)\n\n{caption}\n")
            web.append(f"<figure><img src='figures/{name}.svg' alt='{esc(caption)}'><figcaption>{esc(caption)}</figcaption></figure>")
    def attempts_text(item):
        failures = [a for a in item["attempts"] if a.get("status") != "complete"]
        failures += [a for a in (item["ledger_record"] or {}).get("attempts", []) if a.get("status") != "complete" and a not in failures]
        failures += item["initialization_failures"]
        return "; ".join(f"{a.get('status')}: {a.get('exception', '')} / OOM={a.get('oom', '未記録')} / {str(a.get('message', ''))[:160]}" for a in failures) or "なし"

    heading("Transformerで光通信の誤りを減らせたか", 1)
    paragraph(model["label"])
    synthetic_sources = {d["source"]["fixture"] for d, _ in model["datasets"].values() if d.get("source", {}).get("fixture")}
    if synthetic_sources:
        paragraph("入力fixtureの明示: " + "; ".join(sorted(synthetic_sources)))
    paragraph("16QAMは1偏波の1記号に4 bit、64QAMは6 bitを載せる方式です。2つの偏波X/Yを同時に送ります。長い光ファイバでは雑音と光強度に依存するKerr効果で受信点がずれます。通常DSPは分散や位相などを補正する受信処理、NLCはその後に残るずれを学習モデルで推定し、引き算する補償です。残差には雑音も含まれ、純粋なKerr効果だけを測っているわけではありません。")
    table(("判断の対象", "保存された証拠と状態"), (
        ("学習・固定・評価の完了", f"学習 {len(model['trials'])}/36 run、評価 {len([r for r in model['replicates'] if r['candidate'] != 'no-nlc'])}/36 run。評価status={model['metrics']['status']}"),
        ("数値資格", "; ".join(f"{c.condition}QAM r{c.replicate}: {(d.get('qualification') or {}).get('status', '未測定 / pilot')}" for d, c in model["datasets"].values())),
        ("独立bit再計数", f"評価時={model['metrics'].get('independent_recount', {}).get('status', '未完了')}、今回={model['recount']['status'] if model['recount'] else '未再実行（保存済み証拠を参照）'}"),
        ("論文再現の分類", "主2条件の部分再現。Qが近くても原論文全体の厳密再現成功とはしない。一致の合否は根拠ある数値許容幅がないため未確定。"),
        ("独立validator", "指定 / 試行 / 完了 = 0 / 0 / 0。この報告生成は独立評価ではない。")))
    if model["missing"]:
        paragraph(f"未評価の固定対象（baseline含む）: {model['missing']}。未測定点にゼロの性能を代入していません。")
    heading("どの条件を測ったか")
    rows = []
    for _, config in model["datasets"].values():
        l = config.link
        rows.append((f"{config.condition}QAM r{config.replicate}", f"{l.n_spans}×{l.span_km:g} = {l.total_length_km:g} km", f"{l.symbol_rate_hz/1e9:g} GBd / {config.launch_power_dbm:g} dBm (XY合計)",
                     f"{config.train_frames} / {config.validation_frames} / {config.test_frames} × {config.scored_symbols} DP時刻", f"{l.sps} sps / {config.phase_max_deg:g}° / guard {config.guard_symbols} / RRC {l.rrc_span_symbols}"))
    table(("条件", "距離", "伝送速度 / power", "train / validation / test", "数値設定"), rows)
    table(("条件", "通常DSPの固定値", "normalization / testラベル"),
          ((f"{c.condition}QAM r{c.replicate}", f"FIR {c.equalizer_taps} taps / BPS window {d.get('receiver', {}).get('bps_window', '未記録')} / phases {d.get('receiver', {}).get('bps_phases', '未記録')}",
            "既知XY power→unit-Es、専用pilot校正。test scored labelsでfitしない") for d, c in model["datasets"].values()))
    table(("項目", "原論文と今回の扱い"), (
        ("published", "主16QAM 3200 km・32 GBd、主64QAM 960 km・42 GBd、NF6 dB、roll-off1/16、半量Tx分散事前補償"),
        ("adopted", "loss0.2 dB/km、D17 ps/(nm km)、gamma1.3 /(W km)、1550 nm、PMD/linewidth0、固定128 GHz ASE band、pilotによるFIR/CR、追加validation、BER由来Q"),
        ("author_tex_commented", "代表構造との対応は公式TeXコメントに依存。公開本文で確認された実行configではない。64QAM low-adoptedはt8からt16に変更。"),
        ("unresolved", "原実装のDSP、key size解釈、mask密度、Fig.8色、Q定義、RMPSの全算入規則との同一性は未確定。")))
    heading("誤り率と通信品質")
    paragraph("BERは誤りbit数/全bit数で、小さいほど良い値です。QはBERを非線形変換したdB値で、大きいほど良い値です。表のQ合算は全seedのbitを合算したBERからの変換で、seed Qの平均とは異なります。95%計数区間は独立bitを仮定するClopper–Pearson区間で、相関した誤りやseed間の不確かさを包含しません。")
    figure("ber-q")
    table(("条件 / 手法", "seed", "errors / bits（重複なし）", "合算BER [95%計数区間]", "合算Q [95%計数区間] dB", "seed Q平均 ± 標本SD dB"),
          ((f"{s['condition']}QAM / {_method(s)}", str(s["replicates"]), f"{s['pooled_counting']['bit_errors']} / {s['pooled_counting']['n_bits']}",
            _interval(s["pooled_counting"], "BER"), _interval(s["pooled_counting"], "Q"), f"{_number(s['mean_seed_q_db'])} ± {_number(s['sample_std_seed_q_db'])}") for s in model["summary"]))
    paragraph(f"通常DSPはdata manifest・condition・replicate・frameごとに一度だけ計数しました。完全一致した重複frameの除外数={model['baseline_duplicates_removed']}。本番1 seedの分母は16QAM 2^18×8=2,097,152 bit、64QAM 2^18×12=3,145,728 bitです。pilotの分母をこれらへ拡大していません。X/Y別と全frame/seedの生計数はcounts.csvに保存しました。")
    figure("frame-variation")
    heading("全候補の性能と計算量")
    figure("performance-rmps")
    complexities = {}
    for trial in model["trials"].values():
        complexities[(trial["condition"], trial["candidate"], trial["masked"])] = trial["complexity"]
    table(("条件 / 手法", "理論RMPS / 片偏波symbol", "理論RMPS / DP時刻", "dense backend推定 / 片偏波symbol", "許可attention密度"),
          ((f"{key[0]}QAM {key[1]} / {'maskあり' if key[2] else 'maskなし'}", _number(c["total_per_output_polarization_symbol"]), _number(c["total_per_dp_time"]),
            _number(c["dense_backend_total_per_output_polarization_symbol"]), f"{c['allowed_attention_elements']/c['dense_attention_elements']:.6f}") for key, c in complexities.items()))
    paragraph("RMPSは採用規約による実乗算数の推定です。片偏波出力に対しXYを2回推論するためDP時刻では2倍です。maskありの理論削減は疎な計算を仮定します。実装はdense attention dispatcherを使うため、密度低下から学習時間・速度・VRAMの削減を主張できません。加算、活性化、rsqrt、メモリ移動は計数外で、softmax換算とLayerNormの算入は再現側の規約です。")
    paragraph("primaryはtestを開く前にvalidation MSEで選択した候補です。seedの良いものだけを選ぶ操作はしていません。固定された選択: " + "; ".join(f"{p['condition']}QAM {'maskあり' if p['masked'] else 'maskなし'}={p['candidate']}、seed={p['replicates']}" for p in model["freeze"]["primary"]))
    heading("学習はどこで止まったか")
    figure("learning")
    table(("条件 / 候補 / mask / seed", "学習 / test", "best / 完了epoch", "停止理由 / 収束", "失敗・中断attempt"),
          ((f"{i['condition']} / {i['candidate']} / {i['masked']} / r{i['replicate']}", f"{i['training_status']} / {i['test_status']}", f"{i['best_epoch']} / {i['completed_epochs']}",
            f"{i['stop_reason']} / {i['convergence']}", attempts_text(i)) for i in inventory))
    paragraph("全36の予定条件を表示しました。未実施runは本番完了に数えません。cap到達は収束を意味せず、patienceも100 epoch改善なしという停止規則の達成を表すだけです。上表の失敗messageは先頭160文字までで、失敗・OOM・中断・再開理由の全履歴と設定はrun-inventory.jsonに残しました。")
    if model["metrics"].get("failures"):
        paragraph("評価時の失敗: " + json.dumps(model["metrics"]["failures"], ensure_ascii=False))
    heading("maskが見ている範囲")
    figure("mask")
    paragraph("maskは離れた時刻の組合せの一部をattentionから除外する規則です。m=0の中心行とn=0列を許可する積形式で補完しており、Algorithm 1を文字通り実装したものではありません。Fig.4bとの密度差は未解決です。maskの絵はモデル規則の可視化で、通信品質の測定結果ではありません。")
    heading("実測資源と計測範囲")
    table(("対象", "hardware / dtype / backend", "有効batch / microbatch", "時間 / peak GPU memory", "範囲"),
          ((f"{k[0]} {k[1]} {k[2]} r{k[3]}", f"{r['runtime'].get('gpu_name', r['runtime'].get('platform', 'hardware未記録'))} / {r['runtime'].get('dtype', '未記録')} / {r['runtime'].get('attention_backend', '未記録')}",
            f"{r['config']['effective_batch_blocks']} / {r['config']['microbatch_blocks']}", f"完了epoch合計 {sum(h['seconds'] for h in r['history']):.3g} s / {_number(r.get('peak_allocated_GPU_bytes'))} bytes", "train X + validation XY。失敗・初期化・I/Oの全時間ではない") for k, r in model["runs"].items()))
    table(("test対象", "時間 s（I/O/XY scoring含む）", "peak allocated GPU bytes"),
          ((f"{r['condition']} {_method(r)} r{r['replicate']}", _number(r.get("evaluation_seconds_including_IO_scoring_XY")), _number(r.get("peak_allocated_GPU_bytes"))) for r in model["replicates"] if r["candidate"] != "no-nlc"))
    paragraph("GPU memoryのnullはゼロ使用の実測値ではなく未測定です。通常DSP行の保存時間はTransformerと同じ評価ループに属するため、通常DSP単独のlatencyとして集計しません。未記録のwarm-up/throughput/単独推論latencyは未測定です。評価runtime: " + json.dumps(model["metrics"].get("evaluation_runtime", {}), ensure_ascii=False))
    if model["resources"]:
        table(("専用資源benchmark", "status / freezeとのsource一致", "秒 / allocated / reserved bytes", "計測条件"),
              ((f"{r['config']['condition']} / {r['config']['candidate']} / {r['config']['masked']}", f"{r['status']} / {r['report_source_matches_freeze']}", f"{_number(r.get('seconds'))} / {_number(r.get('peak_allocated_GPU_bytes'))} / {_number(r.get('peak_reserved_GPU_bytes'))}",
                f"{r.get('measurement', '未記録')} / {r.get('runtime', {})}") for r in model["resources"]))
    else:
        paragraph("専用resource proofは未入力です。最大構造のbatch資源資格や本番所要時間をこの報告だけで保証しません。")
    heading("論文の参考値と今回の限界")
    table(("16QAM参考値（Table II）", "通常DSP Q dB", "A / B / C Q dB", "A / B / C RMPS 千 / 片偏波symbol"),
          (("maskなし", "6.67", "8.82 / 8.54 / 7.31", "608 / 129 / 22"), ("maskあり", "6.67", "8.76 / 8.51 / 7.68", "404 / 81 / 9")))
    paragraph("64QAMの本文参考値は通常DSP約6.5 dBから最良約8.1 dBです。今回の各縮小候補の正解値や達成保証ではありません。上の表はpublishedの参考値であり、この報告の実測値ではありません。構造との対応はauthor_tex_commentedに留まります。元論文の図は結果図に転用していません。")
    link("原論文 arXiv:2304.13119v3（2024-08-01）", PAPER)
    comparison = []
    baseline = {s["condition"]: s for s in model["summary"] if s["candidate"] == "no-nlc"}
    for s in model["summary"]:
        if s["candidate"] == "no-nlc":
            continue
        base_q = baseline.get(s["condition"], {}).get("mean_seed_q_db")
        q = s["mean_seed_q_db"]
        delta = q - base_q if q is not None and base_q is not None else None
        reference = {False: {"A": 8.82, "B": 8.54, "C": 7.31}, True: {"A": 8.76, "B": 8.51, "C": 7.68}}[s["masked"]].get(s["candidate"]) if s["condition"] == 16 else None
        matched = []
        for record in model["replicates"]:
            if _key(record)[:3] != (s["condition"], s["candidate"], s["masked"]):
                continue
            same_seed = next((b for b in model["replicates"] if b["candidate"] == "no-nlc" and (b["condition"], b["replicate"]) == (record["condition"], record["replicate"])), None)
            rq = record["metrics"]["total"]["q_ber_db"]
            bq = same_seed["metrics"]["total"]["q_ber_db"] if same_seed else None
            if rq is not None and bq is not None:
                matched.append(rq - bq)
        delta = float(np.mean(matched)) if matched else None
        comparison.append((f"{s['condition']}QAM / {_method(s)}", f"{_number(delta)} (n={len(matched)})", _number(q - reference if q is not None and reference is not None else None),
                           _number(delta - (reference - 6.67) if delta is not None and reference is not None else None),
                           "平均Qで改善" if delta is not None and delta > 0 else "平均Qで改善なし" if delta is not None else "判定不能"))
    table(("手法", "同一seed通常DSPからのΔQ平均 dB", "論文からの絶対Q差 dB", "論文からのΔQ差 dB", "観測結果（有意性判定ではない）"), comparison)
    paragraph("ΔQは同じ条件・seedの差を取り、その全点を平均した記述値で、統計的有意差の宣言ではありません。nは対応するQが定義されたseed数です。数値資格の0.1 dB目安を論文性能の合格幅に転用しません。縮小探索、採用した物理/DSP条件、追加validation、コメント由来構造、mask補完と密度差の限界が残ります。launch-power曲線と最適power、PMD、レート変更、CNN-LSTM/DBP、広範なアブレーションは未実施で、追加前に相談する範囲です。負の結果も隠さず、親Issue #40の全範囲の完了とは扱いません。")
    heading("保存物から再生成する")
    paragraph("学習・test推論を再実行せずcompact JSONを検査して生成しました。checkpointとbulk配列は既定では読みません。モデルのhashはfreezeに保存された識別子を照合したもので、今回モデルbytesを再検証したという意味ではありません。optional --recountだけは保存されたbulk予測をCPUで再計数します。")
    command_text = json.dumps(command, ensure_ascii=False)
    md.append("```json\n" + command_text + "\n```\n")
    web.append("<pre>" + esc(command_text) + "</pre>")
    link("入力・生成コード・出力のhash manifest", "report-manifest.json")
    link("全frame/seed・X/Y/合算の直接計数 CSV", "tables/counts.csv")
    link("合算とseed平均を分けた集約 JSON", "tables/summary.json")
    link("全36予定run・失敗・再開・設定・RMPS内訳 JSON", "tables/run-inventory.json")
    output.joinpath("report.md").write_text("\n".join(md), encoding="utf-8")
    style = "body{font-family:system-ui,'Noto Sans CJK JP',sans-serif;max-width:1120px;margin:2rem auto;padding:0 1rem;color:#172b3a;line-height:1.75}h1,h2{line-height:1.4}h2{margin-top:2.5rem}figure{margin:1.7rem 0}img{width:100%;height:auto}figcaption{font-size:.9rem;color:#455568}table{border-collapse:collapse;font-size:.85rem;width:100%}th,td{border:1px solid #cfdae3;padding:.5rem;text-align:left;vertical-align:top}th{background:#eef3f7}.table{overflow-x:auto;margin:1rem 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef3f7;padding:1rem}a{color:#00659c}"
    output.joinpath("report.html").write_text("<!doctype html><html lang='ja'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Transformer-NLC 保存結果</title><style>" + style + "</style><body>" + "\n".join(web) + "</body></html>\n", encoding="utf-8")


def generate_report(metrics_path, output_dir, *, allow_pilot=False, allow_partial=False, ledger_path=None,
                    resource_paths=(), recount=False, regenerate=False, dry_run=False, command=None):
    """Validate first, render in a sibling temporary directory, publish atomically.

    Regeneration replaces only an unchanged report owned by our hash manifest.
    Any unexpected or modified user file causes refusal before writing output.
    """
    code_paths = [Path(__file__), Path(__file__).with_name("__main__.py"), Path(__file__).with_name("evaluation.py"),
                  Path(__file__).with_name("complexity.py"), Path(__file__).with_name("mask.py"), Path(__file__).with_name("model_config.py"),
                  Path(__file__).with_name("training_config.py"), Path(__file__).with_name("data.py"), Path(__file__).with_name("data_config.py"),
                  Path(__file__).with_name("checkpoint.py"), Path(__file__).with_name("docs") / "reproduction-spec.md"]
    import optical_comm.metrics as shared_metrics
    code_paths.append(Path(shared_metrics.__file__).resolve())
    code = {str(p): file_hash(p) for p in code_paths}
    model = inspect_inputs(metrics_path, allow_pilot=allow_pilot, allow_partial=allow_partial,
                           ledger_path=ledger_path, resource_paths=resource_paths, recount=recount)
    output = _destination(output_dir, model["protected"])
    _owned_output(output, regenerate)
    plan = {"status": "dry_run" if dry_run else "complete", "output": str(output), "label": model["label"],
            "input_files": model["inputs"], "frozen_trials": len(model["trials"]), "scored_replicates": len(model["replicates"]),
            "missing_replicates": [list(k) for k in model["missing"]], "bulk_access": recount,
            "training_or_inference": False, "checkpoint_loading": False}
    if dry_run:
        return plan
    from .checkpoint import advisory_lock
    output.parent.mkdir(parents=True, exist_ok=True)
    repository = next((parent for parent in output.parents if (parent / ".git").exists()), None)
    lock = ((repository / ".work" / "transformer-report-locks" / f"{canonical_hash(str(output))}.lock")
            if repository is not None else output.parent / f".{output.name}.report.lock")
    with advisory_lock(lock):
        _owned_output(output, regenerate)
        temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
        backup = None
        try:
            (temporary / "figures").mkdir()
            figures, font = _figures(model, temporary / "figures")
            inventory = _write_tables(model, temporary)
            _documents(model, figures, inventory, temporary, command)
            # A running evaluation/run must not silently change between our
            # initial read and publication, even with --allow-partial.
            for path, digest in model["inputs"].items():
                if file_hash(path) != digest:
                    raise ValueError(f"Input changed during report generation: {path}")
            for path, digest in code.items():
                if file_hash(path) != digest:
                    raise ValueError(f"Report code changed during generation: {path}")
            import matplotlib
            import platform
            import scipy
            manifest = {"schema": SCHEMA, "generated_utc": datetime.now(timezone.utc).isoformat(), "command": command,
                        "regime": model["regime"], "evaluation_complete": model["complete"], "label": model["label"],
                        "inputs": model["inputs"], "code": code, "code_sha256": canonical_hash(code), "font": font,
                        "renderer_runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__, "platform": platform.platform()},
                        "saved_numerical_source": model["freeze"]["source"],
                        "validation": {"compact_hashes": "passed", "count_summaries_and_coverage": "passed",
                                       "baseline_duplicate_frames_removed": model["baseline_duplicates_removed"],
                                       "independent_recount_this_generation": model["recount"], "checkpoint_bytes": "not_read", "bulk_arrays": "read_for_recount" if recount else "not_read",
                                       "offline_links": "relative figure/table links checked; external paper citation optional",
                                       "independent_validator_requested_attempted_completed": [0, 0, 0]},
                        "outputs": {str(p.relative_to(temporary)): file_hash(p) for p in sorted(temporary.rglob("*")) if p.is_file()}}
            # Check all offline assets actually exist before publishing.
            for name, _ in figures:
                for suffix in ("png", "svg"):
                    if not (temporary / "figures" / f"{name}.{suffix}").is_file():
                        raise ValueError("Missing offline figure")
            (temporary / "report-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            if output.exists():
                _owned_output(output, regenerate)
                backup = Path(tempfile.mkdtemp(prefix=f".{output.name}.previous.", dir=output.parent))
                backup.rmdir()
                os.replace(output, backup)
            try:
                os.replace(temporary, output)
            except BaseException:
                if backup is not None:
                    os.replace(backup, output)
                    backup = None
                raise
            if backup is not None:
                shutil.rmtree(backup)
            return plan | {"markdown": str(output / "report.md"), "html": str(output / "report.html"), "manifest": str(output / "report-manifest.json")}
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
