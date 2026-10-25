"""Validation-only selection freeze and counted, reproducible held-out scoring."""
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.special import erfcinv

from optical_comm.metrics import ber_interval, compute_metrics, symbols_to_bits
from .checkpoint import (advisory_lock, code_identity, configure_runtime, device_lock,
                         load_checkpoint, protect_output)
from .complexity import complexity_breakdown
from .data import atomic_json, atomic_npz, canonical_hash, ensure_bulk_ignored, file_hash, iter_blocks, load_frame
from .training import read_manifest
from .training_config import CANDIDATES, TrainingConfig, fixed_trial_plan


def compensate(received: np.ndarray, residual: np.ndarray, *, train_power_dbm: float, eval_power_dbm: float):
    """Eq. 2 after DSP in unit-Es coordinates; no fitting to evaluation labels."""
    received, residual = np.asarray(received), np.asarray(residual)
    if received.shape != residual.shape or not np.all(np.isfinite(received)) or not np.all(np.isfinite(residual)):
        raise ValueError("Received and predicted residual arrays must match and be finite")
    if not math.isfinite(train_power_dbm) or not math.isfinite(eval_power_dbm):
        raise ValueError("Launch powers must be finite")
    alpha = 10 ** ((eval_power_dbm - train_power_dbm) / 10)
    return received - alpha * residual


def freeze_selection(run_directories: list[str | Path], output_path: str | Path, *, allow_pilot=False, command=None) -> dict:
    """Freeze every predeclared candidate before any test frame is opened."""
    if not run_directories:
        raise ValueError("Freeze requires completed training runs")
    records = []
    for directory in run_directories:
        directory = Path(directory).resolve()
        record = json.loads((directory / "run.json").read_text())
        config = TrainingConfig.from_dict(record["config"])
        if record["status"] != "complete" or "best" not in record.get("checkpoints", {}):
            raise ValueError("Freeze requires completed runs with best-validation checkpoints; failures remain unresolved")
        if config.regime != "production" and not allow_pilot:
            raise ValueError("Pilot freeze must be explicitly marked allow_pilot")
        path, data, data_config = read_manifest(record["data_manifest"])
        if record["identities"]["data_manifest_sha256"] != file_hash(path):
            raise ValueError("Training dataset changed before freeze")
        checkpoint = directory / record["checkpoints"]["best"]["path"]
        state = load_checkpoint(checkpoint, expected_sha256=record["checkpoints"]["best"]["sha256"])
        if canonical_hash(state["config"]) != state["identities"]["config_sha256"] or canonical_hash(record["config"]) != record["identities"]["config_sha256"]:
            raise ValueError("Training/checkpoint configuration hash mismatch")
        checkpoint_config, final_config = dict(state["config"]), dict(record["config"])
        checkpoint_micro, final_micro = checkpoint_config.pop("microbatch_blocks"), final_config.pop("microbatch_blocks")
        approved_micro_change = checkpoint_micro == final_micro or any(
            event.get("microbatch_change", {}).get("to") == final_micro for event in record.get("resume_events", []))
        same_identities = {k: v for k, v in state["identities"].items() if k != "config_sha256"} == {
            k: v for k, v in record["identities"].items() if k != "config_sha256"}
        if (not same_identities or not approved_micro_change or canonical_hash(checkpoint_config) != canonical_hash(final_config)
            or state["epoch"] != record["best_epoch"]):
            raise ValueError("Best checkpoint does not match run identity/selection")
        if state["best_validation_mse"] != record["best_validation_mse"] or not math.isfinite(state["best_validation_mse"]):
            raise ValueError("Nonfinite or inconsistent selection score")
        if record["identities"]["code_content_sha256"] != code_identity()["content_sha256"]:
            raise ValueError("Executable code changed since training; resolve before freezing test")
        if config.condition != data_config.condition:
            raise ValueError("Condition mismatch")
        records.append({"condition": config.condition, "candidate": config.candidate, "masked": config.masked,
            "replicate": data_config.replicate, "regime": config.regime, "run_directory": str(directory),
            "run_sha256": file_hash(directory / "run.json"), "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_hash(checkpoint), "config": config.to_dict(),
            "checkpoint_config": state["config"], "checkpoint_config_sha256": state["identities"]["config_sha256"],
            "operational_resume_events": record.get("resume_events", []),
            "config_sha256": record["identities"]["config_sha256"], "best_epoch": state["epoch"],
            "validation_mse": state["best_validation_mse"], "data_manifest": str(path),
            "data_manifest_sha256": file_hash(path), "data_config_sha256": data["config_sha256"],
            "data_source": data["source"], "training_source": record["source"],
            "training_runtime": record["runtime"], "train_power_dbm": data_config.launch_power_dbm,
            "complexity": complexity_breakdown(config.model).to_dict()})
    keys = [(r["condition"], r["candidate"], r["masked"], r["replicate"]) for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate frozen trial")
    common_data = defaultdict(set)
    for record in records:
        common_data[(record["condition"], record["replicate"])].add(record["data_manifest_sha256"])
    if any(len(values) != 1 for values in common_data.values()):
        raise ValueError("Every candidate/mask pair in one condition/replicate must share the same frozen dataset")
    production = all(r["regime"] == "production" for r in records)
    if any(r["regime"] == "production" for r in records) and not production:
        raise ValueError("Cannot mix pilot and production freezes")
    expected = {(r["condition"], r["candidate"], r["masked"], r["replicate"]) for r in fixed_trial_plan()}
    if production and set(keys) != expected:
        raise ValueError("Production freeze requires the full finite 36-run inventory")
    scores = defaultdict(list)
    for record in records:
        scores[(record["condition"], record["masked"], record["candidate"])].append(record)
    ranking = []
    for (condition, masked, candidate), group in scores.items():
        ranking.append({"condition": condition, "masked": masked, "candidate": candidate,
            "replicates": sorted(r["replicate"] for r in group),
            "mean_validation_mse": float(np.mean([r["validation_mse"] for r in group])),
            "rmps_per_polarization_symbol": group[0]["complexity"]["total_per_output_polarization_symbol"]})
    primary = []
    for condition, masked in sorted({(r["condition"], r["masked"]) for r in records}):
        group = [r for r in ranking if (r["condition"], r["masked"]) == (condition, masked)]
        primary.append(min(group, key=lambda r: (r["mean_validation_mse"], r["rmps_per_polarization_symbol"],
                                                -CANDIDATES[condition].index(r["candidate"]))))
    freeze = {"schema": "transformer-selection-freeze-v1", "status": "frozen", "regime": "production" if production else "pilot",
              "frozen_unix": time.time(), "test_metrics_seen": False, "command": command,
              "source": code_identity(), "trials": records, "ranking": ranking, "primary": primary,
              "selection_rule": "mean validation IQ MSE over all 3 replicates; exact recorded-score tie: lower RMPS, then lower table row",
              "pilot_selection_limit": None if production else "incomplete candidate/replicate inventory; cannot claim production selection",
              "test_protocol": {"power": "same fixed principal launch power", "alpha": "10**((P_eval-P_train)/10)",
                 "normalization": "unit-Es from frozen receiver and known physical power; no test-label fitting",
                 "metrics": "direct Gray BER X/Y/total, Clopper-Pearson95%, BER-derived Q; MSE/EVM/SNR auxiliary",
                 "paper_agreement": "absolute Q and delta-Q, frame/replicate variability; no justified numeric acceptance tolerance"}}
    path = protect_output(output_path, [r["run_directory"] for r in records] + [Path(r["data_manifest"]).parent for r in records])
    path.parent.mkdir(parents=True, exist_ok=True)
    with advisory_lock(path.with_suffix(path.suffix + ".lock")):
        if path.exists():
            raise FileExistsError("Selection freeze is immutable; choose a new path")
        atomic_json(path, freeze)
    return freeze


def _q(ber: float) -> float | None:
    if not 0 < ber < .5:
        return None
    return float(20 * np.log10(np.sqrt(2) * erfcinv(2 * ber)))


def _count_summary(errors: int, n_bits: int) -> dict:
    low, high = ber_interval(errors, n_bits)
    return {"bit_errors": errors, "n_bits": n_bits, "ber": errors / n_bits,
            "ber_ci_low": low, "ber_ci_high": high, "q_ber_db": _q(errors / n_bits),
            "q_ci_low_db": _q(high), "q_ci_high_db": _q(low),
            "q_limit": "zero_errors_Q_lower_bound_only" if errors == 0 else "undefined_BER_ge_half" if errors / n_bits >= .5 else None,
            "interval_scope": "binomial counting only; correlated errors require independent frame/replicate variation"}


def score_symbols(tx_symbols: np.ndarray, predicted_symbols: np.ndarray, tx_bits: np.ndarray, condition: int) -> dict:
    """Score supplied unit-Es coordinates directly; no label-derived rotation/gain."""
    if tx_symbols.ndim != 2 or tx_symbols.shape[1] != 2 or predicted_symbols.shape != tx_symbols.shape:
        raise ValueError("Require scored [time,XY] symbols")
    metrics = {}
    for label, index in (("X", 0), ("Y", 1), ("total", slice(None))):
        raw = compute_metrics(tx_symbols[:, index], predicted_symbols[:, index], tx_bits[:, index], order=condition)
        summary = _count_summary(raw["bit_errors"], raw["n_bits"])
        summary.update({name: value if math.isfinite(value) else None for name, value in raw.items()
                        if name in ("mse", "evm", "effective_snr_db", "signal_power")})
        metrics[label] = summary
    return metrics


def _aggregate_frames(frames: list[dict]) -> dict:
    result = {}
    for label in ("X", "Y", "total"):
        values = [f["metrics"][label] for f in frames]
        errors = sum(v["bit_errors"] for v in values)
        n_bits = sum(v["n_bits"] for v in values)
        result[label] = _count_summary(errors, n_bits)
        mse = sum(v["mse"] * v["n_bits"] for v in values) / n_bits
        power = sum(v["signal_power"] * v["n_bits"] for v in values) / n_bits
        result[label].update(mse=mse, signal_power=power, evm=math.sqrt(mse / power),
                             effective_snr_db=None if mse == 0 else 10 * math.log10(power / mse))
    return result


def _predict_frame(model, frame, config, device):
    import torch
    from .model import predict_dual_polarization
    blocks = list(iter_blocks(frame, block_symbols=config.model.b, context=config.model.t))
    residuals, indexes = [], []
    with torch.inference_mode():
        for offset in range(0, len(blocks), config.microbatch_blocks):
            batch = blocks[offset:offset + config.microbatch_blocks]
            received = np.stack([b["rx"] for b in batch])
            features = np.stack((received[..., 0].real, received[..., 0].imag,
                                 received[..., 1].real, received[..., 1].imag), axis=-1).astype(np.float32)
            predicted = predict_dual_polarization(model, torch.from_numpy(features).to(device)).cpu().numpy()
            residuals.append((predicted[..., 0] + 1j * predicted[..., 1]).reshape(-1, 2).astype(np.complex128))
            indexes.extend(b["symbol_index"] for b in batch)
    return np.concatenate(residuals), np.concatenate(indexes)


def evaluate(freeze_path: str | Path, output_dir: str | Path, *, device="cpu", command=None, gpu_lock=None) -> dict:
    """Reload frozen checkpoints and evaluate the entire fixed candidate set."""
    import torch
    from .model import TransformerNLC
    freeze_path = Path(freeze_path).resolve()
    freeze = json.loads(freeze_path.read_text())
    if freeze.get("schema") != "transformer-selection-freeze-v1" or freeze.get("status") != "frozen" or freeze.get("test_metrics_seen") is not False:
        raise ValueError("Evaluation requires an explicit immutable selection freeze")
    if freeze["source"]["content_sha256"] != code_identity()["content_sha256"]:
        raise ValueError("Executable code differs from selection freeze")
    if not freeze.get("trials"):
        raise ValueError("Freeze contains no trials")
    # Verify the entire inventory before opening even the first test frame.
    # A defect in a later candidate must not cause a partial test unsealing.
    for trial in freeze["trials"]:
        if file_hash(trial["data_manifest"]) != trial["data_manifest_sha256"] or file_hash(Path(trial["run_directory"]) / "run.json") != trial["run_sha256"]:
            raise ValueError("Frozen dataset/training run identity changed")
        if file_hash(trial["checkpoint"]) != trial["checkpoint_sha256"]:
            raise ValueError("Checkpoint hash mismatch")
        read_manifest(trial["data_manifest"])
    inputs = [freeze_path] + [t["run_directory"] for t in freeze["trials"]] + [Path(t["data_manifest"]).parent for t in freeze["trials"]]
    output = protect_output(output_dir, inputs)
    ensure_bulk_ignored(output)
    output.mkdir(parents=True, exist_ok=True)
    with advisory_lock(output / ".evaluation.lock"), device_lock(device, gpu_lock):
        if any(p.name != ".evaluation.lock" for p in output.iterdir()):
            raise FileExistsError("Evaluation output exists; use a fresh output for checkpoint reload")
        bulk = output / "bulk"
        bulk.mkdir()
        result = {"schema": "transformer-evaluation-v1", "status": "in_progress", "regime": freeze["regime"],
                  "freeze_path": str(freeze_path), "freeze_sha256": file_hash(freeze_path), "command": command,
                  "frames": [], "replicates": [], "groups": [], "source": code_identity(), "failures": []}
        atomic_json(output / "metrics.json", result)
        seen_baselines = set()
        try:
            for trial in freeze["trials"]:
                if file_hash(trial["data_manifest"]) != trial["data_manifest_sha256"] or file_hash(Path(trial["run_directory"]) / "run.json") != trial["run_sha256"]:
                    raise ValueError("Frozen dataset/training run identity changed")
                path, manifest, data_config = read_manifest(trial["data_manifest"])
                state = load_checkpoint(trial["checkpoint"], expected_sha256=trial["checkpoint_sha256"])
                if (state["identities"]["config_sha256"] != trial["checkpoint_config_sha256"]
                    or canonical_hash(state["config"]) != canonical_hash(trial["checkpoint_config"])):
                    raise ValueError("Frozen checkpoint configuration mismatch")
                config_values = {**trial["config"], "device": device}
                config = TrainingConfig.from_dict(config_values)
                runtime = configure_runtime(config)
                result["evaluation_runtime"] = runtime
                model = TransformerNLC(config.model).to(device)
                model.load_state_dict(state["model"])
                model.eval()
                trial_id = f"main{trial['condition']}-{trial['candidate']}-{'masked' if trial['masked'] else 'unmasked'}-r{trial['replicate']}"
                first_baseline = trial["data_manifest_sha256"] not in seen_baselines
                frames_for_method, baseline_frames = [], []
                started = time.perf_counter()
                if device.startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats(device)
                for entry in manifest["frames"]:
                    if entry["split"] != "test":
                        continue
                    frame = load_frame(path.parent / entry["path"], expected_sha256=entry["file_sha256"])
                    residual, indexes = _predict_frame(model, frame, config, device)
                    scored = np.flatnonzero(frame.arrays["scoring_mask"])
                    if not np.array_equal(scored, indexes) or len(scored) != data_config.scored_symbols:
                        raise ValueError("Test targets overlap or differ from fixed scoring budget")
                    rx = frame.arrays["rx_nlc"][scored]
                    tx = frame.arrays["tx_symbols"][scored]
                    bits = frame.arrays["tx_bits"][scored]
                    corrected = compensate(rx, residual, train_power_dbm=trial["train_power_dbm"], eval_power_dbm=data_config.launch_power_dbm)
                    methods = [(trial_id, residual, corrected, trial["candidate"], trial["masked"])]
                    if first_baseline:
                        methods.append((f"main{trial['condition']}-no-nlc-r{trial['replicate']}", np.zeros_like(residual), rx, "no-nlc", None))
                    for method, estimated, prediction, candidate, masked in methods:
                        diagnostic = bulk / f"{method}-frame{entry['frame']}.npz"
                        atomic_npz(diagnostic, {"rx_nlc": rx, "predicted_residual": estimated, "tx_symbols": tx,
                            "tx_bits": bits, "symbol_index": indexes,
                            "alpha": np.array(10 ** ((data_config.launch_power_dbm - trial["train_power_dbm"]) / 10))})
                        metrics = score_symbols(tx, prediction, bits, trial["condition"])
                        item = {"method": method, "condition": trial["condition"], "candidate": candidate,
                            "masked": masked, "replicate": trial["replicate"], "frame": entry["frame"],
                            "scored_DP_times": len(scored), "excluded_DP_times": len(frame.arrays["scoring_mask"]) - len(scored),
                            "data_frame_sha256": entry["file_sha256"], "diagnostic_path": str(diagnostic.relative_to(output)),
                            "diagnostic_sha256": file_hash(diagnostic), "metrics": metrics}
                        result["frames"].append(item)
                        (baseline_frames if candidate == "no-nlc" else frames_for_method).append(item)
                if device.startswith("cuda"):
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                for items in (frames_for_method, baseline_frames):
                    if not items:
                        continue
                    first = items[0]
                    aggregate = _aggregate_frames(items)
                    expected_bits = data_config.test_frames * data_config.scored_symbols * 2 * int(math.log2(trial["condition"]))
                    if aggregate["total"]["n_bits"] != expected_bits:
                        raise ValueError("Test bit denominator mismatch")
                    if freeze["regime"] == "production" and expected_bits != 2**18 * 2 * int(math.log2(trial["condition"])):
                        raise ValueError("Production test budget was reduced")
                    is_primary = any((p["condition"], p["masked"], p["candidate"]) == (first["condition"], first["masked"], first["candidate"])
                                     for p in freeze["primary"])
                    result["replicates"].append({k: first[k] for k in ("method", "condition", "candidate", "masked", "replicate")} |
                        {"metrics": aggregate, "primary_selected_before_test": is_primary,
                         "checkpoint_sha256": None if first["candidate"] == "no-nlc" else trial["checkpoint_sha256"],
                         "complexity": None if first["candidate"] == "no-nlc" else trial["complexity"],
                         "evaluation_seconds_including_IO_scoring_XY": elapsed,
                         "peak_allocated_GPU_bytes": torch.cuda.max_memory_allocated(device) if device.startswith("cuda") else None})
                seen_baselines.add(trial["data_manifest_sha256"])
                atomic_json(output / "metrics.json", result)
            grouped = defaultdict(list)
            for item in result["replicates"]:
                grouped[(item["condition"], item["candidate"], item["masked"])].append(item)
            for (condition, candidate, masked), items in grouped.items():
                group = {"condition": condition, "candidate": candidate, "masked": masked,
                         "replicates": [i["replicate"] for i in items], "n_replicates": len(items)}
                for name in ("ber", "q_ber_db", "mse", "effective_snr_db"):
                    values = [item["metrics"]["total"][name] for item in items]
                    finite = all(v is not None and math.isfinite(v) for v in values)
                    group[name] = {"all_points": values, "mean": float(np.mean(values)) if finite else None,
                                   "sample_std": float(np.std(values, ddof=1)) if finite and len(values) > 1 else None}
                result["groups"].append(group)
            result.update(status="complete", independent_recount=independent_recount(result, output))
            atomic_json(output / "metrics.json", result)
            with (output / "metrics.csv").open("w", newline="") as stream:
                columns = ["condition", "candidate", "masked", "replicate", "polarization", "bit_errors", "n_bits", "ber", "ber_ci_low", "ber_ci_high", "q_ber_db", "q_ci_low_db", "q_ci_high_db", "mse", "evm", "effective_snr_db"]
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                for item in result["replicates"]:
                    for pol in ("X", "Y", "total"):
                        writer.writerow({**{k: item[k] for k in columns[:4]}, "polarization": pol,
                                         **{k: item["metrics"][pol][k] for k in columns[5:]}})
            return result
        except BaseException as exc:
            result.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed")
            result["failures"].append({"exception": type(exc).__name__, "message": str(exc), "time_unix": time.time()})
            atomic_json(output / "metrics.json", result)
            raise


def independent_recount(metrics: dict | str | Path, directory: str | Path | None = None) -> dict:
    """Reconstruct Eq.2 from saved residuals and independently recount bits.

    Uses direct decisions/count_nonzero, separately from compute_metrics used
    by evaluation. No model or checkpoint is required for this recount.
    """
    if isinstance(metrics, (str, Path)):
        path = Path(metrics)
        directory, metrics = path.parent, json.loads(path.read_text())
    root = Path(directory)
    totals = defaultdict(lambda: [0, 0])
    for frame in metrics["frames"]:
        path = root / frame["diagnostic_path"]
        if file_hash(path) != frame["diagnostic_sha256"]:
            raise ValueError("Diagnostic hash mismatch")
        with np.load(path, allow_pickle=False) as arrays:
            decisions = symbols_to_bits(arrays["rx_nlc"] - float(arrays["alpha"]) * arrays["predicted_residual"], order=frame["condition"])
            bits = arrays["tx_bits"]
            for label, index in (("X", 0), ("Y", 1), ("total", slice(None))):
                errors = int(np.count_nonzero(bits[:, index] != decisions[:, index]))
                count = int(bits[:, index].size)
                expected = frame["metrics"][label]
                if (errors, count) != (expected["bit_errors"], expected["n_bits"]):
                    raise ValueError("Independent BER recount disagrees")
                totals[(frame["method"], label)][0] += errors
                totals[(frame["method"], label)][1] += count
    for replicate in metrics["replicates"]:
        for label in ("X", "Y", "total"):
            expected = replicate["metrics"][label]
            if totals[(replicate["method"], label)] != [expected["bit_errors"], expected["n_bits"]]:
                raise ValueError("Independent aggregate recount disagrees")
    return {"status": "passed", "frame_predictions": len(metrics["frames"]), "implementation": "saved Eq2 residual reconstruction + direct Gray decisions/count_nonzero"}
