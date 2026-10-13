#!/usr/bin/env python3
"""Audit saved run metadata and arrays without propagation, fitting, or prediction.

Input data stay read-only. An optional JSON output separates deterministic
consistency checks, current audit observations, and retained historical evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import shlex
import sys
import time
from zipfile import BadZipFile

import numpy as np

import optical_comm.experiment as experiment
from optical_comm.experiment import (ExperimentConfig, content_hash, derive_seed,
                                     frame_seeds, pool_metrics, selection_key, software_manifest)
from optical_comm.metrics import ber_interval, bits_to_symbols
from optical_comm.reservoir import ReservoirConfig, load_equalizer


METHODS = ("calibrated_edc", "linear", "esn", "calibrated_dbp")
POWER_FILES = {"seeds.json", "selection.json", "calibration.json", "generation.json",
               "linear.npz", "esn.npz", "metrics.json", "metrics.csv",
               "test_frames.json", "test_frames.csv", "constellations.npz"}


class AuditError(ValueError):
    """A saved-data or provenance inconsistency; never repair it implicitly."""


def require(condition: bool, message: str):
    if not condition:
        raise AuditError(message)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise AuditError(f"cannot read JSON {path}: {error}") from error


def same_number(actual, expected, label: str):
    if expected is None:
        require(actual is None, f"{label}: expected null")
    elif isinstance(expected, int):
        require(isinstance(actual, int) and actual == expected, f"{label}: count differs")
    else:
        require(isinstance(actual, (int, float)) and np.isfinite(actual)
                and np.isclose(actual, expected, rtol=1e-12, atol=1e-15), f"{label}: number differs")


def check_counted_metric(row: dict, expected_bits: int, label: str):
    errors, bits = row["bit_errors"], row["n_bits"]
    require(isinstance(errors, int) and isinstance(bits, int) and bits == expected_bits
            and 0 <= errors <= bits, f"{label}: invalid errors/counting denominator")
    same_number(row["ber"], errors / bits, f"{label} BER")
    low, high = ber_interval(errors, bits)
    same_number(row["ber_ci_low"], low, f"{label} BER low interval")
    same_number(row["ber_ci_high"], high, f"{label} BER high interval")
    require(np.isfinite([row["signal_energy"], row["error_energy"]]).all()
            and row["signal_energy"] > 0 and row["error_energy"] >= 0, f"{label}: invalid energy")
    # Reuse the stored-energy pooling convention; do not recount unseen symbols.
    recomputed = pool_metrics([row])
    same_number(row["evm"], recomputed["evm"], f"{label} EVM")
    same_number(row["effective_snr_db"], recomputed["effective_snr_db"], f"{label} SNR")


def check_csv(path: Path, rows: list[dict]):
    with path.open(newline="", encoding="utf-8") as stream:
        saved = list(csv.DictReader(stream))
    require(len(saved) == len(rows), f"{path.name}: CSV/JSON row count differs")
    for index, (csv_row, json_row) in enumerate(zip(saved, rows)):
        require(set(csv_row) == set(json_row), f"{path.name}: CSV/JSON columns differ")
        for key, expected in json_row.items():
            actual = csv_row[key]
            if expected is None:
                require(actual == "", f"{path.name} row {index} {key}: expected empty null")
            elif isinstance(expected, (float, int)):
                same_number(int(actual) if isinstance(expected, int) else float(actual), expected,
                            f"{path.name} row {index} {key}")
            else:
                require(actual == str(expected), f"{path.name} row {index} {key}: value differs")


def historical_measurements(run: Path) -> dict | None:
    """Retain recorded execution evidence; current clone mtimes are not clocks."""
    path = run / "verification.json"
    if not path.exists():
        return None
    previous = read_json(path)
    if previous.get("schema_version") == 2:
        return previous.get("historical_measurements")
    if "runner_elapsed_manifest_to_last_complete_seconds" not in previous:
        return None
    return {"origin": "Retained from the original saved verification; not recalculated by this audit",
            "prior_record_sha256": file_hash(path), "original_verified_utc": previous.get("verified_utc"),
            "runner_elapsed_manifest_to_last_complete_seconds": previous["runner_elapsed_manifest_to_last_complete_seconds"],
            "wall_time_scope": previous.get("wall_time_scope"),
            "resume_integrity_check": previous.get("resume_integrity_check")}


def audit_run(run: Path) -> dict:
    start = time.perf_counter()
    run = run.resolve()
    repository = Path(__file__).resolve().parents[1]
    imported_repository = Path(experiment.__file__).resolve().parents[2]
    require(repository == imported_repository,
            "audit CLI and imported scientific source are from different trees; install this tree or set PYTHONPATH")
    config = ExperimentConfig.from_dict(read_json(run / "config.json"))
    manifest = read_json(run / "manifest.json")
    compatibility = manifest["compatibility"]
    actual_source = software_manifest()
    require(actual_source["source_hash"] == compatibility["source_hash"], "scientific source hash differs from execution")
    require(actual_source["source_sha256"] == manifest["software"]["source_sha256"],
            "scientific source file hashes differ from the saved manifest")
    require(content_hash(config.to_dict()) == compatibility["config_hash"], "configuration hash differs from execution")
    require(manifest["software"]["source_hash"] == compatibility["source_hash"], "manifest source hash fields differ")
    require(manifest["mode"] == config.mode and compatibility["backend"] == config.backend,
            "manifest configuration/backend differs")
    n_scored = (config.n_symbols - 2 * config.link.guard_symbols - config.context.past_symbols
                - config.context.future_symbols - config.context.washout_symbols)
    expected_bits = n_scored * 8 * config.test_frames
    validation_bits = n_scored * 8 * config.validation_frames
    train_samples = n_scored * config.train_frames
    summary = read_json(run / "summary.json")
    require(summary["run_id"] == manifest["run_id"] and summary["mode"] == config.mode,
            "summary run identifier/mode differs")
    summary_rows = summary["metrics"]
    require(len(summary_rows) == len(config.launch_powers_dbm) * len(METHODS), "summary power/method coverage differs")
    pooled_rows, power_checks, runner_evidence, seeds_seen, errors_by_power = [], [], {}, set(), {}
    input_hashes = {name: file_hash(run / name) for name in ("config.json", "manifest.json", "summary.json", "summary.csv")}
    for power in config.launch_powers_dbm:
        dirname = "power_" + format(float(power), "+09.3f").replace("+", "p").replace("-", "m").replace(".", "p")
        directory = run / dirname
        completion = read_json(directory / "complete.json")
        require(completion["compatibility"] == compatibility and completion["launch_power_dbm"] == power,
                f"{dirname}: completion provenance differs")
        saved_hashes = completion["files_sha256"]
        require(POWER_FILES.issubset(saved_hashes), f"{dirname}: completion omits required artifact hashes")
        for name, expected in saved_hashes.items():
            require(Path(name).name == name, f"{dirname}: completion contains a nonlocal filename")
            digest = file_hash(directory / name)
            require(digest == expected, f"{dirname}/{name}: saved file hash differs")
            input_hashes[f"{dirname}/{name}"] = digest
        input_hashes[f"{dirname}/complete.json"] = file_hash(directory / "complete.json")
        seeds = read_json(directory / "seeds.json")
        require(seeds == frame_seeds(config, power), f"{dirname}: frame seed hierarchy differs")
        for frame in seeds:
            for key in ("bit_seed", "noise_seed"):
                require(frame[key] not in seeds_seen, f"{dirname}: a frame bit/ASE seed is reused")
                seeds_seen.add(frame[key])
        generation = read_json(directory / "generation.json")
        require([{key: row[key] for key in seeds[0]} for row in generation] == seeds,
                f"{dirname}: generation records do not cover the declared frame recipes")
        require(all(np.isfinite(row["channel_generation_seconds"]) and row["channel_generation_seconds"] >= 0
                    for row in generation), f"{dirname}: invalid recorded generation time")
        selection = read_json(directory / "selection.json")
        require(selection["selection_split"] == "validation" and selection["test_labels_used_for_selection"] is False,
                f"{dirname}: selection is not declared validation-only")
        reservoir_seed = derive_seed(config.root_seed, power, "model", 0, "reservoir")
        require(selection["reservoir_seed"] == reservoir_seed, f"{dirname}: reservoir seed differs")
        expected_candidates = {f"linear-ridge-{index:02d}": {"ridge": float(ridge)}
                               for index, ridge in enumerate(config.linear_ridges)}
        for group, candidate in enumerate(config.esn_candidates):
            for index, ridge in enumerate(config.esn_ridges):
                expected_candidates[f"esn-{group:02d}-ridge-{index:02d}"] = asdict(
                    ReservoirConfig(**candidate, seed=reservoir_seed, ridge=float(ridge)))
        candidates = selection["candidates"]
        require(len(candidates) == len(expected_candidates)
                and {row["candidate_id"] for row in candidates} == set(expected_candidates),
                f"{dirname}: candidate grid coverage differs")
        for row in candidates:
            identifier = row["candidate_id"]
            require(row["parameters"] == expected_candidates[identifier]
                    and row["method"] == ("linear" if identifier.startswith("linear-") else "esn"),
                    f"{dirname}: candidate parameters differ from configured grid")
            require(row["n_frames"] == config.validation_frames and row["training_samples"] == train_samples,
                    f"{dirname}: candidate frame/sample counts differ")
            check_counted_metric(row, validation_bits, f"{dirname}/{identifier}")
        calibration = read_json(directory / "calibration.json")
        for kind in ("edc", "dbp"):
            require(calibration[kind]["convention"] == "inverse_of_train_fitted_forward_gain"
                    and calibration[kind]["context"] == asdict(config.context),
                    f"{dirname}: calibration convention/context differs")
        for kind in ("linear", "esn"):
            chosen = min((row for row in candidates if row["method"] == kind), key=selection_key)
            require(selection["selected"][kind] == chosen["candidate_id"], f"{dirname}: validation tie rule differs")
            model = load_equalizer(directory / f"{kind}.npz")
            require(model.kind == kind and model.training_samples == train_samples
                    and model.preprocessor.context == config.context and model.ridge == chosen["parameters"]["ridge"],
                    f"{dirname}: checkpoint kind/settings/sample counts differ")
            if kind == "esn":
                require(asdict(model.config) == chosen["parameters"], f"{dirname}: ESN checkpoint parameters differ")
                require(model.input_weights.shape == (model.config.units, config.context.n_features)
                        and model.bias.shape == (model.config.units,), f"{dirname}: ESN input/bias shapes differ")
                pointer, indices, data = model.recurrent_indptr, model.recurrent_indices, model.recurrent_data
                require(pointer.shape == (model.config.units + 1,) and pointer[0] == 0 and pointer[-1] == len(data)
                        and len(indices) == len(data) and np.all(np.diff(pointer) >= 0)
                        and np.all((indices >= 0) & (indices < model.config.units)), f"{dirname}: invalid sparse recurrence")
                require(all(np.isfinite(array).all() for array in (model.input_weights, model.bias, data)),
                        f"{dirname}: nonfinite reservoir weights")
            edc = calibration["edc"]
            expected_gain = np.array(edc["applied_gain_real"]) + 1j * np.array(edc["applied_gain_imag"])
            require(np.array_equal(model.preprocessor.gain, expected_gain)
                    and np.array_equal(model.preprocessor.mean, edc["input_mean"])
                    and np.array_equal(model.preprocessor.scale, edc["input_scale"]),
                    f"{dirname}: checkpoint transforms differ from saved calibration")
            require(np.isfinite(model.readout).all() and model.readout.shape ==
                    (1 + config.context.n_features + (model.config.units if kind == "esn" else 0), 4),
                    f"{dirname}: invalid checkpoint readout")
            require(model.model_array_bytes == chosen["model_array_bytes"], f"{dirname}: checkpoint array size differs")
        frames = read_json(directory / "test_frames.json")
        require(len(frames) == config.test_frames * len(METHODS), f"{dirname}: test receiver/frame coverage differs")
        for index in range(config.test_frames):
            seed = next(row for row in seeds if row["split"] == "test" and row["frame_index"] == index)
            frame_rows = [row for row in frames if row["frame_index"] == index]
            require(len(frame_rows) == len(METHODS) and {row["method"] for row in frame_rows} == set(METHODS),
                    f"{dirname}: duplicated or missing frame receiver")
            for row in frame_rows:
                require(row["bit_seed"] == seed["bit_seed"] and row["noise_seed"] == seed["noise_seed"]
                        and row["launch_power_dbm"] == power and row["split"] == "test"
                        and row["n_symbols_xy"] == n_scored, f"{dirname}: test seeds/mask denominator differ")
                check_counted_metric(row, n_scored * 8, f"{dirname}/test-{index}/{row['method']}")
        metrics_record = read_json(directory / "metrics.json")
        require(metrics_record["mode"] == config.mode and metrics_record["launch_power_dbm"] == power,
                f"{dirname}: metrics provenance differs")
        metrics = metrics_record["metrics"]
        require(len(metrics) == len(METHODS) and {row["method"] for row in metrics} == set(METHODS),
                f"{dirname}: pooled method coverage differs")
        for row in metrics:
            require(row["launch_power_dbm"] == power, f"{dirname}: pooled power differs")
            check_counted_metric(row, expected_bits, f"{dirname}/pooled/{row['method']}")
            recomputed = pool_metrics(item for item in frames if item["method"] == row["method"])
            for key, expected in recomputed.items():
                same_number(row[key], expected, f"{dirname}/pooled/{row['method']}/{key}")
        check_csv(directory / "metrics.csv", metrics)
        check_csv(directory / "test_frames.csv", frames)
        pooled_rows.extend(metrics)
        first_seed = next(row for row in seeds if row["split"] == "test" and row["frame_index"] == 0)
        with np.load(directory / "constellations.npz", allow_pickle=False) as diagnostic:
            valid_indices = np.arange(config.link.guard_symbols + config.context.past_symbols + config.context.washout_symbols,
                                      config.n_symbols - config.link.guard_symbols - config.context.future_symbols)
            expected_indices = valid_indices[np.linspace(0, len(valid_indices) - 1,
                                              min(config.diagnostic_symbols, len(valid_indices)), dtype=int)]
            require(np.array_equal(diagnostic["indices"], expected_indices), f"{dirname}: diagnostic indices differ")
            require(int(diagnostic["bit_seed"]) == first_seed["bit_seed"]
                    and int(diagnostic["noise_seed"]) == first_seed["noise_seed"], f"{dirname}: diagnostic seeds differ")
            bits = np.random.default_rng(first_seed["bit_seed"]).integers(0, 2, (config.n_symbols, 2, 4), dtype=np.uint8)
            require(np.array_equal(diagnostic["tx_symbols"], bits_to_symbols(bits)[expected_indices]),
                    f"{dirname}: diagnostic transmitted-label recipe differs")
            for name in ("tx_symbols", "raw_edc_symbols", *METHODS):
                require(diagnostic[name].shape == (len(expected_indices), 2) and np.isfinite(diagnostic[name]).all(),
                        f"{dirname}: diagnostic {name} shape/values differ")
            require(np.array_equal(diagnostic["calibrated_edc"], diagnostic["raw_edc_symbols"] * expected_gain),
                    f"{dirname}: diagnostic EDC calibration differs")
        runner_evidence[str(power)] = {
            "selection_declares_validation_only": True,
            "checkpoint_reload_prediction_equal_reported_by_runner": selection["checkpoint_reload_prediction_equal"],
            "prediction_equality_reexecuted_by_this_audit": False,
            "selection_before_test_generation_reexecuted_by_this_audit": False}
        require(selection["checkpoint_reload_prediction_equal"] is True, f"{dirname}: runner reload evidence is absent")
        errors_by_power[str(power)] = {row["method"]: row["bit_errors"] for row in metrics}
        power_checks.append({"power_dbm": power, "pooled_receiver_rows": len(metrics),
            "independent_test_frames": config.test_frames, "counted_bits_per_receiver": expected_bits,
            "candidate_rows": len(candidates), "selected_candidates": selection["selected"],
            "all_complete_hashes_verified": True, "checkpoint_fields_and_arrays_loaded": True,
            "same_frame_denominators_verified": True, "validation_tie_rule_verified": True,
            "diagnostic_indices_tx_recipe_and_scalar_calibration_verified": True})
    require(summary_rows == pooled_rows, "summary differs from per-power metrics or order")
    check_csv(run / "summary.csv", [dict(run_id=manifest["run_id"], mode=config.mode, **row) for row in pooled_rows])
    # Detect a concurrent edit without deriving chronology or execution runtime from mtime.
    for name, expected in input_hashes.items():
        require(file_hash(run / name) == expected, f"input changed during audit: {name}")
    semantic = {"run_id": manifest["run_id"], "mode": config.mode,
        "execution_git_commit_recorded": manifest["software"]["git_commit"],
        "source_hash": actual_source["source_hash"], "config_hash": compatibility["config_hash"],
        "pooled_rows": len(pooled_rows), "per_frame_receiver_rows": len(config.launch_powers_dbm) * config.test_frames * 4,
        "distinct_frame_bit_and_ase_seeds": len(seeds_seen), "expected_test_bits_per_receiver_power": expected_bits,
        "all_source_config_and_complete_hashes_verified": True, "per_power": power_checks,
        "errors_by_power_method": errors_by_power, "input_files_sha256": input_hashes,
        "additional_channel_or_ase_frames_generated": 0, "models_or_hyperparameters_refit": False,
        "transmitted_diagnostic_labels_reconstructed_from_existing_seeds": True,
        "predictions_regenerated": False}
    return {"schema_version": 2, "evaluation_class": "normal saved-artifact consistency audit; no independent validator evaluation",
        "audit_code": {"path": "scripts/audit_run.py", "sha256": file_hash(Path(__file__).resolve())},
        "semantic_checks": semantic, "semantic_checks_sha256": content_hash(semantic),
        "recorded_runner_evidence_observed": runner_evidence,
        "historical_measurements": historical_measurements(run),
        "observations": {"verified_utc": datetime.now(timezone.utc).isoformat(),
            "audit_elapsed_seconds": time.perf_counter() - start,
            "run_path": str(run), "imported_scientific_source_root": str(imported_repository),
            "audit_packages": actual_source["packages"]},
        "limitations": ["Saved counts and energy consistency are checked; bit decisions and channel outputs are not recomputed.",
            "Runner reload/selection chronology is recorded evidence, not newly executed prediction or timing evidence.",
            "Input hashes establish consistency with the saved manifest, not an independent signature of its origin.",
            "Current file mtimes are never used to reconstruct historical experiment duration."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="completed pilot/final run directory")
    parser.add_argument("--output", type=Path, help="JSON output; stdout if omitted; never overwrite science inputs")
    args = parser.parse_args()
    try:
        run = args.run.resolve()
        if args.output:
            output = args.output.resolve()
            require(not output.is_relative_to(run) or output == run / "verification.json",
                    "output inside a run may only be verification.json; science inputs are read-only")
            repository = Path(__file__).resolve().parents[1]
            require(not output.is_relative_to(repository / "src")
                    and not output.is_relative_to(repository / "configs")
                    and output not in {Path(__file__).resolve(), repository / "scripts/run_experiment.py"},
                    "output may not overwrite scientific source/configuration")
        record = audit_run(run)
        def command_path(path):
            try:
                return str(Path(path).relative_to(Path.cwd()))
            except ValueError:
                return str(path)
        command = [command_path(sys.executable), command_path(Path(__file__).resolve()), "--run", str(args.run)]
        if args.output:
            command += ["--output", str(args.output)]
        record["command"] = shlex.join(command)
        rendered = json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_name(args.output.name + ".tmp")
            temporary.write_text(rendered, encoding="utf-8")
            temporary.replace(args.output)
            print(f"Audited {record['semantic_checks']['pooled_rows']} pooled rows; saved {args.output}")
        else:
            print(rendered, end="")
    except (AuditError, OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile) as error:
        print(f"Audit failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
