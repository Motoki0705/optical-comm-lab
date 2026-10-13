#!/usr/bin/env python3
"""Export contiguous 2-sps diagnostics by reconstructing existing saved test frames.

The entire original frame is propagated before slicing. Saved models supply only
their fixed masks/context; fitting, prediction, selection and BER scoring are not
performed. Existing scientific inputs stay read-only.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import sys
import tempfile
import time
from zipfile import BadZipFile

import numpy as np

from audit_run import AuditError, audit_run, file_hash, read_json, require
from optical_comm.channel import simulate_frame
from optical_comm.experiment import ExperimentConfig, software_manifest
from optical_comm.reservoir import SymbolFrame, load_equalizer


STAGE = "EDC + RRC matched filter + nominal launch/RRC alphabet normalization; uncalibrated"
RTOL = 2e-10
ATOL = 2e-11


def _path_for_command(path: Path) -> str:
    # Preserve the venv interpreter symlink: resolving it would record the
    # system Python, which does not have this environment's dependencies.
    path = path.absolute()
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _existing_run_hashes(run: Path, output: Path) -> dict[str, str]:
    return {str(path.relative_to(run)): file_hash(path) for path in sorted(run.rglob("*"))
            if path.is_file() and not path.is_relative_to(output)}


def _validate_output(run: Path, output: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    if output.is_relative_to(run):
        require(output.is_relative_to(run / "waveform_diagnostics"),
                "output inside a run must be in waveform_diagnostics; scientific inputs are read-only")
    else:
        require(not run.is_relative_to(output), "output cannot contain the input run")
    for directory in ("src", "scripts", "tests", "configs"):
        require(not output.is_relative_to(repository / directory), "output cannot overwrite source or configuration")
    require(not output.exists() or output.is_dir(), "output must be a directory")
    require(not output.exists() or not any(output.iterdir()), "output directory must be empty; use a fresh directory")


def _integer(value: int, name: str, minimum: int) -> None:
    require(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
            and value >= minimum, f"{name} must be an integer >= {minimum}")


def _match_constellation(directory: Path, frame, frame_index: int) -> dict:
    with np.load(directory / "constellations.npz", allow_pickle=False) as saved:
        same_frame = int(saved["bit_seed"]) == frame.bit_seed and int(saved["noise_seed"]) == frame.noise_seed
        if not same_frame:
            require(frame_index != 0, "saved constellation seeds differ from the first test-frame recipe")
            return {"available_for_selected_frame": False,
                    "reason": "original constellations store the first test frame only"}
        indices = saved["indices"]
        require(np.array_equal(saved["tx_symbols"], frame.tx_symbols[indices]),
                "regenerated TX symbols differ from saved constellation labels")
        difference = np.abs(frame.rx_symbols[indices] - saved["raw_edc_symbols"])
        require(np.allclose(frame.rx_symbols[indices], saved["raw_edc_symbols"], rtol=RTOL, atol=ATOL),
                "regenerated raw EDC symbols differ from saved constellation points")
        return {"available_for_selected_frame": True, "all_stored_indices_checked": len(indices),
                "tx_symbols_bitwise_equal": True,
                "rx_symbols_bitwise_equal": bool(np.array_equal(frame.rx_symbols[indices], saved["raw_edc_symbols"])),
                "rx_max_absolute_difference": float(np.max(difference)), "rtol": RTOL, "atol": ATOL}


def export_diagnostics(
    run: Path,
    output_dir: Path | None = None,
    *,
    frame_index: int = 0,
    symbols: int = 512,
    start_symbol: int | None = None,
    backend: str | None = None,
) -> dict:
    """Reconstruct one existing test frame per completed configured power."""
    start_time = time.perf_counter()
    run = Path(run).resolve()
    output = (run / "waveform_diagnostics") if output_dir is None else Path(output_dir).resolve()
    _validate_output(run, output)
    config = ExperimentConfig.from_dict(read_json(run / "config.json"))
    _integer(frame_index, "frame_index", 0)
    _integer(symbols, "symbols", 1)
    require(frame_index < config.test_frames, "frame_index is outside the original test split")
    start_symbol = max(0, config.link.guard_symbols - 16) if start_symbol is None else start_symbol
    _integer(start_symbol, "start_symbol", 0)
    require(start_symbol + symbols <= config.n_symbols, "requested slice extends beyond the original frame")
    backend = config.backend if backend is None else backend
    require(backend in {"numpy", "cupy"}, "backend must be numpy or cupy")

    # Reuse the permanent saved-data audit instead of duplicating its contracts.
    audit = audit_run(run)
    original_manifest = read_json(run / "manifest.json")
    input_hashes = _existing_run_hashes(run, output)
    software = software_manifest()
    exporter_path = Path(__file__).resolve()
    exporter_hash = file_hash(exporter_path)
    audit_path = exporter_path.with_name("audit_run.py")
    audit_hash = file_hash(audit_path)
    completion_paths = [Path(name) for name in audit["semantic_checks"]["input_files_sha256"]
                        if Path(name).name == "complete.json"]
    power_directories = {float(read_json(run / name)["launch_power_dbm"]): run / name.parent
                         for name in completion_paths}
    entries = []
    output.mkdir(parents=True, exist_ok=True)
    # A failed reconstruction leaves no successful manifest or partial NPZ files.
    with tempfile.TemporaryDirectory(prefix=".export-", dir=output) as temporary:
        staging = Path(temporary)
        for power in config.launch_powers_dbm:
            directory = power_directories[float(power)]
            record = next(row for row in read_json(directory / "seeds.json")
                          if row["split"] == "test" and row["frame_index"] == frame_index)
            frame_id = f"{original_manifest['run_id']}/{directory.name}/test-{frame_index}"
            print(f"{power:+g} dBm: reconstructing original test frame {frame_index} in full", flush=True)
            frame_start = time.perf_counter()
            frame = simulate_frame(config.link, power, config.n_symbols, record["bit_seed"],
                                   record["noise_seed"], backend=backend)
            generation_seconds = time.perf_counter() - frame_start
            require(frame.rx_2sps.shape == (2 * config.n_symbols, 2), "unexpected full-frame 2-sps shape")
            require(np.array_equal(frame.rx_2sps[::2], frame.rx_symbols), "2-sps even samples are not symbol aligned")
            correspondence = _match_constellation(directory, frame, frame_index)
            # No labels are passed to the mask preparation and no model prediction
            # or fitting method is called. Both fixed checkpoints must agree.
            symbolic = SymbolFrame(frame.rx_symbols, valid_mask=frame.valid_mask, frame_id=frame_id)
            models = [load_equalizer(directory / f"{kind}.npz") for kind in ("linear", "esn")]
            require(all(asdict(model.preprocessor.context) == asdict(config.context) for model in models),
                    "checkpoint context differs from the run")
            score_masks = [model.preprocessor.common_mask(symbolic) for model in models]
            require(np.array_equal(*score_masks), "saved receiver masks disagree")
            score_mask = score_masks[0]
            original_rows = [row for row in read_json(directory / "test_frames.json")
                             if row["frame_index"] == frame_index]
            require(all(row["n_bits"] == 8 * np.count_nonzero(score_mask) for row in original_rows),
                    "reconstructed score mask differs from the original counting denominator")
            stop = start_symbol + symbols
            symbol_indices = np.arange(start_symbol, stop, dtype=np.int64)
            sample_indices = np.arange(2 * start_symbol, 2 * stop, dtype=np.int64)
            arrays = {
                "rx_2sps": frame.rx_2sps[2 * start_symbol:2 * stop].copy(),
                "rx_symbols": frame.rx_symbols[start_symbol:stop].copy(),
                "tx_symbols": frame.tx_symbols[start_symbol:stop].copy(),
                "tx_bits": frame.tx_bits[start_symbol:stop].copy(),
                "symbol_indices": symbol_indices, "sample_indices_2sps": sample_indices,
                "frame_valid_mask": frame.valid_mask[start_symbol:stop].copy(),
                "score_mask": score_mask[start_symbol:stop].copy(),
                "frame_id": np.asarray(frame_id),
                "bit_seed": np.asarray(record["bit_seed"], dtype=np.uint64),
                "noise_seed": np.asarray(record["noise_seed"], dtype=np.uint64),
                "launch_power_dbm": np.asarray(power, dtype=np.float64),
                "frame_n_symbols": np.asarray(config.n_symbols, dtype=np.int64),
            }
            require(arrays["rx_2sps"].dtype == np.complex128, "2-sps samples must be complex128")
            require(np.array_equal(arrays["rx_2sps"][::2], arrays["rx_symbols"]), "sliced sample alignment differs")
            filename = f"{directory.name}_test_frame_{frame_index:04d}.npz"
            np.savez_compressed(staging / filename, **arrays)
            with np.load(staging / filename, allow_pickle=False) as saved:
                require(set(saved.files) == set(arrays), "saved array schema differs")
                require(all(np.array_equal(saved[key], value) for key, value in arrays.items()),
                        "saved diagnostic changed on reload")
            entries.append({"file": filename, "sha256": file_hash(staging / filename), "frame_id": frame_id,
                "launch_power_dbm": power, "split": "test", "original_frame_index": frame_index,
                "bit_seed": record["bit_seed"], "noise_seed": record["noise_seed"],
                "start_symbol": start_symbol, "stop_symbol_exclusive": stop, "symbols": symbols,
                "sample_2sps_start": 2 * start_symbol, "sample_2sps_stop_exclusive": 2 * stop,
                "frame_valid_symbols_in_slice": int(np.count_nonzero(arrays["frame_valid_mask"])),
                "originally_scored_symbols_in_slice": int(np.count_nonzero(arrays["score_mask"])),
                "original_frame_scored_bits_per_receiver": original_rows[0]["n_bits"],
                "alignment_even_2sps_matches_1sps_bitwise": True, "npz_reload_equal": True,
                "saved_constellation_cross_check": correspondence,
                "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in arrays.items()},
                "reconstruction_seconds": generation_seconds})
            del frame, symbolic, score_masks, models

        require(_existing_run_hashes(run, output) == input_hashes, "existing run inputs changed during export")
        require(software_manifest()["source_hash"] == software["source_hash"], "scientific source changed during export")
        require(file_hash(exporter_path) == exporter_hash and file_hash(audit_path) == audit_hash,
                "export/audit CLI changed during export")
        common_command = [_path_for_command(Path(sys.executable)), _path_for_command(exporter_path),
                          "--run", _path_for_command(run)]
        options = ["--frame-index", str(frame_index), "--symbols", str(symbols),
                   "--start-symbol", str(start_symbol), "--backend", backend]
        actual_command = common_command + ["--output-dir", _path_for_command(output)] + options
        reproduce_command = common_command + ["--output-dir", ".work/waveform-diagnostics-reproduction"] + options
        manifest = {
            "schema_version": 1,
            "evidence_scope": "diagnostic reconstruction of existing test frames; not new independent evaluations",
            "run_id": original_manifest["run_id"], "run": _path_for_command(run),
            "command": shlex.join(actual_command), "regeneration_command": shlex.join(reproduce_command),
            "regeneration_output_note": "Choose an empty output directory; existing outputs are never overwritten.",
            "exporter_code": {"path": _path_for_command(exporter_path), "sha256": exporter_hash},
            "saved_data_audit_code": {"path": _path_for_command(audit_path), "sha256": audit_hash},
            "input_files_sha256": input_hashes, "scientific_source_sha256": software["source_sha256"],
            "original_source_hash": original_manifest["compatibility"]["source_hash"],
            "saved_data_audit_semantic_sha256": audit["semantic_checks_sha256"],
            "original_backend": config.backend, "reconstruction_backend": backend,
            "backend_changed_from_original": backend != config.backend,
            "original_packages": original_manifest["software"]["packages"], "reconstruction_software": software,
            "waveform_stage": STAGE, "train_fitted_gain_or_feature_normalization_applied": False,
            "polarizations": ["X", "Y"], "rx_units": "normalized complex TX alphabet; ensemble mean |TX|^2=1 per polarization",
            "symbol_rate_hz": config.link.symbol_rate_hz, "diagnostic_sample_rate_hz": 2 * config.link.symbol_rate_hz,
            "propagation_sample_rate_hz": config.link.sample_rate_hz,
            "ase_bandwidth_hz": config.link.ase_bandwidth_hz,
            "frame_n_symbols_per_polarization": config.n_symbols,
            "frame_n_samples_2sps": 2 * config.n_symbols,
            "frame_boundaries_symbol_indices": [0, config.n_symbols], "periodic_frame": True,
            "time_origin": "symbol index 0 is the known matched-filter symbol instant; t=sample_indices_2sps/(2*Rs)",
            "slice_is_contiguous_nonwrapping": True,
            "mask_definitions": {
                "frame_valid_mask": "original channel guard mask at symbol positions; excludes guard_symbols at both ends",
                "score_mask": "original common receiver mask from saved fixed context and segment washout; no labels used",
                "guard_symbols_per_side": config.link.guard_symbols, "context": asdict(config.context),
                "half_symbol_samples": "masks are symbol-indexed; the second sample is halfway to the following symbol"},
            "existing_input_hashes_unchanged": True, "scientific_source_hash_unchanged": True,
            "diagnostic_frames_reconstructed": len(entries), "new_independent_test_frames": 0,
            "evaluation_bits_added": 0, "fit_selection_prediction_or_training_executed": False,
            "verification_class": "ordinary post-review verification; no independent re-evaluation",
            "files": entries,
            "observations": {"created_utc": datetime.now(timezone.utc).isoformat(),
                             "export_elapsed_seconds": time.perf_counter() - start_time},
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
        for path in staging.iterdir():
            path.replace(output / path.name)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="completed pilot/final run; read-only")
    parser.add_argument("--output-dir", type=Path, help="empty output directory; defaults to RUN/waveform_diagnostics")
    parser.add_argument("--frame-index", type=int, default=0, help="existing test frame index; default 0")
    parser.add_argument("--symbols", type=int, default=512, help="contiguous symbol count; default 512")
    parser.add_argument("--start-symbol", type=int, help="first symbol index; default max(0, guard-16)")
    parser.add_argument("--backend", choices=("numpy", "cupy"), help="explicit reconstruction backend; default original")
    args = parser.parse_args()
    try:
        manifest = export_diagnostics(args.run, args.output_dir, frame_index=args.frame_index, symbols=args.symbols,
                                      start_symbol=args.start_symbol, backend=args.backend)
        output = args.output_dir if args.output_dir is not None else args.run / "waveform_diagnostics"
        print(f"Saved {len(manifest['files'])} existing-frame 2-sps diagnostics to {output}")
        return 0
    except (AuditError, OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile, RuntimeError) as error:
        print(f"Waveform export failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
