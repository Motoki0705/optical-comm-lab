"""Leakage-safe launch-power studies and reproducible compact artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import csv
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import time
from typing import Callable, Iterable

import numpy as np

from .channel import dbp_symbols, simulate_frame
from .config import LinkConfig
from .metrics import ber_interval, compute_metrics
from .reservoir import (ContextConfig, ESNEqualizer, LinearEqualizer, ReservoirConfig,
                        SymbolFrame, TrainPreprocessor, load_equalizer)


@dataclass(frozen=True)
class ExperimentConfig:
    mode: str = "pilot"
    root_seed: int = 20261012
    link: LinkConfig = field(default_factory=LinkConfig)
    launch_powers_dbm: tuple[float, ...] = (-6, -4, -2, 0, 2, 4, 6)
    n_symbols: int = 8192
    train_frames: int = 2
    validation_frames: int = 1
    test_frames: int = 3
    context: ContextConfig = field(default_factory=ContextConfig)
    linear_ridges: tuple[float, ...] = (1e-6, 1e-4, 1e-2, 1.0)
    esn_ridges: tuple[float, ...] = (1e-6, 1e-4, 1e-2, 1.0)
    esn_candidates: tuple[dict, ...] = field(default_factory=lambda: ({"units": 128},))
    dbp_step_km: float = 4.0
    backend: str = "numpy"
    diagnostic_symbols: int = 512

    def __post_init__(self):
        if self.mode not in {"pilot", "final"}:
            raise ValueError("mode must be pilot or final")
        if not isinstance(self.root_seed, int) or self.root_seed < 0:
            raise ValueError("root_seed must be a nonnegative integer")
        for key in ("n_symbols", "train_frames", "validation_frames", "test_frames", "diagnostic_symbols"):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if not self.launch_powers_dbm or len(set(self.launch_powers_dbm)) != len(self.launch_powers_dbm):
            raise ValueError("launch powers must be nonempty and distinct")
        if not np.isfinite(self.launch_powers_dbm).all():
            raise ValueError("launch powers must be finite")
        for ridges in (self.linear_ridges, self.esn_ridges):
            if not ridges or any(not np.isfinite(ridge) or ridge <= 0 for ridge in ridges):
                raise ValueError("ridge paths must be nonempty, finite, and positive")
        if not self.esn_candidates:
            raise ValueError("at least one reservoir candidate is required")
        for candidate in self.esn_candidates:
            if "seed" in candidate or "ridge" in candidate:
                raise ValueError("candidate seed and ridge come from the seed hierarchy and ridge path")
            ReservoirConfig(**candidate)
        if self.dbp_step_km <= 0 or not np.isfinite(self.dbp_step_km):
            raise ValueError("DBP step must be finite and positive")
        if self.backend not in {"numpy", "cupy"}:
            raise ValueError("backend must be numpy or cupy")
        remaining = (self.n_symbols - 2 * self.link.guard_symbols - self.context.past_symbols
                     - self.context.future_symbols - self.context.washout_symbols)
        if remaining <= 0:
            raise ValueError("no scored symbols remain after guards/context/washout")
        if self.mode == "final" and remaining * 8 * self.test_frames < 1_000_000:
            raise ValueError("final configuration must count at least one million bits per power")

    @classmethod
    def from_dict(cls, values: dict) -> ExperimentConfig:
        values = dict(values)
        schema = values.pop("schema_version", 1)
        if schema != 1:
            raise ValueError("unsupported experiment configuration schema")
        if "link" in values:
            values["link"] = LinkConfig.from_dict(values["link"])
        if "context" in values:
            values["context"] = ContextConfig(**values["context"])
        for name in ("launch_powers_dbm", "linear_ridges", "esn_ridges", "esn_candidates"):
            if name in values:
                values[name] = tuple(values[name])
        return cls(**values)

    def to_dict(self) -> dict:
        result = asdict(self)
        result["schema_version"] = 1
        result["link"] = self.link.to_dict()
        return result


def _json_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def content_hash(value) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def derive_seed(root_seed: int, power_dbm: float, split: str, frame_index: int, stream: str) -> int:
    """Stateless namespaces: changing one split's size never moves another seed."""
    if split not in {"train", "validation", "test", "model"} or stream not in {"bits", "ase", "reservoir"}:
        raise ValueError("unknown seed namespace")
    if root_seed < 0 or frame_index < 0:
        raise ValueError("seed root and frame index must be nonnegative")
    key = [root_seed, format(float(power_dbm), ".12g"), split, frame_index, stream]
    return int.from_bytes(hashlib.sha256(_json_bytes(key)).digest()[:8], "little")


def frame_seeds(config: ExperimentConfig, power_dbm: float) -> list[dict]:
    records = []
    for split, count in (("train", config.train_frames), ("validation", config.validation_frames),
                         ("test", config.test_frames)):
        for index in range(count):
            records.append({"split": split, "frame_index": index,
                            "bit_seed": derive_seed(config.root_seed, power_dbm, split, index, "bits"),
                            "noise_seed": derive_seed(config.root_seed, power_dbm, split, index, "ase")})
    seeds = [record[key] for record in records for key in ("bit_seed", "noise_seed")]
    if len(set(seeds)) != len(seeds):
        raise ValueError("seed hash collision: choose a different root seed")
    return records


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def software_manifest() -> dict:
    versions = {}
    for package in ("numpy", "scipy", "numba", "cupy-cuda12x", "cupy-cuda13x", "optical-comm-lab"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    repository = Path(__file__).resolve().parents[2]
    sources = sorted(Path(__file__).resolve().parent.glob("*.py"))
    script = repository / "scripts" / "run_experiment.py"
    if script.exists():
        sources.append(script)
    hashes = {str(path.relative_to(repository)): _file_hash(path) for path in sources}
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository,
                                check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": versions,
            "git_commit": commit, "source_sha256": hashes, "source_hash": content_hash(hashes)}


def _symbol_frame(frame, rx: np.ndarray | None = None, frame_id: str = "") -> SymbolFrame:
    return SymbolFrame(frame.rx_symbols if rx is None else rx, frame.tx_symbols, frame.valid_mask, frame_id)


def score_symbols(frame, prediction: np.ndarray, mask: np.ndarray) -> dict:
    """Metrics-only label use; fixed-mask denominator and exact energy pooling."""
    mask = np.asarray(mask)
    if mask.dtype != np.bool_ or mask.shape != (len(frame.tx_symbols),):
        raise ValueError("score mask must be a boolean vector matching the frame")
    result = compute_metrics(frame.tx_symbols[mask], prediction[mask], tx_bits=frame.tx_bits[mask])
    if not np.isfinite(result["effective_snr_db"]):
        result["effective_snr_db"] = None
    difference = prediction[mask] - frame.tx_symbols[mask]
    result["error_energy"] = float(np.sum(np.abs(difference)**2))
    result["signal_energy"] = float(np.sum(np.abs(frame.tx_symbols[mask])**2))
    result["n_symbols_xy"] = int(np.count_nonzero(mask))
    return result


def pool_metrics(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    errors = sum(int(row["bit_errors"]) for row in rows)
    bits = sum(int(row["n_bits"]) for row in rows)
    if bits == 0:
        raise ValueError("no counted test bits")
    error_energy = sum(float(row["error_energy"]) for row in rows)
    signal_energy = sum(float(row["signal_energy"]) for row in rows)
    evm = float(np.sqrt(error_energy / signal_energy))
    low, high = ber_interval(errors, bits)
    return {"bit_errors": errors, "n_bits": bits, "ber": errors / bits,
            "ber_ci_low": float(low), "ber_ci_high": float(high), "evm": evm,
            "effective_snr_db": float(-20 * np.log10(evm)) if evm > 0 else None,
            "n_frames": len(rows), "frame_ber_min": min(row["ber"] for row in rows),
            "frame_ber_max": max(row["ber"] for row in rows),
            "frame_ber_std": float(np.std([row["ber"] for row in rows], ddof=1)) if len(rows) > 1 else None,
            "error_energy": error_energy, "signal_energy": signal_energy}


def selection_key(row: dict) -> tuple:
    """Frozen rule: pooled validation BER, EVM, stored array size, candidate ID."""
    return row["ber"], row["evm"], row["model_array_bytes"], row["candidate_id"]


def _evaluate_path(models, frames, raw_frames, group_id: str, training_seconds: float) -> list[dict]:
    by_model = [[] for _ in models]
    for frame, raw in zip(frames, raw_frames):
        prepared, design = models[0].design(frame)
        for index, model in enumerate(models):
            prediction = model.predict_prepared(prepared, design)
            by_model[index].append(score_symbols(raw, prediction.symbols, prediction.valid_mask))
    rows = []
    for index, (model, metrics) in enumerate(zip(models, by_model)):
        config = asdict(model.config) if isinstance(model, ESNEqualizer) else {"ridge": model.ridge}
        rows.append({"candidate_id": f"{group_id}-ridge-{index:02d}", "method": model.kind,
                     "parameters": config, **pool_metrics(metrics), "model_array_bytes": model.model_array_bytes,
                     "training_samples": model.training_samples, "shared_group_training_seconds": training_seconds})
    return rows


def _power_directory(power_dbm: float) -> str:
    return "power_" + format(float(power_dbm), "+09.3f").replace("+", "p").replace("-", "m").replace(".", "p")


def _calibration_dict(preprocessor: TrainPreprocessor) -> dict:
    return {"convention": "inverse_of_train_fitted_forward_gain",
            "applied_gain_real": preprocessor.gain.real.tolist(), "applied_gain_imag": preprocessor.gain.imag.tolist(),
            "input_mean": preprocessor.mean.tolist(), "input_scale": preprocessor.scale.tolist(),
            "context": asdict(preprocessor.context)}


def _generate(config: ExperimentConfig, power: float, record: dict):
    start = time.perf_counter()
    frame = simulate_frame(config.link, power, config.n_symbols, record["bit_seed"],
                           record["noise_seed"], backend=config.backend)
    return frame, time.perf_counter() - start


def _run_power(config: ExperimentConfig, power: float, output: Path, compatibility: dict,
               progress: Callable[[str], None] | None = None) -> dict:
    emit = progress if progress is not None else lambda message: None
    emit(f"{power:+g} dBm: generating independent train/validation frames")
    output.mkdir(parents=True, exist_ok=True)
    seeds = frame_seeds(config, power)
    _write_json(output / "seeds.json", seeds)
    generated = {}
    generation_rows = []
    for record in seeds:
        if record["split"] == "test":
            continue
        frame, duration = _generate(config, power, record)
        generated.setdefault(record["split"], []).append(frame)
        generation_rows.append(dict(record, channel_generation_seconds=duration))
    raw_train, raw_validation = generated["train"], generated["validation"]
    train = [_symbol_frame(frame, frame_id=f"train-{index}") for index, frame in enumerate(raw_train)]
    validation = [_symbol_frame(frame, frame_id=f"validation-{index}") for index, frame in enumerate(raw_validation)]
    preprocessor = TrainPreprocessor(config.context).fit(train)
    model_lookup, candidates = {}, []
    tuning_start = time.perf_counter()
    start = time.perf_counter()
    linear_models = LinearEqualizer(preprocessor).fit_ridge_path(train, config.linear_ridges)
    linear_rows = _evaluate_path(linear_models, validation, raw_validation, "linear", time.perf_counter() - start)
    candidates.extend(linear_rows)
    model_lookup.update(zip((row["candidate_id"] for row in linear_rows), linear_models))
    # All hyperparameter candidates use the same per-power random graph/input
    # draw; scales/radii change it deterministically. Test seeds are unrelated.
    reservoir_seed = derive_seed(config.root_seed, power, "model", 0, "reservoir")
    for group_index, candidate in enumerate(config.esn_candidates):
        emit(f"{power:+g} dBm: training reservoir group {group_index + 1}/{len(config.esn_candidates)}")
        start = time.perf_counter()
        reservoir = ESNEqualizer(preprocessor, ReservoirConfig(**candidate, seed=reservoir_seed))
        models = reservoir.fit_ridge_path(train, config.esn_ridges)
        rows = _evaluate_path(models, validation, raw_validation, f"esn-{group_index:02d}", time.perf_counter() - start)
        candidates.extend(rows)
        model_lookup.update(zip((row["candidate_id"] for row in rows), models))
    selected_rows = {kind: min((row for row in candidates if row["method"] == kind), key=selection_key)
                     for kind in ("linear", "esn")}
    selected = {kind: model_lookup[row["candidate_id"]] for kind, row in selected_rows.items()}
    tuning_seconds = time.perf_counter() - tuning_start
    for kind, model in selected.items():
        checkpoint = output / f"{kind}.npz"
        model.save(checkpoint)
        restored = load_equalizer(checkpoint)
        original_prediction, restored_prediction = model.predict(validation[0]), restored.predict(validation[0])
        if not (np.array_equal(original_prediction.symbols, restored_prediction.symbols)
                and np.array_equal(original_prediction.valid_mask, restored_prediction.valid_mask)):
            raise RuntimeError("saved model changed prediction on reload")
        selected[kind] = restored
    # DBP gets its own scalar calibration, estimated on the same training frames.
    start = time.perf_counter()
    dbp_train = [_symbol_frame(frame, dbp_symbols(frame, step_km=config.dbp_step_km, backend=config.backend))
                 for frame in raw_train]
    dbp_preprocessor = TrainPreprocessor(config.context).fit(dbp_train)
    dbp_calibration_seconds = time.perf_counter() - start
    selection = {"selection_split": "validation", "selection_rule": ["lowest pooled direct-count BER", "lowest EVM",
                 "smallest stored-array byte count", "lexicographic candidate ID"],
                 "test_labels_used_for_selection": False, "reservoir_seed": reservoir_seed,
                 "selected": {kind: row["candidate_id"] for kind, row in selected_rows.items()},
                 "candidates": candidates, "total_model_selection_seconds": tuning_seconds,
                 "dbp_train_calibration_seconds": dbp_calibration_seconds,
                 "checkpoint_reload_prediction_equal": True}
    # Lock this record before generating any held-out frame.
    _write_json(output / "selection.json", selection)
    _write_json(output / "calibration.json", {"edc": _calibration_dict(preprocessor),
                                              "dbp": _calibration_dict(dbp_preprocessor)})
    emit(f"{power:+g} dBm: validation selected {selection['selected']}; held-out settings frozen")
    calibration_sha256 = _file_hash(output / "calibration.json")
    selection_sha256 = _file_hash(output / "selection.json")
    test_rows, diagnostics = [], {}
    for record in (record for record in seeds if record["split"] == "test"):
        emit(f"{power:+g} dBm: held-out frame {record['frame_index'] + 1}/{config.test_frames}")
        frame, duration = _generate(config, power, record)
        generation_rows.append(dict(record, channel_generation_seconds=duration))
        symbol_frame = _symbol_frame(frame)
        common_mask = preprocessor.common_mask(symbol_frame)
        predictions, seconds = {}, {}
        start = time.perf_counter()
        predictions["calibrated_edc"] = preprocessor.calibrate(frame.rx_symbols)
        seconds["calibrated_edc"] = time.perf_counter() - start
        for kind, model in selected.items():
            start = time.perf_counter()
            prediction = model.predict(symbol_frame)
            seconds[kind] = time.perf_counter() - start
            if not np.array_equal(prediction.valid_mask, common_mask):
                raise RuntimeError("receiver comparison uses different masks")
            predictions[kind] = prediction.symbols
        start = time.perf_counter()
        predictions["calibrated_dbp"] = dbp_preprocessor.calibrate(
            dbp_symbols(frame, step_km=config.dbp_step_km, backend=config.backend))
        seconds["calibrated_dbp"] = time.perf_counter() - start
        for method, predicted in predictions.items():
            test_rows.append({"launch_power_dbm": power, "split": "test", "frame_index": record["frame_index"],
                              "bit_seed": record["bit_seed"], "noise_seed": record["noise_seed"], "method": method,
                              **score_symbols(frame, predicted, common_mask), "inference_seconds": seconds[method]})
        # Fixed first-frame, evenly spaced, common indices; never pick nice-looking outputs.
        if record["frame_index"] == 0:
            valid_indices = np.flatnonzero(common_mask)
            indices = valid_indices[np.linspace(0, len(valid_indices) - 1,
                                                min(config.diagnostic_symbols, len(valid_indices)), dtype=int)]
            diagnostics = {"indices": indices, "tx_symbols": frame.tx_symbols[indices],
                           "raw_edc_symbols": frame.rx_symbols[indices],
                           **{method: predicted[indices] for method, predicted in predictions.items()},
                           "bit_seed": np.array(record["bit_seed"], dtype=np.uint64),
                           "noise_seed": np.array(record["noise_seed"], dtype=np.uint64)}
        del frame, symbol_frame
    if (_file_hash(output / "selection.json") != selection_sha256
            or _file_hash(output / "calibration.json") != calibration_sha256):
        raise RuntimeError("selection/calibration changed while testing")
    pooled = []
    for method in ("calibrated_edc", "linear", "esn", "calibrated_dbp"):
        rows = [row for row in test_rows if row["method"] == method]
        pooled.append({"launch_power_dbm": power, "method": method, **pool_metrics(rows),
                       "inference_seconds": sum(row["inference_seconds"] for row in rows),
                       "model_bytes": (output / f"{method}.npz").stat().st_size if method in selected else 32,
                       "model_array_bytes": selected[method].model_array_bytes if method in selected else 32,
                       "selected_candidate": selection["selected"].get(method),
                       "shared_group_training_seconds": selected_rows[method]["shared_group_training_seconds"]
                       if method in selected_rows else (dbp_calibration_seconds if method == "calibrated_dbp" else 0.0)})
    result = {"schema_version": 1, "mode": config.mode, "launch_power_dbm": power,
              "evidence_scope": "exploratory pilot" if config.mode == "pilot" else "held-out simulation",
              "numerical_qualification": "not_assessed_by_training_runner", "metrics": pooled,
              "uncertainty_note": "Clopper-Pearson intervals pool bits conditionally; errors may correlate. "
              "Independent frame BERs and their spread are also retained.",
              "runtime_note": "Inference timings are extra processing after shared channel/EDC generation; "
              "DBP includes inverse propagation. They exclude disk I/O and are not end-to-end hardware latency."}
    _write_json(output / "metrics.json", result)
    _write_csv(output / "metrics.csv", pooled)
    _write_json(output / "test_frames.json", test_rows)
    _write_csv(output / "test_frames.csv", test_rows)
    _write_json(output / "generation.json", generation_rows)
    np.savez_compressed(output / "constellations.npz", **diagnostics)
    if software_manifest()["source_hash"] != compatibility["source_hash"]:
        raise RuntimeError("source files changed during the run; do not mark this condition complete")
    _write_json(output / "complete.json", {"compatibility": compatibility,
                "launch_power_dbm": power, "files_sha256": {path.name: _file_hash(path) for path in output.iterdir()
                if path.is_file() and path.name != "complete.json" and not path.name.endswith(".tmp")}})
    emit(f"{power:+g} dBm: completed; counted {pooled[0]['n_bits']} bits per receiver")
    return result


def _verify_completion(path: Path, compatibility: dict, power: float):
    record = json.loads((path / "complete.json").read_text())
    if record["compatibility"] != compatibility or record["launch_power_dbm"] != power:
        raise ValueError("power artifact provenance differs from requested run")
    for filename, expected in record["files_sha256"].items():
        artifact = path / filename
        if not artifact.exists() or _file_hash(artifact) != expected:
            raise ValueError(f"resume rejected: missing or changed artifact {artifact}")


def run_experiment(config: ExperimentConfig, output_directory: str | Path, run_id: str,
                   powers: Iterable[float] | None = None, resume: bool = False,
                   progress: Callable[[str], None] | None = None) -> Path:
    """Fit/select on train/validation, then test once, power by power.

    Complete conditions are reused only after config/source/environment and every
    saved artifact hash agree. A partial condition is regenerated with identical
    seeds/settings when ``resume`` is explicit; no result-based adjustment occurs.
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id) or not run_id.startswith(config.mode + "-"):
        raise ValueError(f"run_id must be a safe name beginning with {config.mode}-")
    powers = list(config.launch_powers_dbm if powers is None else powers)
    if not powers or len(set(powers)) != len(powers) or any(power not in config.launch_powers_dbm for power in powers):
        raise ValueError("requested powers must be a distinct nonempty subset of the configured sweep")
    output = Path(output_directory) / run_id
    software = software_manifest()
    compatibility = {"config_hash": content_hash(config.to_dict()), "source_hash": software["source_hash"],
                     "python": software["python"], "packages": software["packages"], "backend": config.backend}
    all_seeds = [record[key] for power in config.launch_powers_dbm for record in frame_seeds(config, power)
                 for key in ("bit_seed", "noise_seed")]
    if len(set(all_seeds)) != len(all_seeds):
        raise ValueError("cross-power seed collision: choose a different root seed")
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if not resume:
            raise FileExistsError("run already exists; use --resume with unchanged configuration/software")
        if manifest["compatibility"] != compatibility:
            raise ValueError("resume rejected: configuration, source, backend, or software versions changed")
        if content_hash(json.loads((output / "config.json").read_text())) != compatibility["config_hash"]:
            raise ValueError("resume rejected: recorded run configuration was changed")
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("nonempty run directory has no manifest")
        output.mkdir(parents=True, exist_ok=True)
        _write_json(output / "config.json", config.to_dict())
        _write_json(manifest_path, {"schema_version": 1, "run_id": run_id, "mode": config.mode,
            "created_utc": datetime.now(timezone.utc).isoformat(), "compatibility": compatibility, "software": software,
            "future_context_symbols": config.context.future_symbols,
            "future_context_seconds": config.context.future_symbols / config.link.symbol_rate_hz,
            "seed_contract": "Independent whole frames, distinct bit/ASE streams, split before windows",
            "power_generalization": "Models trained separately per power; unseen-power generalization is not assessed"})
    for power in powers:
        power_output = output / _power_directory(power)
        completion = power_output / "complete.json"
        if completion.exists():
            _verify_completion(power_output, compatibility, power)
        else:
            if power_output.exists() and not resume:
                raise FileExistsError("partial power output exists; explicit --resume is required")
            _run_power(config, power, power_output, compatibility, progress)
    # Include every completed configured power, including previous subset calls.
    results = []
    for power in config.launch_powers_dbm:
        path = output / _power_directory(power)
        if (path / "complete.json").exists():
            _verify_completion(path, compatibility, power)
            results.extend(json.loads((path / "metrics.json").read_text())["metrics"])
    _write_csv(output / "summary.csv", [dict(run_id=run_id, mode=config.mode, **row) for row in results])
    _write_json(output / "summary.json", {"run_id": run_id, "mode": config.mode, "metrics": results})
    return output
