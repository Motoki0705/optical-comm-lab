"""Independent physical frames, boundary-safe blocks, and atomic data manifests."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Any, Callable, Iterator

import numpy as np
import scipy

from optical_comm.channel import edfa_ase_psd, propagate_link_adaptive
from optical_comm.dsp import apply_dispersion, pulse_shape, receiver_dsp
from optical_comm.metrics import bits_to_symbols
from experiments.transformer_nlc.data_config import DataConfig, PURPOSE_KEYS, SPLIT_KEYS, scientific_contract
from experiments.transformer_nlc.receiver import PilotFrame, ReceiverFit, fit_receiver, select_receiver_window

SCHEMA_VERSION = "transformer-data-v2"
QUALIFICATION_SCHEMA_VERSION = "transformer-qualification-v2"
SOURCE_SCOPE_VERSION = "transformer-numerical-source-v1"
# Deliberate dependency boundary: direct numerical algorithms, study policy,
# recipe parsing and qualification decisions. Model/training/report files,
# prose and dependency lockfiles remain observed provenance, not physics guards.
NUMERICAL_SOURCE_PATHS = (
    "src/optical_comm/config.py", "src/optical_comm/channel.py",
    "src/optical_comm/dsp.py", "src/optical_comm/metrics.py",
    "experiments/transformer_nlc/data_config.py", "experiments/transformer_nlc/data.py",
    "experiments/transformer_nlc/receiver.py", "experiments/transformer_nlc/data_cli.py",
    "experiments/transformer_nlc/qualification.py",
)
RUNTIME_SCOPE_VERSION = "transformer-numerical-runtime-v1"


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


@contextmanager
def run_lock(output: Path):
    """Nonblocking advisory flock, released by the OS even after SIGKILL.

    The lock inode is persistent and never unlinked: removing a flock path can
    let two writers lock different inodes. A stale PID is informational only.
    """
    descriptor = os.open(output / ".generation.lock", os.O_CREAT | os.O_RDWR, 0o600)
    acquired = False
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise FileExistsError("Another writer holds the run lock") from exc
        acquired = True
        os.ftruncate(descriptor, 0)
        os.write(descriptor, str(os.getpid()).encode())
        yield
    finally:
        if acquired:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def ensure_bulk_ignored(output: Path) -> None:
    """Keep in-repository bulk beneath an actually ignored path."""
    root = Path(__file__).resolve().parents[2]
    target = (output / "bulk" / "recipe-placeholder.npz").resolve()
    if target.is_relative_to(root):
        result = subprocess.run(["git", "check-ignore", "--quiet", str(target.relative_to(root))], cwd=root)
        if result.returncode != 0:
            raise ValueError("Bulk output must be ignored; use .work/ or artifacts/<run>/bulk")


def environment_identity(backend: str) -> dict[str, Any]:
    result = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform(), "machine": platform.machine(), "physical_dtype": "complex128"}
    if backend == "cupy":
        import cupy as cp
        device = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        name = device["name"]
        result.update(cupy=cp.__version__, gpu_name=name.decode() if isinstance(name, bytes) else str(name),
            cuda_runtime=cp.cuda.runtime.runtimeGetVersion(), cuda_driver=cp.cuda.runtime.driverGetVersion(),
            gpu_total_bytes=device["totalGlobalMem"], physical_dtype="complex128")
    return result


def numeric_runtime(environment: dict[str, Any], backend: str) -> dict[str, Any]:
    """Strict resume guard for numerical dependencies; optional Torch excluded."""
    keys = ["python", "numpy", "scipy", "machine", "physical_dtype"]
    if backend == "cupy":
        keys += ["cupy", "cuda_runtime", "cuda_driver", "gpu_name"]
    return {"scope_version": RUNTIME_SCOPE_VERSION, "backend": backend,
            **{key: environment[key] for key in keys}}


def numerical_source_identity(root: Path | None = None) -> dict[str, Any]:
    """Versioned exact dependency set; missing contributing files fail closed."""
    root = Path(__file__).resolve().parents[2] if root is None else Path(root)
    hashes = {relative: file_hash(root / relative) for relative in NUMERICAL_SOURCE_PATHS}
    return {"scope_version": SOURCE_SCOPE_VERSION, "scope_paths": list(NUMERICAL_SOURCE_PATHS),
            "content_sha256": canonical_hash({"scope_version": SOURCE_SCOPE_VERSION, "files": hashes}), "files": hashes}


def validate_qualification(config: DataConfig, gate: dict[str, Any], source: dict[str, Any]) -> None:
    """Validate the complete scientific contract before creating output paths."""
    contract = scientific_contract(config)
    if (gate.get("schema") != QUALIFICATION_SCHEMA_VERSION or gate.get("status") != "passed"
        or gate.get("condition") != config.condition
        or gate.get("source_scope_version") != SOURCE_SCOPE_VERSION
        or gate.get("source_content_sha256") != source["content_sha256"]
        or gate.get("scientific_contract") != contract
        or gate.get("scientific_contract_sha256") != canonical_hash(contract)):
        raise ValueError("Qualification does not match the production scientific contract/source or did not pass")


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def source_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    paths = []
    for directory in (root / "src/optical_comm", root / "experiments/transformer_nlc"):
        paths.extend(p for p in directory.rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".toml", ".md") and "__pycache__" not in p.parts)
    paths.extend(p for p in (root / "pyproject.toml", root / "uv.lock") if p.is_file())
    hashes = {str(p.relative_to(root)): file_hash(p) for p in sorted(paths)}
    return {**numerical_source_identity(root), "commit": commit,
            "observed_provenance": {"scope": "shared_and_study_tree_plus_pyproject_lock",
                "content_sha256": canonical_hash(hashes), "files": hashes},
            "dirty_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).splitlines()}


def seed_sequence(config: DataConfig, split: str, frame: int, purpose: str) -> np.random.SeedSequence:
    """Exact production key; dedicated pilot subframes cannot collide with it."""
    if split not in SPLIT_KEYS or purpose not in PURPOSE_KEYS or not isinstance(frame, int) or not 0 <= frame < 1_000_000:
        raise ValueError("Invalid split, purpose, or frame index")
    key_split, key_frame = SPLIT_KEYS[split], frame
    if config.regime != "production":
        key_split = SPLIT_KEYS["pilot"]
        key_frame += (SPLIT_KEYS[split] * 1_000_000 if split != "pilot" else 3_000_000)
    return np.random.SeedSequence(config.master_entropy,
        spawn_key=(config.condition, config.replicate, key_split, key_frame, PURPOSE_KEYS[purpose]))


def learning_seed_recipe(config: DataConfig) -> dict[str, Any]:
    """Model/framework and PCG64 shuffle seeds shared across a mask pair."""
    result = {}
    for purpose in ("model", "shuffle"):
        sequence = seed_sequence(config, "train", 0, purpose)
        result[purpose] = {"master_entropy": config.master_entropy, "spawn_key": list(sequence.spawn_key),
            "native_seed_uint64": int(sequence.generate_state(1, dtype=np.uint64)[0]),
            "PCG64_initial_state": np.random.PCG64(sequence).state}
    return result


def frame_layout(config: DataConfig) -> dict[str, np.ndarray]:
    edges = np.r_[0, np.cumsum(config.region_lengths)]
    n = config.n_symbols
    pilot = np.zeros(n, dtype=bool)
    scoring = np.zeros(n, dtype=bool)
    context = np.zeros(n, dtype=bool)
    valid = np.ones(n, dtype=bool)
    pilot[edges[1]:edges[2]] = pilot[edges[5]:edges[6]] = True
    scoring[edges[3]:edges[4]] = True
    context[edges[2]:edges[5]] = True
    if config.guard_symbols:
        valid[:config.guard_symbols] = valid[-config.guard_symbols:] = False
    return {"symbol_index": np.arange(n, dtype=np.int64), "pilot_mask": pilot,
            "scoring_mask": scoring, "context_valid_mask": context, "dsp_valid_mask": valid}


@dataclass
class PhysicalFrame:
    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]

    @property
    def pilots(self) -> PilotFrame:
        return PilotFrame(self.arrays["rx_matched"],
            np.where(self.arrays["pilot_mask"][:, None], self.arrays["tx_symbols"], 0),
            self.arrays["pilot_mask"])


def simulate_physical_frame(config: DataConfig, split: str, frame: int, *, backend: str = "numpy",
                            ase_source: Callable | None = None, noise_recipe: dict | None = None) -> PhysicalFrame:
    """Generate all seven independent regions and the symmetric-CDC link.

    Physical arrays stay complex128. No scored labels reach fitted DSP here.
    """
    start = time.perf_counter()
    sequence = seed_sequence(config, split, frame, "bits")
    regions, rng_states = [], []
    for length, child in zip(config.region_lengths, sequence.spawn(7), strict=True):
        rng = np.random.Generator(np.random.PCG64(child))
        initial = rng.bit_generator.state
        region_bits = rng.integers(0, 2, (length, 2, int(np.log2(config.condition))), dtype=np.uint8)
        # Anchor left guard at its pilot-facing edge under guard refinement.
        regions.append(region_bits[::-1].copy() if not regions else region_bits)
        rng_states.append({"spawn_key": list(child.spawn_key), "initial_state": initial, "final_state": rng.bit_generator.state})
    bits = np.concatenate(regions)
    symbols = bits_to_symbols(bits, config.condition)
    field = pulse_shape(symbols, config.link) * np.sqrt(1e-3 * 10**(config.launch_power_dbm / 10) / 2)
    field = apply_dispersion(field, config.link, -config.link.total_length_km / 2)
    noise_sequence = seed_sequence(config, split, frame, "ase")
    received, audit = propagate_link_adaptive(field, config.link, noise_sequence,
        phase_max_deg=config.phase_max_deg, min_step_km=config.min_step_km, backend=backend, ase_source=ase_source)
    post_cdc = apply_dispersion(received, config.link, -config.link.total_length_km / 2)
    matched, two_sps = receiver_dsp(post_cdc, config.link, config.launch_power_dbm, compensate_dispersion=False)
    arrays = {"tx_bits": bits, "tx_symbols": symbols, "rx_matched": matched, **frame_layout(config)}
    arrays["frame_id"] = np.full(config.n_symbols, frame, dtype=np.int64)
    arrays["split_id"] = np.full(config.n_symbols, SPLIT_KEYS[split], dtype=np.int8)
    raw = {}
    for name, array, enabled, stage, unit in (
        ("rx_field", received, config.save_rx_field, "fiber_after_last_EDFA_before_Rx_CDC", "sqrt(W)"),
        ("rx_2sps", two_sps, config.save_rx_2sps, "Rx_half_CDC_matched_RRC_before_FIR_CR", "unit_Es_complex")):
        raw[name] = {"shape": list(array.shape), "dtype": str(array.dtype), "sha256": array_hash(array),
                     "stage": stage, "unit": unit, "storage": "bulk_npz" if enabled else "regeneration_recipe"}
        if enabled:
            arrays[name] = array
    # Expose the exact physical noise grid and span seed recipes without saving
    # bulk noise. RNG initial states reproduce every ordered frequency draw.
    ase_states = []
    for child in np.random.SeedSequence(noise_sequence.entropy, spawn_key=noise_sequence.spawn_key).spawn(config.link.n_spans):
        rng = np.random.Generator(np.random.PCG64(child))
        ase_states.append({"spawn_key": list(child.spawn_key), "initial_state": rng.bit_generator.state})
    sample_rate = config.link.sample_rate_hz
    modes = np.arange(len(received)); modes[modes >= (len(modes) + 1) // 2] -= len(modes)
    bandwidth = config.link.ase_bandwidth_hz or sample_rate
    active = int(np.count_nonzero((modes >= -bandwidth * len(modes) / (2 * sample_rate)) & (modes < bandwidth * len(modes) / (2 * sample_rate))))
    metadata = {"split": split, "frame": frame, "regime": config.regime,
        "rng_engine": "PCG64", "master_entropy": config.master_entropy,
        "bits_seed": {"spawn_key": list(sequence.spawn_key), "regions": rng_states},
        "ase_seed": {"spawn_key": list(noise_sequence.spawn_key), "spans": ase_states},
        "propagation_audit": asdict(audit), "raw_arrays": raw,
        "ase": {"two_sided_bandwidth_hz": bandwidth, "sample_rate_hz": sample_rate,
                "fft_spacing_hz": sample_rate / len(received), "active_bins": active,
                "psd_w_per_hz_per_pol": edfa_ase_psd(config.link),
                "per_span_complex_variance_per_pol": (noise_recipe["reference_complex_variance_per_pol"] if noise_recipe else edfa_ase_psd(config.link) * sample_rate / len(received) * active),
                "grid_role": "reference_recipe_grid_used_for_injection" if noise_recipe else "record_periodic_FFT_grid",
                "injected_reference_grid": noise_recipe},
        "physical_seconds": time.perf_counter() - start,
        "backend": backend, "custom_noise_recipe": noise_recipe, "regeneration": {"config": config.to_dict(), "split": split, "frame": frame,
            "call": ("simulate_physical_frame(DataConfig.from_dict(config), split, frame, backend=backend)" if ase_source is None else
                     "Use qualification_cases(condition), CommonPhysicalNoise(cases,current,frame), then simulate_physical_frame(..., ase_source=source, noise_recipe=source.recipe)"),
            "custom_noise_recipe": noise_recipe}}
    return PhysicalFrame(arrays, metadata)


def _schema(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    descriptions = {
        "tx_bits": ("bit", "TX_Gray_labels"), "tx_symbols": ("unit_Es_complex", "TX_before_RRC"),
        "rx_matched": ("unit_Es_complex", "Rx_half_CDC_matched_RRC_before_FIR_CR"),
        "rx_nlc": ("unit_Es_complex", "Rx_FIR_CR_pilot_calibrated_before_NLC"),
        "rx_field": ("sqrt(W)", "fiber_after_last_EDFA_before_Rx_CDC"),
        "rx_2sps": ("unit_Es_complex", "Rx_half_CDC_matched_RRC_before_FIR_CR"),
        "symbol_index": ("DP_time_index", "within_frame"), "frame_id": ("frame_index", "within_split"),
        "split_id": ("split_key", "train0_validation1_test2_pilot3")}
    return {name: {"shape": list(array.shape), "dtype": str(array.dtype), "sha256": array_hash(array),
            "unit": descriptions.get(name, ("boolean", name))[0], "stage": descriptions.get(name, ("boolean", name))[1],
            **({"true_count": int(array.sum())} if array.dtype == bool else {})}
            for name, array in arrays.items()}


def load_frame(path: str | Path, *, expected_sha256: str | None = None) -> PhysicalFrame:
    path = Path(path)
    if expected_sha256 is not None and file_hash(path) != expected_sha256:
        raise ValueError("Frame file hash mismatch")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    return PhysicalFrame(arrays, {})


def iter_blocks(frame: PhysicalFrame, *, block_symbols: int = 128, context: int = 96) -> Iterator[dict[str, np.ndarray]]:
    """One frame per call; reject holes, periodic wrap, or split/frame leakage.

    Complex128 inputs and residual targets are returned; training owns float32.
    """
    arrays = frame.arrays
    scored = np.flatnonzero(arrays["scoring_mask"])
    if not len(scored) or block_symbols < 1 or context < 0 or len(scored) % block_symbols or np.any(np.diff(scored) != 1):
        raise ValueError("Require one contiguous scoring region divisible by block size")
    for start in range(int(scored[0]), int(scored[-1]) + 1, block_symbols):
        left, right = start - context, start + block_symbols + context
        if left < 0 or right > len(arrays["rx_nlc"]):
            raise ValueError("Context crosses frame boundary")
        if not np.all(arrays["context_valid_mask"][left:right] & arrays["dsp_valid_mask"][left:right]):
            raise ValueError("Context crosses invalid region")
        if len(np.unique(arrays["frame_id"][left:right])) != 1 or len(np.unique(arrays["split_id"][left:right])) != 1 or np.any(np.diff(arrays["symbol_index"][left:right]) != 1):
            raise ValueError("Window crosses frame/index boundary")
        target = slice(start, start + block_symbols)
        yield {"rx": arrays["rx_nlc"][left:right], "tx": arrays["tx_symbols"][target],
               "residual": arrays["rx_nlc"][target] - arrays["tx_symbols"][target],
               "symbol_index": arrays["symbol_index"][target], "frame_id": arrays["frame_id"][target]}


def generation_plan(config: DataConfig) -> dict[str, Any]:
    n_frames = sum(config.frame_counts.values())
    return {"regime": config.regime, "condition": config.condition, "replicate": config.replicate,
            "frames": config.frame_counts, "scored_DP_times": {k: v * config.scored_symbols for k, v in config.frame_counts.items()},
            "frame_DP_times_including_overhead": config.n_symbols,
            "minimum_array_bytes": n_frames * config.n_symbols * (2 * int(np.log2(config.condition)) + 2 * 16 * 3 + 8 * 2 + 4),
            "requires_numerical_qualification": config.regime == "production",
            "test_metrics": "not_computed", "config": config.to_dict()}


def generate(config: DataConfig, output_dir: str | Path, *, backend: str = "numpy", resume: bool = False,
             qualification: str | Path | None = None, command: list[str] | None = None) -> dict[str, Any]:
    """Generate common candidate data; select/freeze receiver before test DSP.

    Writes all arrays under ignored bulk/, plus a compact manifest and receiver.
    Resume verifies source/config/file hashes and deterministic receiver fit.
    Production requires an explicit passing qualification artifact for this
    condition's numerical controls; it is never silently inferred from smoke.
    """
    output = Path(output_dir)
    ensure_bulk_ignored(output)
    source = source_identity()
    config_hash = canonical_hash(config.to_dict())
    gate = None
    if config.regime == "production":
        if qualification is None:
            raise ValueError("Production blocked: full numerical qualification evidence is required")
        gate = json.loads(Path(qualification).read_text())
        validate_qualification(config, gate, source)
    environment = environment_identity(backend)
    runtime = numeric_runtime(environment, backend)
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError("Run directory exists; choose a new run ID or --resume")
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output):
        can_record_failure = False
        try:
            bulk = output / "bulk"
            bulk.mkdir(exist_ok=True)
            manifest_path = output / "manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                if (manifest.get("schema") != SCHEMA_VERSION or manifest["config_sha256"] != config_hash
                    or manifest["source"].get("scope_version") != SOURCE_SCOPE_VERSION
                    or manifest["source"]["content_sha256"] != source["content_sha256"]
                    or manifest.get("numeric_runtime") != runtime):
                    raise ValueError("Resume schema/config/source/numeric runtime/backend mismatch")
            else:
                manifest = {"schema": SCHEMA_VERSION, "status": "in_progress", "plan": generation_plan(config),
                    "config": config.to_dict(), "config_sha256": config_hash, "source": source,
                    "backend": backend, "command": command, "qualification": gate,
                    "environment": environment, "numeric_runtime": runtime,
                    "scientific_contract": scientific_contract(config),
                    "scientific_contract_sha256": canonical_hash(scientific_contract(config)),
                    "frames": [], "test_metrics": "not_computed", "learning_seeds": learning_seed_recipe(config),
                    "stages": ["TX_unit_Es_Gray_QAM", "TX_RRC_sqrtW", "TX_inverse_CDC_half",
                        "Manakov_EDFA_ASE", "ideal_resampling_clock_frequency_identity", "RX_inverse_CDC_half",
                        "RX_matched_RRC_unit_Es", "train_pilot_2x2_FIR", "frame_pilot_gain",
                        "blind_phase_search", "frame_pilot_ambiguity_gain", "rx_nlc_unit_Es"],
                    "feature_order": ["XI", "XQ", "YI", "YQ"],
                    "seed_convention": {"production": "(condition,replicate,split,frame,purpose)",
                        "dedicated_pilot": "split=3; frame=split_id*1000000+frame (pilot named split adds3000000)",
                        "bits_regions": "purpose_bits child region=0..6", "ase": "purpose_ase child span=0..n_spans-1"}}
            can_record_failure = True
            atomic_json(manifest_path, manifest)
            frames = {}
            by_key = {(entry["split"], entry["frame"]): entry for entry in manifest["frames"]}
            for split, count in config.frame_counts.items():
                for index in range(count):
                    key = (split, index)
                    path = bulk / f"{split}-{index:03d}-physical.npz"
                    if key in by_key:
                        entry = by_key[key]
                        path = output / entry["path"]
                        frame = load_frame(path, expected_sha256=entry["file_sha256"])
                        frame.metadata = entry["metadata"]
                    else:
                        frame = simulate_physical_frame(config, split, index, backend=backend)
                        atomic_npz(path, frame.arrays)
                        entry = {"split": split, "frame": index, "path": str(path.relative_to(output)),
                            "file_sha256": file_hash(path), "metadata": frame.metadata, "arrays": _schema(frame.arrays)}
                        manifest["frames"].append(entry)
                        by_key[key] = entry
                        atomic_json(manifest_path, manifest)
                    frames[key] = frame
            fit = fit_receiver(config, [frames[("train", i)].pilots for i in range(config.train_frames)])
            validation = []
            for i in range(config.validation_frames):
                frame = frames[("validation", i)]
                validation.append((frame.pilots, frame.arrays["tx_symbols"], frame.arrays["tx_bits"], frame.arrays["scoring_mask"]))
            fit = select_receiver_window(fit, config, validation)
            receiver_path = output / "receiver.npz"
            receiver_metadata = {"order": fit.order, "bps_window": fit.bps_window, "bps_phases": fit.bps_phases,
                "fit_metadata": fit.fit_metadata, "validation_selection": fit.validation_selection,
                "weights_sha256": array_hash(fit.weights)}
            if "receiver" in manifest:
                if file_hash(receiver_path) != manifest["receiver"]["file_sha256"]:
                    raise ValueError("Receiver file hash mismatch on resume")
                if manifest["receiver"]["weights_sha256"] != receiver_metadata["weights_sha256"]:
                    raise ValueError("Deterministic receiver fit changed on resume")
            else:
                atomic_npz(receiver_path, {"weights": fit.weights})
            manifest["receiver"] = {**receiver_metadata, "path": "receiver.npz", "file_sha256": file_hash(receiver_path)}
            for key, frame in frames.items():
                rx, valid, calibration = fit.apply(frame.pilots)
                if np.any(frame.arrays["scoring_mask"] & ~valid):
                    raise ValueError("Scoring outside DSP-valid mask")
                frame.arrays["rx_nlc"] = rx
                frame.arrays["dsp_valid_mask"] &= valid
                path = bulk / f"{key[0]}-{key[1]:03d}-nlc.npz"
                if by_key[key]["path"] == str(path.relative_to(output)):
                    # Already committed final output: unchanged arrays need no rewrite.
                    if any(_schema(frame.arrays)[name]["sha256"] != by_key[key]["arrays"][name]["sha256"] for name in frame.arrays):
                        raise ValueError("Receiver outputs changed on resume")
                else:
                    atomic_npz(path, frame.arrays)
                by_key[key].update(path=str(path.relative_to(output)), file_sha256=file_hash(path), arrays=_schema(frame.arrays), calibration=calibration)
                atomic_json(manifest_path, manifest)
            manifest["status"] = "complete"
            manifest["receiver_fixed_before_test_metrics"] = True
            atomic_json(manifest_path, manifest)
            return manifest
        except BaseException as exc:
            if can_record_failure:
                manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
                manifest.setdefault("failed_attempts", []).append({"exception": type(exc).__name__, "message": str(exc)})
                try:
                    atomic_json(manifest_path, manifest)
                except OSError:
                    pass  # Preserve the original exception if storage itself failed.
            raise



def load_receiver(output_dir: str | Path) -> ReceiverFit:
    output = Path(output_dir)
    metadata = json.loads((output / "manifest.json").read_text())["receiver"]
    if file_hash(output / metadata["path"]) != metadata["file_sha256"]:
        raise ValueError("Receiver file hash mismatch")
    with np.load(output / metadata["path"], allow_pickle=False) as archive:
        weights = archive["weights"]
    weights.setflags(write=False)
    return ReceiverFit(weights, metadata["order"], metadata["bps_window"], metadata["bps_phases"],
                       metadata["fit_metadata"], tuple(metadata["validation_selection"]))
