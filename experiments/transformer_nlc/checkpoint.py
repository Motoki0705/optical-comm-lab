"""Atomic checkpoints and explicit code/runtime identities; torch imports lazily."""
from contextlib import contextmanager
import fcntl
import importlib
import os
from pathlib import Path
import platform
import sys
import tempfile

import numpy as np
import scipy

from .data import canonical_hash, file_hash, source_identity

ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def advisory_lock(path: str | Path):
    """OS releases this lock even after SIGKILL; stale PID files do not block."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise BlockingIOError(f"Job is busy: {path}") from exc
        try:
            stream.seek(0)
            stream.truncate()
            stream.write(str(os.getpid()))
            stream.flush()
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


@contextmanager
def device_lock(device: str, gpu_lock: str | Path | None = None):
    if device.startswith("cuda"):
        # All worktrees share the coordinating checkout's physical GPU lock.
        shared = Path("/home/kamimura/projects/optical-comm-lab/.work/orchestration/gpu.lock")
        with advisory_lock(gpu_lock or shared):
            yield
    else:
        yield


def code_identity() -> dict:
    """Versioned exact executable dependency set, valid in checkout or wheel.

    README/report additions do not change a model's numerical source identity.
    Git/dependency-lock observations remain literal provenance when available.
    """
    base = source_identity()
    layout = base["source_layout"]
    anchor = Path(layout["module_paths"]["experiments/transformer_nlc/data.py"]).parent
    names = ("model", "model_config", "mask", "complexity",
             "training_config", "training", "evaluation", "checkpoint")
    hashes = dict(base["files"])
    paths = dict(layout["module_paths"])
    for name in ("", *names):
        module = importlib.import_module(f"experiments.transformer_nlc{'.' + name if name else ''}")
        actual = Path(module.__file__).resolve()
        if actual.parent != anchor or actual.suffix != ".py" or not actual.is_file():
            raise ValueError(f"Mixed training imports: {module.__name__} belongs to a different study tree")
        logical = f"experiments/transformer_nlc/{name or '__init__'}.py"
        paths[logical] = str(actual)
        hashes[logical] = file_hash(actual)
    result = {"scope_version": "transformer-training-source-v1", "files": hashes,
              "content_sha256": canonical_hash({"scope_version": "transformer-training-source-v1", "files": hashes}),
              "source_layout": {**layout, "module_paths": paths},
              "observed_provenance": base["observed_provenance"],
              "guard_exclusions": "CLI dispatch, prose, report modules; observed hashes retained, resolved configs guarded",
              "dependency_locks": {name: digest for name, digest in base["observed_provenance"]["files"].items()
                                   if name in ("pyproject.toml", "uv.lock")},
              "commit": base["commit"], "dirty_status": base["dirty_status"], "provenance_kind": layout["kind"]}
    return result


def configure_runtime(config) -> dict:
    import torch
    torch.set_num_threads(config.cpu_threads)
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(config.deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if config.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        # Must be set before a CUDA context is first used, also recorded below.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return runtime_identity(config.device)


def runtime_identity(device: str) -> dict:
    import torch
    result = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__,
              "device": device, "dtype": "float32", "physical_dtype": "complex128",
              "cpu_threads": torch.get_num_threads(), "deterministic": torch.are_deterministic_algorithms_enabled(),
              "default_dtype": str(torch.get_default_dtype()), "float32_matmul_precision": torch.get_float32_matmul_precision(),
              "cuda_build": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
              "tf32": torch.backends.cuda.matmul.allow_tf32,
              "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
              "attention_backend": "torch.scaled_dot_product_attention dense dispatcher",
              "SDPA_backends": {"flash": torch.backends.cuda.flash_sdp_enabled(), "math": torch.backends.cuda.math_sdp_enabled(),
                                "memory_efficient": torch.backends.cuda.mem_efficient_sdp_enabled()},
              "platform": platform.platform(), "byteorder": sys.byteorder}
    if device.startswith("cuda"):
        properties = torch.cuda.get_device_properties(device)
        result.update(gpu_name=properties.name, gpu_total_bytes=properties.total_memory,
                      capability=list(torch.cuda.get_device_capability(device)))
    return result


def rng_state(generator: np.random.Generator) -> dict:
    import torch
    return {"torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() and torch.cuda.is_initialized() else [],
            "numpy_PCG64": generator.bit_generator.state}


def restore_rng(state: dict, generator: np.random.Generator) -> None:
    import torch
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    generator.bit_generator.state = state["numpy_PCG64"]


def save_checkpoint(path: str | Path, state: dict) -> str:
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return file_hash(path)


def load_checkpoint(path: str | Path, *, expected_sha256: str | None = None) -> dict:
    """Only load trusted local study checkpoints; verify frozen hashes first."""
    import torch
    if expected_sha256 is not None and file_hash(path) != expected_sha256:
        raise ValueError("Checkpoint hash mismatch")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("schema") != "transformer-checkpoint-v1":
        raise ValueError("Unknown checkpoint schema")
    return state


def protect_output(output: str | Path, inputs: list[str | Path]) -> Path:
    output = Path(output).resolve()
    for item in inputs:
        source = Path(item).resolve()
        if source.is_relative_to(output) or output.is_relative_to(source):
            raise ValueError(f"Output overlaps protected input: {source}")
    layout = source_identity()["source_layout"]
    executing_directories = {Path(location).resolve().parent for location in layout["module_paths"].values()}
    # Checkout shared modules live under src/optical_comm; wheels place them
    # directly under optical_comm. Guard the actual imported directories in
    # both layouts before callers create any output directory or lock.
    protected = executing_directories | {
        ROOT / name for name in ("src", "experiments", "configs", "tests", "docs", "reports")}
    for source in protected:
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("Output overlaps repository source/report paths")
    return output
