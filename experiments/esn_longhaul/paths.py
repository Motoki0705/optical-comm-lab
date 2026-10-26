"""Locate installed scientific sources without relying on the working directory."""

from pathlib import Path

import optical_comm.config


WORKSPACE = Path(__file__).resolve().parent
LIBRARY = Path(optical_comm.config.__file__).resolve().parent


def repository_root() -> Path | None:
    """Return a source checkout root, or None for an installed wheel."""
    candidate = WORKSPACE.parents[1]
    if (candidate / "src/optical_comm").is_dir():
        return candidate
    return None


def verify_source_layout() -> None:
    """Reject a checkout that accidentally imported another checkout's library."""
    repository = repository_root()
    expected = repository / "src/optical_comm" if repository else WORKSPACE.parents[1] / "optical_comm"
    if LIBRARY != expected.resolve():
        raise ValueError("experiment and imported scientific source are from different trees; install this tree or set PYTHONPATH")


def source_files() -> dict[str, Path]:
    """Actual source bytes, with stable logical paths in checkouts and wheels."""
    verify_source_layout()
    files = {f"src/optical_comm/{path.name}": path for path in sorted(LIBRARY.glob("*.py"))}
    files["experiments/__init__.py"] = WORKSPACE.parent / "__init__.py"
    files.update({f"experiments/esn_longhaul/{path.name}": path for path in sorted(WORKSPACE.glob("*.py"))})
    repository = repository_root()
    if repository and (repository / "scripts/run_experiment.py").is_file():
        files["scripts/run_experiment.py"] = repository / "scripts/run_experiment.py"
    return files


def protected_roots() -> tuple[Path, ...]:
    """Protect study code/configuration as well as the shared numerical library."""
    roots = [LIBRARY, WORKSPACE.parent]
    repository = repository_root()
    if repository:
        roots.extend(repository / name for name in ("src", "scripts", "tests", "configs", "docs"))
    return tuple(roots)
