#!/usr/bin/env python3
"""Generate qualified real SSFM data for the Kerr Blender films.

Usage: .venv/bin/python scripts/simulate_kerr_films.py
Reproduction: append --output .work/kerr-physics-reproduction
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import scipy

from optical_comm.config import LinkConfig
from optical_comm.visualization.pulses import (
    display_data,
    gaussian_pulse,
    propagate_snapshots,
    qualify_pulse_trajectories,
    time_grid,
    trajectory_metrics,
)


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/kerr_films.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/kerr-physics")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("unsupported Kerr film config schema")
    base_link = LinkConfig.from_dict(config["link"])
    if base_link.n_spans != 1 or base_link.ase_enabled:
        raise ValueError("Kerr film configuration must describe one unamplified ASE-free fiber")
    if set(config["cases"]) != {"pure_spm", "dispersion_only", "combined"}:
        raise ValueError("qualification requires the three explicitly separated cases")
    steps = tuple(config["qualification"]["step_sizes_km"])
    if len(steps) != 3 or base_link.step_km != steps[1] or steps[0] != 2 * steps[1] or steps[1] != 2 * steps[2]:
        raise ValueError("production step must be the middle of three halving refinements")
    if config["qualification"]["time_grid_refinement_factor"] != 2 or config["qualification"]["period_refinement_factor"] != 2:
        raise ValueError("this qualification implements independent factor-two grid and period refinements")
    cases = {name: replace(base_link, **settings) for name, settings in config["cases"].items()}
    if cases["pure_spm"].dispersion_ps_nm_km != 0 or cases["dispersion_only"].gamma_per_w_km != 0:
        raise ValueError("pure_spm needs D=0 and dispersion_only needs gamma=0")
    interval = float(config["sampling"]["snapshot_interval_km"])
    if interval <= 0 or not np.isclose(base_link.span_km / interval, round(base_link.span_km / interval), atol=1e-12):
        raise ValueError("snapshot interval must divide fiber length exactly")
    times = time_grid(config["sampling"]["n_samples"], base_link.sample_rate_hz)
    positions = np.arange(round(base_link.span_km / interval) + 1, dtype=np.float64) * interval
    pulse = config["pulse"]
    input_field = gaussian_pulse(times, pulse["peak_xy_power_W"], pulse["intensity_1e_half_width_s"])
    display = config["display"]
    phase_threshold = display["phase_relative_power_threshold"]
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output must be empty; use a fresh reproduction directory: {relative(output)}")
    output.mkdir(parents=True, exist_ok=True)
    source_paths = [
        *sorted((ROOT / "src/optical_comm").glob("*.py")),
        ROOT / "src/optical_comm/visualization/__init__.py",
        ROOT / "src/optical_comm/visualization/pulses.py",
        ROOT / "scripts/simulate_kerr_films.py",
        ROOT / "tests/test_kerr_pulses.py",
        ROOT / "pyproject.toml", ROOT / "uv.lock",
        ROOT / "docs/requirements.md", ROOT / "docs/experiment-plan.md",
        ROOT / "docs/channel-methods.md", ROOT / "docs/kerr-physics.md",
    ]
    source_hashes = {relative(path): digest(path) for path in source_paths}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    manifest = {
        "schema_version": 1,
        "status": "running",
        "purpose": config["purpose"],
        "issue": "https://github.com/Motoki0705/optical-comm-lab/issues/15",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "source_hashes": source_hashes,
        "input_config": {"path": relative(config_path), "sha256": digest(config_path)},
        "input_pulse_sha256": hashlib.sha256(input_field.tobytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "system": platform.system(), "machine": platform.machine(),
            "processor": platform.processor(), "backend": "numpy/scipy FFT, complex128",
        },
        "seed_contract": "No random variables or seeds: deterministic analytic input; no ASE, no train/validation/test split.",
        "solver": {
            "entry_point": "optical_comm.channel.propagate_fiber",
            "equation": "dA/dz=-alpha*A/2-i*beta2*d2A/dt2/2+i*(8/9)*gamma*(|Ax|²+|Ay|²)*A",
            "step_km": base_link.step_km,
            "snapshot_z_km": positions.tolist(),
            "sample_rate_Hz": base_link.sample_rate_hz,
            "n_samples": len(times), "dt_ps": 1e12 / base_link.sample_rate_hz,
            "period_ns": len(times) / base_link.sample_rate_hz * 1e9,
            "field_dtype": "complex128",
            "boundary": "Periodic FFT on a 2 ns isolated-pulse window; independent 4 ns refinement is measured.",
            "beta2_ps2_per_km": base_link.beta2_s2_per_km * 1e24,
            "alpha_per_km": base_link.alpha_per_km,
            "manakov_nonlinearity_factor": 8 / 9,
            "amplifiers_and_noise": "Not applied: propagate_fiber is called, not propagate_link.",
            "resolved_case_configs": {name: value.to_dict() for name, value in cases.items()},
        },
        "display_contract": {
            "axes": "field/observable arrays=(distance,time,polarization); polarization order X,Y",
            "files": {name: name + ".npz" for name in cases},
            "units": {
                "z_km": "km", "t_ps": "ps", "A_xy_sqrtW": "sqrt(W)", "power_xy_W": "W per polarization",
                "phase_xy_rad": "rad", "delta_omega_rad_s": "rad/s", "phase_valid": "bool",
                "frequency_offset_Hz": "Hz physical optical offset", "spectrum_xy_J_per_Hz": "J/Hz per polarization",
                "energy_xy_J": "J per polarization", "attenuation_power_factor": "dimensionless power ratio",
            },
            "phase_mask": "per-polarization power > relative threshold * snapshot XY peak / 2; invalid phase/chirp=NaN; power and spectrum retained",
            "phase_relative_power_threshold": phase_threshold,
            "phase_unwrap": "Each contiguous valid lobe; central lobe anchored at t=0 with phase continuous across z; disconnected lobes locally anchored.",
            "physical_carrier_convention": display["field_convention"],
            "spectral_energy": display["spectrum_units"],
            "polarization_total": "sum(|Ax|²+|Ay|²) and sum(spectra); never FFT(Ax+Ay)",
            "time_view_half_range_ps": display["time_view_half_range_ps"],
            "frequency_view_half_range_GHz": display["frequency_view_half_range_GHz"],
            "cropped_display_energy": "derived_metrics.json retains fractional energy outside the common film view for each z; NPZ retains the full solver window",
            "loss_policy": display["loss_policy"],
            "render_interpolation": "Animation may interpolate between 1 km solver snapshots; intermediate frames are interpolations, not additional SSFM measurements.",
            "carrier_and_geometry": "Optical carrier may be enlarged and geometry/time/z exaggerated only when disclosed; the saved envelope is the actual solver result.",
        },
        "scope": config["scope"],
        "independent_review": {"requested": 0, "attempted": 0, "completed": 0},
        "reproduction_command": ".venv/bin/python scripts/simulate_kerr_films.py --config configs/kerr_films.json --output .work/kerr-physics-reproduction",
        "test_command": ".venv/bin/python -m pytest tests/test_kerr_pulses.py -q",
        "sources": [
            {"title": "Stolen and Lin, Self-phase-modulation in silica optical fibers (1978)", "doi": "10.1103/PhysRevA.17.1448", "url": "https://journals.aps.org/pra/abstract/10.1103/PhysRevA.17.1448", "use": "Physical basis for SPM and frequency broadening; not a validation of this pulse's parameters."},
            {"title": "Marcuse, Menyuk and Wai, Application of the Manakov-PMD equation (1997)", "doi": "10.1109/50.622902", "url": "https://doi.org/10.1109/50.622902", "use": "Averaged polarization nonlinear coupling; existing repository Manakov convention."},
            {"title": "Hasegawa and Tappert, Transmission of stationary nonlinear optical pulses I (1973)", "doi": "10.1063/1.1654836", "url": "https://doi.org/10.1063/1.1654836", "use": "Nonlinearity/dispersion interaction; the present Gaussian/lossy case is not claimed to be a soliton."},
            {"title": "Sinkin et al., Optimization of the split-step Fourier method (2003)", "doi": "10.1109/JLT.2003.808628", "url": "https://umbc.edu/photonics/members/publications/PdfPapers/PAJ167.pdf", "use": "Numerical SSFM step selection/refinement; measured convergence here remains case-specific."},
        ],
    }
    dump_json(output / "manifest.json", manifest)
    dump_json(output / "config.json", config)
    start = time.perf_counter()
    trajectories = {}
    generation_times = {}
    for name, link in cases.items():
        begin = time.perf_counter()
        trajectories[name] = propagate_snapshots(input_field, link, positions)
        generation_times[name] = time.perf_counter() - begin
    metrics = {
        "schema_version": 1,
        "scope": "Deterministic isolated-pulse diagnostics from full-grid complex128 SSFM; distinct from communications validation and testing.",
        "width_definition": "rms_time_ps is standard deviation of XY power over t; rms_frequency_GHz is standard deviation of XY spectral energy. Gaussian T0=sqrt(2)*rms_time.",
        "cases": {name: trajectory_metrics(field, times, positions, cases[name], phase_relative_threshold=phase_threshold, time_view_ps=display["time_view_half_range_ps"], frequency_view_GHz=display["frequency_view_half_range_GHz"]) for name, field in trajectories.items()},
    }
    dump_json(output / "derived_metrics.json", metrics)
    qualify_start = time.perf_counter()
    qualification = qualify_pulse_trajectories(trajectories, cases, times, positions, pulse["peak_xy_power_W"], pulse["intensity_1e_half_width_s"], step_sizes_km=steps, phase_relative_threshold=phase_threshold)
    qualification_time = time.perf_counter() - qualify_start
    export_records = {}
    for name, field in trajectories.items():
        data, export_record = display_data(field, times, positions, cases[name], decimation=config["sampling"]["display_decimation"], phase_relative_threshold=phase_threshold)
        np.savez_compressed(output / (name + ".npz"), **data)
        export_records[name] = export_record
    qualification["display_export"] = {"cases": export_records, "passed": all(record["passed"] for record in export_records.values())}
    qualification["passed"] = qualification["passed"] and qualification["display_export"]["passed"]
    dump_json(output / "qualification.json", qualification)
    checks_command = [sys.executable, "-m", "pytest", "tests/test_kerr_pulses.py", "-q"]
    tests = subprocess.run(checks_command, cwd=ROOT, capture_output=True, text=True)
    (output / "tests.txt").write_text(tests.stdout + tests.stderr, encoding="utf-8")
    dump_json(output / "tests.json", {"command": manifest["test_command"], "exit_code": tests.returncode, "passed": tests.returncode == 0, "log": "tests.txt", "scope": "Substantive analytic, data-unit, polarization and numerical-regression tests; normal verification, not an independent validator."})
    rows = ["case,z_km,rms_time_ps,rms_frequency_GHz,peak_xy_power_W,center_phase_rad,energy_total_J"]
    for name, result in metrics["cases"].items():
        for index, z in enumerate(positions):
            rows.append(",".join([name, f"{z:.8g}", *[f"{result[key][index]:.12g}" for key in ("rms_time_ps", "rms_frequency_GHz", "peak_xy_power_W", "center_phase_rad")], f"{sum(result['energy_xy_J'][index]):.12g}"]))
    (output / "summary.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    endpoints = []
    for name, result in metrics["cases"].items():
        endpoints.append(f"| {name} | {result['rms_time_ps'][-1]:.6f} | {result['rms_frequency_GHz'][-1]:.6f} | {1000*result['peak_xy_power_W'][-1]:.6f} | {result['center_phase_rad'][-1]:.6f} |")
    (output / "README.md").write_text(
        "# Kerr film physical data\n\n"
        "Computed illustrative pulse trajectories for [Issue #15](https://github.com/Motoki0705/optical-comm-lab/issues/15). The existing complex128 Manakov SSFM propagates a 50 mW **XY-total peak** Gaussian with intensity 1/e half-width 20 ps through 80 km of lossy fiber. No amplifier or ASE is applied. This is a separate deterministic demonstration, not a new 1600 km communications test.\n\n"
        "| Case at 80 km | RMS time (ps) | RMS spectral width (GHz) | Raw XY peak (mW) | Center envelope phase (rad) |\n| --- | ---: | ---: | ---: | ---: |\n"
        + "\n".join(endpoints) + "\n\n"
        "All 81 distance snapshots and full-window display fields are saved in the three NPZ files. `derived_metrics.json`/`summary.csv` use full solver precision; `qualification.json` records analytic, energy, step/grid/window, and display-export checks. `manifest.json` hashes numerical inputs and every artifact, declares units/conventions, and gives reproduction commands. `tests.json` and `tests.txt` retain ordinary verification.\n\n"
        "`spectrum_xy_J_per_Hz` is pulse spectral **energy** density per polarization, not W/Hz. Orthogonal XY powers/spectra are summed separately. Raw fields retain 16 dB loss over 80 km; loss-compensated film views must be labeled. Phase/chirp NaNs are low-power mask values, not missing propagation. Optical carrier periods and fiber geometry in the films may be exaggerated; the saved complex envelope is not hand-authored. See [kerr-physics.md](../../docs/kerr-physics.md) for the stable data contract and scope.\n",
        encoding="utf-8",
    )
    if source_hashes != {relative(path): digest(path) for path in source_paths}:
        raise RuntimeError("an input source changed during the run; leave this output incomplete and regenerate")
    manifest.update({
        "status": "complete" if qualification["passed"] and tests.returncode == 0 else "failed_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": {"generation_by_case": generation_times, "numerical_qualification": qualification_time, "total": time.perf_counter() - start},
        "qualification_passed": qualification["passed"], "ordinary_tests_passed": tests.returncode == 0,
        "display_export": export_records,
        "artifact_hashes": {path.name: digest(path) for path in sorted(output.iterdir()) if path.is_file() and path.name != "manifest.json"},
        "artifact_total_bytes_excluding_manifest": sum(path.stat().st_size for path in output.iterdir() if path.is_file() and path.name != "manifest.json"),
        "manifest_hash_policy": "All other artifacts are hashed here; a manifest cannot contain its own hash. Film/explanation manifests hash this manifest externally.",
    })
    dump_json(output / "manifest.json", manifest)
    if manifest["status"] != "complete":
        raise RuntimeError("ordinary physical/display verification failed; inspect qualification.json and tests.txt")
    return manifest


def main() -> None:
    manifest = run(arguments())
    print(json.dumps({"status": manifest["status"], "qualification_passed": manifest["qualification_passed"], "ordinary_tests_passed": manifest["ordinary_tests_passed"], "artifact_bytes": manifest["artifact_total_bytes_excluding_manifest"]}))


if __name__ == "__main__":
    main()
