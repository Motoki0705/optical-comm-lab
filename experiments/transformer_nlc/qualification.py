"""Bounded one-factor numerical qualification; dry planning never propagates.

Initial plan: 6 settings × 3 dedicated pilot frames × 2 principal conditions.
The initial plan is a budget estimate, not a cap on justified refinement.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from optical_comm.channel import ase_noise, edfa_ase_psd, propagate_link_adaptive
from optical_comm.dsp import apply_dispersion, pulse_shape, receiver_dsp
from optical_comm.metrics import bits_to_symbols, symbols_to_bits, compute_metrics, gray_qam_awgn_ber, ber_interval
from experiments.transformer_nlc.data_config import DataConfig, main16, main64, scientific_contract
from experiments.transformer_nlc.data import (PhysicalFrame, atomic_json, atomic_npz, canonical_hash,
    file_hash, load_frame, seed_sequence, simulate_physical_frame, source_identity, run_lock, environment_identity, ensure_bulk_ignored, numeric_runtime, SOURCE_SCOPE_VERSION, QUALIFICATION_SCHEMA_VERSION)
from experiments.transformer_nlc.receiver import fit_receiver, select_receiver_window

CASES = ("baseline", "phase", "sps", "frame", "guard", "rrc")


def qualification_cases(condition: int) -> dict[str, DataConfig]:
    if condition not in (16,64):
        raise ValueError("condition must be16 or64")
    base = replace(main16() if condition == 16 else main64(), regime="qualification",
                   train_frames=1, validation_frames=1, test_frames=1)
    return {"baseline": base,
            "phase": replace(base, phase_max_deg=.025),
            "sps": replace(base, link=replace(base.link, sps=8)),
            "frame": replace(base, scored_symbols=131072),
            "guard": replace(base, guard_symbols=8192, link=replace(base.link, guard_symbols=8192)),
            "rrc": replace(base, link=replace(base.link, rrc_span_symbols=256))}


def qualification_plan(conditions: tuple[int, ...] = (16, 64), frames: int = 3) -> dict[str, Any]:
    return {"schema": "transformer-qualification-plan-v1", "regime": "numerical_qualification",
        "conditions": list(conditions), "independent_frames": frames, "full_link_propagations": len(conditions)*frames*len(CASES),
        "settings": {str(c): {key: cfg.to_dict() for key,cfg in qualification_cases(c).items()} for c in conditions},
        "execution": "Run selected cases explicitly; no propagation occurs in dry-run.",
        "shared_physical_noise": "same long periodic 8-sps Fourier ASE realization, cropped at anchored scored index before injection; fixed128GHz",
        "common_scoring": "first65536 scored DP times in every case; region3 bits are identical prefixes",
        "acceptance": {"max_mean_effective_snr_difference_db": .1,
                       "ber_consistency": "abs(mean frame BER difference)<=max(3 counting-SE,3 frame-mean SE)",
                       "all_cases_required": list(CASES), "analytic_controls": True,
                       "CPU_GPU_required": True},
        "time_estimate": "unmeasured: run --benchmark first; do not infer full-link convergence from a tiny smoke"}


class CommonPhysicalNoise:
    """A common bandlimited noise process across duration and sps refinements.

    Drawing each shorter FFT grid from the same seed would *not* share physical
    noise for unequal periods. Instead all cases sample a single longer 8-sps
    reference process, anchored at scored index0. No noise oracle is removed.
    The longer reference is a reproducible coupling for numerical comparisons,
    not the production generator's periodic noise recipe.
    """
    def __init__(self, cases: dict[str, DataConfig], current: DataConfig, frame: int):
        self.current = current
        self.left = max(c.guard_symbols+c.pilot_symbols+c.context_symbols for c in cases.values())
        self.right = max(c.scored_symbols+c.context_symbols+c.pilot_symbols+c.guard_symbols for c in cases.values())
        self.reference = replace(current.link, sps=8)
        self.sequence = seed_sequence(current, "pilot", frame, "ase")
        self.spans = self.sequence.spawn(current.link.n_spans)
        reference_samples = (self.left+self.right)*8
        reference_df = self.reference.sample_rate_hz/reference_samples
        half_modes = self.reference.ase_bandwidth_hz/(2*reference_df)
        active_bins = int(np.ceil(half_modes)-np.ceil(-half_modes))
        self.recipe = {"reference_DP_times": self.left+self.right,
            "reference_sample_rate_hz": self.reference.sample_rate_hz,
            "reference_fft_spacing_hz": reference_df, "reference_active_bins": active_bins,
            "reference_complex_variance_per_pol": edfa_ase_psd(self.reference)*reference_df*active_bins,
            "reference_sps": 8, "scored_origin_DP_index": self.left,
            "master_entropy": current.master_entropy, "spawn_key": list(self.sequence.spawn_key),
            "physical_bandwidth_hz": current.link.ase_bandwidth_hz,
            "algorithm": "reference8sps_fixed_band_then_crop_decimate_v1"}

    def __call__(self, span: int, n_samples: int, link: Any) -> np.ndarray:
        if 8 % link.sps or link.symbol_rate_hz != self.reference.symbol_rate_hz:
            raise ValueError("Common noise requires integer 8/sps and the same physical rate")
        rng = np.random.Generator(np.random.PCG64(self.spans[span]))
        reference = ase_noise((self.left+self.right)*8, self.reference, rng)
        origin = self.current.guard_symbols+self.current.pilot_symbols+self.current.context_symbols
        start = (self.left-origin)*8
        result = reference[start:start+n_samples*(8//link.sps):8//link.sps]
        if len(result) != n_samples:
            raise ValueError("Common physical noise reference is too short")
        return result


def analytic_controls(condition: int, *, backend: str = "numpy") -> dict[str, Any]:
    """Small analytic/statistical limits; distinct from full-link refinement."""
    base = main16() if condition == 16 else main64()
    rng = np.random.Generator(np.random.PCG64(2026102544+condition))
    bits = rng.integers(0,2,(65536,2,int(np.log2(condition))),dtype=np.uint8)
    tx = bits_to_symbols(bits,condition)
    all_bits = ((np.arange(condition)[:,None] >> np.arange(int(np.log2(condition))-1,-1,-1))&1).astype(np.uint8)
    alphabet = bits_to_symbols(all_bits,condition)
    mapping = np.array_equal(symbols_to_bits(alphabet,condition),all_bits) and abs(float(np.mean(abs(alphabet)**2))-1) < 1e-14
    variance = 10**(-14/10)
    noise = np.sqrt(variance/2)*(rng.normal(size=tx.shape)+1j*rng.normal(size=tx.shape))
    awgn = compute_metrics(tx,tx+noise,bits,order=condition)
    lo,hi = ber_interval(awgn["bit_errors"],awgn["n_bits"],.9999)
    theory = gray_qam_awgn_ber(14,condition)
    link = replace(base.link,n_spans=2,span_km=1,gamma_per_w_km=0,ase_enabled=False)
    physical = pulse_shape(tx[:2048],link)*np.sqrt(1e-3/2)
    pre = apply_dispersion(physical,link,-link.total_length_km/2)
    cpu,_ = propagate_link_adaptive(pre,link,37,backend="numpy")
    recovered = apply_dispersion(cpu,link,-link.total_length_km/2)
    cdc_error = float(np.max(abs(recovered-physical)))
    matched,_ = receiver_dsp(recovered,link,0,compensate_dispersion=False)
    b2b = compute_metrics(tx[:2048],matched,bits[:2048],order=condition)
    noise = ase_noise(65536,base.link,np.random.Generator(np.random.PCG64(53)))
    expected = edfa_ase_psd(base.link)*128e9
    noise_variance = np.mean(abs(noise)**2,axis=0)
    ase_error = float(np.max(abs(noise_variance/expected-1)))
    quadrature_error = float(max(np.max(abs(np.var(noise.real,axis=0)/(expected/2)-1)),np.max(abs(np.var(noise.imag,axis=0)/(expected/2)-1))))
    kerr_link = replace(base.link,n_spans=1,span_km=1,dispersion_ps_nm_km=0,ase_enabled=False)
    constant = np.tile([np.sqrt(.001),1j*np.sqrt(.001)],(128,1))
    actual,audit = propagate_link_adaptive(constant,kerr_link,73)
    effective = -np.expm1(-kerr_link.alpha_per_km*kerr_link.span_km)/kerr_link.alpha_per_km
    expected_field = constant*np.exp(1j*(8/9)*kerr_link.gamma_per_w_km*.002*effective)
    kerr_error = float(np.max(abs(actual-expected_field)))
    parity = {"status": "not_run", "reason": "CuPy backend must be explicitly requested under shared GPU lock"}
    if backend == "cupy":
        gpu,_ = propagate_link_adaptive(pre,link,37,backend="cupy")
        parity = {"status": "passed" if np.max(abs(gpu-cpu)) < 2e-12 else "failed", "max_abs_error": float(np.max(abs(gpu-cpu)))}
    passed = mapping and lo<theory<hi and cdc_error<1e-12 and b2b["bit_errors"]==0 and b2b["mse"]<1e-5 and ase_error<.04 and quadrature_error<.05 and kerr_error<1e-12
    return {"status": "passed" if passed else "failed", "mapping_power": bool(mapping),
            "AWGN": {**awgn,"theoretical_ber":theory,"counting_9999_interval":[lo,hi]},
            "CDC_max_abs_error":cdc_error,"B2B":b2b,"ASE_relative_variance_error":ase_error,
            "ASE_relative_quadrature_error":quadrature_error,"Kerr_max_abs_error":kerr_error,
            "Kerr_phase_audit":audit.spans[0].max_phase_rad,"CPU_GPU":parity}


def _run_qualification(condition: int, output_dir: str | Path, *, backend: str = "numpy",
                      cases: tuple[str, ...] = CASES, frames: int = 3, resume: bool = False) -> dict[str, Any]:
    """Execute only explicitly selected bounded cases; a partial run cannot pass.

    Results and configs are immutable per source/config; additional cases can
    be resumed. Calling with all default cases performs18 full-link propagations
    for one condition and must be launched only after a resource pilot.
    """
    if not cases or set(cases)-set(CASES) or frames != 3:
        raise ValueError("Qualification requires named cases and exactly3 independent frames")
    output = Path(output_dir)
    ensure_bulk_ignored(output)
    output.mkdir(parents=True,exist_ok=True)
    report_path = output/"qualification.json"
    source = source_identity()
    configs = qualification_cases(condition)
    contract = scientific_contract(configs["baseline"])
    environment = environment_identity(backend)
    runtime = numeric_runtime(environment, backend)
    if report_path.exists():
        if not resume:
            raise FileExistsError("Qualification exists; use --resume for additional cases")
        report = json.loads(report_path.read_text())
        if (report.get("schema") != QUALIFICATION_SCHEMA_VERSION
            or report.get("source_scope_version") != SOURCE_SCOPE_VERSION
            or report["source_content_sha256"] != source["content_sha256"] or report["condition"] != condition
            or report.get("scientific_contract") != contract
            or report.get("scientific_contract_sha256") != canonical_hash(contract)
            or report.get("numeric_runtime") != runtime):
            raise ValueError("Qualification resume schema/source/scientific contract/numeric runtime/backend mismatch")
    else:
        report = {"schema":QUALIFICATION_SCHEMA_VERSION,"status":"incomplete","condition":condition,
            "source_scope_version":SOURCE_SCOPE_VERSION,
            "source_content_sha256":source["content_sha256"],"source":source,"backend":backend,"environment":environment,
            "numeric_runtime":runtime,"scientific_contract":contract,"scientific_contract_sha256":canonical_hash(contract),
            "physical_config_sha256":canonical_hash(configs["baseline"].link.to_dict()),
            "phase_max_deg":configs["baseline"].phase_max_deg,"plan":qualification_plan((condition,),frames),
            "analytic_controls":analytic_controls(condition,backend=backend),"frames":{},"comparisons":{}}
    atomic_json(report_path,report)
    bulk = output/"bulk"; bulk.mkdir(exist_ok=True)
    # All cases use the complete reference geometry, even when executing a subset.
    for case in cases:
        for index in range(frames):
            key = f"{case}-{index}"
            if key in report["frames"]:
                entry = report["frames"][key]
                if file_hash(output/entry["path"]) != entry["file_sha256"]:
                    raise ValueError("Qualification frame hash mismatch")
                continue
            config = configs[case]
            noise = CommonPhysicalNoise(configs,config,index)
            try:
                frame = simulate_physical_frame(config,"pilot",index,backend=backend,ase_source=noise,noise_recipe=noise.recipe)
            except BaseException as exc:
                report.setdefault("failed_attempts", []).append({"case":case,"frame":index,"exception":type(exc).__name__,"message":str(exc)})
                report["status"] = "incomplete"
                atomic_json(report_path,report)
                raise
            path = bulk/f"{key}.npz"
            atomic_npz(path,frame.arrays)
            report["frames"][key] = {"case":case,"frame":index,"path":str(path.relative_to(output)),
                "file_sha256":file_hash(path),"metadata":frame.metadata}
            atomic_json(report_path,report)
    if all(f"baseline-{i}" in report["frames"] for i in range(frames)):
        baseline = [load_frame(output/report["frames"][f"baseline-{i}"]["path"],expected_sha256=report["frames"][f"baseline-{i}"]["file_sha256"]) for i in range(frames)]
        fit = fit_receiver(configs["baseline"],[f.pilots for f in baseline])
        fit = select_receiver_window(fit,configs["baseline"],[(f.pilots,f.arrays["tx_symbols"],f.arrays["tx_bits"],f.arrays["scoring_mask"]) for f in baseline])
        report["receiver"] = {"bps_window":fit.bps_window,"selection":fit.validation_selection,"fit":fit.fit_metadata,
            "policy":"dedicated qualification pilots; preliminary window selected on dedicated pilot scored data, never production/test"}
        for entry in report["frames"].values():
            frame = load_frame(output/entry["path"],expected_sha256=entry["file_sha256"])
            rx,valid,calibration = fit.apply(frame.pilots)
            common = np.flatnonzero(frame.arrays["scoring_mask"])[:65536]
            mask = np.zeros(len(rx),bool); mask[common] = True
            if np.any(mask & ~valid):
                raise ValueError("Qualification scoring exceeds valid receiver support")
            entry["metrics"] = compute_metrics(frame.arrays["tx_symbols"],rx,frame.arrays["tx_bits"],mask,order=condition)
            entry["calibration"] = calibration
            entry["common_scored_relative_index"] = [0,65536]
        for case in CASES[1:]:
            if not all(f"{case}-{i}" in report["frames"] for i in range(frames)):
                continue
            deltas, ber_deltas, count_vars = [], [], []
            for i in range(frames):
                base = report["frames"][f"baseline-{i}"]["metrics"]
                changed = report["frames"][f"{case}-{i}"]["metrics"]
                deltas.append(changed["effective_snr_db"]-base["effective_snr_db"])
                ber_deltas.append(changed["ber"]-base["ber"])
                count_vars.append(sum(m["ber"]*(1-m["ber"])/m["n_bits"] for m in (base,changed)))
                # Prove the scored bits are identical at the compared indices.
                b = baseline[i].arrays
                other = load_frame(output/report["frames"][f"{case}-{i}"]["path"]).arrays
                if not np.array_equal(b["tx_bits"][b["scoring_mask"]],other["tx_bits"][other["scoring_mask"]][:65536]):
                    raise ValueError("Qualification common scored bits differ")
            counting_se = float(np.sqrt(sum(count_vars))/frames)
            frame_se = float(np.std(ber_deltas,ddof=1)/np.sqrt(frames))
            tolerance = 3*max(counting_se,frame_se)
            passed = abs(float(np.mean(deltas)))<=.1 and abs(float(np.mean(ber_deltas)))<=tolerance
            report["comparisons"][case] = {"status":"passed" if passed else "failed",
                "effective_snr_difference_db_per_frame":deltas,"mean_effective_snr_difference_db":float(np.mean(deltas)),
                "BER_difference_per_frame":ber_deltas,"mean_BER_difference":float(np.mean(ber_deltas)),
                "BER_counting_SE":counting_se,"BER_frame_mean_SE":frame_se,"BER_3SE_tolerance":tolerance,
                "noise_coupling":"same reference bandlimited process at matching scored physical times"}
    complete = len(report["comparisons"]) == 5 and len(report["frames"]) == 18
    controls = report["analytic_controls"]
    if complete:
        passed = all(row["status"]=="passed" for row in report["comparisons"].values()) and controls["status"]=="passed" and controls["CPU_GPU"]["status"]=="passed"
        report["status"] = "passed" if passed else "not_qualified"
    report["limitations"] = ["Numerical precision qualification only; not held-out testing or paper-performance reproduction",
        "Counting intervals condition on independent-bit assumptions; independent-frame differences are retained",
        "0.1dB is a numerical convergence guide, not a paper reproduction acceptance threshold"]
    atomic_json(report_path,report)
    return report


def run_qualification(condition: int, output_dir: str | Path, *, backend: str = "numpy",
                      cases: tuple[str, ...] = CASES, frames: int = 3, resume: bool = False) -> dict[str, Any]:
    """Lock the run and execute only the explicitly requested bounded cases."""
    output = Path(output_dir)
    ensure_bulk_ignored(output)
    output.mkdir(parents=True,exist_ok=True)
    with run_lock(output):
        return _run_qualification(condition,output,backend=backend,cases=cases,frames=frames,resume=resume)


def resource_benchmark(condition: int, *, backend: str = "numpy", scored_symbols: int = 2048,
                       spans: int = 2) -> dict[str, Any]:
    """A short explicit pilot for timing/storage; extrapolation is disclosed."""
    from experiments.transformer_nlc.data_config import pilot_config
    if not isinstance(spans,int) or spans < 1:
        raise ValueError("Resource pilot needs positive span count")
    config = pilot_config(condition,scored_symbols=scored_symbols)
    config = replace(config,link=replace(config.link,n_spans=spans))
    start = time.perf_counter()
    frame = simulate_physical_frame(config,"pilot",0,backend=backend)
    elapsed = time.perf_counter()-start
    steps = sum(span["count"] for span in frame.metadata["propagation_audit"]["spans"])
    base = qualification_cases(condition)["baseline"]
    # FFT NlogN and distance scaling are estimates, not a promised runtime.
    current_n, target_n = config.n_symbols*config.link.sps, base.n_symbols*base.link.sps
    scale = (target_n*np.log2(target_n))/(current_n*np.log2(current_n))*base.link.n_spans/spans
    return {"regime":"resource_pilot_not_qualification","config":config.to_dict(),"backend":backend,
        "environment":environment_identity(backend),"elapsed_seconds":elapsed,"accepted_steps":steps,"physical_seconds":frame.metadata["physical_seconds"],
        "array_bytes":sum(a.nbytes for a in frame.arrays.values()),
        "estimated_baseline_frame_seconds":elapsed*scale,
        "estimated18_propagation_seconds":elapsed*scale*18,
        "estimate_assumptions":"FFT NlogN and span count scaling only; adaptive peak/step count, GPU occupancy, longer-case/noise-reference cost are not modeled",
        "propagation_audit":frame.metadata["propagation_audit"]}
