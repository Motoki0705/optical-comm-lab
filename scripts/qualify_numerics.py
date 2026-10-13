#!/usr/bin/env python3
"""Reproducible numerical qualification; these are not held-out model scores.

Only qualification-specific orchestration lives here. Physical propagation,
receiver DSP, demapping and uncertainty use the owning package modules.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import itertools
import json
import math
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
from scipy.special import ndtr
from scipy.stats import binomtest

from optical_comm.channel import (SimulatedFrame, ase_noise, dbp_symbols,
                                  propagate_fiber, simulate_frame)
from optical_comm.config import LinkConfig
from optical_comm.dsp import pulse_shape, receiver_dsp
from optical_comm.metrics import ber_interval, compute_metrics, symbols_to_bits


ROOT = Path(__file__).resolve().parents[1]


def json_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def content_hash(value) -> str:
    return hashlib.sha256(json_bytes(value)).hexdigest()


def seed(root: int, *namespace) -> int:
    return int.from_bytes(hashlib.sha256(json_bytes([root, *namespace])).digest()[:8], "little")


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def guarded_mask(n_symbols: int, guard: int) -> np.ndarray:
    if guard < 0 or 2 * guard >= n_symbols:
        raise ValueError("guard must leave at least one scored symbol")
    mask = np.ones(n_symbols, dtype=bool)
    if guard:
        mask[:guard] = False
        mask[-guard:] = False
    return mask


def forward_training_gain(tx: np.ndarray, rx: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fit RX=h*TX on a separate calibration frame, independently per XY.

    Inverting this forward LS gain avoids the MMSE contraction obtained by
    fitting TX=g*RX. No evaluation labels enter the estimated gain.
    """
    if tx.shape != rx.shape or tx.ndim != 2 or tx.shape[1] != 2:
        raise ValueError("TX/RX must have the same (time,2) shape")
    if mask.dtype != bool or mask.shape != (len(tx),) or not np.any(mask):
        raise ValueError("calibration mask must select training symbols")
    numerator = np.sum(rx[mask] * tx[mask].conj(), axis=0)
    denominator = np.sum(np.abs(tx[mask]) ** 2, axis=0)
    gain = numerator / denominator
    if not np.all(np.isfinite(gain)) or np.any(np.abs(gain) < 1e-12):
        raise ValueError("singular forward calibration gain")
    return gain


def gain_record(gain: np.ndarray) -> dict:
    return {"forward_h_real": gain.real.tolist(), "forward_h_imag": gain.imag.tolist(),
            "convention": "RX=h*TX fit on independent calibration; evaluation RX/h"}


def score(frame: SimulatedFrame, gain: np.ndarray, mask: np.ndarray | None = None) -> dict:
    mask = frame.valid_mask if mask is None else mask
    prediction = frame.rx_symbols / gain
    row = compute_metrics(frame.tx_symbols, prediction, frame.tx_bits, mask)
    row["signal_energy"] = float(np.sum(np.abs(frame.tx_symbols[mask]) ** 2))
    row["error_energy"] = float(np.sum(np.abs(prediction[mask] - frame.tx_symbols[mask]) ** 2))
    return row


def pool(rows: list[dict]) -> dict:
    errors = sum(row["bit_errors"] for row in rows)
    bits = sum(row["n_bits"] for row in rows)
    signal = sum(row["signal_energy"] for row in rows)
    error = sum(row["error_energy"] for row in rows)
    lo, hi = ber_interval(errors, bits)
    return {"bit_errors": errors, "n_bits": bits, "ber": errors / bits,
            "ber_ci_low": lo, "ber_ci_high": hi, "signal_energy": signal,
            "error_energy": error, "evm": math.sqrt(error / signal),
            "effective_snr_db": 10 * math.log10(signal / error), "n_frames": len(rows)}


def independent_gray16qam_theory(snr_db: float) -> float:
    """Independent exact decision-region integration, not the package formula.

    Enumerate four equiprobable real 4PAM levels, all decision bins and the
    two-bit Gray-label Hamming distances. I and Q have the same bit BER.
    """
    levels = np.array([-3, -1, 1, 3], dtype=float) / math.sqrt(10)
    labels = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    edges = np.array([-math.inf, -2, 0, 2, math.inf], dtype=float) / math.sqrt(10)
    sigma = math.sqrt(10 ** (-snr_db / 10) / 2)
    total = 0.0
    for origin, decision in itertools.product(range(4), repeat=2):
        probability = (ndtr((edges[decision + 1] - levels[origin]) / sigma)
                       - ndtr((edges[decision] - levels[origin]) / sigma))
        distance = np.count_nonzero(labels[origin] != labels[decision])
        total += float(probability) * int(distance) / 8
    return total


def independent_edfa_psd(config: LinkConfig) -> float:
    gain = 10 ** (config.alpha_db_per_km * config.span_km / 10)
    photon = 6.62607015e-34 * 299792458.0 / config.wavelength_m
    return 0.0 if gain == 1 else photon * (gain * 10 ** (config.edfa_noise_figure_db / 10) - 1) / 2


def timed_frame(config, power, n_symbols, bit_seed, noise_seed, backend):
    start = time.perf_counter()
    frame = simulate_frame(config, power, n_symbols, bit_seed, noise_seed, backend=backend)
    return frame, time.perf_counter() - start


def compare_frames(left: SimulatedFrame, right: SimulatedFrame, left_gain: np.ndarray,
                   right_gain: np.ndarray, mask: np.ndarray, confidence: float) -> dict:
    """Same transmitted symbols; raw waveform and trained-calibration metrics.

    The right record can be longer but must contain the same TX prefix. Higher
    sampling is compared at identical physical time instants, never interpolated
    by fitting labels or aligning a test-dependent phase.
    """
    n = len(left.tx_symbols)
    np.testing.assert_array_equal(left.tx_symbols, right.tx_symbols[:n])
    if right.config.sps % left.config.sps:
        raise ValueError("right sps must be an integer multiple of left sps")
    ratio = right.config.sps // left.config.sps
    right_raw = right.rx_waveform[: n * right.config.sps : ratio]
    sample_mask = np.repeat(mask, left.config.sps)
    raw_difference = left.rx_waveform[sample_mask] - right_raw[sample_mask]
    raw_reference_energy = float(np.sum(np.abs(right_raw[sample_mask]) ** 2))
    raw_relative = math.sqrt(float(np.sum(np.abs(raw_difference) ** 2)) / raw_reference_energy)
    a = left.rx_symbols / left_gain
    b = right.rx_symbols[:n] / right_gain
    pair_evm = math.sqrt(float(np.sum(np.abs(a[mask] - b[mask]) ** 2))
                         / float(np.sum(np.abs(left.tx_symbols[mask]) ** 2)))
    ma = score(left, left_gain, mask)
    mb = compute_metrics(left.tx_symbols, b, left.tx_bits, mask)
    alo, ahi = ber_interval(ma["bit_errors"], ma["n_bits"], confidence)
    blo, bhi = ber_interval(mb["bit_errors"], mb["n_bits"], confidence)
    a_error = symbols_to_bits(a[mask]) != left.tx_bits[mask]
    b_error = symbols_to_bits(b[mask]) != left.tx_bits[mask]
    only_a = int(np.count_nonzero(a_error & ~b_error))
    only_b = int(np.count_nonzero(b_error & ~a_error))
    discordant = only_a + only_b
    paired_p = 1.0 if not discordant else float(binomtest(only_a, discordant, 0.5).pvalue)
    return {"left_effective_snr_db": ma["effective_snr_db"],
            "right_effective_snr_db": mb["effective_snr_db"],
            "snr_difference_db": ma["effective_snr_db"] - mb["effective_snr_db"],
            "left_bit_errors": ma["bit_errors"], "right_bit_errors": mb["bit_errors"],
            "n_bits": ma["n_bits"], "left_ber": ma["ber"], "right_ber": mb["ber"],
            "ber_intervals_overlap": max(alo, blo) <= min(ahi, bhi),
            "ber_interval_confidence": confidence, "paired_only_left_errors": only_a,
            "paired_only_right_errors": only_b, "paired_ber_pvalue": paired_p,
            "paired_ber_consistent": paired_p >= 1 - confidence,
            "raw_waveform_relative_rms_difference": raw_relative,
            "calibrated_symbol_pair_evm": pair_evm,
            "n_scored_symbols": int(np.count_nonzero(mask))}


def run_controls(spec: dict, output: Path):
    print("controls: full-waveform back-to-back and linear EDFA/AWGN", flush=True)
    base = LinkConfig.from_dict(spec["link"])
    control = replace(base, n_spans=1, step_km=80, gamma_per_w_km=0,
                      dispersion_ps_nm_km=0, guard_symbols=256)
    n = spec["n_symbols"]
    root = spec["root_seed"]
    rows, summaries = [], []
    back, seconds = timed_frame(replace(control, n_spans=0), 0, n,
                                seed(root, "b2b", "bits"), seed(root, "b2b", "ase"), "numpy")
    b2b = score(back, np.ones(2))
    b2b.update(runtime_seconds=seconds, accepted=b2b["bit_errors"] == 0 and b2b["evm"] < 0.002)
    psd = independent_edfa_psd(control)
    for snr_db in spec["awgn_snr_db"]:
        power_per_pol = psd * control.symbol_rate_hz * 10 ** (snr_db / 10)
        power_dbm = 10 * math.log10(2 * power_per_pol / 1e-3)
        entries = []
        for repeat in range(spec["awgn_repeats"]):
            bits = seed(root, "awgn", snr_db, repeat, "bits")
            noise = seed(root, "awgn", snr_db, repeat, "ase")
            frame, seconds = timed_frame(control, power_dbm, n, bits, noise, "numpy")
            entry = dict(target_esn0_db=snr_db, launch_power_dbm=power_dbm, repeat=repeat,
                         bit_seed=bits, noise_seed=noise, runtime_seconds=seconds, **score(frame, np.ones(2)))
            rows.append(entry)
            entries.append(entry)
        summary = dict(target_esn0_db=snr_db, launch_power_dbm=power_dbm, **pool(entries))
        theory = independent_gray16qam_theory(snr_db)
        lo, hi = ber_interval(summary["bit_errors"], summary["n_bits"], spec["ber_consistency_confidence"])
        summary.update(theoretical_ber=theory, independent_expected_complex_noise_variance=10 ** (-snr_db / 10),
                       theory_inside_counting_interval=lo <= theory <= hi,
                       qualification_ci_low=lo, qualification_ci_high=hi,
                       snr_difference_db=summary["effective_snr_db"] - snr_db)
        summary["accepted"] = (summary["theory_inside_counting_interval"]
                               and abs(summary["snr_difference_db"]) <= spec["snr_tolerance_db"])
        summaries.append(summary)
    result = {"kind": "numerical_verification_control", "not_held_out_testing": True,
              "calibration": "none; known terminal power normalization", "config": control.to_dict(),
              "independent_psd_w_per_hz_per_pol": psd, "back_to_back": b2b,
              "theory_method": "Exact independent 4PAM Gray-label decision-bin integration",
              "confidence_caveat": "Counting intervals assume independent bits; repetitions are also retained",
              "frames": rows, "summary": summaries,
              "all_accepted": b2b["accepted"] and all(row["accepted"] for row in summaries)}
    write_json(output / "controls.json", result)
    write_csv(output / "controls_frames.csv", rows)
    write_csv(output / "controls_summary.csv", summaries)
    print(f"controls accepted={result['all_accepted']}", flush=True)


def run_refinement(spec: dict, output: Path):
    base = LinkConfig.from_dict(spec["link"])
    root, backend, n = spec["root_seed"], spec["backend"], spec["n_symbols"]
    cases = {entry["name"]: entry for entry in spec["refinement_cases"]}
    rows, comparisons, calibrations, summaries = [], [], [], []
    for power, ase in itertools.product(spec["powers_dbm"], spec["ase_modes"]):
        print(f"refinement {power:+g} dBm ASE={ase}: independent calibration", flush=True)
        gains = {}
        calibration_bit = seed(root, "refinement", power, ase, "calibration", "bits")
        calibration_noise = seed(root, "refinement", power, ase, "calibration", "ase")
        configurations = {}
        for name, entry in cases.items():
            config = replace(base, ase_enabled=ase, sps=entry["sps"], step_km=entry["step_km"])
            configurations[name] = config
            frame, seconds = timed_frame(config, power, spec["calibration_symbols"],
                                         calibration_bit, calibration_noise, backend)
            gains[name] = forward_training_gain(frame.tx_symbols, frame.rx_symbols, frame.valid_mask)
            calibrations.append(dict(case=name, power_dbm=power, ase_enabled=ase,
                                     bit_seed=calibration_bit, noise_seed=calibration_noise,
                                     n_symbols=spec["calibration_symbols"], runtime_seconds=seconds,
                                     **gain_record(gains[name])))
        repeats = spec["noisy_repeats"] if ase else spec["deterministic_repeats"]
        for repeat in range(repeats):
            print(f"refinement {power:+g} dBm ASE={ase}: repeat {repeat + 1}/{repeats}", flush=True)
            frames = {}
            bits = seed(root, "refinement", power, ase, "evaluation", repeat, "bits")
            noise = seed(root, "refinement", power, ase, "evaluation", repeat, "ase")
            assert bits != calibration_bit and noise != calibration_noise
            for name, config in configurations.items():
                frame, seconds = timed_frame(config, power, n, bits, noise, backend)
                frames[name] = frame
                raw = compute_metrics(frame.tx_symbols, frame.rx_symbols, frame.tx_bits, frame.valid_mask)
                row = dict(case=name, power_dbm=power, ase_enabled=ase, repeat=repeat, sps=config.sps,
                           step_km=config.step_km, bit_seed=bits, noise_seed=noise, runtime_seconds=seconds,
                           raw_effective_snr_db=raw["effective_snr_db"], raw_ber=raw["ber"],
                           raw_waveform_sha256=hashlib.sha256(frame.rx_waveform.tobytes()).hexdigest(),
                           **score(frame, gains[name]))
                rows.append(row)
                if repeat == 0:
                    middle = n // 2
                    end = middle + spec["diagnostic_symbols"]
                    np.savez_compressed(output / f"diagnostic_{power:+g}dBm_ase{int(ase)}_{name}.npz",
                                        tx=frame.tx_symbols[middle:end], rx=frame.rx_symbols[middle:end],
                                        raw_at_symbol_times=frame.rx_waveform[middle * config.sps:end * config.sps:config.sps],
                                        forward_training_gain=gains[name])
            for left_name, right_name, kind in spec["refinement_pairs"]:
                left, right = frames[left_name], frames[right_name]
                result = compare_frames(left, right, gains[left_name], gains[right_name],
                                        left.valid_mask, spec["ber_consistency_confidence"])
                result.update(power_dbm=power, ase_enabled=ase, repeat=repeat, kind=kind,
                              left_case=left_name, right_case=right_name)
                result["snr_target_met"] = abs(result["snr_difference_db"]) <= spec["snr_tolerance_db"]
                result["accepted"] = result["snr_target_met"] and result["ber_intervals_overlap"]
                comparisons.append(result)
            # Finest, independently discretized inverse checks supplement raw
            # forward differences; same-grid DBP cancellation is not the criterion.
            if not ase and repeat == 0:
                for name in ("sps4_step1", "sps4_step0p25"):
                    frame = frames[name]
                    start = time.perf_counter()
                    inverse = dbp_symbols(frame, step_km=0.125, backend=backend)
                    metric = compute_metrics(frame.tx_symbols, inverse, frame.tx_bits, frame.valid_mask)
                    write_json(output / f"independent_inverse_{power:+g}dBm_{name}.json",
                               {"forward_case": name, "forward_config": frame.config.to_dict(),
                                "inverse_step_km": 0.125, "calibration": "none", "metrics": metric,
                                "runtime_seconds": time.perf_counter() - start})
            del frames
            write_json(output / "refinement.json", {"frames": rows, "comparisons": comparisons,
                                                     "calibration": calibrations})
    for power, ase, name in itertools.product(spec["powers_dbm"], spec["ase_modes"], cases):
        entries = [row for row in rows if row["power_dbm"] == power and row["ase_enabled"] == ase and row["case"] == name]
        summaries.append(dict(power_dbm=power, ase_enabled=ase, case=name, **pool(entries)))
    pair_summary = []
    for power, ase, pair in itertools.product(spec["powers_dbm"], spec["ase_modes"], spec["refinement_pairs"]):
        left, right, kind = pair
        entries = [row for row in comparisons if row["power_dbm"] == power and row["ase_enabled"] == ase
                   and row["left_case"] == left and row["right_case"] == right]
        pair_summary.append(dict(power_dbm=power, ase_enabled=ase, left_case=left, right_case=right, kind=kind,
            mean_snr_difference_db=float(np.mean([row["snr_difference_db"] for row in entries])),
            maximum_abs_snr_difference_db=max(abs(row["snr_difference_db"]) for row in entries),
            mean_raw_waveform_relative_rms_difference=float(np.mean([row["raw_waveform_relative_rms_difference"] for row in entries])),
            mean_calibrated_symbol_pair_evm=float(np.mean([row["calibrated_symbol_pair_evm"] for row in entries])),
            all_ber_intervals_overlap=all(row["ber_intervals_overlap"] for row in entries),
            all_paired_ber_consistent=all(row["paired_ber_consistent"] for row in entries),
            accepted=all(row["accepted"] for row in entries), n_repeats=len(entries)))
    write_json(output / "refinement.json", {"kind": "numerical_qualification_not_held_out_testing",
              "configs": {name: config.to_dict() for name, config in configurations.items()},
              "noise_coupling": "Same ordered physical-frequency ASE draws, fixed two-sided 128 GHz band",
              "gain_fitting": "Separate calibration frame per case; no evaluation labels fitted",
              "frames": rows, "comparisons": comparisons, "calibration": calibrations,
              "summary": summaries, "pair_summary": pair_summary,
              "acceptance": "Every repeat <=0.1 dB and overlapping 99.9% counting BER intervals; paired bit test also retained"})
    write_csv(output / "refinement_frames.csv", rows)
    write_csv(output / "refinement_comparisons.csv", comparisons)
    write_csv(output / "refinement_summary.csv", summaries)
    write_csv(output / "refinement_pairs.csv", pair_summary)
    print("refinement complete", flush=True)


def common_noise_length_frames(config: LinkConfig, power: float, lengths: list[int],
                               bit_seed: int, noise_seed: int, backend: str) -> tuple[dict, dict]:
    """Common TX prefix and common physical ASE crop, with no shared test fit.

    Short-period crops introduce an end discontinuity. This deliberately tests
    periodic-block perturbations; it is not an independent short FFT-grid draw.
    The longest noise period and its per-span seeds are the generation recipe.
    """
    longest = max(lengths)
    tx = simulate_frame(replace(config, n_spans=0), power, longest, bit_seed, noise_seed, backend="numpy")
    fields = {n: pulse_shape(tx.tx_symbols[:n], config) * math.sqrt(1e-3 * 10 ** (power / 10) / 2)
              for n in lengths}
    duration = {n: 0.0 for n in lengths}
    span_sequences = np.random.SeedSequence(noise_seed).spawn(config.n_spans)
    for span in range(config.n_spans):
        noise = (ase_noise(longest * config.sps, config, np.random.default_rng(span_sequences[span]))
                 if config.ase_enabled else None)
        for n in lengths:
            start = time.perf_counter()
            fields[n] = propagate_fiber(fields[n], config, backend=backend) * math.sqrt(config.edfa_gain)
            if noise is not None:
                fields[n] += noise[: n * config.sps]
            duration[n] += time.perf_counter() - start
    frames = {}
    for n in lengths:
        start = time.perf_counter()
        symbols, two_sps = receiver_dsp(fields[n], config, power)
        duration[n] += time.perf_counter() - start
        frames[n] = SimulatedFrame(tx_symbols=tx.tx_symbols[:n].copy(), tx_bits=tx.tx_bits[:n].copy(),
                                  tx_indices=tx.tx_indices[:n].copy(), rx_symbols=symbols,
                                  rx_2sps=two_sps, rx_waveform=fields[n],
                                  valid_mask=guarded_mask(n, config.guard_symbols), config=config,
                                  launch_power_dbm=power, bit_seed=bit_seed, noise_seed=noise_seed, backend=backend)
    return frames, duration


def run_boundaries(spec: dict, output: Path):
    base = LinkConfig.from_dict(spec["link"])
    case = next(row for row in spec["refinement_cases"] if row["name"] == spec["boundary_case"])
    base = replace(base, sps=case["sps"], step_km=case["step_km"])
    lengths = spec["frame_lengths"]
    longest = max(lengths)
    rows, comparisons, calibrations, profiles = [], [], [], []
    for power, ase in itertools.product(spec["powers_dbm"], spec["ase_modes"]):
        print(f"periodic lengths/guards {power:+g} dBm ASE={ase}", flush=True)
        config = replace(base, ase_enabled=ase)
        root = spec["root_seed"]
        cal_bits = seed(root, "boundaries", power, ase, "calibration", "bits")
        cal_noise = seed(root, "boundaries", power, ase, "calibration", "ase")
        calibration, seconds = timed_frame(config, power, spec["calibration_symbols"],
                                           cal_bits, cal_noise, spec["backend"])
        gain = forward_training_gain(calibration.tx_symbols, calibration.rx_symbols, calibration.valid_mask)
        calibrations.append(dict(power_dbm=power, ase_enabled=ase, bit_seed=cal_bits,
                                 noise_seed=cal_noise, runtime_seconds=seconds, **gain_record(gain)))
        repeats = spec["boundary_repeats"] if ase else 1
        for repeat in range(repeats):
            bits = seed(root, "boundaries", power, ase, "evaluation", repeat, "bits")
            noise = seed(root, "boundaries", power, ase, "evaluation", repeat, "ase")
            frames, durations = common_noise_length_frames(config, power, lengths, bits, noise, spec["backend"])
            reference = frames[longest]
            for n in lengths:
                frame = frames[n]
                for guard in spec["guards"]:
                    mask = guarded_mask(n, guard)
                    rows.append(dict(power_dbm=power, ase_enabled=ase, repeat=repeat, n_symbols=n,
                                     guard_symbols=guard, bit_seed=bits, noise_seed=noise,
                                     runtime_seconds=durations[n], **score(frame, gain, mask)))
                    if n == longest:
                        continue
                    comparison = compare_frames(frame, reference, gain, gain, mask,
                                                spec["ber_consistency_confidence"])
                    comparison.update(power_dbm=power, ase_enabled=ase, repeat=repeat,
                                      n_symbols=n, reference_n_symbols=longest, guard_symbols=guard)
                    comparison["snr_target_met"] = abs(comparison["snr_difference_db"]) <= spec["snr_tolerance_db"]
                    comparison["accepted"] = comparison["snr_target_met"] and comparison["ber_intervals_overlap"]
                    comparisons.append(comparison)
                if n != longest:
                    distance = np.minimum(np.arange(n), n - 1 - np.arange(n))
                    error = np.mean(np.abs((frame.rx_symbols - reference.rx_symbols[:n]) / gain) ** 2, axis=1)
                    for low, high in ((0, 64), (64, 128), (128, 256), (256, 512),
                                      (512, 1024), (1024, 2048), (2048, n // 2 + 1)):
                        selected = (distance >= low) & (distance < high)
                        if np.any(selected):
                            profiles.append(dict(power_dbm=power, ase_enabled=ase, repeat=repeat,
                                                 n_symbols=n, edge_distance_low_symbols=low,
                                                 edge_distance_high_symbols=high,
                                                 pair_rms_error=float(np.sqrt(np.mean(error[selected]))),
                                                 n_positions=int(np.count_nonzero(selected))))
            write_json(output / "boundaries.json", {"frames": rows, "comparisons": comparisons,
                                                    "calibration": calibrations, "edge_profiles": profiles})
    summary = []
    for power, ase, n, guard in itertools.product(spec["powers_dbm"], spec["ase_modes"],
                                                lengths[:-1], spec["guards"]):
        entries = [row for row in comparisons if row["power_dbm"] == power and row["ase_enabled"] == ase
                   and row["n_symbols"] == n and row["guard_symbols"] == guard]
        summary.append(dict(power_dbm=power, ase_enabled=ase, n_symbols=n, reference_n_symbols=longest,
                            guard_symbols=guard, n_repeats=len(entries),
                            maximum_abs_snr_difference_db=max(abs(row["snr_difference_db"]) for row in entries),
                            mean_raw_waveform_relative_rms_difference=float(np.mean([row["raw_waveform_relative_rms_difference"] for row in entries])),
                            mean_calibrated_symbol_pair_evm=float(np.mean([row["calibrated_symbol_pair_evm"] for row in entries])),
                            all_ber_intervals_overlap=all(row["ber_intervals_overlap"] for row in entries),
                            all_paired_ber_consistent=all(row["paired_ber_consistent"] for row in entries),
                            accepted=all(row["accepted"] for row in entries)))
    write_json(output / "boundaries.json", {"kind": "numerical_qualification_not_held_out_testing",
              "config": base.to_dict(), "lengths": lengths, "guards": spec["guards"],
              "noise_recipe": "Longest-period ordered spectral ASE per span; exact prefix crops shared by shorter periods",
              "caveat": "Cropped noise is periodized with an end discontinuity; it is a boundary perturbation, not a fresh short-grid noise draw",
              "gain_fitting": "One separate calibration frame/gain per power/ASE, shared across all lengths and guards",
              "runtime_caveat": "Per-length durations include propagation and receiver, shared TX/noise generation excluded",
              "frames": rows, "comparisons": comparisons, "calibration": calibrations,
              "edge_profiles": profiles, "summary": summary})
    write_csv(output / "boundaries_frames.csv", rows)
    write_csv(output / "boundaries_comparisons.csv", comparisons)
    write_csv(output / "boundaries_summary.csv", summary)
    write_csv(output / "boundary_edge_profiles.csv", profiles)
    print("length/guard qualification complete", flush=True)


def software_manifest(spec: dict) -> dict:
    sources = [ROOT / "scripts/qualify_numerics.py", ROOT / "configs/numerical.json"]
    sources += [ROOT / "src/optical_comm" / name for name in ("channel.py", "config.py", "dsp.py", "metrics.py")]
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    versions = {}
    for package in ("numpy", "scipy", "cupy-cuda13x", "optical-comm-lab"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    gpu = None
    if spec["backend"] == "cupy":
        import cupy as cp
        gpu = {"name": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
               "driver_version": cp.cuda.runtime.driverGetVersion(),
               "runtime_version": cp.cuda.runtime.runtimeGetVersion(), "dtype": "complex128"}
    return {"created_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
            "platform": platform.platform(), "packages": versions, "gpu": gpu,
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                                          capture_output=True, text=True).stdout.strip(),
            "source_sha256": hashes, "config_hash": content_hash(spec),
            "evaluation_class": "numerical qualification, not independent validator or held-out learning testing"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/numerical.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/numerical-qualification")
    parser.add_argument("--stage", choices=("all", "controls", "refinement", "boundaries"), default="all")
    args = parser.parse_args()
    spec = json.loads(args.config.read_text())
    if spec["schema_version"] != 1:
        raise ValueError("unsupported qualification config schema")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    manifest = software_manifest(spec)
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text())
        if prior["config_hash"] != manifest["config_hash"] or prior["source_sha256"] != manifest["source_sha256"]:
            raise ValueError("refusing to mix qualification artifacts from different config/source versions")
        manifest = prior
    else:
        write_json(output / "config.json", spec)
        write_json(manifest_path, manifest)
    start = time.perf_counter()
    for stage, runner in (("controls", run_controls), ("refinement", run_refinement), ("boundaries", run_boundaries)):
        if args.stage in ("all", stage):
            runner(spec, output)
    duration = time.perf_counter() - start
    completion = {"stage": args.stage, "wall_seconds": duration, "completed_utc": datetime.now(timezone.utc).isoformat()}
    write_json(output / f"completion_{args.stage}.json", completion)
    print(json.dumps(completion), flush=True)


if __name__ == "__main__":
    main()
