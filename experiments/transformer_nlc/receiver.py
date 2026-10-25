"""Study policy for frozen train-pilot FIR and validation-selected blind CR."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import numpy as np
from optical_comm.dsp import (apply_fir, blind_phase_search, fit_pilot_fir, pilot_gain_calibration)
from optical_comm.metrics import compute_metrics
from experiments.transformer_nlc.data_config import DataConfig


@dataclass(frozen=True)
class PilotFrame:
    rx_matched: np.ndarray
    pilot_targets: np.ndarray
    pilot_mask: np.ndarray


@dataclass(frozen=True)
class ReceiverFit:
    weights: np.ndarray
    order: int
    bps_window: int
    bps_phases: int
    fit_metadata: dict[str, Any]
    validation_selection: tuple[dict[str, Any], ...] = ()

    def apply(self, frame: PilotFrame) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """No scored TX symbols/bits are accepted by the receiver interface."""
        equalized, fir_valid = apply_fir(frame.rx_matched, self.weights)
        # Gain before BPS places the decision metric on the known unit-Es scale.
        usable_pilots = frame.pilot_mask & fir_valid
        targets = np.where(usable_pilots[:, None], frame.pilot_targets, 0)
        calibrated, pre_gain = pilot_gain_calibration(equalized, targets, usable_pilots)
        recovered, phase, cr_valid = blind_phase_search(calibrated, self.order,
            window=self.bps_window, n_phases=self.bps_phases)
        usable_pilots &= cr_valid
        targets = np.where(usable_pilots[:, None], frame.pilot_targets, 0)
        recovered, post_gain = pilot_gain_calibration(recovered, targets, usable_pilots)
        return recovered, fir_valid & cr_valid, {
            "pre_bps_gain": [[float(x.real), float(x.imag)] for x in pre_gain],
            "post_bps_gain": [[float(x.real), float(x.imag)] for x in post_gain],
            "pilot_rows": int(usable_pilots.sum()),
            "phase_min_rad": phase.min(axis=0).tolist(), "phase_max_rad": phase.max(axis=0).tolist(),
            "calibration_labels": "dedicated_frame_pilots_only", "normalization": "known_nominal_XY_power_then_pilot_LS"}


def fit_receiver(config: DataConfig, training_pilots: list[PilotFrame]) -> ReceiverFit:
    """Fit the 2×2 FIR on dedicated train pilots; initial CR window is provisional."""
    weights, metadata = fit_pilot_fir([f.rx_matched for f in training_pilots],
        [f.pilot_targets for f in training_pilots], [f.pilot_mask for f in training_pilots],
        taps=config.equalizer_taps, ridge_relative=config.ridge_relative)
    weights.setflags(write=False)
    return ReceiverFit(weights, config.condition, 65 if 65 in config.bps_windows else config.bps_windows[0],
        config.bps_phases, {**metadata, "fit_split": "train", "labels": "dedicated_pilots_only"})


def select_receiver_window(
    fit: ReceiverFit, config: DataConfig,
    validation: list[tuple[PilotFrame, np.ndarray, np.ndarray, np.ndarray]],
) -> ReceiverFit:
    """Select ≤3 CR windows by aggregate validation BER, then MSE; never test."""
    from dataclasses import replace
    if not validation:
        raise ValueError("Independent validation frames are required to fix CR")
    records = []
    for window in config.bps_windows:
        candidate = replace(fit, bps_window=window)
        errors = n_bits = n_rows = 0
        squared_error = 0.
        for pilots, tx, bits, scoring in validation:
            rx, valid, _ = candidate.apply(pilots)
            if np.any(scoring & ~valid):
                raise ValueError("Scored region exceeds DSP-valid support")
            metrics = compute_metrics(tx, rx, bits, scoring, order=config.condition)
            errors += metrics["bit_errors"]
            n_bits += metrics["n_bits"]
            squared_error += metrics["mse"] * int(scoring.sum())
            n_rows += int(scoring.sum())
        records.append({"window": window, "bit_errors": errors, "n_bits": n_bits,
                        "ber": errors / n_bits, "mse": squared_error / n_rows})
    best = min(records, key=lambda row: (row["ber"], row["mse"], row["window"]))
    return replace(fit, bps_window=best["window"], validation_selection=tuple(records))
