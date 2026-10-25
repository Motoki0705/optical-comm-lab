"""Explicit Transformer data recipes; no inherited ESN science defaults."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import math
from typing import Any

from optical_comm.config import LinkConfig

SPLIT_KEYS = {"train": 0, "validation": 1, "test": 2, "pilot": 3}
PURPOSE_KEYS = {"bits": 0, "ase": 1, "model": 2, "shuffle": 3}
MASTER_ENTROPY = 2026102543


@dataclass(frozen=True)
class DataConfig:
    condition: int = 16
    replicate: int = 0
    master_entropy: int = MASTER_ENTROPY
    regime: str = "production"
    launch_power_dbm: float = 2.0
    scored_symbols: int = 65536
    guard_symbols: int = 4096
    pilot_symbols: int = 2048  # each side
    context_symbols: int = 96  # each side
    train_frames: int = 8
    validation_frames: int = 2
    test_frames: int = 4
    phase_max_deg: float = .05
    min_step_km: float = 1e-6
    equalizer_taps: int = 11
    ridge_relative: float = 1e-6
    bps_windows: tuple[int, ...] = (33, 65, 129)
    bps_phases: int = 64
    save_rx_field: bool = False
    save_rx_2sps: bool = False
    link: LinkConfig = field(default_factory=lambda: LinkConfig(
        n_spans=40, span_km=80, alpha_db_per_km=.2,
        dispersion_ps_nm_km=17, gamma_per_w_km=1.3, symbol_rate_hz=32e9,
        wavelength_m=1550e-9, sps=4, step_km=1, rolloff=1/16,
        rrc_span_symbols=128, guard_symbols=4096, edfa_noise_figure_db=6,
        ase_enabled=True, ase_bandwidth_hz=128e9))

    def __post_init__(self) -> None:
        if self.condition not in (16, 64) or self.regime not in ("production", "pilot", "qualification"):
            raise ValueError("Only main16/main64 and explicit production/pilot/qualification regimes are supported")
        if isinstance(self.replicate, bool) or not isinstance(self.replicate, int) or self.replicate not in range(3):
            raise ValueError("replicate must be 0, 1, or 2")
        for key in ("master_entropy", "scored_symbols", "guard_symbols", "pilot_symbols", "context_symbols", "train_frames", "validation_frames", "test_frames", "equalizer_taps", "bps_phases"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if key in ("master_entropy", "guard_symbols") else 1):
                raise ValueError(f"Invalid integer {key}")
        if self.link.guard_symbols != self.guard_symbols:
            raise ValueError("Study and physical guard lengths must agree")
        if self.bps_phases < 2:
            raise ValueError("BPS requires at least two phase candidates")
        if self.equalizer_taps % 2 != 1 or self.pilot_symbols < self.equalizer_taps * 2:
            raise ValueError("Require odd FIR length and sufficient dedicated pilots")
        if not self.bps_windows or len(self.bps_windows) > 3 or any(not isinstance(w, int) or w < 1 or w % 2 == 0 for w in self.bps_windows):
            raise ValueError("Specify at most three odd BPS windows")
        if not 0 < self.phase_max_deg or not math.isfinite(self.phase_max_deg) or not 0 < self.min_step_km <= self.link.step_km:
            raise ValueError("Invalid adaptive phase/step bound")
        if not math.isfinite(self.launch_power_dbm) or not math.isfinite(self.ridge_relative) or self.ridge_relative < 0:
            raise ValueError("Invalid launch power/ridge")
        if self.regime == "production":
            if self.master_entropy != MASTER_ENTROPY:
                raise ValueError("Production master entropy is fixed by the seed contract")
            if (self.scored_symbols, self.pilot_symbols, self.context_symbols, self.train_frames, self.validation_frames, self.test_frames) != (65536, 2048, 96, 8, 2, 4):
                raise ValueError("Production scored/pilot/context/frame budgets are fixed; use explicit pilot regime for small fixtures")
            if self.phase_max_deg > .05 or self.guard_symbols < 4096 or self.link.sps < 4 or self.link.rrc_span_symbols < 128 or self.link.step_km > 1:
                raise ValueError("Production numerical settings may only refine the specification baseline")
            expected = (40, 32e9, 2.) if self.condition == 16 else (12, 42e9, 3.)
            if (self.link.n_spans, self.link.symbol_rate_hz, self.launch_power_dbm) != expected:
                raise ValueError("Production condition must retain principal distance/rate/power")
            physical = (self.link.span_km, self.link.alpha_db_per_km, self.link.dispersion_ps_nm_km, self.link.gamma_per_w_km,
                        self.link.wavelength_m, self.link.rolloff, self.link.edfa_noise_figure_db, self.link.ase_enabled, self.link.ase_bandwidth_hz)
            if physical != (80., .2, 17., 1.3, 1550e-9, 1/16, 6., True, 128e9):
                raise ValueError("Production physical system is fixed by reproduction-spec")

    @property
    def region_lengths(self) -> tuple[int, ...]:
        return (self.guard_symbols, self.pilot_symbols, self.context_symbols,
                self.scored_symbols, self.context_symbols, self.pilot_symbols, self.guard_symbols)

    @property
    def n_symbols(self) -> int:
        return sum(self.region_lengths)

    @property
    def frame_counts(self) -> dict[str, int]:
        return {name: getattr(self, f"{name}_frames") for name in ("train", "validation", "test")}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> DataConfig:
        values = dict(values)
        if "link" in values:
            values["link"] = LinkConfig.from_dict(values["link"])
        if "bps_windows" in values:
            values["bps_windows"] = tuple(values["bps_windows"])
        return cls(**values)


def main16(**overrides: Any) -> DataConfig:
    return replace(DataConfig(), **overrides)


def main64(**overrides: Any) -> DataConfig:
    reference = DataConfig()
    reference = replace(reference, condition=64, launch_power_dbm=3.,
        link=replace(reference.link, n_spans=12, symbol_rate_hz=42e9))
    return replace(reference, **overrides)


def pilot_config(condition: int = 16, *, scored_symbols: int = 4096,
                 guard_symbols: int = 256, pilot_symbols: int = 256) -> DataConfig:
    """Dedicated split=pilot seeds; short fixtures never satisfy production."""
    reference = main16() if condition == 16 else main64() if condition == 64 else None
    if reference is None:
        raise ValueError("condition must be 16 or 64")
    return replace(reference, regime="pilot", scored_symbols=scored_symbols,
        guard_symbols=guard_symbols, pilot_symbols=pilot_symbols,
        train_frames=1, validation_frames=1, test_frames=1,
        link=replace(reference.link, guard_symbols=guard_symbols))


SCIENTIFIC_CONTRACT_VERSION = "transformer-data-science-v1"


def scientific_contract(config: DataConfig) -> dict[str, Any]:
    """Numerical/receiver geometry qualified before production generation.

    Regime, split counts, replicate/seed, storage choices and output locations
    are excluded: independent production streams legitimately differ from the
    dedicated qualification streams. Every physical or fitted-DSP input is
    included, including the full LinkConfig and maximum/minimum SSFM bounds.
    """
    return {"schema": SCIENTIFIC_CONTRACT_VERSION,
        "condition": config.condition, "launch_power_dbm_XY_total": config.launch_power_dbm,
        "link": config.link.to_dict(),
        "geometry": {"scored_symbols": config.scored_symbols, "guard_symbols_per_side": config.guard_symbols,
            "pilot_symbols_per_side": config.pilot_symbols, "context_symbols_per_side": config.context_symbols,
            "regions": ["guard", "pilot", "context", "scored", "context", "pilot", "guard"]},
        "adaptive_SSFM": {"phase_max_deg": config.phase_max_deg, "min_step_km": config.min_step_km,
            "max_step_km": config.link.step_km},
        "receiver": {"equalizer_taps": config.equalizer_taps, "ridge_relative": config.ridge_relative,
            "bps_windows": list(config.bps_windows), "bps_phases": config.bps_phases,
            "FIR_fit": "dedicated_train_pilots_only", "CR_selection": "validation_BER_then_MSE",
            "frame_calibration": "dedicated_pilot_gain_before_and_after_blind_phase_search"}}
