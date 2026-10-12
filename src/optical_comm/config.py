"""Validated, serializable physical link configuration with explicit SI units."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class LinkConfig:
    """Single-wavelength ideal DP link; distance is in km and power in W.

    ``ase_bandwidth_hz`` is a fixed two-sided complex-envelope bandwidth,
    rather than an sps-dependent optical noise bandwidth. ``None`` deliberately
    enables noise over the entire sampled Nyquist band, changing the physical
    bandwidth when sps changes. Amplifier gain exactly restores span loss.
    """

    n_spans: int = 20
    span_km: float = 80.0
    alpha_db_per_km: float = 0.2
    dispersion_ps_nm_km: float = 17.0
    gamma_per_w_km: float = 1.3
    symbol_rate_hz: float = 32e9
    wavelength_m: float = 1550e-9
    sps: int = 4
    step_km: float = 1.0
    rolloff: float = 0.1
    rrc_span_symbols: int = 64
    guard_symbols: int = 256
    edfa_noise_figure_db: float = 5.0
    ase_enabled: bool = True
    ase_bandwidth_hz: float | None = 128e9

    def __post_init__(self) -> None:
        for name in ("n_spans", "sps", "rrc_span_symbols", "guard_symbols"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.n_spans < 0 or self.guard_symbols < 0:
            raise ValueError("n_spans and guard_symbols must be nonnegative")
        if self.sps < 2 or self.sps % 2:
            raise ValueError("sps must be a positive even integer >= 2")
        if self.rrc_span_symbols < 2 or self.rrc_span_symbols % 2:
            raise ValueError("rrc_span_symbols must be an even integer >= 2")
        for name in ("span_km", "step_km", "symbol_rate_hz", "wavelength_m"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("alpha_db_per_km", "gamma_per_w_km"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not math.isfinite(self.dispersion_ps_nm_km):
            raise ValueError("dispersion_ps_nm_km must be finite")
        if not 0 <= self.rolloff <= 1:
            raise ValueError("rolloff must be in [0, 1]")
        if not math.isfinite(self.edfa_noise_figure_db) or self.edfa_noise_figure_db < 0:
            raise ValueError("edfa_noise_figure_db must be finite and nonnegative")
        if self.ase_bandwidth_hz is not None and (
            not math.isfinite(self.ase_bandwidth_hz)
            or self.ase_bandwidth_hz <= 0
            or self.ase_bandwidth_hz > self.sample_rate_hz * (1 + 1e-12)
        ):
            raise ValueError("ase_bandwidth_hz must be positive and <= sample_rate_hz")

    @property
    def sample_rate_hz(self) -> float:
        return self.symbol_rate_hz * self.sps

    @property
    def total_length_km(self) -> float:
        return self.n_spans * self.span_km

    @property
    def alpha_per_km(self) -> float:
        """Power attenuation coefficient [neper/km]."""
        return self.alpha_db_per_km * math.log(10) / 10

    @property
    def beta2_s2_per_km(self) -> float:
        """D [ps/(nm km)] -> beta2 [s^2/km], including its physical sign."""
        c_m_per_s = 299_792_458.0
        return -self.dispersion_ps_nm_km * 1e-3 * self.wavelength_m**2 / (
            2 * math.pi * c_m_per_s
        )

    @property
    def edfa_gain_db(self) -> float:
        return self.alpha_db_per_km * self.span_km

    @property
    def edfa_gain(self) -> float:
        return 10 ** (self.edfa_gain_db / 10)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> LinkConfig:
        known = {field.name for field in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"Unknown LinkConfig keys: {sorted(unknown)}")
        return cls(**dict(values))
