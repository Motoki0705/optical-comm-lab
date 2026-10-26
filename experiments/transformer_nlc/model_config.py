"""Fixed single-polarization architectures from docs/reproduction-spec.md.

The six structures are comment-TeX candidates, not verified author configs.
In particular main64/low-adopted changes the commented t=8 to t=16. The key
dimension is adopted as the *total* attention projection dimension, d, not d/h.
"""

from dataclasses import asdict, dataclass, fields
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class ModelConfig:
    t: int = 16
    b: int = 128
    k: int = 9
    d_model: int = 16
    attention_projection_dim: int = 16
    h: int = 4
    layers: int = 2
    d_ff: int = 32
    output_window: int = 7
    mask_rho: float | None = 2.6
    dropout: float = 0.0
    layer_norm_eps: float = 1e-5
    negative_slope: float = 0.2
    initialization: str = "xavier_uniform"

    def __post_init__(self):
        for name in ("t", "b", "k", "d_model", "attention_projection_dim", "h",
                     "layers", "d_ff", "output_window"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.k != 9:
            raise ValueError("the fixed study requires k=9; redefine mask coordinates before changing it")
        if self.output_window % 2 != 1:
            raise ValueError("output_window must be odd")
        if self.attention_projection_dim % self.h:
            raise ValueError("attention_projection_dim must be divisible by h")
        if self.t < (self.k - 1) // 2 + self.w:
            raise ValueError("t must cover the valid CNN offset and output neighborhood")
        if self.mask_rho is not None:
            if isinstance(self.mask_rho, bool) or not isinstance(self.mask_rho, (int, float)):
                raise ValueError("mask_rho must be a finite nonnegative number or None")
            if not math.isfinite(self.mask_rho) or self.mask_rho < 0:
                raise ValueError("mask_rho must be a finite nonnegative number or None")
        if self.dropout != 0.0 or isinstance(self.dropout, bool):
            raise ValueError("the fixed study requires dropout=0")
        if self.layer_norm_eps != 1e-5:
            raise ValueError("the fixed study requires layer_norm_eps=1e-5")
        if self.negative_slope != 0.2:
            raise ValueError("the fixed study requires negative_slope=0.2")
        if self.initialization != "xavier_uniform":
            raise ValueError("the fixed study requires xavier_uniform initialization")

    @property
    def ell(self) -> int:
        return 2 * self.t - self.k + 1

    @property
    def sequence_length(self) -> int:
        return self.b + self.ell

    @property
    def input_length(self) -> int:
        return self.b + 2 * self.t

    @property
    def head_dim(self) -> int:
        return self.attention_projection_dim // self.h

    @property
    def w(self) -> int:
        return self.output_window // 2

    @property
    def first_center(self) -> int:
        return self.t - (self.k - 1) // 2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ModelConfig":
        if not isinstance(values, Mapping):
            raise ValueError("model configuration must be a mapping")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"unknown model configuration fields: {sorted(unknown)}")
        return cls(**dict(values))

    @classmethod
    def candidate(cls, condition: str, name: str, *, masked: bool = True) -> "ModelConfig":
        """Return one of the six fixed candidates (k9/b128/h4).

        Table IV rate/PMD structures are deliberately absent. ``masked=False``
        creates the truly unmasked partner with identical trainable parameters.
        """
        candidates = {
            "main16": {"A": (96, 96, 64, 3, 64, 15),
                       "B": (64, 64, 32, 2, 32, 7),
                       "C": (16, 16, 16, 2, 32, 7)},
            "main64": {"high": (64, 96, 32, 3, 64, 11),
                       "mid": (32, 64, 32, 1, 64, 11),
                       "low-adopted": (16, 16, 16, 1, 64, 7)},
        }
        if type(masked) is not bool:
            raise ValueError("masked must be a bool")
        try:
            t, hidden, d, layers, ff, window = candidates[condition][name]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"unknown fixed candidate {condition}/{name}") from exc
        return cls(t=t, d_model=hidden, attention_projection_dim=d, layers=layers,
                   d_ff=ff, output_window=window, mask_rho=2.6 if masked else None)
