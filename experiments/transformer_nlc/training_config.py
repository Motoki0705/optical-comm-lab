"""Finite, explicit training budgets for the principal Transformer study."""
from dataclasses import asdict, dataclass, fields
import math
from typing import Any, Mapping

from .model_config import ModelConfig

CANDIDATES = {16: ("A", "B", "C"), 64: ("high", "mid", "low-adopted")}


@dataclass(frozen=True)
class TrainingConfig:
    condition: int = 16
    candidate: str = "C"
    masked: bool = True
    regime: str = "production"
    model: ModelConfig | None = None
    effective_batch_blocks: int = 512
    microbatch_blocks: int = 512
    pilot_validation_symbols: int | None = None
    max_epochs: int = 500
    patience: int = 100
    warmup_epochs: int = 10
    max_lr: float = 1e-3
    adam_betas: tuple[float, float] = (.9, .999)
    adam_eps: float = 1e-8
    weight_decay: float = 0.0
    device: str = "cpu"
    cpu_threads: int = 1
    deterministic: bool = True
    gpu_resume_atol: float = 1e-6
    gpu_resume_rtol: float = 1e-5

    def __post_init__(self):
        if self.condition not in CANDIDATES or self.candidate not in CANDIDATES[self.condition]:
            raise ValueError("Unknown fixed condition/candidate")
        if type(self.masked) is not bool or self.regime not in ("production", "pilot", "smoke"):
            raise ValueError("Require boolean masked and explicit production/pilot/smoke regime")
        resolved = ModelConfig.candidate(f"main{self.condition}", self.candidate, masked=self.masked)
        if self.model is None:
            object.__setattr__(self, "model", resolved)
        if not isinstance(self.model, ModelConfig) or (self.model.mask_rho is not None) != self.masked:
            raise ValueError("Model and mask identity disagree")
        if self.regime == "production" and self.model != resolved:
            raise ValueError("Production requires the fixed candidate architecture")
        if self.regime == "pilot" and self.pilot_validation_symbols is None:
            object.__setattr__(self, "pilot_validation_symbols", 2048)
        if self.regime != "pilot" and self.pilot_validation_symbols is not None:
            raise ValueError("Validation slicing is permitted only for dedicated learning pilots")
        if self.pilot_validation_symbols is not None and self.pilot_validation_symbols != 2048:
            raise ValueError("Learning pilot validation budget is fixed at 2048 DP times per frame")
        for name in ("effective_batch_blocks", "microbatch_blocks", "max_epochs", "patience", "warmup_epochs", "cpu_threads"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.microbatch_blocks > self.effective_batch_blocks:
            raise ValueError("microbatch cannot exceed effective batch")
        if self.max_epochs > 500 or self.patience != 100 or self.warmup_epochs != 10:
            raise ValueError("Fixed cap<=500, patience=100, and warmup=10 epochs are required")
        if self.regime == "production" and self.max_epochs != 500:
            raise ValueError("Production epoch cap is fixed at 500")
        if self.regime == "pilot" and self.max_epochs > 5:
            raise ValueError("Dedicated learning pilot is limited to five epochs")
        if self.regime == "production" and self.effective_batch_blocks != 512:
            raise ValueError("Production effective batch is 512 blocks")
        if (self.max_lr, tuple(self.adam_betas), self.adam_eps, self.weight_decay) != (1e-3, (.9, .999), 1e-8, 0.):
            raise ValueError("Adam and learning-rate assumptions are fixed by reproduction-spec")
        if self.device != "cpu" and not (self.device == "cuda" or self.device.startswith("cuda:")):
            raise ValueError("device must be cpu, cuda, or cuda:<index>")
        if self.deterministic is not True:
            raise ValueError("The initial study requires deterministic algorithms")
        if any(not math.isfinite(v) or v < 0 for v in (self.gpu_resume_atol, self.gpu_resume_rtol)):
            raise ValueError("Resume tolerances must be finite and nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "TrainingConfig":
        if not isinstance(values, Mapping):
            raise ValueError("Training config must be an object")
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown training config fields: {sorted(unknown)}")
        values = dict(values)
        if values.get("model") is not None:
            values["model"] = ModelConfig.from_dict(values["model"])
        if "adam_betas" in values:
            values["adam_betas"] = tuple(values["adam_betas"])
        return cls(**values)


def fixed_trial_plan() -> list[dict[str, Any]]:
    """Exactly 36 entries, including every mask and all three replicates."""
    return [{"trial_id": f"main{condition}-{candidate}-{'masked' if masked else 'unmasked'}-r{replicate}",
             "condition": condition, "candidate": candidate, "masked": masked, "replicate": replicate}
            for condition, names in CANDIDATES.items() for candidate in names
            for masked in (False, True) for replicate in range(3)]


def learning_rate(step: int, *, warmup_steps: int, max_lr: float = 1e-3) -> float:
    if step < 1 or warmup_steps < 1:
        raise ValueError("Scheduler step and warmup_steps start at one")
    return max_lr * min(step / warmup_steps, math.sqrt(warmup_steps / step))
