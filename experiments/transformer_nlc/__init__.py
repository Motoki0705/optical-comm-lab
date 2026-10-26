"""Transformer nonlinear compensation, separate from the ESN study.

Configuration, masks and theoretical complexity need only NumPy. PyTorch is
loaded only when a model symbol is requested (install the ``transformer`` extra).
"""

from .model_config import ModelConfig

__all__ = ["ModelConfig", "TransformerNLC", "predict_dual_polarization", "swap_xy"]


def __getattr__(name):
    if name in {"TransformerNLC", "predict_dual_polarization", "swap_xy"}:
        from . import model

        return getattr(model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
