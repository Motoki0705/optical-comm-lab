"""Componentwise theoretical real multiplications under the fixed spec.

This is the adopted counting convention, not an exact reconstruction of all
paper code. Mask savings assume a sparse implementation; our SDPA backend is
dense. Count parameters separately; do not interpret RMPS as latency/FLOPs.
"""

from dataclasses import asdict, dataclass

from .mask import block_mask
from .model_config import ModelConfig


@dataclass(frozen=True)
class ComplexityBreakdown:
    sequence_length: int
    allowed_attention_elements: int
    dense_attention_elements: int
    components_per_output_polarization_symbol: dict[str, float]
    dense_backend_components_per_output_polarization_symbol: dict[str, float]
    total_per_output_polarization_symbol: float
    total_per_dp_time: float
    dense_backend_total_per_output_polarization_symbol: float
    dense_backend_total_per_dp_time: float
    parameter_count: int
    convention: str = "adopted reproduction-spec RMPS; sparse mask theoretical only"
    backend: str = "dense attention (mask does not imply sparse execution)"
    excluded: tuple[str, ...] = ("addition/bias", "ReLU/LeakyReLU", "precomputed PE",
                                 "rsqrt", "memory movement")

    def to_dict(self):
        return asdict(self)


def analytic_parameter_count(config: ModelConfig) -> int:
    """Trainable weights, biases and both affine LayerNorms, independent of mask."""
    c = config
    m, d, f = c.d_model, c.attention_projection_dim, c.d_ff
    cnn = m * (4 * c.k + 1)
    projections = 3 * (m * d + d) + d * m + m
    ffn = m * f + f + f * m + m
    norms = 4 * m
    head = c.output_window * m * 2 + 2 + 2 * 10 + 10 + 10 * 2 + 2
    return cnn + c.layers * (projections + ffn + norms) + head


def count_parameters(model) -> int:
    """Count actual trainable model parameters without importing PyTorch here."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def complexity_breakdown(config: ModelConfig) -> ComplexityBreakdown:
    c = config
    n, b = c.sequence_length, c.b
    allowed = int(block_mask(c.t, c.b, c.mask_rho, k=c.k).sum())

    def components(elements):
        # Separate Eq. 6 attention from linear projections and added LN terms.
        return {
            "cnn": n * 4 * c.k * c.d_model / b,
            "qkv_output_projection": c.layers * 4 * n * c.d_model * c.attention_projection_dim / b,
            "attention_eq6": c.layers * (2 * c.attention_projection_dim + 3 * c.h) * elements / b,
            "ffn": c.layers * 2 * n * c.d_model * c.d_ff / b,
            "layer_norm_added": c.layers * n * (6 * c.d_model + 4) / b,
            "head": 2 * c.output_window * c.d_model + 2 * 10 + 10 * 2,
        }

    sparse = components(allowed)
    dense = components(n * n)
    total, dense_total = sum(sparse.values()), sum(dense.values())
    return ComplexityBreakdown(n, allowed, n * n, sparse, dense, total, 2 * total,
                               dense_total, 2 * dense_total, analytic_parameter_count(c))
