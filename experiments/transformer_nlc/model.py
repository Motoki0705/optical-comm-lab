"""CNN/Post-LN Transformer for one polarization's received residual.

All numerical assumptions follow ``docs/reproduction-spec.md``. The default
SDPA backend computes dense attention, even when a physical mask is supplied;
theoretical sparse RMPS is not a claim about runtime. Requesting attention
uses an explicit dense reference path and returns tensors to the caller only.
"""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .mask import additive_mask, block_mask
from .model_config import ModelConfig


def sinusoidal_encoding(length: int, width: int, *, dtype=torch.float32) -> Tensor:
    """Positions 0..N-1, without an extra sqrt(d_model) embedding multiplier."""
    positions = torch.arange(length, dtype=torch.float64)[:, None]
    frequencies = torch.exp(torch.arange(0, width, 2, dtype=torch.float64)
                            * (-math.log(10000.0) / width))
    angles = positions * frequencies
    encoding = torch.empty(length, width, dtype=torch.float64)
    encoding[:, 0::2] = torch.sin(angles)
    encoding[:, 1::2] = torch.cos(angles[:, :width // 2])
    return encoding.to(dtype=dtype)


class MultiHeadAttention(nn.Module):
    """d_model -> total d -> h heads of d/h, then d -> d_model."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.h = config.h
        self.head_dim = config.head_dim
        d = config.attention_projection_dim
        self.query = nn.Linear(config.d_model, d, bias=True)
        self.key = nn.Linear(config.d_model, d, bias=True)
        self.value = nn.Linear(config.d_model, d, bias=True)
        self.output = nn.Linear(d, config.d_model, bias=True)

    def forward(self, x: Tensor, mask: Tensor | None = None, *, return_attention=False):
        batch, length, _ = x.shape

        def split(projection):
            return projection(x).reshape(batch, length, self.h, self.head_dim).transpose(1, 2)

        query, key, value = (split(projection) for projection in (self.query, self.key, self.value))
        weights = None
        if return_attention:
            scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask
            weights = torch.softmax(scores, dim=-1)
            attended = weights @ value
        else:
            attended = F.scaled_dot_product_attention(query, key, value, attn_mask=mask,
                                                     dropout_p=0.0, is_causal=False)
        combined = attended.transpose(1, 2).reshape(batch, length, self.h * self.head_dim)
        return self.output(combined), weights


class EncoderLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.attention_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps,
                                           elementwise_affine=True, bias=True)
        self.ffn = nn.Sequential(nn.Linear(config.d_model, config.d_ff, bias=True), nn.ReLU(),
                                 nn.Linear(config.d_ff, config.d_model, bias=True))
        self.ffn_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps,
                                     elementwise_affine=True, bias=True)

    def forward(self, x: Tensor, mask: Tensor | None, *, return_attention=False):
        attended, weights = self.attention(x, mask, return_attention=return_attention)
        y = self.attention_norm(x + attended)
        return self.ffn_norm(y + self.ffn(y)), weights


class TransformerNLC(nn.Module):
    """features [B,b+2t,4] (XI,XQ,YI,YQ) -> X residual [B,b,2].

    ``forward(mask=None)`` uses the configured mask. To run truly unmasked,
    instantiate ``ModelConfig(mask_rho=None)`` (or pass an all-allowed mask).
    An explicit mask is a bool allowed array or additive 0/-inf square [N,N].
    ``return_attention=True`` returns (prediction, tuple(layer attention)),
    each attention shaped [B,h,N,N], and switches to the dense reference path.
    Attention is never retained on the module or written to disk.
    """

    attention_backend = "torch.scaled_dot_product_attention dense dispatcher"

    def __init__(self, config: ModelConfig):
        super().__init__()
        if not isinstance(config, ModelConfig):
            raise TypeError("config must be ModelConfig")
        self.config = config
        self.embedding = nn.Conv1d(4, config.d_model, config.k, stride=1, padding=0, bias=True)
        self.embedding_activation = nn.LeakyReLU(config.negative_slope)
        self.encoders = nn.ModuleList(EncoderLayer(config) for _ in range(config.layers))
        self.head = nn.Sequential(
            nn.Linear(config.output_window * config.d_model, 2, bias=True),
            nn.LeakyReLU(config.negative_slope), nn.Linear(2, 10, bias=True),
            nn.LeakyReLU(config.negative_slope), nn.Linear(10, 2, bias=True))
        # Derived buffers are recomputed from the saved ModelConfig, not learned.
        self.register_buffer("position_encoding",
                             sinusoidal_encoding(config.sequence_length, config.d_model),
                             persistent=False)
        configured_mask = None if config.mask_rho is None else torch.from_numpy(
            additive_mask(block_mask(config.t, config.b, config.mask_rho, k=config.k)))
        self.register_buffer("configured_mask", configured_mask, persistent=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Adopted explicit initialization; no framework default weights survive."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _prepare_mask(self, mask, features: Tensor) -> Tensor | None:
        if mask is None:
            return (None if self.configured_mask is None
                    else self.configured_mask.to(dtype=features.dtype))
        mask = torch.as_tensor(mask, device=features.device)
        n = self.config.sequence_length
        if tuple(mask.shape) != (n, n):
            raise ValueError(f"attention mask must have shape {(n, n)}")
        if mask.dtype == torch.bool:
            allowed = mask
            result = torch.zeros((n, n), device=features.device, dtype=features.dtype)
            result.masked_fill_(~allowed, -torch.inf)
        elif torch.is_floating_point(mask):
            if not torch.all((mask == 0) | torch.isneginf(mask)).item():
                raise ValueError("additive mask may contain only 0 and -inf")
            allowed = mask == 0
            result = mask.to(dtype=features.dtype)
        else:
            raise ValueError("mask must be bool allowed positions or floating 0/-inf")
        if not torch.all(allowed.any(dim=-1)).item():
            raise ValueError("every attention row must permit at least one key")
        return result

    def extract_neighborhoods(self, encoded: Tensor) -> Tensor:
        """Flatten [c_i-w:c_i+w+1] in (position, feature) order.

        c_i=t-(k-1)/2+i maps to the input target index t+i. Both edge
        neighborhoods are guaranteed in range by ModelConfig validation.
        """
        cfg = self.config
        if encoded.ndim != 3 or tuple(encoded.shape[1:]) != (cfg.sequence_length, cfg.d_model):
            raise ValueError("encoded tensor must have shape [B,N,d_model]")
        start = cfg.first_center - cfg.w
        segment = encoded[:, start:start + cfg.b + cfg.output_window - 1]
        windows = segment.unfold(1, cfg.output_window, 1).transpose(-1, -2)
        return windows.reshape(encoded.shape[0], cfg.b, cfg.output_window * cfg.d_model)

    def forward(self, features: Tensor, mask=None, *, return_attention=False):
        cfg = self.config
        if features.ndim != 3 or tuple(features.shape[1:]) != (cfg.input_length, 4):
            raise ValueError(f"features must have shape [B,{cfg.input_length},4]")
        if not torch.is_floating_point(features):
            raise ValueError("features must have a floating dtype")
        attention_mask = self._prepare_mask(mask, features)
        encoded = self.embedding_activation(self.embedding(features.transpose(1, 2))).transpose(1, 2)
        encoded = encoded + self.position_encoding.to(dtype=encoded.dtype)
        attention = []
        for encoder in self.encoders:
            encoded, weights = encoder(encoded, attention_mask, return_attention=return_attention)
            if return_attention:
                attention.append(weights)
        prediction = self.head(self.extract_neighborhoods(encoded))
        return (prediction, tuple(attention)) if return_attention else prediction


def swap_xy(features: Tensor) -> Tensor:
    """XI,XQ,YI,YQ -> YI,YQ,XI,XQ, without changing time or IQ order."""
    if features.ndim != 3 or features.shape[-1] != 4:
        raise ValueError("features must have shape [B,time,4]")
    return features[..., [2, 3, 0, 1]]


def predict_dual_polarization(model: TransformerNLC, features: Tensor, mask=None) -> Tensor:
    """Two calls with shared weights -> [B,b,2,2], axes (batch,time,XY,IQ).

    Returns received residuals, not compensated symbols. No no_grad context is
    imposed: validation callers choose inference_mode; training still uses X.
    """
    return torch.stack((model(features, mask), model(swap_xy(features), mask)), dim=-2)
