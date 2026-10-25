from copy import deepcopy
from dataclasses import replace
import importlib.util
import math

import numpy as np
import pytest

if importlib.util.find_spec("torch") is None:
    pytest.skip("install the optional transformer extra for model tests", allow_module_level=True)
import torch
from torch import nn
from torch.nn import functional as F

from experiments.transformer_nlc.complexity import analytic_parameter_count, count_parameters
from experiments.transformer_nlc.mask import additive_mask, block_mask
from experiments.transformer_nlc.model import (TransformerNLC, predict_dual_polarization,
                                               sinusoidal_encoding, swap_xy)
from experiments.transformer_nlc.model_config import ModelConfig


@pytest.fixture(autouse=True)
def limit_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_config(**overrides):
    return replace(ModelConfig(t=8, b=3, d_model=6, attention_projection_dim=4,
                               h=2, layers=2, d_ff=7, output_window=3), **overrides)


def manual_forward(model, features, additive):
    """Independent dense equations, explicit Post-LN and per-target slicing."""
    cfg = model.config
    x = F.conv1d(features.transpose(1, 2), model.embedding.weight, model.embedding.bias)
    x = torch.where(x >= 0, x, 0.2 * x).transpose(1, 2) + model.position_encoding

    def linear(tensor, module):
        return tensor @ module.weight.T + module.bias

    def layer_norm(tensor, module):
        centered = tensor - tensor.mean(dim=-1, keepdim=True)
        variance = centered.square().mean(dim=-1, keepdim=True)
        return centered / torch.sqrt(variance + 1e-5) * module.weight + module.bias

    for encoder in model.encoders:
        attention = encoder.attention
        q, k, v = [linear(x, projection).reshape(x.shape[0], cfg.sequence_length,
                   cfg.h, cfg.head_dim).permute(0, 2, 1, 3)
                   for projection in (attention.query, attention.key, attention.value)]
        score = (q @ k.transpose(-1, -2)) / math.sqrt(cfg.head_dim)
        if additive is not None:
            score = score + additive
        attended = (score.softmax(-1) @ v).transpose(1, 2).reshape(x.shape[0], cfg.sequence_length, -1)
        y = layer_norm(x + linear(attended, attention.output), encoder.attention_norm)
        ffn = linear(torch.clamp_min(linear(y, encoder.ffn[0]), 0), encoder.ffn[2])
        x = layer_norm(y + ffn, encoder.ffn_norm)
    windows = torch.stack([x[:, cfg.first_center + i - cfg.w:cfg.first_center + i + cfg.w + 1]
                           .reshape(x.shape[0], -1) for i in range(cfg.b)], dim=1)
    x = linear(windows, model.head[0])
    x = torch.where(x >= 0, x, 0.2 * x)
    x = linear(x, model.head[2])
    x = torch.where(x >= 0, x, 0.2 * x)
    return linear(x, model.head[4])


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("b", [1, 3])
def test_dense_equation_reference_output_and_all_gradients_agree(masked, b):
    torch.manual_seed(482)
    config = tiny_config(b=b, mask_rho=2.6 if masked else None)
    model = TransformerNLC(config).double()
    reference = deepcopy(model)
    x = torch.randn(2, config.input_length, 4, dtype=torch.float64, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    mask = None if not masked else torch.from_numpy(additive_mask(block_mask(config.t, config.b),
                                                                               dtype=np.float64))
    prediction = model(x)
    expected = manual_forward(reference, reference_x, mask)
    torch.testing.assert_close(prediction, expected, rtol=1e-10, atol=1e-11)
    target = torch.randn_like(prediction)
    (prediction - target).square().mean().backward()
    (expected - target).square().mean().backward()
    assert torch.isfinite(x.grad).all()
    torch.testing.assert_close(x.grad, reference_x.grad, rtol=1e-9, atol=1e-10)
    for (name, parameter), (reference_name, reference_parameter) in zip(
            model.named_parameters(), reference.named_parameters()):
        assert name == reference_name
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        torch.testing.assert_close(parameter.grad, reference_parameter.grad, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("condition,name", [("main16", "A"), ("main16", "B"), ("main16", "C"),
                                            ("main64", "high"), ("main64", "mid"),
                                            ("main64", "low-adopted")])
def test_fixed_candidate_shape_independent_projection_and_parameter_count(condition, name):
    torch.manual_seed(917)
    config = ModelConfig.candidate(condition, name)
    model = TransformerNLC(config)
    with torch.inference_mode():
        output = model(torch.randn(1, config.input_length, 4))
    assert output.shape == (1, 128, 2)
    assert torch.isfinite(output).all()
    assert model.encoders[0].attention.query.weight.shape == (config.attention_projection_dim,
                                                            config.d_model)
    assert count_parameters(model) == analytic_parameter_count(config)
    assert model.config.dropout == 0


def test_tagged_index_crop_neighborhood_order_and_raw_input_alignment():
    cfg = tiny_config()
    model = TransformerNLC(cfg)
    # Each encoded tag carries its raw center (j+4) and feature index.
    raw_centers = torch.arange(cfg.sequence_length) + (cfg.k - 1) // 2
    tagged = 100 * raw_centers[:, None] + torch.arange(cfg.d_model)[None, :]
    selected = model.extract_neighborhoods(tagged[None]).reshape(1, cfg.b, cfg.output_window, cfg.d_model)
    for i in range(cfg.b):
        for offset in range(-cfg.w, cfg.w + 1):
            torch.testing.assert_close(selected[0, i, offset + cfg.w],
                                       100 * (cfg.t + i + offset) + torch.arange(cfg.d_model))
    assert selected[0, 0, cfg.w, 0] == 100 * cfg.t
    assert selected[0, -1, cfg.w, 0] == 100 * (cfg.t + cfg.b - 1)
    # Independently trace a real valid convolution's middle tap through crop.
    with torch.no_grad():
        model.embedding.weight.zero_()
        model.embedding.bias.zero_()
        model.embedding.weight[:, 0, 4] = 1
        features = torch.zeros(1, cfg.input_length, 4)
        features[0, :, 0] = torch.arange(cfg.input_length)
        embedded = model.embedding(features.transpose(1, 2)).transpose(1, 2)
    neighborhoods = model.extract_neighborhoods(embedded).reshape(1, cfg.b, cfg.output_window, cfg.d_model)
    torch.testing.assert_close(neighborhoods[0, :, cfg.w, 0], torch.arange(cfg.t, cfg.t + cfg.b).float())


def test_xy_swap_and_two_call_prediction_use_the_same_weights_and_axis_order():
    torch.manual_seed(61)
    cfg = tiny_config()
    model = TransformerNLC(cfg)
    features = torch.randn(2, cfg.input_length, 4)
    swapped = swap_xy(features)
    torch.testing.assert_close(swap_xy(swapped), features)
    torch.testing.assert_close(swapped[..., :2], features[..., 2:])
    prediction = predict_dual_polarization(model, features)
    assert prediction.shape == (2, cfg.b, 2, 2)
    torch.testing.assert_close(prediction[:, :, 0], model(features))
    torch.testing.assert_close(prediction[:, :, 1], model(swapped))
    torch.testing.assert_close(predict_dual_polarization(model, swapped), prediction.flip(-2))


def test_explicit_mask_none_uses_config_and_unmasked_partner_is_truly_dense():
    torch.manual_seed(7)
    cfg = tiny_config(mask_rho=0)
    masked = TransformerNLC(cfg).double()
    unmasked = TransformerNLC(replace(cfg, mask_rho=None)).double()
    unmasked.load_state_dict(masked.state_dict(), strict=True)
    x = torch.randn(2, cfg.input_length, 4, dtype=torch.float64)
    allowed = block_mask(cfg.t, cfg.b, 0)
    torch.testing.assert_close(masked(x), masked(x, None), rtol=0, atol=0)
    torch.testing.assert_close(masked(x), masked(x, allowed), rtol=0, atol=0)
    torch.testing.assert_close(masked(x, additive_mask(allowed)), masked(x), rtol=0, atol=0)
    torch.testing.assert_close(masked(x, np.ones_like(allowed)), unmasked(x), rtol=1e-12, atol=1e-13)
    assert not torch.allclose(masked(x), unmasked(x), rtol=1e-5, atol=1e-6)
    assert unmasked.configured_mask is None
    prediction, attention = masked(x, return_attention=True)
    torch.testing.assert_close(prediction, masked(x), rtol=1e-10, atol=1e-11)
    assert len(attention) == cfg.layers
    for layer in attention:
        assert layer.shape == (2, cfg.h, cfg.sequence_length, cfg.sequence_length)
        assert torch.isfinite(layer).all()
        torch.testing.assert_close(layer.sum(-1), torch.ones_like(layer.sum(-1)))
        assert (layer[..., ~torch.from_numpy(allowed)] == 0).all()
    assert not hasattr(masked, "last_attention")


def test_explicit_initialization_position_encoding_and_no_extra_embedding_scale():
    cfg = tiny_config(d_model=5)
    torch.manual_seed(13)
    model = TransformerNLC(cfg)
    torch.manual_seed(13)
    duplicate = TransformerNLC(cfg)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, duplicate.state_dict()[name], rtol=0, atol=0)
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            assert torch.count_nonzero(module.bias) == 0
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(module.weight)
            assert module.weight.abs().max() <= math.sqrt(6 / (fan_in + fan_out))
        elif isinstance(module, nn.LayerNorm):
            assert torch.equal(module.weight, torch.ones_like(module.weight))
            assert torch.count_nonzero(module.bias) == 0 and module.eps == 1e-5
    encoding = sinusoidal_encoding(4, 5, dtype=torch.float64)
    for position in range(4):
        for axis in range(5):
            angle = position / 10000 ** (2 * (axis // 2) / 5)
            assert encoding[position, axis].item() == pytest.approx(
                math.sin(angle) if axis % 2 == 0 else math.cos(angle), abs=1e-14)
    # Hook the first encoder: it must see CNN+PE with unit embedding scale.
    captured = []
    hook = model.encoders[0].register_forward_pre_hook(lambda _, arguments: captured.append(arguments[0]))
    x = torch.randn(1, cfg.input_length, 4)
    model(x)
    hook.remove()
    expected = model.embedding_activation(model.embedding(x.transpose(1, 2))).transpose(1, 2)
    torch.testing.assert_close(captured[0], expected + model.position_encoding)


def test_invalid_features_and_attention_masks_are_rejected_before_softmax():
    cfg = tiny_config()
    model = TransformerNLC(cfg)
    x = torch.zeros(1, cfg.input_length, 4)
    for invalid in (x.squeeze(0), x[:, :-1], x[..., :3], x.long()):
        with pytest.raises(ValueError, match="features"):
            model(invalid)
    n = cfg.sequence_length
    empty_row = np.ones((n, n), dtype=bool)
    empty_row[-1] = False
    for mask in (np.ones((n, n - 1), dtype=bool), np.ones((n, n), dtype=int), empty_row,
                 np.full((n, n), np.nan), np.full((n, n), np.inf), np.full((n, n), -1.0),
                 np.full((n, n), -np.inf)):
        with pytest.raises(ValueError):
            model(x, mask)
    with pytest.raises(ValueError):
        model.extract_neighborhoods(torch.zeros(1, n - 1, cfg.d_model))
    with pytest.raises(ValueError):
        swap_xy(x[..., :2])


def test_cpu_float32_matches_float64_at_disclosed_precision_and_optimizer_step_is_finite():
    torch.manual_seed(937)
    cfg = tiny_config()
    model = TransformerNLC(cfg)
    reference = deepcopy(model).double()
    features = torch.randn(4, cfg.input_length, 4)
    expected = reference(features.double()).float()
    prediction = model(features)
    torch.testing.assert_close(prediction, expected, rtol=2e-5, atol=2e-6)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    previous = model.head[-1].weight.detach().clone()
    prediction.square().mean().backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    optimizer.step()
    assert not torch.equal(previous, model.head[-1].weight)
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
