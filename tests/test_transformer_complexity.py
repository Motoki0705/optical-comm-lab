from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.transformer_nlc.complexity import analytic_parameter_count, complexity_breakdown
from experiments.transformer_nlc.model_config import ModelConfig


def test_tiny_architecture_has_independently_hand_counted_components_and_parameters():
    config = ModelConfig(t=5, b=2, d_model=3, attention_projection_dim=2,
                         h=1, layers=1, d_ff=4, output_window=1, mask_rho=0)
    # N=4; two shifted 3x3 crosses have 5+5-2=8 allowed entries.
    result = complexity_breakdown(config)
    assert result.sequence_length == 4
    assert result.allowed_attention_elements == 8
    assert result.dense_attention_elements == 16
    assert result.components_per_output_polarization_symbol == {
        "cnn": 216.0, "qkv_output_projection": 48.0, "attention_eq6": 28.0,
        "ffn": 48.0, "layer_norm_added": 44.0, "head": 46,
    }
    assert result.total_per_output_polarization_symbol == 430
    assert result.total_per_dp_time == 860
    assert result.dense_backend_total_per_output_polarization_symbol == 458
    assert result.dense_backend_total_per_dp_time == 916
    # CNN111; QKV24 + output9; FFN31; LN12; head60.
    assert result.parameter_count == 247
    assert analytic_parameter_count(config) == 247
    assert json.loads(json.dumps(result.to_dict()))["parameter_count"] == 247


@pytest.mark.parametrize("condition,name", [("main16", "A"), ("main16", "B"), ("main16", "C"),
                                            ("main64", "high"), ("main64", "mid"),
                                            ("main64", "low-adopted")])
def test_mask_changes_theoretical_attention_but_never_dense_backend_or_parameters(condition, name):
    masked = complexity_breakdown(ModelConfig.candidate(condition, name))
    unmasked = complexity_breakdown(ModelConfig.candidate(condition, name, masked=False))
    assert masked.parameter_count == unmasked.parameter_count
    assert masked.total_per_output_polarization_symbol < unmasked.total_per_output_polarization_symbol
    assert (masked.dense_backend_total_per_output_polarization_symbol
            == unmasked.total_per_output_polarization_symbol)
    assert unmasked.allowed_attention_elements == unmasked.dense_attention_elements
    for key in masked.components_per_output_polarization_symbol:
        if key != "attention_eq6":
            assert (masked.components_per_output_polarization_symbol[key]
                    == unmasked.components_per_output_polarization_symbol[key])


def test_configuration_roundtrip_and_candidates_do_not_load_optional_torch():
    repository = Path(__file__).resolve().parents[1]
    program = """
import importlib.abc, json, sys
class DenyTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('base configuration unexpectedly imports torch')
sys.meta_path.insert(0, DenyTorch())
from experiments.transformer_nlc import ModelConfig
from experiments.transformer_nlc.complexity import complexity_breakdown
from experiments.transformer_nlc.mask import block_mask
c = ModelConfig.candidate('main64', 'low-adopted')
assert ModelConfig.from_dict(json.loads(json.dumps(c.to_dict()))) == c
assert c.t == 16 and 'torch' not in sys.modules
assert complexity_breakdown(c).parameter_count > 0
assert block_mask(c.t, c.b).any()
"""
    environment = {**os.environ, "PYTHONPATH": f"{repository / 'src'}:{repository}"}
    subprocess.run([sys.executable, "-c", program], check=True, env=environment, cwd=repository)


@pytest.mark.parametrize("options", [{"b": 0}, {"t": 0}, {"t": True}, {"k": 7}, {"k": 8},
                                     {"d_model": 0}, {"attention_projection_dim": 10},
                                     {"h": 0}, {"layers": 0}, {"d_ff": 0},
                                     {"output_window": 2}, {"t": 5, "output_window": 7},
                                     {"dropout": 0.1}, {"dropout": True},
                                     {"layer_norm_eps": 1e-6}, {"negative_slope": 0.1},
                                     {"initialization": "default"}, {"mask_rho": -1},
                                     {"mask_rho": float("nan")}, {"mask_rho": float("inf")},
                                     {"mask_rho": True}])
def test_invalid_model_configuration_rejected(options):
    with pytest.raises(ValueError):
        ModelConfig(**options)


def test_unknown_fields_candidates_and_types_fail_loudly():
    with pytest.raises(ValueError, match="unknown model configuration"):
        ModelConfig.from_dict({"key_size": 16})
    with pytest.raises(ValueError, match="mapping"):
        ModelConfig.from_dict([])
    for condition, name in (("16", "A"), ("main16", "Model1"), ("main64", "low")):
        with pytest.raises(ValueError):
            ModelConfig.candidate(condition, name)
    with pytest.raises(ValueError):
        ModelConfig.candidate("main16", "A", masked="False")
    assert replace(ModelConfig(), mask_rho=None).mask_rho is None
    assert ModelConfig.candidate("main16", "A").to_dict() == {
        "t": 96, "b": 128, "k": 9, "d_model": 96, "attention_projection_dim": 64,
        "h": 4, "layers": 3, "d_ff": 64, "output_window": 15, "mask_rho": 2.6,
        "dropout": 0.0, "layer_norm_eps": 1e-5, "negative_slope": 0.2,
        "initialization": "xavier_uniform",
    }
