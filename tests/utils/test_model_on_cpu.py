# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace  # Or use a mock object library

import pytest

from verl.utils import model as model_utils
from verl.utils.model import update_model_config


# Parametrize with different override scenarios
@pytest.mark.parametrize(
    "override_kwargs",
    [
        {"param_a": 5, "new_param": "plain_added"},
        {"param_a": 2, "nested_params": {"sub_param_x": "updated_x", "sub_param_z": True}},
    ],
)
def test_update_model_config(override_kwargs):
    """
    Tests that update_model_config correctly updates attributes,
    handling both plain and nested overrides via parametrization.
    """
    # Create a fresh mock config object for each test case
    mock_config = SimpleNamespace(
        param_a=1, nested_params=SimpleNamespace(sub_param_x="original_x", sub_param_y=100), other_param="keep_me"
    )
    # Apply the updates using the parametrized override_kwargs
    update_model_config(mock_config, override_kwargs)

    # Assertions to check if the config was updated correctly
    if "nested_params" in override_kwargs:  # Case 2: Nested override
        override_nested = override_kwargs["nested_params"]
        assert mock_config.nested_params.sub_param_x == override_nested["sub_param_x"], "Nested sub_param_x mismatch"
        assert mock_config.nested_params.sub_param_y == 100, "Nested sub_param_y should be unchanged"
        assert hasattr(mock_config.nested_params, "sub_param_z"), "Expected nested sub_param_z to be added"
        assert mock_config.nested_params.sub_param_z == override_nested["sub_param_z"], "Value of sub_param_z mismatch"
    else:  # Case 1: Plain override (nested params untouched)
        assert mock_config.nested_params.sub_param_x == "original_x", "Nested sub_param_x should be unchanged"
        assert mock_config.nested_params.sub_param_y == 100, "Nested sub_param_y should be unchanged"
        assert not hasattr(mock_config.nested_params, "sub_param_z"), "Nested sub_param_z should not exist"


def test_qwen4_exp_text_model_takes_precedence_over_vlm_config_mapping(monkeypatch):
    config = SimpleNamespace(
        architectures=["Qwen4ExpForCausalLM"],
        model_type="qwen4_exp",
    )
    monkeypatch.setattr(
        model_utils,
        "AutoModelForImageTextToText",
        SimpleNamespace(_model_mapping={type(config): object()}),
    )

    assert model_utils.get_hf_auto_model_class(config) is model_utils.AutoModelForCausalLM


def test_vlm_causal_lm_architecture_keeps_vlm_config_mapping(monkeypatch):
    config = SimpleNamespace(
        architectures=["ExampleVisionForCausalLM"],
        model_type="example_vision",
    )
    vlm_auto_class = SimpleNamespace(_model_mapping={type(config): object()})
    monkeypatch.setattr(model_utils, "AutoModelForImageTextToText", vlm_auto_class)

    assert model_utils.get_hf_auto_model_class(config) is vlm_auto_class


def test_qwen4_exp_multimodal_architecture_keeps_vlm_config_mapping(monkeypatch):
    config = SimpleNamespace(
        architectures=["Qwen4ExpForConditionalGeneration"],
        model_type="qwen4_exp",
    )
    vlm_auto_class = SimpleNamespace(_model_mapping={type(config): object()})
    monkeypatch.setattr(model_utils, "AutoModelForImageTextToText", vlm_auto_class)

    assert model_utils.get_hf_auto_model_class(config) is vlm_auto_class


def test_qwen4_exp_without_a_text_architecture_keeps_vlm_config_mapping(monkeypatch):
    config = SimpleNamespace(architectures=None, model_type="qwen4_exp")
    vlm_auto_class = SimpleNamespace(_model_mapping={type(config): object()})
    monkeypatch.setattr(model_utils, "AutoModelForImageTextToText", vlm_auto_class)

    assert model_utils.get_hf_auto_model_class(config) is vlm_auto_class


def test_remote_code_mapping_takes_precedence(monkeypatch):
    config = SimpleNamespace(
        architectures=["ExampleForCausalLM"],
        auto_map={"AutoModelForVision2Seq": "example.ExampleForCausalLM"},
    )
    monkeypatch.setattr(
        model_utils,
        "AutoModelForImageTextToText",
        SimpleNamespace(_model_mapping={type(config): object()}),
    )

    assert model_utils.get_hf_auto_model_class(config) is model_utils.AutoModelForVision2Seq


def test_unknown_remote_code_auto_class_falls_back_to_auto_model(monkeypatch):
    config = SimpleNamespace(
        architectures=["ExampleForCausalLM"],
        auto_map={"AutoModelForSeq2SeqLM": "example.ExampleForCausalLM"},
    )
    monkeypatch.setattr(
        model_utils,
        "AutoModelForImageTextToText",
        SimpleNamespace(_model_mapping={}),
    )

    assert model_utils.get_hf_auto_model_class(config) is model_utils.AutoModel


def test_vlm_config_mapping_is_used_without_a_known_architecture(monkeypatch):
    architecture = "ExampleForConditionalGeneration"
    assert not any(key in architecture for key in model_utils._architecture_to_auto_class)
    config = SimpleNamespace(architectures=[architecture])
    vlm_auto_class = SimpleNamespace(_model_mapping={type(config): object()})
    monkeypatch.setattr(model_utils, "AutoModelForImageTextToText", vlm_auto_class)

    assert model_utils.get_hf_auto_model_class(config) is vlm_auto_class


def test_unknown_architecture_falls_back_to_auto_model(monkeypatch):
    config = SimpleNamespace(architectures=["ExampleForUnsupportedTask"])
    monkeypatch.setattr(
        model_utils,
        "AutoModelForImageTextToText",
        SimpleNamespace(_model_mapping={}),
    )

    assert model_utils.get_hf_auto_model_class(config) is model_utils.AutoModel


@pytest.mark.parametrize("architectures", [None, []])
def test_missing_architecture_falls_back_to_auto_model(monkeypatch, architectures):
    config = SimpleNamespace(
        architectures=architectures,
        auto_map=None,
    )
    monkeypatch.setattr(
        model_utils,
        "AutoModelForImageTextToText",
        SimpleNamespace(_model_mapping={}),
    )

    assert model_utils.get_hf_auto_model_class(config) is model_utils.AutoModel


def test_remote_code_without_an_architecture_falls_back_to_auto_model(monkeypatch):
    config = SimpleNamespace(
        architectures=None,
        auto_map={"AutoModelForCausalLM": "example.ExampleForCausalLM"},
    )
    monkeypatch.setattr(
        model_utils,
        "AutoModelForImageTextToText",
        SimpleNamespace(_model_mapping={}),
    )

    assert model_utils.get_hf_auto_model_class(config) is model_utils.AutoModel
