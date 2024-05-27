#  Copyright 2024 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
Utility functions to provide FSDPv2 configuration for TPU training.
"""
from typing import Dict, List, Union

from transformers.modeling_utils import PreTrainedModel


def use_fsdp_v2():
    """
    Enable FSDPv2 for TPU training.
    """
    import torch_xla.runtime as xr

    # FSDPv2 requires SPMD to be enabled.
    xr.use_spmd()


def get_fsdp_config(*cls_to_wrap: Union[str | List[str]]) -> Dict:
    """
    Returns the FSDPv2 configuration for a given class to wrap.

    Args:
        cls_to_wrap: One or more class names to wrap with FSDPv2.

    Returns:
        A dictionary with the FSDPv2 configuration.
    """
    return {
        "transformer_layer_cls_to_wrap": [*cls_to_wrap],
        "xla": True,
        "xla_fsdp_v2": True,
        "xla_fsdp_grad_ckpt": True,
    }


def get_fsdp_training_args(model: PreTrainedModel) -> Dict:
    """
    Returns the default FSDPv2 training arguments for a model of a known class.

    Args:
        model: The model to train with FSDPv2.

    Returns:
        A dictionary with the FSDPv2 training arguments.
    """
    model_type = model.config.model_type
    matched_model = True
    if model_type == "gemma":
        from .modeling_gemma import TpuGemmaForCausalLM

        if isinstance(model, TpuGemmaForCausalLM):
            cls_to_wrap = "TpuGemmaDecoderLayer"
            matched_model = True
    elif model_type == "llama":
        from .modeling_llama import LlamaForCausalLM

        if isinstance(model, LlamaForCausalLM):
            cls_to_wrap = "LlamaDecoderLayer"
            matched_model = True

    if not matched_model:
        raise ValueError(f"Model {model} configuration cannot be auto-generated, use get_fsdp_config instead.")

    fsdp_training_args = {
        "fsdp": "full_shard",
        "fsdp_config": get_fsdp_config(cls_to_wrap),
    }
    return fsdp_training_args
