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
Utility functions to provide FSDPv2 sharding on TPU.
"""
import functools
from typing import Dict, List, Union

import numpy as np
import torch
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)
from torch_xla.experimental.spmd_fully_sharded_data_parallel import (
    SpmdFullyShardedDataParallel as FSDPv2,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer_pt_utils import get_module_class_from_name


def setup_mesh():
    """
    Set up the global mesh for FSDPv2 sharding.
    """
    num_devices = xr.global_runtime_device_count()
    xs.set_global_mesh(xs.Mesh(np.array(range(num_devices)), (num_devices, 1), axis_names=("fsdp", "tensor")))


def use_fsdp_v2():
    """
    Enable FSDPv2 for TPU and setup global mesh.
    """
    # FSDPv2 requires SPMD to be enabled.
    xr.use_spmd()
    setup_mesh()


def class_to_wrap(model: PreTrainedModel) -> Union[str | None]:
    """
    Returns the class name to wrap with FSDPv2 for a given model.

    Args:
        model: The model to wrap with FSDPv2.

    Returns:
        The class name to wrap with FSDPv2.
    """
    model_type = model.config.model_type
    if model_type == "gemma":
        from .modeling_gemma import GemmaForCausalLM

        if isinstance(model, GemmaForCausalLM):
            return "GemmaDecoderLayer"
    elif model_type == "llama":
        from .modeling_llama import LlamaForCausalLM

        if isinstance(model, LlamaForCausalLM):
            return "LlamaDecoderLayer"
    return None

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


def _shard_output(output, mesh):

    real_output = None
    if isinstance(output, torch.Tensor):
        real_output = output
    elif isinstance(output, tuple):
        real_output = output[0]
    elif isinstance(output, CausalLMOutputWithPast):
        real_output = output.logits

    if real_output is None:
        raise ValueError("Something went wrong, the output of the model shouldn't be `None`")
    xs.mark_sharding(real_output, mesh, ("fsdp", None, None))


def wrap_model(model: PreTrainedModel) -> FSDPv2:
    """
    Wraps a model with FSDPv2 so it can get sharded across multiple TPUs.

    Args:
        model: The model to wrap.

    Returns:
        The model wrapped with FSDPv2.
    """
    fsdp_transformer_layer_cls_to_wrap = class_to_wrap(model)
    transformer_cls_to_wrap = {get_module_class_from_name(model, fsdp_transformer_layer_cls_to_wrap)}

    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        # Transformer layer class to wrap
        transformer_layer_cls=transformer_cls_to_wrap,
    )
    wrapped_model = FSDPv2(
        model,
        shard_output=_shard_output,
        auto_wrap_policy=auto_wrap_policy,
    )
    return wrapped_model

def _unwrap_model(model: PreTrainedModel) -> PreTrainedModel:
    """
    Unwraps the model from the PeftModel wrapper.

    Args:
        model: The model to unwrap.

    Returns:
        The unwrapped model.
    """
    try:
        from peft.peft_model import LoraModel, PeftModel

        if isinstance(model, PeftModel) and isinstance(model.base_model, LoraModel):
            return model.base_model.model
        return model
    except ImportError:
        return model


def get_fsdp_training_args(model: PreTrainedModel) -> Dict:
    """
    Returns the default FSDPv2 training arguments for a model of a known class.

    Args:
        model: The model to train with FSDPv2.

    Returns:
        A dictionary with the FSDPv2 training arguments.
    """
    model = _unwrap_model(model)
    cls_to_wrap = class_to_wrap(model)
    if not cls_to_wrap:
        raise ValueError(f"Model {model} configuration cannot be auto-generated, use get_fsdp_config instead.")

    fsdp_training_args = {
        "fsdp": "full_shard",
        "fsdp_config": get_fsdp_config(cls_to_wrap),
    }
    return fsdp_training_args
