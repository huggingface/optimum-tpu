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
from typing import Any, Dict, List, Union

from transformers.utils import logging


PreTrainedModel = Any
# NOTE: instead of the above, modeling_utils.PreTrainedModel should be used, but since the usage is only for type
# hinting, it is not imported here, so to avoid pulling imports of torch_xla.


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
    model_type = model.config.model_type
    matched_model = False
    if model_type == "gemma":
        from transformers import GemmaForCausalLM as HFGemmaForCausalLLM

        from .modeling_gemma import GemmaForCausalLM

        if isinstance(model, GemmaForCausalLM) or isinstance(model, HFGemmaForCausalLLM):
            logger = logging.get_logger(__name__)
            from torch_xla import __version__ as xla_version

            if xla_version == "2.3.0":
                logger.warning_once(
                    "Fine-tuning Gemma on Pytorch XLA 2.3.0 might raise some issues. In case of any "
                    "issues consider using the nightly version, and report the issue on the optimum-tpu "
                    "GitHub repository: https://github.com/huggingface/optimum-tpu/issues/new."
                )
            cls_to_wrap = "GemmaDecoderLayer"
            matched_model = True
    elif model_type == "llama":
        from transformers import LlamaForCausalLM as HFLlamaForCausalLLM

        from .modeling_llama import LlamaForCausalLM

        if isinstance(model, LlamaForCausalLM) or isinstance(model, HFLlamaForCausalLLM):
            cls_to_wrap = "LlamaDecoderLayer"
            matched_model = True

    if not matched_model:
        raise ValueError(f"Model {model} configuration cannot be auto-generated, use get_fsdp_config instead.")

    fsdp_training_args = {
        "fsdp": "full_shard",
        "fsdp_config": get_fsdp_config(cls_to_wrap),
    }
    return fsdp_training_args
