# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""TpuModelForXXX classes for inference on TPU devices using the same API as
Transformers."""

from os import PathLike, environ
from typing import Any

from loguru import logger
from transformers import AutoConfig
from transformers import AutoModelForCausalLM as BaseAutoModelForCausalLM


def config_name_to_class(pretrained_model_name_or_path: str):
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
    if config.model_type == "gemma":
        from .modeling_gemma import GemmaForCausalLM

        return GemmaForCausalLM
    if config.model_type == "llama":
        from .modeling_llama import LlamaForCausalLM

        return LlamaForCausalLM
    if config.model_type == "mistral":
        from .modeling_mistral import MistralForCausalLM

        return MistralForCausalLM
    return BaseAutoModelForCausalLM


class AutoModelForCausalLM(BaseAutoModelForCausalLM):

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | PathLike[str],
        task: str = None,
        batch_size: int = None,
        sequence_length: int = None,
        *model_args: Any,
        **kwargs: Any,
    ):
        if "PJRT_DEVICE" not in environ:
            logger.info("PJRT_DEVICE environment variable not found. Setting it to 'TPU'.")
            environ["PJRT_DEVICE"] = "TPU"
        if "DBG_DEVICE" in environ:
            device = environ["DBG_DEVICE"]
            logger.debug(f"Device set to: {device}")
        else:
            device = "xla"
        cls = config_name_to_class(pretrained_model_name_or_path)
        model = cls.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model.to(device)

        # Update config with specific data)
        if task is not None or getattr(model.config, "task", None) is None:
            model.config.task = task
        if batch_size is not None or getattr(model.config, "batch_size", None) is None:
            model.config.batch_size = batch_size
        if sequence_length is not None or getattr(model.config, "sequence_length", None) is None:
            model.config.sequence_length = sequence_length
        # Do eval
        model.eval()

        return model
