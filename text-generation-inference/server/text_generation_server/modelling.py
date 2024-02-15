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

from os import PathLike
from typing import Any

from transformers import AutoModelForCausalLM


# TODO: For now TpuModelForCausalLM is just a shallow wrapper of
# AutoModelForCausalLM, later this could be replaced by a custom class.
class TpuModelForCausalLM(AutoModelForCausalLM):

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
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        # Update config with specific data)
        if task is not None or getattr(model.config, "task", None) is None:
            model.config.task = task
        if batch_size is not None or getattr(model.config, "batch_size", None) is None:
            model.config.batch_size = batch_size
        if sequence_length is not None or getattr(model.config, "sequence_length", None) is None:
            model.config.sequence_length = sequence_length
        return model
