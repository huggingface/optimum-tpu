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

import os
from typing import Any

from transformers import AutoConfig


def verify_imports() -> bool:
    """Check if the necessary imports to use jetstream_pt are available.
    """
    try:
        # Import torch_xla2 first!
        import torch_xla2  # noqa: F401, isort:skip

        import jetstream_pt  # noqa: F401

        return True
    except ImportError:
        return False

def check(model_path: str) -> bool:
    """Checks if the model is supported by Jetstream Pytorch on Optimum TPU and if the required dependencies to provide
    the engine are installed.
    """
    config = AutoConfig.from_pretrained(model_path)
    # For now only Llama 2 with tokenizer.model is supported
    if config.model_type == "llama" and os.path.exists(
        os.path.join(model_path, "tokenizer.model")
    ):
        return verify_imports()
    return False


def create_engine(
    model_path: str,
    batch_size: int,
    sequence_length: int,
    max_input_tokens: int,
    max_output_tokens: int,
) -> Any:
    if not check(model_path):
        # The model is not compatible with Jetstream PyTorch, just exit
        return None

    # Now import engine_loader to prevent importing it at the top when not supported
    from .engine_loader import create_engine
    return create_engine(
        model_path, batch_size, sequence_length, max_input_tokens, max_output_tokens
    )
