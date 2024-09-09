from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import torch
from jetstream.engine import engine_api
from jetstream_pt import engine


class HfEngine(engine.PyTorchEngine):
    def __init__(
        self,
        pt_model: torch.nn.Module,
        env: engine.JetEngineEnvironment,
        weights=None,
    ):
        super().__init__(pt_model, env, weights)
