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

    def generate_ex(
        self, params: Any, decode_state: engine.DecodeState, sampling_fn: Callable[[Any, int], jax.Array]
    ) -> tuple[engine.DecodeState, engine_api.ResultTokens]:
        sampling_fn_backup = self._sampling
        self._sampling = sampling_fn
        new_decode_state, result_tokens = self.generate(params, decode_state)
        self._sampling = sampling_fn_backup
        return new_decode_state, result_tokens
