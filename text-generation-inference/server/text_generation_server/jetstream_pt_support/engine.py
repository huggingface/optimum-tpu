from jetstream_pt import engine
import torch
import jax
from jax import numpy as jnp

from typing import Optional, Any, Tuple
from jetstream.engine import engine_api

class HfEngine(engine.Engine):
    def __init__(
        self,
        pt_model: torch.nn.Module,
        env: engine.JetEngineEnvironment,
        weights=None,
    ):
        super().__init__(pt_model, env, weights)

    def decode_ex(self, input_data, state):
        pass

    def prefill_ex(
        self,
        *,
        params: Any,  # Weights
        existing_prefix: Optional[engine.Prefix] = None,
        padded_tokens: engine.PrefillInputs,  # PrefillInputs[jax.Array],
        true_length: int,
    ) -> Tuple[engine.Prefix, engine_api.ResultTokens]:
        if isinstance(padded_tokens, jax.Array):
            batched_token = padded_tokens.reshape(1, -1)
        else:
            raise TypeError(
                "Input tokens should be of type Jax Array, but receiving:"
                " {prefill_inputs}"
            )
        seq_len = padded_tokens.shape[0]
        input_indexes = jnp.arange(0, seq_len)
        logits, updated_caches = self._call_model_prefill(
            params,
            batched_token,
            input_indexes,
        )
        if len(logits.shape) == 3:  # b, seqlen, num words
            logits = logits[0]  # seqlen, num words

        # TODO: replace this sampling with the one from Generator
        token = sampling_utils.sampling(
            logits[true_length - 1],
            self.rng,
            self.env.sampling_algorithm,
            self.env.topk,
            self.env.nucleus_topp,
            self.env.temperature,
        )
        token_out = jnp.reshape(token, (1, 1))
        data = jnp.concatenate(
            [
                token_out,  # First token
                jnp.ones_like(token_out),  # validity of first token
                jnp.zeros((1, 1), dtype=jnp.int32),  # length = 0
            ],
            axis=-1,
        )
        length = token_out.shape[1]
        result = engine_api.ResultTokens(
            data=data,
            tokens_idx=(0, length),
            valid_idx=(length, 2 * length),
            length_idx=(2 * length, 2 * length + 1),
            samples_per_slot=1,
        )
        # truncate to true_length didnt work need to be out side of jit
        # caches = [
        #   (jax.lax.dynamic_slice_in_dim(
        #       k, seq_len - true_length, true_length, axis=2),
        #    jax.lax.dynamic_slice_in_dim(
        #       v, seq_len - true_length, true_length, axis=2))
        #   for k, v in updated_caches
        # ]
        return Prefix(token, updated_caches, true_length), result
