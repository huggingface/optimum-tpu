import copy
import dataclasses
import math
from contextlib import contextmanager
from functools import partial

import torch
from jetstream_pt.third_party.llama import model_exportable
from jetstream_pt.third_party.llama.model_exportable import Transformer, model_args, TransformerBlock
from transformers import GenerationConfig, GenerationMixin, LlamaConfig
import jax
import torch
import torch.nn.functional as F
from typing import Any, List, Optional
from jetstream_pt.layers import (
  AttentionKernel,
  Int8KVAttentionKernel,
  RMSNorm,
  apply_rotary_emb,
  get_quantized_embedding_layer,
  get_quantized_linear_layer,
)
from jetstream_pt.model_base import ModuleBase
from transformers import GenerationConfig, GenerationMixin, Qwen2Config




def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
):
  freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
  t = torch.arange(end, device=freqs.device, dtype=torch.float32)
  freqs = torch.outer(t, freqs)
  freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
  return freqs_cis


class Qwen2Model(ModuleBase, GenerationMixin):
  """Transformer module that uses HF LlamaConfig instead of Jetstream Pytorch ModelArgs + device.

  Note that this class also derives from GenerationMixin, so that we can use its methods.
  """


  def __init__(
      self,
      config: Qwen2Config,
      device,
      env,
  ):
    super().__init__()
    if config.sliding_window is not None:
        raise ValueError("Sliding window is not supported for Qwen2 model")
    if config.rope_scaling is not None:
        raise ValueError("Rope scaling is not supported for Qwen2 model")

    self.config = config
    self.generation_config = GenerationConfig.from_model_config(config)

    # NOTE: these parameters are deduced from the config's intermediate_size and hidden_size, so to be compatible
    # with the original Jestream/Pytorch model.
    ffn_dim_multiplier = config.intermediate_size / int(8 * config.hidden_size / 3)
    multiple_of = 1

    params = model_args.ModelArgs(
        dim=config.hidden_size,
        n_layers=config.num_hidden_layers,
        n_heads=config.num_attention_heads,
        n_kv_heads=config.num_key_value_heads,
        vocab_size=config.vocab_size,
        multiple_of=multiple_of,
        ffn_dim_multiplier=ffn_dim_multiplier,
        norm_eps=config.rms_norm_eps,
        max_seq_len=env.cache_len,
        bf16_enable=env.bf16_enable,
        rope_theta=config.rope_theta,
    )
    params.device = device

    self.env = env
    self.params = params
    self.vocab_size = config.vocab_size
    self.n_layers = config.num_hidden_layers

    Embedding = get_quantized_embedding_layer(env.quant_config)
    self.tok_embeddings = Embedding(
        config.vocab_size,
        config.hidden_size,
        device=device,
    )

    self.layers = torch.nn.ModuleList()
    for layer_id in range(config.num_hidden_layers):
      self.layers.append(TransformerBlock(layer_id, params, env))
    self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=params.device)

    LinearLayer = get_quantized_linear_layer(env.quant_config)
    linear_kwargs = {}
    if LinearLayer != torch.nn.Linear:
      linear_kwargs["quant_config"] = env.quant_config

    self.output = LinearLayer(
        config.hidden_size,
        config.vocab_size,
        bias=False,
        device=device,
        **linear_kwargs,
    )
    freqs_cis = precompute_freqs_cis(
        config.hidden_size // config.num_attention_heads,
        self.params.max_seq_len * 2,
        theta=config.rope_theta,
    )

    self.register_buffer("freqs_cis", freqs_cis)

    self.hf_name("output", "lm_head")
    self.hf_name("norm", "model.norm")
    self.hf_name("layers", "model.layers")
    self.hf_name("tok_embeddings", "model.embed_tokens")

    self.annotate_sharding("tok_embeddings.weight", 1)
    self.annotate_sharding("output.weight", 0)


  @torch.no_grad()
  def forward(
      self,
      tokens: torch.Tensor,
      input_pos: torch.Tensor,
      caches: List[Any],
      mask,
      start=None,
      ragged_batch_index=None,
      ragged_block_index=None,
  ):
    """
    tokens: the input token for decoding
    input_pos: the decoding position relative to the start, which is the length of the decoding results
    caches: kv caches
    mask: causal mask to filter the attention results
    start: the starting position for each slot
    ragged_batch_index: precomputed batch index for ragged attention
    ragged_block_index: precomputed block index for ragged attention
    """
    with jax.named_scope("transformer_tok"):
      seqlen = tokens.shape[-1]
      h = self.tok_embeddings(tokens)

    with jax.named_scope("transformer_freq"):
      bsz, seqlen = tokens.shape
      freqs_cis = self.freqs_cis[input_pos]
      freqs_cis = freqs_cis.reshape(bsz, seqlen, -1)

    end = None if start is None else (start + input_pos) % self.env.cache_len
    # For stacked case, cannot get cache inside the loop which will cause cache copy
    for layer_id, layer in enumerate(self.layers):
      if caches[0].stacked:
        cache = caches[0]
      else:
        cache = caches[layer_id]
      # else:  # For stacked case, there is only 1 yer of kv cache

      with jax.named_scope("TransformerBlock_Layer_" + str(layer_id)):
        h = layer(
            h,
            freqs_cis,
            mask,
            cache,
            start,
            end,
            ragged_batch_index,
            ragged_block_index,
        )

    with jax.named_scope("transformer_norm"):
      h = self.norm(h)
      output = self.output(h).float()
    return output


  def convert_hf_weights(self, hf_weights):

    def transform(val, n_heads):
      dim1, dim2 = val.shape
      return (
          val.reshape(n_heads, 2, dim1 // n_heads // 2, dim2)
          .transpose(1, 2)
          .reshape(dim1, dim2)
      )

    updated = copy.copy(hf_weights)

    for key, value in hf_weights.items():
      if "bias" in key and ("q_proj" in key or "k_proj" in key):
        continue
      if "q_proj" in key:
        updated[key] = transform(value, self.params.n_heads)
      if "k_proj" in key:
        updated[key] = transform(
            value, self.params.n_kv_heads or self.params.n_heads
        )
    res = super().convert_hf_weights(updated)
    res["freqs_cis"] = self.freqs_cis
    return res

  @classmethod
  def from_config(cls, config, env):
      device = "meta"
      model = cls(config, device, env)
      return model
