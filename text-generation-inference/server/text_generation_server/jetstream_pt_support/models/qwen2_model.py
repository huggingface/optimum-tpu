"""
Qwen2 model implementation, based on Jetstream implementation of Llama model.
"""

from typing import Any, List, Optional
import copy
import jax
import math
import torch
import torch.nn.functional as F
from jetstream_pt.model_base import ModuleBase
from jetstream_pt.layers import (
    Attention,
    RMSNorm,
    get_quantized_embedding_layer,
    get_quantized_linear_layer,
)
from torch import nn

from . import model_args


class FeedForward(ModuleBase):
  """Feed-forward module."""

  def __init__(
      self,
      dim: int,
      hidden_dim: int,
      multiple_of: int,
      ffn_dim_multiplier: Optional[float],
      device="meta",
      env=None,
  ):
    super().__init__()
    self.env = env
    hidden_dim = int(2 * hidden_dim / 3)
    # custom dim factor multiplier
    if ffn_dim_multiplier is not None:
      hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

    LinearLayer = get_quantized_linear_layer(env.quant_config)
    linear_kwargs = {}
    if LinearLayer != torch.nn.Linear:
      linear_kwargs["quant_config"] = env.quant_config

    self.w1 = LinearLayer(
        dim,
        hidden_dim,
        bias=False,
        device=device,
        **linear_kwargs,
    )
    self.w2 = LinearLayer(
        hidden_dim,
        dim,
        bias=False,
        device=device,
        **linear_kwargs,
    )
    self.w3 = LinearLayer(
        dim,
        hidden_dim,
        bias=False,
        device=device,
        **linear_kwargs,
    )
    self.hf_name("w1", "gate_proj")
    self.hf_name("w2", "down_proj")
    self.hf_name("w3", "up_proj")

    self.annotate_sharding("w1.weight", 0)
    self.annotate_sharding("w2.weight", 1)
    self.annotate_sharding("w3.weight", 0)
    if LinearLayer != torch.nn.Linear:
      self.annotate_sharding("w1.weight_scaler", 0)
      self.annotate_sharding("w2.weight_scaler", 0)
      self.annotate_sharding("w3.weight_scaler", 0)

  def forward(self, x):
    result = self.w2(F.silu(self.w1(x)) * self.w3(x))
    return result


class TransformerBlock(ModuleBase):
  """Transformer block."""

  def __init__(
      self,
      layer_id: int,
      args: model_args.ModelArgs,
      env,
  ):
    super().__init__()
    self.env = env
    self.n_heads = args.n_heads
    self.dim = args.dim
    self.head_dim = args.dim // args.n_heads
    self.args = args

    self.attention = Attention(
        args.n_heads,
        args.n_kv_heads or args.n_heads,
        args.dim // args.n_heads,
        args.dim,
        env=env,
        device=args.device,
        layer_id=layer_id,
    )
    self.feed_forward = FeedForward(
        dim=args.dim,
        hidden_dim=4 * args.dim,
        multiple_of=args.multiple_of,
        ffn_dim_multiplier=args.ffn_dim_multiplier,
        device=args.device,
        env=env,
    )
    self.layer_id = layer_id
    self.attention_norm = RMSNorm(
        args.dim, eps=args.norm_eps, device=args.device
    )
    self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps, device=args.device)

    self.hf_name("attention", "self_attn")
    self.attention.hf_name("wq", "q_proj")
    self.attention.hf_name("wk", "k_proj")
    self.attention.hf_name("wv", "v_proj")
    self.attention.hf_name("wo", "o_proj")

    self.attention.annotate_sharding("wq.weight", 0)
    self.attention.annotate_sharding("wk.weight", 0)
    self.attention.annotate_sharding("wv.weight", 0)
    self.attention.annotate_sharding("wo.weight", 1)

    self.hf_name("feed_forward", "mlp")
    self.hf_name("attention_norm", "input_layernorm")
    self.hf_name("ffn_norm", "post_attention_layernorm")

  def forward(
      self,
      x: torch.Tensor,
      freqs_cis: torch.Tensor,
      mask: Optional[torch.Tensor],
      cache,
      start=None,
      end=None,
      ragged_batch_index=None,
      ragged_block_index=None,
  ):
    with jax.named_scope("Attention"):
      attn = self.attention.forward(
          self.attention_norm(x),
          freqs_cis,
          mask,
          cache,
          start,
          end,
          ragged_batch_index,
          ragged_block_index,
      )
    with jax.named_scope("ffn_norm"):
      h = x + attn
      ffns = self.ffn_norm(h)

    with jax.named_scope("ffn"):
      out = h + self.feed_forward.forward(ffns)
      return out


def apply_scaling(freqs: torch.Tensor, config: model_args.RopeScalingArgs):
  # Values obtained from grid search
  scale_factor = config.factor
  low_freq_factor = config.low_freq_factor
  high_freq_factor = config.high_freq_factor
  old_context_len = config.original_max_position_embeddings

  low_freq_wavelen = old_context_len / low_freq_factor
  high_freq_wavelen = old_context_len / high_freq_factor
  new_freqs = []
  for freq in freqs:
    wavelen = 2 * math.pi / freq
    if wavelen < high_freq_wavelen:
      new_freqs.append(freq)
    elif wavelen > low_freq_wavelen:
      new_freqs.append(freq / scale_factor)
    else:
      assert low_freq_wavelen != high_freq_wavelen
      smooth = (old_context_len / wavelen - low_freq_factor) / (
          high_freq_factor - low_freq_factor
      )
      new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
  return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
    rope_scaling_config: model_args.RopeScalingArgs = None,
):
  freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
  t = torch.arange(end, device=freqs.device, dtype=torch.float32)
  if rope_scaling_config is not None:
    freqs = apply_scaling(freqs, rope_scaling_config)
  freqs = torch.outer(t, freqs)
  freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
  return freqs_cis


class Transformer(ModuleBase):
  """Transformer module."""

  def __init__(
      self,
      params: model_args.ModelArgs,
      env,
  ):
    super().__init__()
    self.env = env
    self.params = params
    self.vocab_size = params.vocab_size
    self.n_layers = params.n_layers

    Embedding = get_quantized_embedding_layer(env.quant_config)
    self.tok_embeddings = Embedding(
        params.vocab_size,
        params.dim,
        device=params.device,
    )

    self.layers = torch.nn.ModuleList()
    for layer_id in range(params.n_layers):
      self.layers.append(TransformerBlock(layer_id, params, env))
    self.norm = RMSNorm(params.dim, eps=params.norm_eps, device=params.device)

    LinearLayer = get_quantized_linear_layer(env.quant_config)
    linear_kwargs = {}
    if LinearLayer != torch.nn.Linear:
      linear_kwargs["quant_config"] = env.quant_config

    self.output = LinearLayer(
        params.dim,
        params.vocab_size,
        bias=False,
        device=params.device,
        **linear_kwargs,
    )
    # TODO what to do with this
    freqs_cis = precompute_freqs_cis(
        self.params.dim // self.params.n_heads,
        self.params.max_seq_len * 2,
        theta=self.params.rope_theta,
        rope_scaling_config=self.params.rope_scaling_args,
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

  @classmethod
  def from_hf_model_id(cls, model_id, env, is_tiny=False):
    if is_tiny:
      name = "llama-2-tiny"
    else:
      name = {
          "meta-llama/Llama-2-7b-chat-hf": "llama-2-7b",
          "meta-llama/Llama-2-7b-hf": "llama-2-7b",
          "meta-llama/Llama-2-13b-chat-hf": "llama-2-13b",
          "meta-llama/Llama-2-13b-hf": "llama-2-13b",
          "meta-llama/Llama-2-70b-hf": "llama-2-70b",
          "meta-llama/Llama-2-70b-chat-hf": "llama-2-70b",
          "meta-llama/Meta-Llama-3-8B": "llama-3-8b",
          "meta-llama/Meta-Llama-3-8B-Instruct": "llama-3-8b",
          "meta-llama/Meta-Llama-3-70B": "llama-3-70b",
          "meta-llama/Meta-Llama-3-70B-Instruct": "llama-3-70b",
          "meta-llama/Llama-3.1-8B": "llama-3.1-8b",
          "meta-llama/Llama-3.1-8B-Instruct": "llama-3.1-8b",
          "meta-llama/Llama-3.2-1B": "llama-3.2-1b",
          "meta-llama/Llama-3.2-1B-Instruct": "llama-3.2-1b",
          "meta-llama/Llama-3.3-70B": "llama-3.3-70b",
          "meta-llama/Llama-3.3-70B-Instruct": "llama-3.3-70b",
      }.get(model_id)
    assert name
    args = model_args.get_model_args(
        name, env.cache_len, env.batch_size, env.bf16_enable
    )
    args.device = "meta"
    model = cls(args, env)
    return model

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
      if "q_proj" in key:
        updated[key] = transform(value, self.params.n_heads)
      if "k_proj" in key:
        updated[key] = transform(
            value, self.params.n_kv_heads or self.params.n_heads
        )
    res = super().convert_hf_weights(updated)
    res["freqs_cis"] = self.freqs_cis
    return res
