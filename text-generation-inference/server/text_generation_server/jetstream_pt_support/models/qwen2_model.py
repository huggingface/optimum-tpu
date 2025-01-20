"""
Qwen2 model implementation, based on Jetstream implementation of Llama model.
"""

import copy
from typing import Any, List, Optional

import jax
import torch
import torch.nn.functional as F
from jetstream_pt.layers import (
  AttentionKernel,
  Int8KVAttentionKernel,
  RMSNorm,
  apply_rotary_emb,
  get_quantized_embedding_layer,
  get_quantized_linear_layer,
)
from jetstream_pt.model_base import ModuleBase

# Use llama's functions and classes that are the same as in Qwen2
from jetstream_pt.third_party.llama.model_exportable import model_args
from transformers import GenerationConfig, GenerationMixin, Qwen2Config


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

class QwenAttention(ModuleBase):
  """Attention module."""

  def __init__(
      self, n_heads, n_kv_heads, head_dim, hidden_size, device, env, layer_id
  ):
    super().__init__()
    self.n_heads = n_heads
    self.n_kv_heads = n_kv_heads
    self.head_dim = head_dim
    self.n_rep = self.n_heads // self.n_kv_heads
    self.env = env
    self.hidden_size = hidden_size
    self.layer_id = layer_id

    LinearLayer = get_quantized_linear_layer(env.quant_config)
    linear_kwargs = {}
    if LinearLayer != torch.nn.Linear:
      linear_kwargs = {"quant_config": env.quant_config}

    self.wo = LinearLayer(
        n_heads * self.head_dim,
        hidden_size,
        bias=False,
        device=device,
        **linear_kwargs,
    )

    Kernel = (
        Int8KVAttentionKernel
        if env.quant_config.enable_kv_quantization
        else AttentionKernel
    )
    self.attention_kernel = Kernel(env, self.layer_id)

    self.q_size = n_heads * self.head_dim
    self.kv_size = self.n_kv_heads * self.head_dim
    if self.env.qkv_fusion:
      self._register_load_state_dict_pre_hook(self.load_hook)
      self.wqkv = LinearLayer(
          hidden_size,
          (n_heads + 2 * self.n_kv_heads) * self.head_dim,
          bias=True,
          device=device,
          **linear_kwargs,
      )
    else:
      self.wq = LinearLayer(
          hidden_size,
          n_heads * self.head_dim,
          bias=True,
          device=device,
          **linear_kwargs,
      )
      self.wk = LinearLayer(
          hidden_size,
          self.n_kv_heads * self.head_dim,
          bias=True,
          device=device,
          **linear_kwargs,
      )
      self.wv = LinearLayer(
          hidden_size,
          self.n_kv_heads * self.head_dim,
          bias=True,
          device=device,
          **linear_kwargs,
      )

  def load_hook(self, state_dict, prefix, *args):
    if prefix + "wq.weight" in state_dict:
      wq = state_dict.pop(prefix + "wq.weight")
      wk = state_dict.pop(prefix + "wk.weight")
      wv = state_dict.pop(prefix + "wv.weight")
      state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

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
    with jax.named_scope("attn_linear_before_cache"):
      bsz, seqlen = x.shape[0], x.shape[-2]

      # qkv fuse
      if self.env.qkv_fusion:
        xq, xk, xv = self.wqkv(x).split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1
        )
      else:
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
      xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
      xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
      xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

      shard_axis = 0 if self.env.shard_on_batch else 2
      self.env.apply_sharding(xq, axis=shard_axis)
      self.env.apply_sharding(xk, axis=shard_axis)
      self.env.apply_sharding(xv, axis=shard_axis)

    with jax.named_scope("attn_rope"):
      xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

    xk = xk.transpose(1, 2)
    xv = xv.transpose(1, 2)
    xq = xq.transpose(1, 2)

    if mask.ndim == 2:
      if seqlen == 1:
        mask = mask[:, None, None, :]
      else:
        mask = mask[None, None, :, :]

    # if cache is not None and cache.cache_k is not None:
    # print(f"xq {xq.shape} xk {xk.shape} cache shape {cache.cache_k.shape}")
    output = self.attention_kernel(
        xq=xq,
        xk=xk,
        xv=xv,
        mask=mask,
        # cache[self.layer_id],
        cache=cache,
        start=start,
        end=end,
        ragged_batch_index=ragged_batch_index,
        ragged_block_index=ragged_block_index,
    ).type_as(xq)
    # print(f"output {output.shape}")
    output = output.transpose(-3, -2).contiguous().view(bsz, seqlen, -1)
    return self.wo(output)


class Qwen2DecoderLayer(ModuleBase):
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

    self.attention = QwenAttention(
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
    self.attention.annotate_sharding("wq.weight.bias", 0)
    self.attention.annotate_sharding("wk.weight.bias", 0)
    self.attention.annotate_sharding("wv.weight.bias", 0)
    self.attention.annotate_sharding("wo.weight.bias", -1)

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
  """Qwen2 module."""

  def __init__(
      self,
      config: Qwen2Config,
      device,
      env,
  ):
    if config.sliding_window is not None:
        raise ValueError("Sliding window is not supported for Qwen2 model")
    if config.rope_scaling is not None:
        raise ValueError("Rope scaling is not supported for Qwen2 model")

    super().__init__()
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
      self.layers.append(Qwen2DecoderLayer(layer_id, params, env))
    self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=params.device)

    LinearLayer = get_quantized_linear_layer(env.quant_config)
    linear_kwargs = {}
    if LinearLayer != torch.nn.Linear:
      linear_kwargs["quant_config"] = env.quant_config

    self.output = LinearLayer(
        config.hidden_size,
        config.vocab_size,
        bias=False,
        device=params.device,
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
