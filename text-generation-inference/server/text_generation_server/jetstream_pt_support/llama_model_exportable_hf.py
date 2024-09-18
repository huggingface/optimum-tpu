import copy
from typing import Any, List, Optional

import jax
import torch
import torch.nn.functional as F
from jetstream_pt.layers import (
    Attention,
    RMSNorm,
    get_quantized_embedding_layer,
    get_quantized_linear_layer,
)
from jetstream_pt.model_base import ModuleBase
from transformers import GenerationConfig, GenerationMixin, LlamaConfig


class FeedForward(ModuleBase):
    """Feed-forward module, AKA LlamaMLP on HuggingFace.

    Note the main difference is that it uses intermediate_size instead of multiple_of and ffn_dim_multiplier.
    The parameter dim here corresponds to hidden_size in HuggingFace's Llama model, and hidden_dim is not really used,
    because intermediate_size is used instead.
    """

    def __init__(
        self,
        dim: int,
        intermediate_size: int,
        device="meta",
        env=None,
    ):
        super().__init__()
        self.env = env

        LinearLayer = get_quantized_linear_layer(env.quant_config)
        linear_kwargs = {}
        if LinearLayer != torch.nn.Linear:
            linear_kwargs["quant_config"] = env.quant_config

        self.w1 = LinearLayer(
            dim,
            intermediate_size,
            bias=False,
            device=device,
            **linear_kwargs,
        )
        self.w2 = LinearLayer(
            intermediate_size,
            dim,
            bias=False,
            device=device,
            **linear_kwargs,
        )
        self.w3 = LinearLayer(
            dim,
            intermediate_size,
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

    def forward(self, x):
        result = self.w2(F.silu(self.w1(x)) * self.w3(x))
        return result


class TransformerBlockHf(ModuleBase):
    """This is essentially the same as the JetstreamPytoch Transformer, but it avoids using multiple_of and
    ffn_dim_multiplier that are not available in HuggingFace's Llama model, and it uses intermediate_size instead.
    """

    def __init__(
        self,
        layer_id: int,
        config: LlamaConfig,
        device,
        env,
    ):
        super().__init__()
        self.env = env
        self.n_heads = config.num_attention_heads
        self.dim = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.attention = Attention(
            config.num_attention_heads,
            config.num_key_value_heads or config.num_attention_heads,
            config.hidden_size // config.num_attention_heads,
            config.hidden_size,
            env=env,
            device=device,
            layer_id=layer_id,
        )
        self.feed_forward = FeedForward(
            dim=config.hidden_size,
            intermediate_size=config.intermediate_size,
            device=device,
            env=env,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, device=device
        )
        self.ffn_norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, device=device
        )

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


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


class TransformerHf(ModuleBase, GenerationMixin):
    """Transformer module that uses HF LlamaConfig instead of Jetstream Pytorch ModelArgs + device.

    Note that this class also derives from GenerationMixin, so that we can use its methods.
    """

    def __init__(
        self,
        config: LlamaConfig,
        device,
        env,
    ):
        super().__init__()
        self.env = env
        self.config = config
        self.generation_config = GenerationConfig.from_model_config(config)
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
            self.layers.append(TransformerBlockHf(layer_id, config, device, env))
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device)

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
        # TODO what to do with this
        freqs_cis = precompute_freqs_cis(
            config.hidden_size // config.num_attention_heads,
            env.cache_len * 2,
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

    @classmethod
    def from_config(cls, config, env):
        device = "meta"
        model = cls(config, device, env)
        return model

    def drop_weight(self, key):
        return key.startswith("model")

    def shard_weights(self, _weights_dict):
        """Shards the weights

        Assumes the weights_dict is a list of XLATensor2
        """

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
                updated[key] = transform(value, self.config.num_attention_heads)
            if "k_proj" in key:
                updated[key] = transform(
                    value, self.config.num_key_value_heads or self.config.num_attention_heads
                )
        res = super().convert_hf_weights(updated)
        res["freqs_cis"] = self.freqs_cis
        return res
