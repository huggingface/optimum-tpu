import dataclasses
import math
from contextlib import contextmanager
from functools import partial

import torch
from jetstream_pt.third_party.llama import model_exportable
from jetstream_pt.third_party.llama.model_exportable import Transformer, model_args
from transformers import GenerationConfig, GenerationMixin, LlamaConfig


# TODO: it would be better to have RoPE scaling code in Jetstream Pytorch, but until that is not done,
# we add it here. Note that this is the reason why we define a new class RopeScalingArgs, instead of using the
# config from transformers.
@dataclasses.dataclass
class RopeScalingArgs:
  """Rope scaling configuration parameters."""

  factor: float = 8.0
  low_freq_factor: float = 1.0
  high_freq_factor: float = 4.0
  original_max_position_embeddings: int = 8192


def apply_scaling(freqs: torch.Tensor, config: RopeScalingArgs):
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
    rope_scaling_config: RopeScalingArgs = None,
):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    if rope_scaling_config is not None:
        freqs = apply_scaling(freqs, rope_scaling_config)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


@contextmanager
def patch_precompute_freqs_cis(rope_scaling_js: RopeScalingArgs):
    # NOTE: This is a workaround to pass the rope scaling configuration when it is called in the original
    # Jetstream/Pytorch model. The function is monkey-patched to include the rope scaling configuration.
    original_precompute_freqs_cis = model_exportable.precompute_freqs_cis
    precompute_freqs_cis_partial = partial(precompute_freqs_cis, rope_scaling_config=rope_scaling_js)
    model_exportable.precompute_freqs_cis = precompute_freqs_cis_partial

    yield

    # Original function is restored.
    model_exportable.precompute_freqs_cis = original_precompute_freqs_cis


class TransformerHf(Transformer, GenerationMixin):
    """Transformer module that uses HF LlamaConfig instead of Jetstream Pytorch ModelArgs + device.

    Note that this class also derives from GenerationMixin, so that we can use its methods.
    """

    def __init__(
        self,
        config: LlamaConfig,
        device,
        env,
    ):
        self.config = config
        self.generation_config = GenerationConfig.from_model_config(config)

        # NOTE: these parameters are deduced from the config's intermediate_size and hidden_size, so to be compatible
        # with the original Jestream/Pytorch model.
        ffn_dim_multiplier = config.intermediate_size / int(8 * config.hidden_size / 3)
        multiple_of = 1

        if config.mlp_bias:
            raise ValueError("MLP bias is not supported in the on Jetstream Pytorch."
                             + "If your model requires it, you can open an issue.")

        rope_scaling_js = None
        rope_scaling = config.rope_scaling
        # The original Llama2 and Llama3 models do not have rope scaling configuration, while newer models do.
        if rope_scaling is not None:
            # Some models use "type" instead of "rope_type" in the configuration for historical reasons.
            rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))
            if rope_type != "llama3":
                raise ValueError(f"Unsupported rope type {rope_type} in rope scaling configuration.")

            rope_scaling_js = RopeScalingArgs(
                factor=rope_scaling["factor"],
                low_freq_factor=rope_scaling["low_freq_factor"],
                high_freq_factor=rope_scaling["high_freq_factor"],
                original_max_position_embeddings=rope_scaling["original_max_position_embeddings"],
            )

        args = model_args.ModelArgs(
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
        args.device = device

        with patch_precompute_freqs_cis(rope_scaling_js):
            super().__init__(args, env)


    @classmethod
    def from_config(cls, config, env):
        device = "meta"
        model = cls(config, device, env)
        return model
