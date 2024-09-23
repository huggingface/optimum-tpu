
from jetstream_pt.third_party.llama.model_exportable import Transformer, model_args
from transformers import GenerationConfig, GenerationMixin, LlamaConfig


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
        super().__init__(args, env)


    @classmethod
    def from_config(cls, config, env):
        device = "meta"
        model = cls(config, device, env)
        return model
