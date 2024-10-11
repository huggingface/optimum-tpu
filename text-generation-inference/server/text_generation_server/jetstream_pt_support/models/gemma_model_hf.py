
from jetstream_pt.third_party.gemma import config as gemma_config
from jetstream_pt.third_party.gemma.model import GemmaModel
from transformers import GemmaConfig, GenerationConfig, GenerationMixin


class GemmaModelHf(GemmaModel, GenerationMixin):
    """Transformer module that uses HF GemmaConfig instead of Jetstream Pytorch GemmaConfig + device.

    Note that this class also derives from GenerationMixin, so that we can use its methods.
    """

    def __init__(
        self,
        config: GemmaConfig,
        device,
        env,
    ):
        self.config = config
        self.generation_config = GenerationConfig.from_model_config(config)

        args = gemma_config.GemmaConfig(
            vocab_size=config.vocab_size,
            max_position_embeddings=config.max_position_embeddings,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            head_dim=config.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            dtype="bfloat16",
            quant=False, # No quantization support for now
            tokenizer=None,
        )

        args.device = device
        super().__init__(args, env)


    @classmethod
    def from_config(cls, config, env):
        device = "meta"
        model = cls(config, device, env)
        return model
