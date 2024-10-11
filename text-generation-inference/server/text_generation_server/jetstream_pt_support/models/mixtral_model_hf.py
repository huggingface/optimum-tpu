
from jetstream_pt.third_party.mixtral import config as mixtral_config
from jetstream_pt.third_party.mixtral.model import Transformer
from transformers import GenerationConfig, GenerationMixin, MixtralConfig


class MixtralModelHf(Transformer, GenerationMixin):
    """Transformer module that uses HF MixtralConfig instead of Jetstream Pytorch MixtralConfig + device.
    """

    def __init__(
        self,
        config: MixtralConfig,
        device,
        env,
    ):
        self.config = config
        self.generation_config = GenerationConfig.from_model_config(config)

        args = mixtral_config.ModelArgs(
            block_size=config.max_position_embeddings,
            vocab_size=config.vocab_size,
            n_layer=config.num_hidden_layers,
            n_head=config.num_attention_heads,
            dim=config.hidden_size,
            intermediate_size=config.intermediate_size,
            n_local_heads=config.num_local_experts or config.num_attention_heads,
            num_activated_experts=config.num_experts_per_tok,
            device=device,
        )
        super().__init__(args, env)


    @classmethod
    def from_config(cls, config, env):
        device = "meta"
        model = cls(config, device, env)
        return model
