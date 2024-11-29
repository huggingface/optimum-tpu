from jetstream_pt.third_party.mixtral import config as mixtral_config
from jetstream_pt.third_party.mixtral.model import Transformer
from transformers import GenerationConfig, GenerationMixin, MixtralConfig


class MixtralConfigHf(MixtralConfig, mixtral_config.ModelArgs):
    """This class is used to support both the HF MixtralConfig and the Jetstream Pytorch ModelArgs at the same time."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__post_init__()

    @property
    def block_size(self):
        return self.max_position_embeddings

    @property
    def n_layer(self):
        return self.num_hidden_layers

    @property
    def n_head(self):
        return self.num_attention_heads

    @property
    def dim(self):
        return self.hidden_size

    @property
    def n_local_heads(self):
        return self.num_local_experts or self.num_attention_heads

    @property
    def num_activated_experts(self):
        return self.num_experts_per_tok


class MixtralModelHf(Transformer, GenerationMixin):
    """Transformer module that uses HF MixtralConfig instead of Jetstream Pytorch MixtralConfig + device."""

    def __init__(
        self,
        config: MixtralConfig,
        device,
        env,
    ):
        self.generation_config = GenerationConfig.from_model_config(config)
        args = MixtralConfigHf(**config.to_dict())
        args.device = device
        super().__init__(args, env)


    @classmethod
    def from_config(cls, config, env):
        device = "meta"
        model = cls(config, device, env)
        return model
