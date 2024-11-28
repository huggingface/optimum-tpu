
from jetstream_pt.third_party.mixtral import config as mixtral_config
from jetstream_pt.third_party.mixtral.model import Transformer
from transformers import GenerationConfig, GenerationMixin, MixtralConfig


class MixtralConfigHf(MixtralConfig, mixtral_config.ModelArgs):
    """This class is used to support both the HF MixtralConfig and the Jetstream Pytorch ModelArgs at the same time.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = self.max_position_embeddings
        self.n_layer = self.num_hidden_layers
        self.n_head = self.num_attention_heads
        self.dim = self.hidden_size
        self.n_local_heads = self.num_local_experts or self.num_attention_heads
        self.num_activated_experts = self.num_experts_per_tok
        self.__post_init__()

class MixtralModelHf(Transformer, GenerationMixin):
    """Transformer module that uses HF MixtralConfig instead of Jetstream Pytorch MixtralConfig + device.
    """

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
