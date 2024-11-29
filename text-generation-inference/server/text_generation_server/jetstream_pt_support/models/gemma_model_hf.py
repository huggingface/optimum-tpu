
from jetstream_pt.third_party.gemma import config as gemma_config
from jetstream_pt.third_party.gemma.model import GemmaModel
from transformers import GemmaConfig, GenerationConfig, GenerationMixin


class GemmaConfigHf(GemmaConfig, gemma_config.GemmaConfig):
    """This class is used to support both the HF GemmaConfig and the Jetstream Pytorch GemmaConfig at the same time.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = None


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
        self.generation_config = GenerationConfig.from_model_config(config)
        args = GemmaConfigHf(**config.to_dict())
        args.device = device
        super().__init__(args, env)


    @classmethod
    def from_config(cls, config, env):
        device = "meta"
        model = cls(config, device, env)
        return model
