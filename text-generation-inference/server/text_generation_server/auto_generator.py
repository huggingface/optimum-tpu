from .generator_base import Generator
from .jetstream_pt_support import model_can_use_jetstream_pt


class AutoGenerator:

    @staticmethod
    def from_pretrained(
        model_path: str, revision: str, max_batch_size: int, max_sequence_length: int
    ) -> Generator:
        """Instantiate a Generator for TPU using Jetstream Pytorch or Pytorch/XLA.

        Args:
            model_path (`str`):
                The path to a local model. This path must also contain a Tokenizer.
            revision (`str`):
                The revision of the model.
            max_batch_size (`int`):
                The maximum batch size.
            max_sequence_length (`int`):
                The maximum sequence length.

        Returns:
            A TpuGenerator.
        """
        if check(model_path):
            from .jetstream_pt_support.generator import TpuGeneratorJetStream
            return TpuGeneratorJetStream.from_pretrained(
                model_path, revision=revision, max_batch_size=max_batch_size, max_sequence_length=max_sequence_length
            )
        else:
            from .generator import TpuGenerator
            return TpuGenerator.from_pretrained(
                model_path, revision=revision, max_batch_size=max_batch_size, max_sequence_length=max_sequence_length
            )
