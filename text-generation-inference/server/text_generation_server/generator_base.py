from abc import ABC
from typing import List, Optional, Tuple

from .pb.generate_pb2 import (
    Batch,
    CachedBatch,
    Generation,
    InfoResponse,
)


class Generator(ABC):
    """An abstract class to represent the workhorse behind TextGenerationService.

    Ideally, it should not rely on protobuf constructs, but in a first step it does.
    Implementations would typically need a model and a tokenizer to implement the Generator methods.
    """

    @property
    def info(self) -> InfoResponse:
        """This should simply return the expected InfoResponse"""
        raise NotImplementedError

    def warmup(self, batch: Batch) -> int:
        """Verify if the hardware can support the target load.

        Args:
            batch (`Batch`):
                A batch corresponding to the maximum number of concurrent requests.

        Return:
            The maximum number of tokens the model supports.
        """
        raise NotImplementedError

    def prefill(self, batch: Batch) -> Tuple[List[Generation], CachedBatch]:
        """Prefill is called whenever new requests need to be added.

        When this method returns successfully, a decode method will follow
        with both the current and newly prefilled batch(es).

        Args:
            batch (`Batch`):
                A batch containing the new requests.

        Return:
            A list of `Generation` for each request and a `CachedBatch` containing all pending requests.
        """
        raise NotImplementedError

    def decode(self, batches: List[Batch]) -> Tuple[List[Generation], CachedBatch]:
        """Decode after a prefill or another decode."""
        raise NotImplementedError

    def filter(self, batch_id: int, request_ids: List[int]) -> CachedBatch:
        """Remove requests that are not listed from the specified batch"""
        raise NotImplementedError

    def clear(self, batch_id: Optional[int] = None):
        """Remove all requests from the generator"""
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, model_id: str, revision: Optional[str]):
        """Factory method "a la transformers" """
        raise NotImplementedError
