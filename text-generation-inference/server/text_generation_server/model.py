import os
import time
from typing import Optional

from huggingface_hub import snapshot_download
from loguru import logger

from .modelling import TpuModelForCausalLM


def get_export_kwargs_from_env():
    batch_size = os.environ.get("HF_BATCH_SIZE", None)
    if batch_size is not None:
        batch_size = int(batch_size)
    sequence_length = os.environ.get("HF_SEQUENCE_LENGTH", None)
    if sequence_length is not None:
        sequence_length = int(sequence_length)
    return {
        "task": "text-generation",
        "batch_size": batch_size,
        "sequence_length": sequence_length,
    }


def fetch_model(
    model_id: str,
    revision: Optional[str] = None,
) -> str:
    """Fetch a model to local cache.

    Args:
        model_id (`str`):
            The *model_id* of a model on the HuggingFace hub or the path to a local model.
        revision (`Optional[str]`, defaults to `None`):
            The revision of the model on the HuggingFace hub.

    Returns:
        Local folder path (string) of the model.
    """

    # TODO: add check to verify this is a TPU VM
    if os.path.isdir(model_id):
        if revision is not None:
            logger.warning("Revision {} ignored for local model at {}".format(revision, model_id))
        return model_id

    # This will be retrieving the model snapshot and cache it.
    start = time.time()
    logger.info(f"Fetching revision {revision} of model {model_id}.")
    model_path = snapshot_download(model_id, revision=revision)
    end = time.time()
    logger.info(f"Model successfully fetched in {end - start:.2f} s.")
    # This will allow to set config to update specific config such as
    # batch_size and sequence_length.
    export_kwargs = get_export_kwargs_from_env()
    model = TpuModelForCausalLM.from_pretrained(model_path, **export_kwargs)
    model.save_pretrained(model_path)

    return model_path
