import os
import time
from typing import Optional

from loguru import logger

from .modeling import TpuModelForCausalLM


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
    model_path_or_id: str,
    revision: Optional[str] = None,
) -> str:
    """Fetch a model to local cache.

    Args:
        model_path_or_id (`str`):
            The *model_id* of a model on the HuggingFace hub or the path to a local model.
        revision (`Optional[str]`, defaults to `None`):
            The revision of the model on the HuggingFace hub.

    Returns:
        Model ID or path of the model available in cache.
    """
    if os.path.isdir(model_path_or_id):
        if revision is not None:
            logger.warning("Revision {} ignored for local model at {}".format(revision, model_path_or_id))
        return model_path_or_id

    # This will allow to set config to update specific config such as
    # batch_size and sequence_length.
    export_kwargs = get_export_kwargs_from_env()
    # Call from_pretrained so to download to local cache
    start = time.time()
    TpuModelForCausalLM.from_pretrained(model_path_or_id, **export_kwargs)
    end = time.time()
    logger.info(f"Model successfully fetched in {end - start:.2f} s.")

    return model_path_or_id
